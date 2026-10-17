"""Bounded process counters and durable workflow health; no fraud labels required."""
import logging
import threading
import time
from collections import Counter


class RequestMetrics:
    BOUNDS=(5,25,100,500,2000,10000)

    def __init__(self):
        self.started=time.monotonic()
        self.lock=threading.Lock()
        self.routes={}

    def record(self, method, route, status, milliseconds):
        # Caller passes a route template, never IDs, query strings, bodies or notes.
        with self.lock:
            key=(method,route)
            row=self.routes.setdefault(key,{'requests':0,'errors_4xx':0,'errors_5xx':0,'total_ms':0.,'max_ms':0.,'latency_buckets':[0]*7})
            row['requests']+=1
            row['errors_4xx']+=int(400<=status<500)
            row['errors_5xx']+=int(status>=500)
            row['total_ms']+=milliseconds
            row['max_ms']=max(row['max_ms'],milliseconds)
            bucket=next((i for i,upper in enumerate(self.BOUNDS) if milliseconds<=upper),6)
            row['latency_buckets'][bucket]+=1
        logging.getLogger('graphshield.requests').info('%s %s status=%s duration_ms=%.2f',method,route,status,milliseconds)

    def snapshot(self):
        with self.lock:
            return {'uptime_seconds':time.monotonic()-self.started,'scope':'Since this application process started',
                    'latency_bucket_upper_ms':[*self.BOUNDS,None],
                    'routes':[dict(method=method,route=route,**{k:v for k,v in row.items() if k!='total_ms'},
                                   mean_ms=row['total_ms']/row['requests']) for (method,route),row in sorted(self.routes.items())]}


def workflow_health(service):
    with service.store.connection() as db:
        cases={r[0]:r[1] for r in db.execute('SELECT status,count(*) FROM cases GROUP BY status')}
        jobs=[dict(r) for r in db.execute('SELECT replay_id,status,completed_snapshots,total_snapshots,error FROM replays')]
        score_rows=db.execute('''SELECT c.model,r.bundle_version,CAST(min(c.score*10,9) AS INTEGER) AS bucket,count(*)
            FROM cases c JOIN replays r ON c.replay_id=r.replay_id GROUP BY c.model,r.bundle_version,bucket''').fetchall()
    with service.lock:
        active=set(service.active)
    issues=[]
    for job in jobs:
        if job['status']=='failed':issues.append({'replay_id':job['replay_id'],'condition':'replay_failed','action':'Inspect the replay error, repair its cause, then request the same replay to resume'})
        elif job['status'] in ('pending','running') and job['replay_id'] not in active:
            issues.append({'replay_id':job['replay_id'],'condition':'replay_not_active','action':'Request the same replay to resume committed checkpoints'})
    histograms={}
    for model,version,bucket,count in score_rows:
        histograms.setdefault((model,version),[0]*10)[bucket]+=count
    return {'status':'attention' if issues else 'ok','case_status_counts':cases,
            'replay_status_counts':dict(Counter(j['status'] for j in jobs)),'active_replays':len(active),
            'issues':issues,'database_bytes':service.store.path.stat().st_size,
            'saved_case_score_histograms':[{'model':m,'bundle_version':v,'counts':counts} for (m,v),counts in sorted(histograms.items())],
            'interpretation':'Operational counts and saved-case score distributions only; not accuracy, calibration or a drift test',
            'reviewer_identity_authenticated':False}
