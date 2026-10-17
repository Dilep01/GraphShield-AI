"""Local SQLite event history, replay checkpoints and separate reviewer decisions."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .graph_rules import jaccard
from .paths import ROOT, inside_root


class Conflict(ValueError):
    pass


class Missing(KeyError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path, root=ROOT):
        self.path = inside_root(path, Path(root))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata(version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS worlds(
                    world_id TEXT PRIMARY KEY, label TEXT NOT NULL, provenance TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, watermark INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS transactions(
                    world_id TEXT NOT NULL REFERENCES worlds, transaction_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL, receiver_id TEXT NOT NULL, amount TEXT NOT NULL,
                    step INTEGER NOT NULL, available_step INTEGER NOT NULL, type TEXT NOT NULL,
                    PRIMARY KEY(world_id,transaction_id));
                CREATE TABLE IF NOT EXISTS observations(
                    world_id TEXT NOT NULL REFERENCES worlds, observation_id TEXT NOT NULL,
                    account_id TEXT NOT NULL, device_id TEXT NOT NULL, step INTEGER NOT NULL,
                    available_step INTEGER NOT NULL, PRIMARY KEY(world_id,observation_id));
                CREATE INDEX IF NOT EXISTS tx_history ON transactions(world_id,step,available_step);
                CREATE INDEX IF NOT EXISTS obs_history ON observations(world_id,step,available_step);
                CREATE TABLE IF NOT EXISTS receipts(
                    world_id TEXT NOT NULL REFERENCES worlds, receipt_key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(world_id,receipt_key));
                CREATE TABLE IF NOT EXISTS replays(
                    replay_id TEXT PRIMARY KEY, world_id TEXT NOT NULL REFERENCES worlds,
                    model TEXT NOT NULL, policy TEXT NOT NULL, bundle_version TEXT NOT NULL,
                    status TEXT NOT NULL, completed_snapshots INTEGER NOT NULL DEFAULT 0,
                    total_snapshots INTEGER NOT NULL, last_cutoff INTEGER, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots(
                    replay_id TEXT NOT NULL REFERENCES replays, cutoff INTEGER NOT NULL,
                    snapshot_id TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(replay_id,cutoff));
                CREATE TABLE IF NOT EXISTS cases(
                    case_id TEXT PRIMARY KEY, replay_id TEXT NOT NULL REFERENCES replays,
                    world_id TEXT NOT NULL, model TEXT NOT NULL, policy TEXT NOT NULL,
                    cutoff INTEGER NOT NULL, score REAL NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new', review_version INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS reviews(
                    case_id TEXT NOT NULL REFERENCES cases, request_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
                    note TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(case_id,request_id));
            ''')
            row = db.execute('SELECT version FROM metadata').fetchone()
            if row is None:
                db.execute('INSERT INTO metadata VALUES(1)')
            elif row['version'] != 1:
                raise ValueError('Unsupported local database version')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def register_world(self, world_id, label, provenance):
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO worlds(world_id,label,provenance) VALUES(?,?,?)',
                       (world_id,label,provenance))

    def world(self, world_id, db=None):
        if db is None:
            with self.connection() as connection:
                return self.world(world_id, connection)
        row = db.execute('SELECT * FROM worlds WHERE world_id=?',(world_id,)).fetchone()
        if row is None:
            raise Missing('Unknown network')
        return dict(row)

    def worlds(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute('''SELECT w.*,
                (SELECT count(*) FROM transactions t WHERE t.world_id=w.world_id) AS transactions,
                (SELECT count(*) FROM observations o WHERE o.world_id=w.world_id) AS device_observations
                FROM worlds w ORDER BY w.label''')]

    def ingest(self, world_id, receipt_key, transactions, observations):
        payload = {'transactions':sorted(transactions,key=lambda r:r['transaction_id']),
                   'observations':sorted(observations,key=lambda r:r['observation_id'])}
        digest = fingerprint(payload)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            world = self.world(world_id,db)
            prior = db.execute('SELECT * FROM receipts WHERE world_id=? AND receipt_key=?',(world_id,receipt_key)).fetchone()
            if prior:
                if prior['payload_hash'] != digest:
                    raise Conflict('Idempotency key was already used with different events')
                return dict(json.loads(prior['response']), replayed=True)
            inserted, duplicates = 0, 0
            definitions = (
                ('transactions','transaction_id',('transaction_id','sender_id','receiver_id','amount','step','available_step','type'), payload['transactions']),
                ('observations','observation_id',('observation_id','account_id','device_id','step','available_step'), payload['observations']),
            )
            for table, id_column, columns, rows in definitions:
                for row in rows:
                    values = tuple(row[c] for c in columns)
                    old = db.execute(f"SELECT {','.join(columns)} FROM {table} WHERE world_id=? AND {id_column}=?",
                                     (world_id,row[id_column])).fetchone()
                    if old:
                        if tuple(old) != values:
                            raise Conflict(f'Conflicting event ID: {row[id_column]}')
                        duplicates += 1
                    else:
                        if row['available_step'] < world['watermark']:
                            raise Conflict('New event predates committed history; use its actual later availability step')
                        db.execute(f"INSERT INTO {table}(world_id,{','.join(columns)}) VALUES({','.join('?' for _ in range(len(columns)+1))})",
                                   (world_id,*values))
                        inserted += 1
            revision = world['revision'] + int(inserted > 0)
            db.execute('UPDATE worlds SET revision=? WHERE world_id=?',(revision,world_id))
            result = {'inserted':inserted,'duplicates':duplicates,'history_revision':revision,'replayed':False}
            db.execute('INSERT INTO receipts VALUES(?,?,?,?)',(world_id,receipt_key,digest,canonical(result)))
            return result

    def history(self, world_id, cutoff, window):
        with self.connection() as db:
            db.execute('BEGIN')
            world = self.world(world_id,db)
            lower = max(1,cutoff-window)
            args = (world_id,lower,cutoff,cutoff)
            tx = [tuple(r) for r in db.execute('''SELECT transaction_id,sender_id,receiver_id,amount,step,available_step
                FROM transactions WHERE world_id=? AND step>=? AND step<? AND available_step<? ORDER BY transaction_id''',args)]
            obs = [tuple(r) for r in db.execute('''SELECT observation_id,account_id,device_id,step,available_step
                FROM observations WHERE world_id=? AND step>=? AND step<? AND available_step<? ORDER BY observation_id''',args)]
            return world,tx,obs

    def ensure_replay(self, world_id, model, policy, bundle_version, total):
        replay_id = 'replay_' + fingerprint([world_id,model,policy,bundle_version])[:24]
        with self.connection() as db:
            self.world(world_id,db)
            db.execute('''INSERT OR IGNORE INTO replays(replay_id,world_id,model,policy,bundle_version,status,
                total_snapshots,created_at,updated_at) VALUES(?,?,?,?,?,'pending',?,?,?)''',
                       (replay_id,world_id,model,policy,bundle_version,total,now(),now()))
        return self.replay(replay_id)

    def replay(self, replay_id):
        with self.connection() as db:
            row = db.execute('''SELECT r.*,(SELECT count(*) FROM cases c WHERE c.replay_id=r.replay_id) AS alerts
                FROM replays r WHERE replay_id=?''',(replay_id,)).fetchone()
            if row is None:
                raise Missing('Unknown replay')
            return dict(row)

    def replays(self):
        with self.connection() as db:
            ids = [r[0] for r in db.execute('SELECT replay_id FROM replays ORDER BY created_at DESC')]
        return [self.replay(i) for i in ids]

    def replay_status(self, replay_id, status, error=None):
        with self.connection() as db:
            db.execute('UPDATE replays SET status=?,error=?,updated_at=? WHERE replay_id=?',
                       (status,error,now(),replay_id))

    def committed(self, replay_id, cutoff):
        with self.connection() as db:
            return db.execute('SELECT 1 FROM snapshots WHERE replay_id=? AND cutoff=?',(replay_id,cutoff)).fetchone() is not None

    def commit_snapshot(self, replay_id, response, duplicate_jaccard, capacity):
        """Atomically checkpoint decisions and immutable cases; resume cannot duplicate them."""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute('SELECT * FROM replays WHERE replay_id=?',(replay_id,)).fetchone()
            if not job:
                raise Missing('Unknown replay')
            cutoff = response['cutoff_step']
            previous = db.execute('SELECT response FROM snapshots WHERE replay_id=? AND cutoff=?',(replay_id,cutoff)).fetchone()
            if previous:
                return json.loads(previous[0])
            if job['last_cutoff'] is not None and cutoff <= job['last_cutoff']:
                raise Conflict('Replay snapshots must be committed in increasing cutoff order')
            world = self.world(job['world_id'],db)
            if world['revision'] != response['history_revision']:
                raise Conflict('History changed while scoring; retry this snapshot')
            emitted = [json.loads(r[0]) for r in db.execute('SELECT payload FROM cases WHERE replay_id=? ORDER BY cutoff,case_id',(replay_id,))]
            decisions, new_cases = [], []
            for candidate in sorted(response['candidates'],key=lambda c:(-c['score'],c['candidate_id'])):
                duplicate = None
                if candidate['score'] < response['threshold']:
                    decision = 'below_score'
                else:
                    duplicate = next((old['case_id'] for old in emitted if jaccard(old['members'],candidate['members']) >= duplicate_jaccard),None)
                    if duplicate:
                        decision = 'duplicate'
                    elif len(new_cases) >= capacity:
                        decision = 'capacity'
                    else:
                        decision = 'alert'
                        case_id = 'case_' + fingerprint([replay_id,candidate['candidate_id']])[:24]
                        case = dict(candidate,case_id=case_id,replay_id=replay_id,world_id=job['world_id'],
                                    model=job['model'],policy=job['policy'],bundle_version=job['bundle_version'],
                                    snapshot_id=response['snapshot_id'],cutoff_step=cutoff,window_start=response['window_start'],
                                    history_revision=response['history_revision'],threshold=response['threshold'],
                                    provenance=response['provenance'],score_semantics=response['score_semantics'],
                                    embedding_mode=response['embedding_mode'])
                        db.execute('INSERT INTO cases(case_id,replay_id,world_id,model,policy,cutoff,score,payload) VALUES(?,?,?,?,?,?,?,?)',
                                   (case_id,replay_id,job['world_id'],job['model'],job['policy'],cutoff,case['score'],canonical(case)))
                        emitted.append(case)
                        new_cases.append(case_id)
                decisions.append({'candidate_id':candidate['candidate_id'],'decision':decision,'duplicate_of':duplicate})
            result = {k:v for k,v in response.items() if k != 'candidates'}
            result.update(decisions=decisions,case_ids=new_cases,candidate_count=len(response['candidates']))
            db.execute('INSERT INTO snapshots VALUES(?,?,?,?)',(replay_id,cutoff,response['snapshot_id'],canonical(result)))
            db.execute('UPDATE worlds SET watermark=max(watermark,?) WHERE world_id=?',(cutoff,job['world_id']))
            db.execute('UPDATE replays SET completed_snapshots=completed_snapshots+1,last_cutoff=?,updated_at=? WHERE replay_id=?',
                       (cutoff,now(),replay_id))
            return result

    def list_cases(self, world_id=None, model=None, status=None, replay_id=None, limit=50, offset=0, policy=None, bundle_version=None):
        clauses, args = [], []
        for column,value in (('world_id',world_id),('model',model),('status',status),('replay_id',replay_id),('policy',policy)):
            if value is not None:
                clauses.append(column+'=?')
                args.append(value)
        if bundle_version is not None:
            clauses.append('replay_id IN (SELECT replay_id FROM replays WHERE bundle_version=?)')
            args.append(bundle_version)
        where = ' WHERE '+' AND '.join(clauses) if clauses else ''
        with self.connection() as db:
            count = db.execute('SELECT count(*) FROM cases'+where,args).fetchone()[0]
            rows = db.execute('SELECT * FROM cases'+where+' ORDER BY score DESC,case_id LIMIT ? OFFSET ?',(*args,limit,offset)).fetchall()
            result = []
            for row in rows:
                payload = json.loads(row['payload'])
                result.append({k:payload[k] for k in ('case_id','world_id','model','policy','cutoff_step','score','members','reasons','snapshot_id')}
                              | {'status':row['status'],'review_version':row['review_version']})
            return {'total':count,'items':result,'limit':limit,'offset':offset}

    def case(self, case_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM cases WHERE case_id=?',(case_id,)).fetchone()
            if row is None:
                raise Missing('Unknown case')
            history = [dict(r) for r in db.execute('SELECT version,status,note,created_at FROM reviews WHERE case_id=? ORDER BY version',(case_id,))]
            return dict(json.loads(row['payload']),status=row['status'],review_version=row['review_version'],reviews=history)

    def review(self, case_id, request_id, expected_version, status, note):
        digest = fingerprint([expected_version,status,note])
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM reviews WHERE case_id=? AND request_id=?',(case_id,request_id)).fetchone()
            if old:
                if old['payload_hash'] != digest:
                    raise Conflict('Review request ID was already used with different content')
                return {'version':old['version'],'status':old['status'],'replayed':True}
            case = db.execute('SELECT review_version FROM cases WHERE case_id=?',(case_id,)).fetchone()
            if case is None:
                raise Missing('Unknown case')
            if case[0] != expected_version:
                raise Conflict('This case has a newer review; refresh before saving')
            version = expected_version+1
            db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)',(case_id,request_id,digest,version,status,note,now()))
            db.execute('UPDATE cases SET status=?,review_version=? WHERE case_id=?',(status,version,case_id))
            return {'version':version,'status':status,'replayed':False}
