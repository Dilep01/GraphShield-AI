import json
import math
from pathlib import Path

import duckdb
import pytest

from graphshield.features import FEATURES,build_features
from graphshield.final_features import build_final_features
from graphshield.final_evaluation import verify_seal,final_source
from graphshield.monitoring import RequestMetrics,workflow_health
from graphshield.pipeline import literal,sha256,write_json
from test_phase2 import source_parquet
from test_phase6 import service6,client6


def test_final_transform_matches_frozen_development_transform(project):
    source=project/'source.parquet'
    with duckdb.connect() as con:
        source_parquet(con,source)
        build_final_features(con,source,project/'final.parquet',2,26)
        actual=con.execute('SELECT '+','.join(FEATURES)+' FROM features').fetchall()
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT * REPLACE ('validation' AS split) FROM read_parquet({literal(source)}) WHERE split='test' UNION ALL SELECT * FROM read_parquet({literal(source)}) WHERE split!='test') TO {literal(project/'reference.parquet')} (FORMAT PARQUET)")
        build_features(con,project/'reference.parquet',project/'reference-features.parquet',2,27)
        expected=con.execute('SELECT '+','.join(FEATURES)+" FROM features WHERE transaction_id='f'").fetchall()
    assert actual==expected


def test_final_history_excludes_same_step_and_future_and_includes_prior_test(project):
    result=[]
    for index,amount in enumerate((50,999999)):
        with duckdb.connect() as con:
            source=project/f'source{index}.parquet'
            source_parquet(con,project/f'original{index}.parquet',100)
            con.execute("INSERT INTO tx VALUES ('g',7,27,'TRANSFER',?,'A','B','test'),('h',8,28,'TRANSFER',10,'A','B','test'),('i',9,30,'TRANSFER',?,'A','B','test')",[amount,amount])
            con.execute(f'COPY tx TO {literal(source)} (FORMAT PARQUET)')
            build_final_features(con,source,project/f'final{index}.parquet',2,26)
            result.append(con.execute("SELECT sender_log_out_count,sender_log_out_amount FROM features WHERE transaction_id='f'").fetchone())
            after=con.execute("SELECT sender_log_out_count FROM features WHERE transaction_id='h'").fetchone()[0]
            assert after==pytest.approx(math.log1p(5))
    assert result[0]==result[1]


def test_final_feature_split_mismatch_fails(project):
    with duckdb.connect() as con:
        source_parquet(con,project/'source.parquet')
        with pytest.raises(ValueError,match='boundaries'):
            build_final_features(con,project/'source.parquet',project/'out.parquet',2,27)


def test_final_seal_required_and_tampering_refused(project):
    with pytest.raises(FileNotFoundError):verify_seal(project)
    write_json(project/'protocol.json',{'selected':'frozen'})
    (project/'predictions.json').write_text('[]')
    write_json(project/'prediction_seal.json',{'protocol_sha256':sha256(project/'protocol.json'),
        'files':{'predictions.json':sha256(project/'predictions.json')}})
    verify_seal(project)
    (project/'predictions.json').write_text('[1]')
    with pytest.raises(ValueError,match='Checksum'):verify_seal(project)


def test_final_input_gate_does_not_accept_development_or_paths(project):
    with pytest.raises(ValueError,match='explicit final'):
        final_source(project,{'world_id':'world_x','split':'train'},'observed','transactions',{})
    with pytest.raises(ValueError):
        final_source(project,{'world_id':'../escape','split':'test_known'},'observed','transactions',{})


def test_monitoring_route_templates_and_validation_errors(client6):
    client6.get('/api/v1/cases/private-case-id')
    client6.post('/api/v1/worlds',json={'world_id':'invalid','label':'private note'})
    response=client6.get('/api/v1/monitoring')
    assert response.status_code==200
    body=response.json()
    text=json.dumps(body)
    assert 'private-case-id' not in text and 'private note' not in text
    rows=body['requests']['routes']
    assert any(r['route']=='/api/v1/cases/{case_id}' and r['errors_4xx']==1 for r in rows)
    assert any(r['route']=='/api/v1/worlds' and r['errors_4xx']==1 for r in rows)
    assert body['workflow']['status']=='ok'


def test_monitoring_inactive_replay_requires_resume(service6):
    job=service6.request_replay('world_validation')
    service6.store.replay_status(job['replay_id'],'running')
    health=workflow_health(service6)
    assert health['status']=='attention' and health['issues'][0]['condition']=='replay_not_active'
    service6.run_replay(job['replay_id'])
    assert workflow_health(service6)['status']=='ok'


def test_request_metrics_bucket_totals():
    metrics=RequestMetrics()
    for latency,status in ((1,200),(30,409),(15000,500)):
        metrics.record('POST','/api/v1/score',status,latency)
    row=metrics.snapshot()['routes'][0]
    assert row['requests']==sum(row['latency_buckets'])==3
    assert row['errors_4xx']==row['errors_5xx']==1
