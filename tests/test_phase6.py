import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graphshield.api import create_app
from graphshield.pipeline import sha256, write_json
from graphshield.service_store import Conflict, Store
from graphshield.serving import Bundle, InvestigationService
from test_phase4 import cycle, fixture_source


@pytest.fixture
def service6(project):
    data,source=fixture_source(project)
    rules=json.loads((Path(__file__).resolve().parents[1]/'config/phase4.json').read_text())
    worlds=[]
    for world in source['worlds'][:2]:
        observed={}
        for table in ('transactions','device_observations'):
            path=data/'observed'/world['split']/world['world_id']/f'{table}.parquet'
            observed[table]={'path':str(path.relative_to(project)),'sha256':sha256(path)}
        worlds.append(dict(world,label=world['split'],observed=observed))
    folder=project/'artifacts/phase6/bundles/fixture'
    folder.mkdir(parents=True)
    write_json(folder/'bundle.json',{'status':'complete','schema_version':1,'final_test_included':False,
        'name':'fixture','learned_run':'fixture-not-a-trained-model','files':{},'source_sha256':{},
        'embedding_cache':[],'config':{'threads':1},'rule_config':rules,'worlds':worlds,'cutoffs':[3,10,20],
        'default_model':'graph_only','default_world':'world_validation','models':{
            'graph_only':{'label':'Rules','threshold':.55,'selected_on_tuning':True}},'provenance':'Synthetic fixture'})
    return InvestigationService(Bundle('fixture',project),Store('local.sqlite',project))


@pytest.fixture
def client6(service6,project):
    with TestClient(create_app(root=project,service=service6)) as client:
        yield client


def events(prefix='demo'):
    return [dict(zip(('transaction_id','sender_id','receiver_id','amount','step','available_step'),
                     (r[0],r[1],r[2],'100.00',r[4],r[5]))) for r in cycle(prefix)]


def create_demo(client):
    assert client.post('/api/v1/worlds',json={'world_id':'demo_test','label':'Test network'}).status_code==201


def test_atomic_idempotent_ingestion_and_conflicts(client6,service6):
    create_demo(client6)
    body={'world_id':'demo_test','idempotency_key':'batch1','transactions':events()}
    first=client6.post('/api/v1/ingest',json=body)
    assert first.status_code==200 and first.json()['inserted']==10
    assert client6.post('/api/v1/ingest',json=body).json()['replayed']
    changed=dict(body,idempotency_key='batch2')
    assert client6.post('/api/v1/ingest',json=changed).json()['duplicates']==10
    changed['transactions']=[dict(events()[0],transaction_id='new'),dict(events()[1],amount='101.00')]
    assert client6.post('/api/v1/ingest',json=changed).status_code==409
    world,tx,_=service6.store.history('demo_test',10,168)
    assert len(tx)==10 and world['revision']==1
    body['transactions'][0]['amount']='1.00'
    assert client6.post('/api/v1/ingest',json=body).status_code==409


@pytest.mark.parametrize('change',[{'amount':'NaN'},{'amount':'-1'},{'amount':'0.001'},
                                   {'step':True},{'available_step':1},{'is_fraud':True}])
def test_ingestion_rejects_bad_contracts(client6,change):
    create_demo(client6)
    body={'world_id':'demo_test','idempotency_key':'bad','transactions':[dict(events()[0],**change)]}
    assert client6.post('/api/v1/ingest',json=body).status_code==422


def test_scoring_is_read_only_and_excludes_current_and_late_events(client6,service6):
    create_demo(client6)
    client6.post('/api/v1/ingest',json={'world_id':'demo_test','idempotency_key':'first','transactions':events()})
    body={'world_id':'demo_test','cutoff_step':10}
    before=client6.post('/api/v1/score',json=body).json()
    assert len(before['candidates'])==1
    revision=service6.store.world('demo_test')['revision']
    assert client6.post('/api/v1/score',json=body).json()==before
    assert service6.store.world('demo_test')['revision']==revision
    assert service6.store.list_cases()['total']==0
    extra=[dict(events()[0],transaction_id='late',amount='9000',available_step=10),
           dict(events()[0],transaction_id='current',step=10,available_step=10)]
    client6.post('/api/v1/ingest',json={'world_id':'demo_test','idempotency_key':'later','transactions':extra})
    after=client6.post('/api/v1/score',json=body).json()
    assert before['candidates']==after['candidates'] and before['snapshot_id']==after['snapshot_id']
    assert after['history_revision']>before['history_revision']


def test_replay_resume_duplicate_suppression_and_feedback_are_separate(client6,service6,monkeypatch):
    job=service6.request_replay('world_validation')
    original=service6.score
    def fail_once(world,cutoff,model,policy):
        if cutoff==20:
            raise ValueError('Simulated interruption')
        return original(world,cutoff,model,policy)
    monkeypatch.setattr(service6,'score',fail_once)
    with pytest.raises(ValueError):
        service6.run_replay(job['replay_id'])
    assert service6.store.replay(job['replay_id'])['completed_snapshots']==2
    monkeypatch.setattr(service6,'score',original)
    final=service6.run_replay(job['replay_id'])
    assert final['status']=='complete' and final['alerts']==2
    assert service6.run_replay(job['replay_id'])==final
    case_id=service6.store.list_cases()['items'][0]['case_id']
    case=client6.get('/api/v1/cases/'+case_id).json()
    payload={'request_id':'review1','expected_version':0,'status':'in_review','note':'Observed a repeated transfer pattern.'}
    endpoint='/api/v1/cases/'+case_id+'/reviews'
    assert client6.post(endpoint,json=payload).json()['version']==1
    assert client6.post(endpoint,json=payload).json()['replayed']
    assert client6.post(endpoint,json=dict(payload,request_id='review2')).status_code==409
    reviewed=client6.get('/api/v1/cases/'+case_id).json()
    assert reviewed['evidence']==case['evidence'] and reviewed['score']==case['score']
    assert len(reviewed['reviews'])==1 and reviewed['review_version']==1
    assert 'is_fraud' not in json.dumps(reviewed) and 'group_id' not in json.dumps(reviewed)
    # A late arrival may refer to an old event, but cannot rewrite availability before committed history.
    row=dict(events()[0],transaction_id='new-backdated',type='TRANSFER')
    with pytest.raises(Conflict):
        service6.store.ingest('world_validation','backdated',[row],[])
    row['available_step']=20
    assert service6.store.ingest('world_validation','late-valid',[row],[])['inserted']==1


def test_concurrent_ingest_same_key_is_once(service6):
    rows=[dict(r,type='TRANSFER') for r in events()]
    def run(_):return service6.store.ingest('world_train','parallel',rows,[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,range(2)))
    assert sorted(r['replayed'] for r in results)==[False,True]
    assert service6.store.world('world_train')['revision']==1


def test_api_boundaries_catalog_and_static_dashboard(client6):
    assert client6.get('/').status_code==200
    assert client6.get('/assets/app.js').status_code==200
    catalog=client6.get('/api/v1/catalog').json()
    assert catalog['default_model']=='graph_only' and not catalog['final_test_included']
    assert client6.get('/api/v1/openapi.json').status_code==200
    assert client6.get('/api/v1/cases/missing').status_code==404
    assert client6.get('/api/v1/cases?limit=999').status_code==422
    assert client6.post('/api/v1/replays',json={'world_id':'missing_test'}).status_code==400
    assert client6.post('/api/v1/ingest',json={'world_id':'world_validation','idempotency_key':'external','transactions':events()}).status_code==409
    assert client6.post('/api/v1/worlds',json={'world_id':'demo_a','label':'Demo'},headers={'Origin':'https://outside.example'}).status_code==403
    assert client6.get('/',headers={'Host':'outside.example'}).status_code==400
    assert client6.post('/api/v1/replays',json={'world_id':'world_validation'}).status_code==202
    jobs=client6.get('/api/v1/replays').json()['items']
    assert jobs[0]['status']=='complete'


def test_checksum_failure_and_path_escape(project,service6):
    folder=service6.bundle.folder
    metadata=json.loads((folder/'bundle.json').read_text())
    metadata['files']={'weights.pt':'not-the-real-hash'}
    (folder/'weights.pt').write_bytes(b'invalid')
    write_json(folder/'bundle.json',metadata)
    with pytest.raises(ValueError,match='checksum'):
        Bundle('fixture',project)
    with pytest.raises(ValueError,match='inside'):
        Store('../outside.sqlite',project)
