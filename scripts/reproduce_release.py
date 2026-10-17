"""Run with the fresh environment's Python -I to verify the installed wheel."""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args()
    project=args.project.resolve()
    workspace=project/'artifacts/phase7/reproduction/workspace'
    if workspace.exists():raise ValueError('Reproduction workspace already exists')
    workspace.mkdir(parents=True)
    os.environ['GRAPHSHIELD_ROOT']=str(workspace)
    for key in ('TEMP','TMP','TORCH_HOME'):
        path=workspace/'.cache'/key.lower()
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(path)
    import graphshield.api
    from fastapi.testclient import TestClient
    from graphshield.pipeline import sha256,write_json
    from graphshield.service_store import Store
    from graphshield.serving import Bundle,InvestigationService
    module=Path(graphshield.api.__file__).resolve()
    if not module.is_relative_to(Path(sys.prefix).resolve()) or module.is_relative_to(project/'src'):
        raise ValueError('Reproduction must import the installed wheel from its fresh environment')
    release=project/'artifacts/phase7/release'
    with zipfile.ZipFile(release/'graphshield-demo-data-v0.7.0.zip') as archive:
        for name in archive.namelist():
            if not (workspace/name).resolve().is_relative_to(workspace):raise ValueError('Unsafe archive member')
        archive.extractall(workspace)
    manifest=json.loads((release/'demo-data-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha256(workspace/name)!=digest:raise ValueError('Extracted artifact checksum mismatch')
    start=time.perf_counter()
    bundle=Bundle(root=workspace)
    original=Store('artifacts/phase6/demo.sqlite',workspace)
    service=InvestigationService(bundle,Store('artifacts/phase6/reproduced.sqlite',workspace))
    job=service.request_replay(bundle.meta['default_world'])
    final=service.run_replay(job['replay_id'])
    def candidates(store):
        return {store.case(r['case_id'])['candidate_id']:r['score'] for r in store.list_cases(limit=200)['items']}
    assert candidates(original)==candidates(service.store)
    assert final['status']=='complete' and final['alerts']==17 and final['completed_snapshots']==27
    with TestClient(graphshield.api.create_app(root=workspace,service=service)) as client:
        for path in ('/','/assets/app.js','/assets/app.css','/api/v1/health','/api/v1/catalog','/api/v1/monitoring','/api/v1/openapi.json'):
            assert client.get(path).status_code==200,path
        assert client.get('/api/v1/cases').json()['total']==17
        assert client.post('/api/v1/replays',json={'world_id':bundle.meta['default_world']}).json()['alerts']==17
    result={'status':'passed','python':sys.version,'environment':sys.prefix,'installed_module':str(module),
        'workspace':str(workspace),'bundle_version':bundle.version,'reproduced_snapshots':27,'reproduced_cases':17,
        'exact_score_and_candidate_parity':True,'api_and_packaged_assets':'passed','final_test_loaded':False,
        'seconds':time.perf_counter()-start}
    write_json(project/'artifacts/phase7/reproduction/report.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
