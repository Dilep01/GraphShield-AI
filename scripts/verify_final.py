"""Reload frozen models and reconstruct final predictions and evaluation, read-only."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'.cache/tmp')
import duckdb
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from graphshield.baselines import predict_bundle
from graphshield.evaluation import evaluate as transaction_metrics
from graphshield.features import FEATURES
from graphshield.final_evaluation import load_protocol,verify_seal,read,verified,final_source
from graphshield.graph_rules import snapshot
from graphshield.learned_rings import select_cases
from graphshield.pipeline import literal,records,sha256,write_json
from graphshield.ring_evaluation import evaluate,aggregate
from graphshield.rings import read_jsonl


def verify(run):
    folder,protocol,bundle=load_protocol(run)
    seal=verify_seal(folder)
    verified(folder/'results.json',read(folder/'status.json')['results_sha256'])
    result=read(folder/'results.json')
    config=protocol['config']
    checks={'sealed_files_and_protocol':True}
    with duckdb.connect() as con:
        f=con.execute(f'SELECT * FROM read_parquet({literal(folder/"paysim/features.parquet")}) ORDER BY source_row_number').fetchnumpy()
        p=con.execute(f'SELECT * FROM read_parquet({literal(folder/"paysim/predictions.parquet")}) ORDER BY source_row_number').fetchnumpy()
        label_path=ROOT/'data/processed'/config['paysim_source']/'labels.parquet'
        source=read(ROOT/'artifacts/phase1'/config['paysim_source']/'manifest.json')
        verified(label_path,source['files']['labels.parquet']['sha256'])
        y=con.execute(f'SELECT l.is_fraud FROM read_parquet({literal(folder/"paysim/predictions.parquet")}) p JOIN read_parquet({literal(label_path)}) l USING(transaction_id) ORDER BY p.source_row_number').fetchnumpy()['is_fraud'].astype(np.int8)
    assert np.array_equal(f['transaction_id'],p['transaction_id'])
    x=np.column_stack([f[n] for n in FEATURES]).astype(np.float32)
    model_folder=ROOT/'artifacts/phase2/models'/config['transaction_models']
    models=read(model_folder/'manifest.json')
    numerical_checks={}
    with threadpool_limits(limits=4):
        for name,threshold in protocol['transaction_thresholds'].items():
            path=verified(model_folder/(name+'.joblib'),models['files'][name+'.joblib'])
            scores=predict_bundle(joblib.load(path),x)
            # LR computes in float32; cross-process BLAS/serialization differences
            # must be bounded at that precision, not tested as float64 identity.
            tolerance=1e-6 if scores.dtype==np.float32 else 1e-12
            difference=float(np.max(np.abs(scores-p[name])))
            assert np.allclose(scores,p[name],atol=tolerance,rtol=0)
            assert np.array_equal(scores>=threshold,p[name]>=threshold)
            numerical_checks[name]={'dtype':str(scores.dtype),'maximum_absolute_difference':difference,
                                    'absolute_tolerance':tolerance,'threshold_decisions_equal':True}
            cfg=protocol['transaction_config']
            metrics=transaction_metrics(y,p[name],p['step'],p['source_row_number'],threshold,cfg['alert_budget_per_step'],cfg['global_top_k'])
            assert metrics=={k:v for k,v in result['paysim']['models'][name].items() if k!='cohorts'}
    checks.update(paysim_all_predictions_reproduce=True,paysim_headline_metrics_reproduce=True)
    cases=read_jsonl(folder/'rings/candidates.jsonl')
    alerts=read_jsonl(folder/'rings/alerts.jsonl')
    source=read(ROOT/'artifacts/phase3'/config['synthetic_source']/'manifest.json')
    data=ROOT/source['output_path']
    evaluated=[]
    for world in seal['rings']['worlds']:
        with duckdb.connect() as con:
            tx=con.execute(f'SELECT transaction_id,sender_id,receiver_id,amount,step,available_step FROM read_parquet({literal(final_source(data,world,"observed","transactions",source))}) ORDER BY transaction_id').fetchall()
            obs=con.execute(f'SELECT observation_id,account_id,device_id,step,available_step FROM read_parquet({literal(final_source(data,world,"observed","device_observations",source))}) ORDER BY observation_id').fetchall()
            groups=records(con,f'SELECT group_id,scenario_family,is_fraud,start_step,end_step FROM read_parquet({literal(final_source(data,world,"ground_truth","groups",source))}) ORDER BY group_id')
            members=con.execute(f'SELECT group_id,account_id FROM read_parquet({literal(final_source(data,world,"ground_truth","group_memberships",source))})').fetchall()
            event_groups=dict(con.execute(f'SELECT transaction_id,group_id FROM read_parquet({literal(final_source(data,world,"ground_truth","transaction_labels",source))}) WHERE group_id IS NOT NULL').fetchall())
        for group in groups:group['members']=[a for gid,a in members if gid==group['group_id']]
        world_cases=[c for c in cases if c['world_id']==world['world_id']]
        for cutoff in bundle.meta['cutoffs']:
            expected=[c for c in world_cases if c['cutoff_step']==cutoff]
            actual,_=snapshot(tx,obs,cutoff,protocol['rule_config'])
            assert actual==[{k:v for k,v in c.items() if k not in ('world_id','split','model_scores')} for c in expected]
            if actual:
                path=folder/'rings/graphs'/f'{world["world_id"]}-{cutoff}.npz'
                bundle.cache[(world['world_id'],cutoff)]={'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}
            for model in config['ring_models']:
                scored,_=bundle.score(actual,tx,obs,world['world_id'],cutoff,model)
                assert np.allclose([c['score'] for c in scored],[c['model_scores'][model] for c in expected],atol=1e-6,rtol=0)
        coverage=evaluate(world_cases,groups,event_groups,protocol['rule_config'])
        assert coverage==next(c['metrics'] for c in result['candidate_coverage'] if c['world_id']==world['world_id'])
        for model in config['ring_models']:
            for policy in config['ring_policies']:
                threshold=protocol['ring_thresholds'][model] if policy=='tuned_threshold' else 0
                selected,_=select_cases(world_cases,np.asarray([c['model_scores'][model] for c in world_cases]),threshold,protocol['rule_config'])
                saved=[c for c in alerts if c['world_id']==world['world_id'] and c['model']==model and c['policy']==policy]
                assert [c['candidate_id'] for c in selected]==[c['candidate_id'] for c in saved]
                metrics=evaluate(selected,groups,event_groups,protocol['rule_config'])
                assert metrics==next(r['metrics'] for r in result['worlds'] if r['world_id']==world['world_id'] and r['model']==model and r['policy']==policy)
                evaluated.append(dict(world,model=model,policy=policy,metrics=metrics))
        print('Verified final world '+world['world_id'],flush=True)
    for split in config['ring_splits']:
        for model in config['ring_models']:
            for policy in config['ring_policies']:
                assert aggregate([r['metrics'] for r in evaluated if r['split']==split and r['model']==model and r['policy']==policy])==result['rings'][split][model][policy]
    checks.update(all_final_snapshots_reconstructed=True,all_seven_ring_model_scores_reproduce=True,
                  all_final_alert_selections_reproduce=True,all_ring_metrics_and_coverage_reproduce=True)
    report={'status':'passed','checks':checks,'transaction_numerical_checks':numerical_checks,'paysim_rows':len(y),'ring_snapshots':seal['rings']['snapshots'],
            'candidate_observations':len(cases),'results_sha256':sha256(folder/'results.json'),
            'verifier_sha256':sha256(Path(__file__)),'scope':'Reloaded models and reconstructed observed snapshots; cached embeddings checksummed, not retrained'}
    write_json(folder/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='final-v1')
    verify(parser.parse_args().run_name)
