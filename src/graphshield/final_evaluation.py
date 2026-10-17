"""Explicit final gate: freeze choices, seal predictions, then open held-out truth."""
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .baselines import predict_bundle
from .evaluation import classification, evaluate as transaction_metrics, paired_block_intervals, per_step_selection, reference_policies
from .features import FEATURES
from .final_features import build_final_features
from .graph_rules import snapshot
from .learned_graph import historical_graph, node2vec
from .learned_rings import MODELS, seed_for, select_cases
from .paths import ROOT, inside_root
from .pipeline import literal, records, sha256, write_json
from .ring_evaluation import aggregate, eligible, evaluate
from .rings import read_jsonl, write_jsonl
from .serving import Bundle
from .synthetic import checked_name

IMPLEMENTATION=('final_evaluation.py','final_features.py','features.py','evaluation.py','baselines.py',
                'serving.py','graph_rules.py','learned_graph.py','learned_rings.py','ring_evaluation.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified(path, digest):
    if sha256(path)!=digest:
        raise ValueError(f'Checksum changed: {path.name}')
    return path


def freeze(run, root=ROOT):
    root=Path(root).resolve()
    folder=inside_root(Path('artifacts/phase7')/checked_name(run),root)
    if folder.exists():
        raise ValueError('Final run already exists; do not overwrite evaluation evidence')
    config=read(root/'config/phase7.json')
    if config['version']!='final-evaluation-v1' or config['ring_models']!=list(MODELS):
        raise ValueError('Unexpected final protocol/model order')
    bundle=Bundle(config['serving_bundle'],root)
    transaction_folder=root/'artifacts/phase2/models'/checked_name(config['transaction_models'])
    tm=read(transaction_folder/'manifest.json')
    p1=root/'artifacts/phase1'/checked_name(config['paysim_source'])/'manifest.json'
    p3=root/'artifacts/phase3'/checked_name(config['synthetic_source'])/'manifest.json'
    if any(read(p)['status']!='complete' for p in (p1,p3,transaction_folder/'manifest.json')):
        raise ValueError('Completed source runs are required')
    for name,digest in tm['files'].items():
        verified(inside_root(name,transaction_folder),digest)
    metrics=read(transaction_folder/'metrics.json')
    # These are development-selected values; no final records have been opened.
    protocol={'version':config['version'],'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'bundle_version':bundle.version,'transaction_selected':tm['selected_model'],
        'ring_selected':bundle.meta['default_model'],
        'transaction_thresholds':{n:m['tuning']['threshold'] for n,m in metrics['models'].items()},
        'ring_thresholds':{n:m['threshold'] for n,m in bundle.meta['models'].items()},
        'transaction_config':tm['config'],'rule_config':bundle.meta['rule_config'],
        'source_manifests':{str(p.relative_to(root)):sha256(p) for p in (p1,p3,transaction_folder/'manifest.json')},
        'implementation':{name:sha256(Path(__file__).parent/name) for name in IMPLEMENTATION},
        'selection_rule':'Preserve development choices; no threshold search, fitting or model promotion on final labels',
        'uncertainty':'PaySim paired whole-step bootstrap; ring paired whole-world bootstrap with only two worlds per split, descriptive only'}
    folder.mkdir(parents=True)
    write_json(folder/'protocol.json',protocol)
    return folder,protocol


def load_protocol(run, root=ROOT):
    root=Path(root).resolve()
    folder=inside_root(Path('artifacts/phase7')/checked_name(run),root)
    protocol=read(folder/'protocol.json')
    for path,digest in protocol['source_manifests'].items():
        verified(inside_root(path,root),digest)
    for name,digest in protocol['implementation'].items():
        verified(Path(__file__).parent/name,digest)
    bundle=Bundle(protocol['config']['serving_bundle'],root)
    if bundle.version!=protocol['bundle_version']:
        raise ValueError('Serving bundle changed after protocol freeze')
    return folder,protocol,bundle


def final_source(data, world, kind, table, source):
    if world['split'] not in ('test_known','test_unseen') or (kind,table) not in (
        ('observed','transactions'),('observed','device_observations'),('ground_truth','groups'),
        ('ground_truth','group_memberships'),('ground_truth','transaction_labels')):
        raise ValueError('Only explicit final inputs may be opened here')
    path=inside_root(Path(kind)/world['split']/checked_name(world['world_id'])/(table+'.parquet'),data)
    return verified(path,source['files'][str(path.relative_to(data))])


def predict_transactions(folder, protocol, root):
    config=protocol['config']
    source=read(root/'artifacts/phase1'/config['paysim_source']/'manifest.json')
    tx=verified(root/'data/processed'/config['paysim_source']/'transactions.parquet',source['files']['transactions.parquet']['sha256'])
    work=folder/'paysim'
    work.mkdir()
    model_folder=root/'artifacts/phase2/models'/config['transaction_models']
    model_manifest=read(model_folder/'manifest.json')
    start=time.perf_counter()
    print('Building final PaySim features from strictly earlier event buckets...',flush=True)
    with duckdb.connect(str(work/'features.duckdb'),config={'memory_limit':'1GB','threads':4,'temp_directory':str(work/'spill')}) as con:
        count=build_final_features(con,tx,work/'features.parquet',source['train_end_step'],source['validation_end_step'])
        arrays=con.execute('SELECT * FROM features ORDER BY source_row_number').fetchnumpy()
    x=np.column_stack([arrays[name] for name in FEATURES]).astype(np.float32)
    scores={}
    with threadpool_limits(limits=4):
        for name,threshold in protocol['transaction_thresholds'].items():
            path=verified(model_folder/(name+'.joblib'),model_manifest['files'][name+'.joblib'])
            model=joblib.load(path)
            if model['threshold']!=threshold:
                raise ValueError('Transaction threshold differs from frozen protocol')
            scores[name]=predict_bundle(model,x)
    path=work/'predictions.csv'
    with path.open('w',newline='',encoding='utf-8') as out:
        writer=csv.writer(out)
        writer.writerow(['transaction_id','source_row_number','step','either_unseen_at_train_cutoff',*scores])
        writer.writerows(zip(arrays['transaction_id'],arrays['source_row_number'],arrays['step'],arrays['either_unseen_at_train_cutoff'],*scores.values()))
    with duckdb.connect() as con:
        con.execute(f'COPY (SELECT * FROM read_csv_auto({literal(path)})) TO {literal(work/"predictions.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    path.unlink()
    # The persisted working database is disposable and contains only observations/features.
    (work/'features.duckdb').unlink()
    return {'rows':count,'seconds':time.perf_counter()-start}


def predict_rings(folder, protocol, bundle, root):
    config=protocol['config']
    source=read(root/'artifacts/phase3'/config['synthetic_source']/'manifest.json')
    data=inside_root(source['output_path'],root)
    worlds=[{k:w[k] for k in ('world_id','split')} for w in source['worlds'] if w['split'] in config['ring_splits']]
    if len(worlds)!=4 or len({w['world_id'] for w in worlds})!=4:
        raise ValueError('Expected the four pre-generated final worlds')
    work=folder/'rings'
    (work/'graphs').mkdir(parents=True)
    all_cases,all_alerts,decisions,timings=[],[],[],[]
    cfg=bundle.meta['config']
    for world in worlds:
        with duckdb.connect() as con:
            tx=con.execute(f'SELECT transaction_id,sender_id,receiver_id,amount,step,available_step FROM read_parquet({literal(final_source(data,world,"observed","transactions",source))}) ORDER BY transaction_id').fetchall()
            obs=con.execute(f'SELECT observation_id,account_id,device_id,step,available_step FROM read_parquet({literal(final_source(data,world,"observed","device_observations",source))}) ORDER BY observation_id').fetchall()
        cases=[]
        for cutoff in bundle.meta['cutoffs']:
            start=time.perf_counter()
            current,summary=snapshot(tx,obs,cutoff,protocol['rule_config'])
            embedding_seconds=0.0
            if current:
                embedding_start=time.perf_counter()
                graph=historical_graph(tx,obs,cutoff,protocol['rule_config']['window_steps'],cfg['device_projection_limit'])
                embedding,_=node2vec(graph['edges'],len(graph['nodes']),cfg['node2vec'],seed_for(cfg['seed'],world['world_id'],cutoff))
                path=work/'graphs'/f'{world["world_id"]}-{cutoff}.npz'
                np.savez_compressed(path,nodes=graph['nodes'],x=graph['x'],edges=graph['edges'],embedding=embedding)
                bundle.cache[(world['world_id'],cutoff)]={'path':str(path.relative_to(root)),'sha256':sha256(path)}
                embedding_seconds=time.perf_counter()-embedding_start
            scores={}
            for name in config['ring_models']:
                scored,_=bundle.score(current,tx,obs,world['world_id'],cutoff,name)
                scores[name]=[r['score'] for r in scored]
            cases.extend(dict(c,**world,model_scores={n:scores[n][i] for n in scores}) for i,c in enumerate(current))
            timings.append(dict(world,cutoff_step=cutoff,candidates=len(current),seconds=time.perf_counter()-start,
                                graph_and_embedding_seconds=embedding_seconds,summary=summary))
        all_cases.extend(cases)
        for name in config['ring_models']:
            for policy in config['ring_policies']:
                threshold=protocol['ring_thresholds'][name] if policy=='tuned_threshold' else 0.0
                alerts,current_decisions=select_cases(cases,np.asarray([c['model_scores'][name] for c in cases]),threshold,protocol['rule_config'])
                all_alerts.extend(dict(c,model=name,policy=policy,score=c['model_scores'][name]) for c in alerts)
                decisions.extend(dict(d,**world,model=name,policy=policy) for d in current_decisions)
        print(f'Final observed network {world["world_id"]}: {len(cases)} candidates scored; labels still unopened.',flush=True)
    write_jsonl(work/'candidates.jsonl',all_cases)
    write_jsonl(work/'alerts.jsonl',all_alerts)
    write_jsonl(work/'decisions.jsonl',decisions)
    write_json(work/'timings.json',timings)
    return {'worlds':worlds,'candidate_observations':len(all_cases),'snapshots':len(timings)}


def predict(run, root=ROOT):
    root=Path(root).resolve()
    folder,protocol,bundle=load_protocol(run,root)
    if (folder/'paysim').exists() or (folder/'rings').exists():
        raise ValueError('Prediction output already exists; preserve it and investigate before a new run')
    status={'status':'predicting','protocol_sha256':sha256(folder/'protocol.json'),'truth_opened':False}
    write_json(folder/'status.json',status)
    start=time.perf_counter()
    try:
        paysim=predict_transactions(folder,protocol,root)
        rings=predict_rings(folder,protocol,bundle,root)
        seal={'protocol_sha256':sha256(folder/'protocol.json'),'sealed_at_utc':datetime.now(timezone.utc).isoformat(),
              'truth_opened':False,'paysim':paysim,'rings':rings,'seconds':time.perf_counter()-start,
              'files':{str(p.relative_to(folder)):sha256(p) for part in ('paysim','rings') for p in sorted((folder/part).rglob('*')) if p.is_file()}}
        write_json(folder/'prediction_seal.json',seal)
        write_json(folder/'status.json',dict(status,status='predictions_sealed'))
    except Exception as exc:
        write_json(folder/'status.json',dict(status,status='prediction_failed',error=str(exc)))
        raise
    return seal


def verify_seal(folder):
    seal=read(folder/'prediction_seal.json')
    verified(folder/'protocol.json',seal['protocol_sha256'])
    for path,digest in seal['files'].items():
        verified(inside_root(path,folder),digest)
    return seal


def paired_world_intervals(rows, selected, repeats=2000, seed=42):
    worlds=sorted({r['world_id'] for r in rows})
    by_key={(r['world_id'],r['model'],r['policy']):r['metrics'] for r in rows}
    rng=np.random.default_rng(seed)
    samples={name:[] for name in ('case_precision','ring_recall','recall_difference_vs_graph_lgb')}
    for _ in range(repeats):
        chosen=[worlds[i] for i in rng.integers(0,len(worlds),len(worlds))]
        a=aggregate([by_key[w,selected,'tuned_threshold'] for w in chosen])
        b=aggregate([by_key[w,'graph_lgb','tuned_threshold'] for w in chosen])
        if a['case_precision'] is not None:samples['case_precision'].append(a['case_precision'])
        if a['ring_recall'] is not None:
            samples['ring_recall'].append(a['ring_recall'])
            samples['recall_difference_vs_graph_lgb'].append(a['ring_recall']-b['ring_recall'])
    return {'independent_worlds':len(worlds),'repeats':repeats,'seed':seed,
            'method':'Paired whole-world bootstrap, percentile 95% intervals',
            'limitation':'Only two worlds per split: descriptive resampling ranges, not reliable population confidence or significance evidence',
            'intervals':{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in samples.items()}}


def evaluate_final(run, root=ROOT):
    root=Path(root).resolve()
    folder,protocol,bundle=load_protocol(run,root)
    seal=verify_seal(folder)
    if (folder/'results.json').exists():
        raise ValueError('Final results already exist; verify them without overwriting')
    config=protocol['config']
    # Persist the transition before the first access to any held-out label file.
    write_json(folder/'status.json',{'status':'evaluating','truth_opened':True,'prediction_seal_sha256':sha256(folder/'prediction_seal.json')})
    p1=read(root/'artifacts/phase1'/config['paysim_source']/'manifest.json')
    label_path=verified(root/'data/processed'/config['paysim_source']/'labels.parquet',p1['files']['labels.parquet']['sha256'])
    with duckdb.connect() as con:
        prediction_path=folder/'paysim/predictions.parquet'
        arrays=con.execute(f'SELECT p.*,l.is_fraud FROM read_parquet({literal(prediction_path)}) p JOIN read_parquet({literal(label_path)}) l USING(transaction_id) ORDER BY source_row_number').fetchnumpy()
    if len(arrays['is_fraud'])!=seal['paysim']['rows']:
        raise ValueError('Label join changed final row count')
    y=arrays['is_fraud'].astype(np.int8)
    cfg=protocol['transaction_config']
    tx_results={}
    for name,threshold in protocol['transaction_thresholds'].items():
        result=transaction_metrics(y,arrays[name],arrays['step'],arrays['source_row_number'],threshold,cfg['alert_budget_per_step'],cfg['global_top_k'])
        selected=per_step_selection(arrays[name],arrays['step'],arrays['source_row_number'],cfg['alert_budget_per_step'])
        result['cohorts']={}
        unseen=arrays['either_unseen_at_train_cutoff'].astype(bool)
        for cohort,mask in (('either_account_unseen',unseen),('both_accounts_seen',~unseen)):
            result['cohorts'][cohort]={'rows':int(mask.sum()),'positives':int(y[mask].sum()),
                'threshold_policy':classification(y[mask],arrays[name][mask]>=threshold),
                'full_queue_budget_policy':classification(y[mask],selected[mask])}
        tx_results[name]=result
    dev=read(root/'artifacts/phase2/models'/config['transaction_models']/'metrics.json')
    tx_report={'selected_model':protocol['transaction_selected'],'models':tx_results,
        'paired_intervals':paired_block_intervals(y,{n:arrays[n] for n in tx_results},arrays['step'],cfg['bootstrap_repeats'],cfg['seed']),
        'references':reference_policies(y,arrays['step'],cfg['alert_budget_per_step'],dev['data']['train_positives']/dev['data']['train_rows'])}
    candidates=read_jsonl(folder/'rings/candidates.jsonl')
    alerts=read_jsonl(folder/'rings/alerts.jsonl')
    source=read(root/'artifacts/phase3'/config['synthetic_source']/'manifest.json')
    data=inside_root(source['output_path'],root)
    rows,coverage=[],[]
    for world in seal['rings']['worlds']:
        with duckdb.connect() as con:
            groups=records(con,f'SELECT group_id,scenario_family,is_fraud,start_step,end_step FROM read_parquet({literal(final_source(data,world,"ground_truth","groups",source))}) ORDER BY group_id')
            memberships=con.execute(f'SELECT group_id,account_id FROM read_parquet({literal(final_source(data,world,"ground_truth","group_memberships",source))}) ORDER BY group_id,account_id').fetchall()
            event_groups=dict(con.execute(f'SELECT transaction_id,group_id FROM read_parquet({literal(final_source(data,world,"ground_truth","transaction_labels",source))}) WHERE group_id IS NOT NULL').fetchall())
        for group in groups:group['members']=[a for gid,a in memberships if gid==group['group_id']]
        current=[c for c in candidates if c['world_id']==world['world_id']]
        target=np.asarray([int(any(g['is_fraud'] and eligible(c,g,event_groups,protocol['rule_config']) for g in groups)) for c in current])
        coverage.append(dict(world,metrics=evaluate(current,groups,event_groups,protocol['rule_config'])))
        for model in config['ring_models']:
            for policy in config['ring_policies']:
                selected=[c for c in alerts if c['world_id']==world['world_id'] and c['model']==model and c['policy']==policy]
                result=evaluate(selected,groups,event_groups,protocol['rule_config'])
                rows.append(dict(world,model=model,policy=policy,metrics=result,
                    candidate_ap=float(average_precision_score(target,[c['model_scores'][model] for c in current])) if target.sum() else None))
    ring_summary={split:{model:{policy:aggregate([r['metrics'] for r in rows if r['split']==split and r['model']==model and r['policy']==policy]) for policy in config['ring_policies']} for model in config['ring_models']} for split in config['ring_splits']}
    results={'protocol_sha256':sha256(folder/'protocol.json'),'prediction_seal_sha256':sha256(folder/'prediction_seal.json'),
        'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),'selected_ring_model':protocol['ring_selected'],
        'paysim':tx_report,'rings':ring_summary,'worlds':rows,'candidate_coverage':coverage,
        'ring_uncertainty':{split:paired_world_intervals([r for r in rows if r['split']==split],protocol['ring_selected'],config['bootstrap_world_repeats'],config['bootstrap_seed']) for split in config['ring_splits']},
        'limitations':['All data is synthetic','Two final worlds per ring split; limited uncertainty evidence',
                       'Final labels are now consumed; later changes need a new benchmark','Models, thresholds and weights were not fitted on final labels']}
    write_json(folder/'results.json',results)
    write_json(folder/'status.json',{'status':'complete','truth_opened':True,'results_sha256':sha256(folder/'results.json')})
    return results
