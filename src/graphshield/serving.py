"""Versioned, observed-only inference and resumable development-data replay."""
import json
import shutil
import threading
from pathlib import Path

import duckdb
import joblib
import numpy as np
import torch

from .graph_rules import snapshot
from .learned_graph import (ACTIVITY_FEATURES, CASE_FEATURES, EMBEDDING_FEATURES, CaseNetwork,
                            case_matrix, embedding_features, historical_graph, member_indices, node2vec)
from .learned_rings import seed_for
from .paths import ROOT, inside_root
from .pipeline import literal, sha256, write_json
from .service_store import Conflict, Missing, Store, fingerprint
from .synthetic import checked_name

LABELS = {'activity_lgb':'Activity LightGBM','graph_lgb':'Graph features + LightGBM',
          'node2vec_lgb':'Node2Vec + LightGBM','mlp':'Neural model without messages',
          'graphsage':'GraphSAGE','graph_only':'Repeated-transfer rules','graph_device':'Transfer + device rules'}
INFERENCE_LOCK = threading.RLock()


def prepare_bundle(learned_run, bundle_name, root=ROOT):
    root = Path(root).resolve()
    source = inside_root(Path('artifacts/phase5')/checked_name(learned_run),root)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('A completed learned model run is required')
    for name,digest in manifest['files'].items():
        if sha256(inside_root(name,source)) != digest:
            raise ValueError(f'Learned artifact checksum mismatch: {name}')
    for name,digest in manifest['source_sha256'].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError(f'Training source changed: {name}')
    phase4_path = inside_root(Path('artifacts/phase4')/manifest['rule_run']/'manifest.json',root)
    if sha256(phase4_path) != manifest['rule_manifest_sha256']:
        raise ValueError('Phase 4 manifest changed')
    phase4 = json.loads(phase4_path.read_text(encoding='utf-8'))
    phase3_path = inside_root(Path('artifacts/phase3')/phase4['source_run']/'manifest.json',root)
    if sha256(phase3_path) != phase4['source_manifest_sha256']:
        raise ValueError('Scenario source manifest changed')
    phase3 = json.loads(phase3_path.read_text(encoding='utf-8'))
    evaluation = json.loads((source/'evaluation.json').read_text(encoding='utf-8'))
    destination = inside_root(Path('artifacts/phase6/bundles')/checked_name(bundle_name),root)
    if destination.exists():
        raise ValueError('Bundle already exists; choose a new name')
    (destination/'models').mkdir(parents=True)
    files = {}
    for path in sorted((source/'models').iterdir()):
        if path.suffix in ('.joblib','.pt'):
            copied = destination/'models'/path.name
            shutil.copyfile(path,copied)
            files[str(copied.relative_to(destination))] = sha256(copied)
    worlds = []
    data = inside_root(phase3['output_path'],root)
    for w in manifest['worlds']:
        if w['split'] not in ('train','validation'):
            raise ValueError('Final-test worlds cannot be served in this phase')
        role = 'Training network' if w['split']=='train' else 'Tuning network' if w['world_id']==evaluation['tuning_world'] else 'Comparison network'
        observed = {}
        for table in ('transactions','device_observations'):
            path = inside_root(Path('observed')/w['split']/w['world_id']/f'{table}.parquet',data)
            digest = phase3['files'][str(path.relative_to(data))]
            if sha256(path) != digest:
                raise ValueError('Observed source checksum mismatch')
            observed[table] = {'path':str(path.relative_to(root)),'sha256':digest}
        worlds.append(dict(w,label=role,observed=observed))
    cache = []
    for item in json.loads((source/'graph_log.json').read_text(encoding='utf-8')):
        path = source/'graphs'/item['filename']
        cache.append({'world_id':item['world_id'],'cutoff_step':item['cutoff_step'],
                      'path':str(path.relative_to(root)),'sha256':sha256(path)})
    first,last = phase3['config']['warmup_steps']+1,phase3['config']['steps']+1
    cutoffs = sorted(set(range(first,last+1,phase4['config']['snapshot_stride']))|{last})
    bundle = {'status':'complete','schema_version':1,'name':bundle_name,'learned_run':learned_run,
              'learned_manifest_sha256':sha256(source/'manifest.json'),'default_model':manifest['selected_on_tuning'],
              'default_world':evaluation['comparison_world'],'models':{
                  name:{'label':LABELS[name],'threshold':row['threshold'],'selected_on_tuning':name==manifest['selected_on_tuning']}
                  for name,row in evaluation['thresholds'].items()},
              'config':manifest['config'],'rule_config':phase4['config'],'worlds':worlds,'cutoffs':cutoffs,
              'files':files,'embedding_cache':cache,'source_sha256':manifest['source_sha256'],
              'provenance':'Fully synthetic development networks','experimental':True,'final_test_included':False}
    write_json(destination/'bundle.json',bundle)
    return bundle


class Bundle:
    def __init__(self, name='serving-v1', root=ROOT):
        self.root = Path(root).resolve()
        self.folder = inside_root(Path('artifacts/phase6/bundles')/checked_name(name),self.root)
        self.meta = json.loads((self.folder/'bundle.json').read_text(encoding='utf-8'))
        if self.meta['status']!='complete' or self.meta['schema_version']!=1 or self.meta['final_test_included']:
            raise ValueError('Unsupported serving bundle')
        for path,digest in self.meta['files'].items():
            if sha256(inside_root(path,self.folder)) != digest:
                raise ValueError('Model checksum mismatch; loading refused')
        for path,digest in self.meta['source_sha256'].items():
            if sha256(inside_root(path,Path(__file__).parent)) != digest:
                raise ValueError('Inference implementation differs from the trained version')
        if any(w['split'] not in ('train','validation') for w in self.meta['worlds']):
            raise ValueError('Final-test worlds are excluded')
        self.version = sha256(self.folder/'bundle.json')
        self.models = {}
        self.cache = {(c['world_id'],c['cutoff_step']):c for c in self.meta['embedding_cache']}
        torch.set_num_threads(self.meta['config']['threads'])
        torch.use_deterministic_algorithms(True)

    def catalog(self):
        return {'bundle':self.meta['name'],'bundle_version':self.version,'learned_run':self.meta['learned_run'],
                'default_model':self.meta['default_model'],'default_world':self.meta['default_world'],
                'models':self.meta['models'],'provenance':self.meta['provenance'],'experimental':True,
                'score_semantics':'Investigation priority, not fraud probability',
                'final_test_included':False,'cutoffs':self.meta['cutoffs'],
                'worlds':[{k:w[k] for k in ('world_id','split','label')} for w in self.meta['worlds']]}

    def model(self, name):
        if name not in self.meta['models']:
            raise ValueError('Unknown model')
        if name not in self.models:
            path = self.folder/'models'
            if name.endswith('_lgb'):
                self.models[name] = joblib.load(path/f'{name}.joblib')
            elif name in ('graphsage','mlp'):
                network = CaseNetwork(self.meta['config']['neural']['hidden_dimensions'],name=='graphsage')
                network.load_state_dict(torch.load(path/f'{name}.pt',map_location='cpu',weights_only=True))
                network.eval()
                self.models[name] = network
            else:
                self.models[name] = None
        return self.models[name]

    def score(self, cases, tx, obs, world, cutoff, model):
        with INFERENCE_LOCK:
            if model not in self.meta['models']:
                raise ValueError('Unknown model')
            if not cases:
                return [],'not_required'
            matrix = case_matrix(cases)
            graph = None
            embedding_mode = 'not_required'
            contributions = None
            features = CASE_FEATURES
            if model in ('node2vec_lgb','graphsage','mlp'):
                graph = historical_graph(tx,obs,cutoff,self.meta['rule_config']['window_steps'],
                                         self.meta['config']['device_projection_limit'])
                members = member_indices(cases,graph['nodes'])
                if (members == -2).any():
                    raise ValueError('Candidate members are absent from observed history')
            if model == 'node2vec_lgb':
                cached = self.cache.get((world,cutoff))
                embedding = None
                if cached:
                    path = inside_root(cached['path'],self.root)
                    if sha256(path) != cached['sha256']:
                        raise ValueError('Embedding cache checksum mismatch')
                    with np.load(path,allow_pickle=False) as saved:
                        if all(np.array_equal(saved[k],graph[k]) for k in ('nodes','x','edges')):
                            embedding = saved['embedding']
                            embedding_mode = 'verified_historical_cache'
                if embedding is None:
                    cfg = self.meta['config']
                    embedding,_ = node2vec(graph['edges'],len(graph['nodes']),cfg['node2vec'],seed_for(cfg['seed'],world,cutoff))
                    embedding_mode = 'fitted_on_visible_snapshot'
                matrix = np.column_stack((matrix,embedding_features(embedding,members)))
                features = CASE_FEATURES+EMBEDDING_FEATURES
            if model == 'activity_lgb':
                matrix = case_matrix(cases,ACTIVITY_FEATURES)
                features = ACTIVITY_FEATURES
            if model.endswith('_lgb'):
                classifier = self.model(model).booster_
                scores = classifier.predict(matrix)
                contributions = classifier.predict(matrix,pred_contrib=True)
            elif model in ('graphsage','mlp'):
                if 'scalers' not in self.models:
                    self.models['scalers'] = joblib.load(self.folder/'models/scalers.joblib')
                scalers = self.models['scalers']
                with torch.no_grad():
                    scores = torch.sigmoid(self.model(model)(
                        torch.tensor(scalers['node'].transform(graph['x']),dtype=torch.float32),
                        torch.from_numpy(graph['edges']),torch.from_numpy(members),
                        torch.tensor(scalers['case'].transform(matrix),dtype=torch.float32))).numpy()
            else:
                scores = np.asarray([c['scores'][model] for c in cases])
            if not np.isfinite(scores).all():
                raise ValueError('Model produced a non-finite score')
            tx_by_id = {r[0]:r for r in tx}
            obs_by_id = {r[0]:r for r in obs}
            result = []
            for i,case in enumerate(cases):
                contribution = {'available':False,'reason':'This model has no local feature-contribution export'}
                if contributions is not None:
                    values = {name:float(v) for name,v in zip(features,contributions[i,:-1])}
                    contribution = {'available':True,'units':'raw model margin (log-odds), not percentage points',
                                    'bias':float(contributions[i,-1]),'features':values}
                transfers = [dict(zip(('transaction_id','sender_id','receiver_id','amount','step','available_step'),tx_by_id[k])) for k in case['transaction_ids']]
                devices = [dict(zip(('observation_id','account_id','device_id','step','available_step'),obs_by_id[k])) for k in case['device_observation_ids']]
                result.append(dict(case,score=float(scores[i]),contributions=contribution,
                                   evidence={'transactions':transfers,'device_observations':devices}))
            return result,embedding_mode


class InvestigationService:
    def __init__(self, bundle, store):
        self.bundle,self.store = bundle,store
        self.active = set()
        self.lock = threading.RLock()
        for world in bundle.meta['worlds']:
            store.register_world(world['world_id'],world['label'],'Fully synthetic development network')

    def score(self, world_id, cutoff, model=None, policy='tuned_threshold'):
        model = model or self.bundle.meta['default_model']
        if model not in self.bundle.meta['models'] or policy not in ('tuned_threshold','budget_only'):
            raise ValueError('Unknown model or policy')
        cfg = self.bundle.meta['rule_config']
        world,tx,obs = self.store.history(world_id,cutoff,cfg['window_steps'])
        cases,summary = snapshot(tx,obs,cutoff,cfg)
        scored,mode = self.bundle.score(cases,tx,obs,world_id,cutoff,model)
        threshold = self.bundle.meta['models'][model]['threshold'] if policy=='tuned_threshold' else 0.0
        return {'world_id':world_id,'model':model,'policy':policy,'bundle_version':self.bundle.version,
                'snapshot_id':'snapshot_'+fingerprint([world_id,cutoff,model,policy,self.bundle.version,tx,obs])[:24],
                'cutoff_step':cutoff,'window_start':summary['window_start'],'history_revision':world['revision'],
                'threshold':threshold,'score_semantics':'Investigation priority, not fraud probability',
                'provenance':world['provenance'],'embedding_mode':mode,'fallback_reason':None,
                'summary':summary,'candidates':scored}

    def request_replay(self, world_id, model=None, policy='tuned_threshold'):
        model = model or self.bundle.meta['default_model']
        if world_id not in {w['world_id'] for w in self.bundle.meta['worlds']}:
            raise ValueError('Replay supports only bundled development networks')
        if model not in self.bundle.meta['models'] or policy not in ('tuned_threshold','budget_only'):
            raise ValueError('Unknown model or policy')
        return self.store.ensure_replay(world_id,model,policy,self.bundle.version,len(self.bundle.meta['cutoffs']))

    def claim(self, replay_id):
        with self.lock:
            if replay_id in self.active or self.store.replay(replay_id)['status']=='complete':
                return False
            self.active.add(replay_id)
            return True

    def run_replay(self, replay_id, already_claimed=False):
        if not already_claimed and not self.claim(replay_id):
            return self.store.replay(replay_id)
        try:
            job = self.store.replay(replay_id)
            world = next(w for w in self.bundle.meta['worlds'] if w['world_id']==job['world_id'])
            self.store.replay_status(replay_id,'running')
            paths = {}
            for name,entry in world['observed'].items():
                path = inside_root(entry['path'],self.bundle.root)
                if sha256(path)!=entry['sha256']:
                    raise ValueError('Observed replay source changed')
                paths[name] = path
            with duckdb.connect() as con:
                tx = con.execute(f'SELECT transaction_id,sender_id,receiver_id,amount,step,available_step,type FROM read_parquet({literal(paths["transactions"])}) ORDER BY available_step,transaction_id').fetchall()
                obs = con.execute(f'SELECT observation_id,account_id,device_id,step,available_step FROM read_parquet({literal(paths["device_observations"])}) ORDER BY available_step,observation_id').fetchall()
            lower = 0
            cfg = self.bundle.meta['rule_config']
            for cutoff in self.bundle.meta['cutoffs']:
                current_tx = [dict(zip(('transaction_id','sender_id','receiver_id','amount','step','available_step','type'),
                                      (*r[:3],format(r[3],'.2f'),*r[4:]))) for r in tx if lower<=r[5]<cutoff]
                current_obs = [dict(zip(('observation_id','account_id','device_id','step','available_step'),r)) for r in obs if lower<=r[4]<cutoff]
                lower = cutoff
                if self.store.committed(replay_id,cutoff):
                    continue
                self.store.ingest(job['world_id'],f'{replay_id}:{cutoff}',current_tx,current_obs)
                response = self.score(job['world_id'],cutoff,job['model'],job['policy'])
                self.store.commit_snapshot(replay_id,response,cfg['duplicate_jaccard'],cfg['alerts_per_snapshot'])
            self.store.replay_status(replay_id,'complete')
            return self.store.replay(replay_id)
        except Exception as exc:
            self.store.replay_status(replay_id,'failed',str(exc))
            raise
        finally:
            with self.lock:
                self.active.discard(replay_id)
