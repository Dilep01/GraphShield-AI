"""Controlled development-only ranking comparisons on frozen Phase 4 candidates."""

import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from .graph_rules import jaccard
from .learned_graph import (ACTIVITY_FEATURES, CASE_FEATURES, EMBEDDING_FEATURES, NODE_FEATURES,
                            CaseNetwork, case_matrix, embedding_features, historical_graph,
                            member_indices, node2vec)
from .paths import ROOT, inside_root
from .pipeline import sha256, write_json
from .ring_evaluation import aggregate, eligible, evaluate
from .rings import checked_inputs, development_source, read_jsonl, read_observed, read_truth, write_jsonl
from .synthetic import checked_name

VERSION = 'learned-rings-v1'
MODELS = ('activity_lgb', 'graph_lgb', 'node2vec_lgb', 'mlp', 'graphsage', 'graph_only', 'graph_device')


def validate_config(config):
    if config.get('version') != VERSION:
        raise ValueError('Unsupported learned-ring version')
    for key in ('seed', 'threads', 'threshold_quantiles', 'device_projection_limit'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['threads'] > 8 or config['threshold_quantiles'] > 101:
        raise ValueError('Excessive thread or threshold search budget')
    integers = {'lightgbm': ('n_estimators','num_leaves','min_child_samples'),
                'node2vec': ('dimensions','walks_per_node','walk_length','context_size','negative_samples','epochs','batch_size'),
                'neural': ('hidden_dimensions','epochs')}
    for section, keys in integers.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    for section, keys in {'lightgbm': ('learning_rate','reg_lambda'),
                          'node2vec': ('learning_rate','p','q'),
                          'neural': ('learning_rate','weight_decay')}.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{section}.{key} must be finite and positive')
    if config['node2vec']['walk_length'] < 2 or config['node2vec']['context_size'] >= config['node2vec']['walk_length']:
        raise ValueError('Walk length must exceed the context size and be at least two')


def seed_for(seed, world, cutoff):
    return int.from_bytes(hashlib.sha256(f'{seed}:{world}:{cutoff}'.encode()).digest()[:4], 'little') % (2**31-1)


def load_source(rule_run, root):
    folder = inside_root(Path('artifacts/phase4') / checked_name(rule_run), root)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('A completed Phase 4 run is required')
    for name, digest in manifest['files'].items():
        if sha256(folder / name) != digest:
            raise ValueError(f'Phase 4 artifact checksum mismatch: {name}')
    source, data, worlds, source_path = development_source(manifest['source_run'], root)
    if sha256(source_path) != manifest['source_manifest_sha256'] or checked_inputs(data, worlds, source) != manifest['inputs']:
        raise ValueError('Phase 4 source data changed')
    if worlds != manifest['worlds'] or sum(w['split'] == 'validation' for w in worlds) != 2:
        raise ValueError('Exactly two validation worlds and matching development identities are required')
    cases = read_jsonl(folder / 'candidates.jsonl')
    allowed = {(w['world_id'], w['split']) for w in worlds}
    if any((c['world_id'], c['split']) not in allowed for c in cases):
        raise ValueError('Candidate outside development worlds')
    return folder, manifest, source, data, worlds, cases


def candidate_targets(cases, truths, rule_config):
    """Retrospective archival targets, separate from observations.

    Dominant planted-group clusters share unit sample mass, including partial variants;
    background cases share mass across repeated exact member sets. No validation class
    or weight is used in training. All labels describe fully adjudicated offline worlds.
    """
    labels, clusters = [], []
    for case in cases:
        groups, events = truths[case['world_id']]
        labels.append(int(any(g['is_fraud'] and eligible(case, g, events, rule_config) for g in groups)))
        members = set(case['members'])
        overlaps = sorted(((len(members & set(g['members'])) / len(members), g['group_id']) for g in groups), reverse=True)
        if overlaps and overlaps[0][0] >= .5:
            cluster = overlaps[0][1]
        else:
            cluster = 'background:' + ':'.join(sorted(members))
        clusters.append(case['world_id'] + '/' + cluster)
    return np.asarray(labels, np.int64), clusters


def training_weights(labels, clusters):
    counts = Counter(clusters)
    weights = np.asarray([1 / counts[c] for c in clusters], np.float32)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Training candidates must contain both classes')
    for label in (0, 1):
        mask = labels == label
        weights[mask] *= .5 / weights[mask].sum()
    return weights * len(weights)


def select_cases(cases, scores, threshold, rule_config):
    if len(cases) != len(scores) or not np.isfinite(scores).all() or not math.isfinite(threshold):
        raise ValueError('Case scores must be finite and aligned')
    by_time = defaultdict(list)
    for i, case in enumerate(cases):
        by_time[case['cutoff_step']].append(i)
    alerts, decisions = [], []
    for cutoff, indices in sorted(by_time.items()):
        used = 0
        for i in sorted(indices, key=lambda n: (-float(scores[n]), cases[n]['candidate_id'])):
            case = cases[i]
            duplicate = None
            if scores[i] < threshold:
                decision = 'below_score'
            else:
                duplicate = next((old['candidate_id'] for old in alerts
                                  if jaccard(old['members'], case['members']) >= rule_config['duplicate_jaccard']), None)
                if duplicate:
                    decision = 'duplicate'
                elif used >= rule_config['alerts_per_snapshot']:
                    decision = 'capacity'
                else:
                    decision = 'alert'
                    used += 1
                    alerts.append(case)
            decisions.append({'candidate_id': case['candidate_id'], 'cutoff_step': cutoff,
                              'decision': decision, 'duplicate_of': duplicate})
    return alerts, decisions


def ring_f1(result):
    denominator = result['alerts'] + result['fraud_groups']
    return 2 * result['matched_groups'] / denominator if denominator else 0.0


def choose_threshold(cases, scores, groups, events, rule_config, quantiles):
    candidates = sorted(set([0.0, float(max(scores, default=0) + 1e-6)]
                            + np.quantile(scores, np.linspace(0, 1, quantiles)).tolist())) if len(scores) else [0.0]
    search = []
    for threshold in candidates:
        alerts, _ = select_cases(cases, scores, threshold, rule_config)
        result = evaluate(alerts, groups, events, rule_config)
        search.append({'threshold': threshold, 'case_ring_f1': ring_f1(result),
                       'precision': result['case_precision'] or 0.0, 'recall': result['ring_recall'] or 0.0,
                       'alerts': result['alerts'], 'matched_groups': result['matched_groups']})
    best = max(search, key=lambda r: (r['case_ring_f1'], r['precision'], r['threshold']))
    return best, search


def prepare_graphs(con, data, worlds, cases, config, rule_config, output):
    folder = output / 'graphs'
    folder.mkdir()
    graphs, log = [], []
    all_embeddings = np.zeros((len(cases), len(EMBEDDING_FEATURES)), np.float32)
    for world in worlds:
        tx, obs = read_observed(con, data, world)
        by_time = defaultdict(list)
        for i, case in enumerate(cases):
            if case['world_id'] == world['world_id']:
                by_time[case['cutoff_step']].append(i)
        for number, (cutoff, indices) in enumerate(sorted(by_time.items()), 1):
            started = time.perf_counter()
            graph = historical_graph(tx, obs, cutoff, rule_config['window_steps'], config['device_projection_limit'])
            current = [cases[i] for i in indices]
            graph['members'] = member_indices(current, graph['nodes'])
            if np.any(graph['members'] == -2):
                raise ValueError('Candidate member missing from its observed historical graph')
            graph['indices'] = np.asarray(indices, np.int64)
            graph['case_x'] = case_matrix(current)
            graph['embedding'], diagnostics = node2vec(graph['edges'], len(graph['nodes']), config['node2vec'],
                                                      seed_for(config['seed'], world['world_id'], cutoff))
            graph['embedding_x'] = embedding_features(graph['embedding'], graph['members'])
            all_embeddings[indices] = graph['embedding_x']
            name = f"{world['world_id']}-{cutoff}.npz"
            np.savez_compressed(folder / name, **graph)
            metadata = dict(world, cutoff_step=cutoff, filename=name, nodes=len(graph['nodes']),
                            directed_edges=graph['edges'].shape[1], candidates=len(indices),
                            oversized_devices_skipped=int(graph['oversized_devices_skipped']),
                            elapsed_seconds=round(time.perf_counter()-started, 4), **diagnostics)
            log.append(metadata)
            graphs.append(dict(graph, **world))
            if number % 5 == 0 or number == len(by_time):
                print(f"Embedded {world['split']} {world['world_id']}: {number}/{len(by_time)} snapshots", flush=True)
    write_json(output / 'graph_log.json', log)
    return graphs, all_embeddings


def graph_tensors(graphs, node_scaler, case_scaler):
    return [dict(g, x_t=torch.tensor(node_scaler.transform(g['x']), dtype=torch.float32),
                 case_t=torch.tensor(case_scaler.transform(g['case_x']), dtype=torch.float32),
                 edge_t=torch.from_numpy(g['edges']), member_t=torch.from_numpy(g['members'])) for g in graphs]


def neural_predict(model, graphs, count):
    scores = np.zeros(count, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for g in graphs:
            scores[g['indices']] = torch.sigmoid(model(g['x_t'], g['edge_t'], g['member_t'], g['case_t'])).numpy()
    return scores


def fit_models(graphs, cases, labels, clusters, embedding_x, config, output):
    train = np.asarray([c['split'] == 'train' for c in cases])
    if not train.any():
        raise ValueError('No training cases')
    weights = training_weights(labels[train], [c for c, yes in zip(clusters, train) if yes])
    matrix = case_matrix(cases)
    models_dir = output / 'models'
    models_dir.mkdir()
    predictions, timings = {}, {}
    activities = [CASE_FEATURES.index(f) for f in ACTIVITY_FEATURES]
    matrices = {'activity_lgb':matrix[:, activities], 'graph_lgb':matrix,
                'node2vec_lgb':np.column_stack((matrix, embedding_x))}
    for name, features in matrices.items():
        start = time.perf_counter()
        model = lgb.LGBMClassifier(**config['lightgbm'], random_state=config['seed'],
                                  n_jobs=config['threads'], verbosity=-1, deterministic=True, force_col_wise=True)
        model.fit(features[train], labels[train], sample_weight=weights)
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        predictions[name] = model.booster_.predict(features)
        predict_seconds = time.perf_counter()-start
        joblib.dump(model, models_dir / f'{name}.joblib')
        model.booster_.save_model(str(models_dir / f'{name}.txt'))
        names = ACTIVITY_FEATURES if name == 'activity_lgb' else CASE_FEATURES + (EMBEDDING_FEATURES if name == 'node2vec_lgb' else ())
        write_json(models_dir / f'{name}_importance.json', dict(zip(names, model.booster_.feature_importance('gain').tolist())))
        timings[name] = {'fit_seconds':fit_seconds, 'score_all_candidates_seconds':predict_seconds}
    node_scaler = StandardScaler().fit(np.concatenate([g['x'] for g in graphs if g['split'] == 'train']))
    case_scaler = StandardScaler().fit(matrix[train])
    joblib.dump({'node':node_scaler, 'case':case_scaler}, models_dir / 'scalers.joblib')
    tensors = graph_tensors(graphs, node_scaler, case_scaler)
    train_graphs = [g for g in tensors if g['split'] == 'train']
    all_weights = np.zeros(len(cases), np.float32)
    all_weights[train] = weights
    for name in ('mlp', 'graphsage'):
        started = time.perf_counter()
        torch.manual_seed(config['seed'])
        rng = np.random.default_rng(config['seed'])
        model = CaseNetwork(config['neural']['hidden_dimensions'], use_edges=name == 'graphsage')
        optimizer = torch.optim.Adam(model.parameters(), lr=config['neural']['learning_rate'],
                                     weight_decay=config['neural']['weight_decay'])
        history = []
        for epoch in range(config['neural']['epochs']):
            model.train()
            total = 0.0
            for gi in rng.permutation(len(train_graphs)):
                graph = train_graphs[gi]
                indices = graph['indices']
                logits = model(graph['x_t'], graph['edge_t'], graph['member_t'], graph['case_t'])
                loss = (F.binary_cross_entropy_with_logits(logits, torch.tensor(labels[indices], dtype=torch.float32), reduction='none')
                        * torch.tensor(all_weights[indices])).sum() / weights.sum()
                optimizer.zero_grad()
                (loss * len(train_graphs)).backward()
                optimizer.step()
                total += loss.item()
            history.append(total)
            if (epoch+1) % 10 == 0:
                print(f"{name}: epoch {epoch+1}/{config['neural']['epochs']}, weighted loss {total:.4f}", flush=True)
        fit_seconds = time.perf_counter()-started
        started = time.perf_counter()
        predictions[name] = neural_predict(model, tensors, len(cases))
        timings[name] = {'fit_seconds':fit_seconds, 'score_all_candidates_seconds':time.perf_counter()-started,
                         'parameter_count':sum(p.numel() for p in model.parameters())}
        torch.save(model.state_dict(), models_dir / f'{name}.pt')
        write_json(models_dir / f'{name}_loss.json', history)
    for name in ('graph_only', 'graph_device'):
        predictions[name] = np.asarray([c['scores'][name] for c in cases])
    return predictions, timings


def evaluate_models(cases, predictions, truths, worlds, labels, config, rule_config):
    validation = [w['world_id'] for w in worlds if w['split'] == 'validation']
    tuning_world, comparison_world = validation
    tuning_indices = [i for i, c in enumerate(cases) if c['world_id'] == tuning_world]
    tuning_cases = [cases[i] for i in tuning_indices]
    thresholds, searches = {}, {}
    for name in MODELS:
        thresholds[name], searches[name] = choose_threshold(tuning_cases, predictions[name][tuning_indices],
                                                            *truths[tuning_world], rule_config, config['threshold_quantiles'])
    # Freeze the choice using the tuning world only; comparison results cannot alter it.
    selected = max(MODELS, key=lambda name: (thresholds[name]['case_ring_f1'], thresholds[name]['precision'], -MODELS.index(name)))
    rows, alert_log, decision_log = [], [], []
    for world in worlds:
        indices = [i for i, c in enumerate(cases) if c['world_id'] == world['world_id']]
        current = [cases[i] for i in indices]
        role = 'train' if world['split'] == 'train' else 'tuning' if world['world_id'] == tuning_world else 'comparison'
        for name in MODELS:
            scores = predictions[name][indices]
            for policy, threshold in (('budget_only', 0.0), ('tuned_threshold', thresholds[name]['threshold'])):
                alerts, decisions = select_cases(current, scores, threshold, rule_config)
                result = evaluate(alerts, *truths[world['world_id']], rule_config)
                result['case_ring_f1'] = ring_f1(result)
                result['candidate_ap'] = float(average_precision_score(labels[indices], scores)) if labels[indices].any() else None
                rows.append(dict(world, role=role, model=name, policy=policy, threshold=threshold, metrics=result))
                alert_log.extend(dict(c, model=name, policy=policy) for c in alerts)
                decision_log.extend(dict(d, **world, model=name, policy=policy) for d in decisions)
    summary = {}
    for role in ('train', 'tuning', 'comparison'):
        summary[role] = {}
        for name in MODELS:
            summary[role][name] = {}
            for policy in ('budget_only', 'tuned_threshold'):
                relevant = [r for r in rows if r['role'] == role and r['model'] == name and r['policy'] == policy]
                metrics = aggregate([r['metrics'] for r in relevant])
                metrics['case_ring_f1'] = ring_f1(metrics)
                indices = [i for i, c in enumerate(cases) if any(c['world_id'] == r['world_id'] for r in relevant)]
                metrics['candidate_ap'] = float(average_precision_score(labels[indices], predictions[name][indices])) if labels[indices].any() else None
                metrics['candidate_prevalence'] = float(labels[indices].mean()) if indices else None
                summary[role][name][policy] = metrics
    return {'summary':summary, 'worlds':rows, 'thresholds':thresholds, 'threshold_searches':searches,
            'tuning_world':tuning_world, 'comparison_world':comparison_world, 'selected_on_tuning':selected,
            'final_test_evaluated':False}, alert_log, decision_log


def report_text(result):
    lines = ['# Phase 5 learned ring-ranking comparison', '',
             'Fully synthetic development results. The comparison world was already inspected in Phase 4;',
             'it was excluded from Phase 5 fitting, threshold selection and model selection.', '',
             f"Model selected on the tuning world: **{result['selected_on_tuning']}**.", '',
             '| Model | Policy | Alerts | Recovered groups | Case precision | Ring recall | Case/ring F1 | Candidate AP |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        for policy, r in result['summary']['comparison'][name].items():
            precision = f"{r['case_precision']:.2%}" if r['case_precision'] is not None else 'n/a'
            ap = f"{r['candidate_ap']:.4f}" if r['candidate_ap'] is not None else 'n/a'
            lines.append(f"| {name} | {policy} | {r['alerts']} | {r['matched_groups']}/{r['fraud_groups']} | {precision} | "
                         f"{r['ring_recall']:.2%} | {r['case_ring_f1']:.4f} | {ap} |")
    lines += ['', 'All methods use the same candidates, historical boundaries, duplicate policy and maximum',
              'three alerts per world/snapshot. Budget-only uses no score floor; tuned thresholds can abstain.',
              'Candidate AP counts repeated snapshot observations and is not unique-ring precision.',
              'Only one network (12 planted fraud groups) is used for the Phase 5 comparison.',
              'Scores are uncalibrated priorities. Neither final-test split was evaluated.', '']
    return '\n'.join(lines)


def train_rings(rule_run, run_name, config_path=Path('config/phase5.json'), root=ROOT):
    root = Path(root).resolve()
    config = json.loads(inside_root(config_path, root).read_text(encoding='utf-8'))
    validate_config(config)
    folder, rule_manifest, source, data, worlds, cases = load_source(rule_run, root)
    output = inside_root(Path('artifacts/phase5') / checked_name(run_name), root)
    if output.exists():
        raise ValueError('Run already exists; use a new run name')
    output.mkdir(parents=True)
    work = inside_root(Path('.cache/phase5') / run_name, root)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    manifest = {'status':'running', 'version':VERSION, 'rule_run':rule_run, 'config':config,
                'rule_manifest_sha256':sha256(folder / 'manifest.json'), 'worlds':worlds,
                'source_sha256':{name:sha256(Path(__file__).parent / name) for name in
                                 ('learned_graph.py','learned_rings.py','ring_evaluation.py','rings.py','graph_rules.py')},
                'versions':{name:importlib.metadata.version(name) for name in ('torch','numpy','lightgbm','scikit-learn','duckdb','joblib')},
                'device':'cpu', 'final_test_evaluated':False}
    write_json(output / 'manifest.json', manifest)
    try:
        with duckdb.connect(config={'memory_limit':'512MB', 'threads':'2', 'temp_directory':str(work)}) as con:
            feature_start = time.perf_counter()
            graphs, embedding_x = prepare_graphs(con, data, worlds, cases, config, rule_manifest['config'], output)
            feature_seconds = time.perf_counter()-feature_start
            # Observed feature artifacts are frozen before any truth is used for learning.
            truths = {w['world_id']:read_truth(con, data, w) for w in worlds}
        labels, clusters = candidate_targets(cases, truths, rule_manifest['config'])
        write_jsonl(output / 'targets.jsonl', [dict(candidate_id=c['candidate_id'], world_id=c['world_id'],
                                                   label=int(y), training_cluster=cluster)
                                              for c, y, cluster in zip(cases, labels, clusters)])
        predictions, timings = fit_models(graphs, cases, labels, clusters, embedding_x, config, output)
        write_jsonl(output / 'predictions.jsonl', [dict(candidate_id=c['candidate_id'], world_id=c['world_id'],
                                                       scores={name:float(values[i]) for name, values in predictions.items()})
                                                  for i, c in enumerate(cases)])
        evaluation, alerts, decisions = evaluate_models(cases, predictions, truths, worlds, labels, config, rule_manifest['config'])
        write_json(output / 'evaluation.json', evaluation)
        write_jsonl(output / 'alerts.jsonl', alerts)
        write_jsonl(output / 'decisions.jsonl', decisions)
        (output / 'report.md').write_text(report_text(evaluation), encoding='utf-8')
        write_json(output / 'feature_contract.json', {'case_features':list(CASE_FEATURES), 'activity_features':list(ACTIVITY_FEATURES),
                                                     'node_features':list(NODE_FEATURES), 'embedding_features':list(EMBEDDING_FEATURES),
                                                     'labels':'Retrospective, fully adjudicated archival development worlds',
                                                     'scores':'Uncalibrated investigation priorities',
                                                     'node2vec_transfer':'Snapshot-local embeddings; only pair cosine summaries enter classifier',
                                                     'graphsage':'Two full-neighbor mean layers on an undirected transfer/device projection'})
        timings['historical_graph_and_node2vec_seconds'] = feature_seconds
        write_json(output / 'timings.json', timings)
        manifest.update(status='complete', elapsed_seconds=round(time.perf_counter()-started, 2),
                        selected_on_tuning=evaluation['selected_on_tuning'])
        manifest['files'] = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                             if p.is_file() and p.name != 'manifest.json'}
        write_json(output / 'manifest.json', manifest)
        return evaluation
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        write_json(output / 'manifest.json', manifest)
        raise


def verify_learned(run_name, root=ROOT):
    root = Path(root).resolve()
    output = inside_root(Path('artifacts/phase5') / checked_name(run_name), root)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Only completed learned runs can be verified')
    checks = {
        'output_hashes_match':all(sha256(inside_root(name, output)) == digest for name, digest in manifest['files'].items()),
        'source_code_matches':all(sha256(Path(__file__).parent / name) == digest for name, digest in manifest['source_sha256'].items()),
    }
    if not all(checks.values()):
        raise ValueError('Artifact or source checksum mismatch; model loading refused')
    folder, rule_manifest, source, data, worlds, cases = load_source(manifest['rule_run'], root)
    config = manifest['config']
    validate_config(config)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    checks['phase4_manifest_matches'] = sha256(folder / 'manifest.json') == manifest['rule_manifest_sha256']
    checks['development_worlds_only'] = worlds == manifest['worlds'] and not manifest['final_test_evaluated']
    log = json.loads((output / 'graph_log.json').read_text(encoding='utf-8'))
    graphs, coverage = [], []
    embedding_x = np.zeros((len(cases), len(EMBEDDING_FEATURES)), np.float32)
    graph_ok, member_ok, embedding_ok, case_ok = True, True, True, True
    work = inside_root(Path('.cache/phase5') / run_name, root)
    work.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={'temp_directory':str(work), 'threads':'2', 'memory_limit':'512MB'}) as con:
        for world in worlds:
            tx, obs = read_observed(con, data, world)
            for entry in (r for r in log if r['world_id'] == world['world_id']):
                with np.load(inside_root(Path('graphs') / entry['filename'], output), allow_pickle=False) as saved:
                    graph = {key:saved[key] for key in saved.files}
                rebuilt = historical_graph(tx, obs, entry['cutoff_step'], rule_manifest['config']['window_steps'],
                                           config['device_projection_limit'])
                graph_ok &= all(np.array_equal(graph[k], v) for k, v in rebuilt.items())
                indices = graph['indices'].tolist()
                expected = [i for i, c in enumerate(cases) if c['world_id'] == world['world_id'] and c['cutoff_step'] == entry['cutoff_step']]
                member_ok &= indices == expected and np.array_equal(graph['members'], member_indices([cases[i] for i in indices], graph['nodes']))
                case_ok &= np.array_equal(graph['case_x'], case_matrix([cases[i] for i in indices]))
                embedding_ok &= np.isfinite(graph['embedding']).all() and np.allclose(graph['embedding_x'], embedding_features(graph['embedding'], graph['members']), atol=1e-7)
                coverage.extend(indices)
                embedding_x[indices] = graph['embedding_x']
                graphs.append(dict(graph, **world))
        truths = {w['world_id']:read_truth(con, data, w) for w in worlds}
    checks.update(historical_graphs_reproduce=bool(graph_ok), member_indices_reproduce=bool(member_ok),
                  explicit_case_features_match=bool(case_ok), embedding_summaries_reproduce=bool(embedding_ok),
                  every_candidate_exactly_once=sorted(coverage) == list(range(len(cases))))
    labels, clusters = candidate_targets(cases, truths, rule_manifest['config'])
    targets = [dict(candidate_id=c['candidate_id'], world_id=c['world_id'], label=int(y), training_cluster=cluster)
               for c, y, cluster in zip(cases, labels, clusters)]
    checks['retrospective_targets_reproduce'] = targets == read_jsonl(output / 'targets.jsonl')
    matrix = case_matrix(cases)
    train = np.asarray([c['split'] == 'train' for c in cases])
    scalers = joblib.load(output / 'models/scalers.joblib')
    reference_node = StandardScaler().fit(np.concatenate([g['x'] for g in graphs if g['split'] == 'train']))
    reference_case = StandardScaler().fit(matrix[train])
    checks['scalers_fit_training_only'] = all(np.allclose(getattr(scalers[name], stat), getattr(ref, stat))
                                             for name, ref in (('node',reference_node), ('case',reference_case))
                                             for stat in ('mean_','scale_'))
    activities = [CASE_FEATURES.index(f) for f in ACTIVITY_FEATURES]
    matrices = {'activity_lgb':matrix[:, activities], 'graph_lgb':matrix,
                'node2vec_lgb':np.column_stack((matrix, embedding_x))}
    predictions = {name:joblib.load(output / 'models' / f'{name}.joblib').booster_.predict(x) for name, x in matrices.items()}
    tensors = graph_tensors(graphs, scalers['node'], scalers['case'])
    for name in ('mlp','graphsage'):
        model = CaseNetwork(config['neural']['hidden_dimensions'], use_edges=name == 'graphsage')
        model.load_state_dict(torch.load(output / 'models' / f'{name}.pt', map_location='cpu', weights_only=True))
        predictions[name] = neural_predict(model, tensors, len(cases))
    for name in ('graph_only','graph_device'):
        predictions[name] = np.asarray([c['scores'][name] for c in cases])
    saved = read_jsonl(output / 'predictions.jsonl')
    checks['saved_prediction_identity_matches'] = [(r['world_id'],r['candidate_id']) for r in saved] == [(c['world_id'],c['candidate_id']) for c in cases]
    checks['saved_model_predictions_reproduce'] = all(np.allclose(values, [r['scores'][name] for r in saved], atol=1e-7, rtol=1e-6) for name, values in predictions.items())
    evaluation, alerts, decisions = evaluate_models(cases, predictions, truths, worlds, labels, config, rule_manifest['config'])
    saved_eval = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
    checks['thresholds_selection_and_metrics_reproduce'] = evaluation == saved_eval
    checks['alerts_reproduce'] = alerts == read_jsonl(output / 'alerts.jsonl')
    checks['decisions_reproduce'] = decisions == read_jsonl(output / 'decisions.jsonl')
    checks['report_reproduces'] = report_text(evaluation) == (output / 'report.md').read_text(encoding='utf-8')
    checks['selection_uses_tuning_only'] = manifest['selected_on_tuning'] == evaluation['selected_on_tuning']
    counts = Counter((c['world_id'], c['model'], c['policy'], c['cutoff_step']) for c in alerts)
    checks['review_budget_respected'] = all(v <= rule_manifest['config']['alerts_per_snapshot'] for v in counts.values())
    checks['matches_are_one_to_one'] = all(len(r['metrics']['matches']) == len({m['group_id'] for m in r['metrics']['matches']}) == len({m['candidate_id'] for m in r['metrics']['matches']}) for r in saved_eval['worlds'])
    write_json(output / 'verification.json', checks)
    if not all(checks.values()):
        raise ValueError(f'Learned verification failed: {[k for k, v in checks.items() if not v]}')
    return checks
