"""Phase 4 runner and reproducibility checks. Test worlds are never opened here."""

import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .graph_rules import POLICIES, VERSION, select_alerts, snapshot, validate_config
from .paths import ROOT, inside_root
from .pipeline import literal, records, sha256, write_json
from .ring_evaluation import aggregate, evaluate
from .synthetic import checked_name

DEVELOPMENT_SPLITS = ('train', 'validation')
SOURCE_FILES = {'observed': ('transactions', 'device_observations'),
                'ground_truth': ('groups', 'group_memberships', 'transaction_labels')}


def development_source(source_run, root):
    path = inside_root(Path('artifacts/phase3') / checked_name(source_run) / 'manifest.json', root)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Phase 4 requires a completed synthetic source run')
    data = inside_root(manifest['output_path'], root)
    worlds = [{'world_id': checked_name(w['world_id']), 'split': w['split']} for w in manifest['worlds']
              if w['split'] in DEVELOPMENT_SPLITS]
    if {w['split'] for w in worlds} != set(DEVELOPMENT_SPLITS):
        raise ValueError('Training and validation worlds are required')
    if len({w['world_id'] for w in worlds}) != len(worlds):
        raise ValueError('Duplicate development world IDs')
    return manifest, data, worlds, path


def source_file(data, world, kind, table):
    # Direct callers cannot switch this runner to a test split.
    if world['split'] not in DEVELOPMENT_SPLITS or table not in SOURCE_FILES.get(kind, ()):
        raise ValueError('Only explicitly supported development inputs may be opened')
    return inside_root(Path(kind) / world['split'] / checked_name(world['world_id']) / f'{table}.parquet', data)


def checked_inputs(data, worlds, source):
    hashes = {}
    for world in worlds:
        for kind, tables in SOURCE_FILES.items():
            for table in tables:
                path = source_file(data, world, kind, table)
                key = str(path.relative_to(data))
                checksum = sha256(path)
                if checksum != source['files'].get(key):
                    raise ValueError(f'Synthetic source checksum mismatch: {key}')
                hashes[key] = checksum
    return hashes


def read_observed(con, data, world):
    tx = source_file(data, world, 'observed', 'transactions')
    obs = source_file(data, world, 'observed', 'device_observations')
    return (
        con.execute(f'SELECT transaction_id, sender_id, receiver_id, amount, step, available_step '
                    f'FROM read_parquet({literal(tx)}) ORDER BY transaction_id').fetchall(),
        con.execute(f'SELECT observation_id, account_id, device_id, step, available_step '
                    f'FROM read_parquet({literal(obs)}) ORDER BY observation_id').fetchall(),
    )


def read_truth(con, data, world):
    group_path = source_file(data, world, 'ground_truth', 'groups')
    member_path = source_file(data, world, 'ground_truth', 'group_memberships')
    event_path = source_file(data, world, 'ground_truth', 'transaction_labels')
    groups = records(con, f'SELECT group_id, scenario_family, is_fraud, start_step, end_step '
                     f'FROM read_parquet({literal(group_path)}) ORDER BY group_id')
    memberships = con.execute(f'SELECT group_id, account_id FROM read_parquet({literal(member_path)}) '
                              'ORDER BY group_id, account_id').fetchall()
    members = {}
    for gid, account in memberships:
        members.setdefault(gid, []).append(account)
    for group in groups:
        group['members'] = members[group['group_id']]
    event_groups = dict(con.execute(f'SELECT transaction_id, group_id FROM read_parquet({literal(event_path)}) '
                                   'WHERE group_id IS NOT NULL').fetchall())
    return groups, event_groups


def cutoffs(source, config):
    first, last = source['config']['warmup_steps'] + 1, source['config']['steps'] + 1
    return sorted(set(range(first, last + 1, config['snapshot_stride'])) | {last})


def build_world(con, data, world, source, config):
    transactions, observations = read_observed(con, data, world)
    cases, summaries = [], []
    for cutoff in cutoffs(source, config):
        current, summary = snapshot(transactions, observations, cutoff, config)
        cases.extend(dict(case, **world) for case in current)
        summaries.append(dict(summary, **world))
    return cases, summaries


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_worlds(con, data, worlds, candidates, alerts, config):
    evaluations = []
    for world in worlds:
        groups, event_groups = read_truth(con, data, world)
        world_cases = [c for c in candidates if c['world_id'] == world['world_id']]
        coverage = evaluate(world_cases, groups, event_groups, config)
        policies = {}
        for policy in POLICIES:
            emitted = [c for c in alerts if c['world_id'] == world['world_id'] and c['policy'] == policy]
            policies[policy] = evaluate(emitted, groups, event_groups, config)
        evaluations.append(dict(world, policies=policies,
                                candidate_coverage={'fraud_groups': coverage['fraud_groups'],
                                                    'reachable_groups': coverage['matched_groups'],
                                                    'recall': coverage['ring_recall']}))
    summary = {}
    for split in DEVELOPMENT_SPLITS:
        relevant = [r for r in evaluations if r['split'] == split]
        summary[split] = {policy: aggregate([r['policies'][policy] for r in relevant]) for policy in POLICIES}
        total = sum(r['candidate_coverage']['fraud_groups'] for r in relevant)
        reachable = sum(r['candidate_coverage']['reachable_groups'] for r in relevant)
        summary[split]['candidate_coverage'] = {'fraud_groups': total, 'reachable_groups': reachable,
                                               'recall': reachable / total if total else None}
    return {'summary': summary, 'worlds': evaluations,
            'provenance': 'Fully synthetic development networks; retrospective group matching',
            'final_test_evaluated': False}


def report_markdown(evaluation, snapshots, candidates, decisions):
    lines = ['# Phase 4 graph-rule development results', '',
             'Fully synthetic training/validation networks only. No final-test evaluation.', '',
             f'{len(snapshots):,} historical snapshots; {len(candidates):,} candidate observations.', '',
             '| Split | Policy | Alerts | Matched / groups | Case precision | Ring recall | Benign groups alerted |',
             '|---|---|---:|---:|---:|---:|---:|']
    def percent(value):
        return f'{value:.2%}' if value is not None else 'n/a'
    for split, policies in evaluation['summary'].items():
        for policy in POLICIES:
            r = policies[policy]
            lines.append(f"| {split} | {policy} | {r['alerts']} | {r['matched_groups']} / {r['fraud_groups']} | "
                         f"{percent(r['case_precision'])} | {percent(r['ring_recall'])} | "
                         f"{r['benign_groups_alerted']} / {r['benign_groups']} |")
    lines.extend(['', 'Candidate coverage before scoring, capacity and duplicate suppression:', ''])
    for split, policies in evaluation['summary'].items():
        r = policies['candidate_coverage']
        lines.append(f"- {split}: {r['reachable_groups']} / {r['fraud_groups']} ({percent(r['recall'])}).")
    lines.extend(['', 'Policy decision counts:', ''])
    for policy in POLICIES:
        counts = dict(sorted(Counter(d['decision'] for d in decisions if d['policy'] == policy).items()))
        lines.append(f'- {policy}: {counts}.')
    largest = max((s['largest_weak_component'] for s in snapshots), default=0)
    lines.extend(['', f'Largest historical weak component: {largest:,} accounts. Weak components are not cases.', '',
                  'Case precision counts each planted fraudulent group at most once; unmatched alerts include',
                  'legitimate lookalikes, background matches and repeated matches to an already matched group.',
                  'Matching requires the saved member precision and recall thresholds, plus an observed planted',
                  'event in the alert evidence. Config values are saved in manifest.json.', '',
                  'Scores express heuristic investigation priority, not fraud probability. Shared devices and',
                  'repeated patterns also occur in legitimate groups. This run does not establish real-world accuracy.', '',
                  'Both policies and all constants were specified before this run. No validation-driven winner',
                  'or threshold was selected. Candidate coverage is an oracle upper bound, not an alert-policy result.', ''])
    return '\n'.join(lines)


def run_rules(source_run, run_name, config_path=Path('config/phase4.json'), root=ROOT):
    root = Path(root).resolve()
    source, data, worlds, source_manifest_path = development_source(source_run, root)
    config = json.loads(inside_root(config_path, root).read_text(encoding='utf-8'))
    validate_config(config)
    output = inside_root(Path('artifacts/phase4') / checked_name(run_name), root)
    if output.exists():
        raise ValueError('Run already exists; choose a new run name')
    inputs = checked_inputs(data, worlds, source)
    output.mkdir(parents=True)
    work = inside_root(Path('.cache/phase4') / run_name, root)
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_dir = Path(__file__).resolve().parent
    manifest = {
        'status': 'running', 'version': VERSION, 'source_run': source_run,
        'source_manifest_sha256': sha256(source_manifest_path), 'config': config, 'worlds': worlds,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'python_version': platform.python_version(),
        'duckdb_version': duckdb.__version__, 'inputs': inputs,
        'source_sha256': {name: sha256(source_dir / name) for name in
                          ('graph_rules.py', 'ring_evaluation.py', 'rings.py')},
        'final_test_evaluated': False, 'score_semantics': 'uncalibrated investigation priority',
    }
    write_json(output / 'manifest.json', manifest)
    candidates, summaries, alerts, decisions = [], [], [], []
    try:
        with duckdb.connect(config={'memory_limit': '512MB', 'threads': '2', 'temp_directory': str(work)}) as con:
            # Finish and save all observed-only results before loading evaluation truth.
            for world in worlds:
                current, snapshots = build_world(con, data, world, source, config)
                candidates.extend(current)
                summaries.extend(snapshots)
                for policy in POLICIES:
                    emitted, log = select_alerts(current, config, policy)
                    alerts.extend(dict(case, policy=policy) for case in emitted)
                    decisions.extend(dict(row, **world, policy=policy) for row in log)
                print(f"Built {world['split']} {world['world_id']}: {len(current)} candidates", flush=True)
            for filename, rows in (('candidates.jsonl', candidates), ('snapshots.jsonl', summaries),
                                   ('alerts.jsonl', alerts), ('decisions.jsonl', decisions)):
                write_jsonl(output / filename, rows)
            evaluation = score_worlds(con, data, worlds, candidates, alerts, config)
        write_json(output / 'evaluation.json', evaluation)
        (output / 'report.md').write_text(report_markdown(evaluation, summaries, candidates, decisions), encoding='utf-8')
        manifest['files'] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != 'manifest.json'}
        manifest.update(status='complete', elapsed_seconds=round(time.perf_counter() - started, 2))
        write_json(output / 'manifest.json', manifest)
        return evaluation
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        write_json(output / 'manifest.json', manifest)
        raise


def verify_rules(run_name, root=ROOT):
    """Replay every development snapshot and metric; validate saved evidence directly."""
    root = Path(root).resolve()
    output = inside_root(Path('artifacts/phase4') / checked_name(run_name), root)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Only completed runs can be verified')
    config = manifest['config']
    validate_config(config)
    source, data, worlds, source_path = development_source(manifest['source_run'], root)
    checks = {
        'source_manifest_unchanged': sha256(source_path) == manifest['source_manifest_sha256'],
        'selected_inputs_unchanged': checked_inputs(data, worlds, source) == manifest['inputs'],
        'only_development_worlds': worlds == manifest['worlds'] and not manifest['final_test_evaluated'],
        'source_code_unchanged': all(sha256(Path(__file__).parent / name) == expected
                                     for name, expected in manifest['source_sha256'].items()),
        'output_checksums_match': all(sha256(output / name) == expected for name, expected in manifest['files'].items()),
    }
    candidates = read_jsonl(output / 'candidates.jsonl')
    snapshots = read_jsonl(output / 'snapshots.jsonl')
    alerts = read_jsonl(output / 'alerts.jsonl')
    decisions = read_jsonl(output / 'decisions.jsonl')
    reconstructed, summaries, emitted, logs = [], [], [], []
    temporal_ok, membership_ok, capacity_ok = True, True, True
    work = inside_root(Path('.cache/phase4') / run_name, root)
    work.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={'memory_limit': '512MB', 'threads': '2', 'temp_directory': str(work)}) as con:
        for world in worlds:
            current, summary = build_world(con, data, world, source, config)
            reconstructed.extend(current)
            summaries.extend(summary)
            tx, obs = read_observed(con, data, world)
            tx_by_id, obs_by_id = {r[0]: r for r in tx}, {r[0]: r for r in obs}
            for case in (c for c in candidates if c['world_id'] == world['world_id']):
                lower, cutoff = case['window_start'], case['cutoff_step']
                members = set(case['members'])
                for event in case['transaction_ids']:
                    row = tx_by_id[event]
                    temporal_ok &= lower <= row[4] < cutoff and row[5] < cutoff
                    membership_ok &= row[1] in members and row[2] in members
                for event in case['device_observation_ids']:
                    row = obs_by_id[event]
                    temporal_ok &= lower <= row[3] < cutoff and row[4] < cutoff
                    membership_ok &= row[1] in members and row[2] in case['shared_device_ids']
            for policy in POLICIES:
                selected, log = select_alerts(current, config, policy)
                emitted.extend(dict(c, policy=policy) for c in selected)
                logs.extend(dict(d, **world, policy=policy) for d in log)
                saved = [c for c in alerts if c['world_id'] == world['world_id'] and c['policy'] == policy]
                capacity_ok &= all(count <= config['alerts_per_snapshot'] for count in
                                   Counter(c['cutoff_step'] for c in saved).values())
        evaluation = score_worlds(con, data, worlds, candidates, alerts, config)
    saved_eval = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
    checks.update(
        historical_candidates_reproduce=candidates == reconstructed,
        snapshot_summaries_reproduce=snapshots == summaries,
        alert_selection_reproduces=alerts == emitted,
        suppression_decisions_reproduce=decisions == logs,
        all_evidence_is_historical=bool(temporal_ok), evidence_matches_members=bool(membership_ok),
        review_capacity_respected=bool(capacity_ok), evaluation_reproduces=evaluation == saved_eval,
        report_reproduces=(output / 'report.md').read_text(encoding='utf-8') == report_markdown(saved_eval, snapshots, candidates, decisions),
        unique_candidate_ids=len({(c['world_id'], c['candidate_id']) for c in candidates}) == len(candidates),
        group_matches_are_one_to_one=all(
            len({m['group_id'] for m in r['matches']}) == len(r['matches']) == len({m['candidate_id'] for m in r['matches']})
            for world in saved_eval['worlds'] for r in world['policies'].values()),
    )
    write_json(output / 'verification.json', checks)
    if not all(checks.values()):
        raise ValueError(f'Phase 4 verification failed: {[k for k, value in checks.items() if not value]}')
    return checks
