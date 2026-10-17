"""A separate, reproducible scenario benchmark. Never reads or alters PaySim."""

import csv
import hashlib
import json
import math
import platform
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .paths import ROOT, inside_root
from .pipeline import literal, sha256, write_json

VERSION = 'coordinated-v1'
SPLITS = ('train', 'validation', 'test_known', 'test_unseen')
FAMILIES = {'cycle', 'fan_in', 'fan_out', 'relay_chain'}
SCHEMAS = {
    'accounts': {'account_id': 'VARCHAR', 'created_step': 'BIGINT', 'available_step': 'BIGINT'},
    'devices': {'device_id': 'VARCHAR', 'device_type': 'VARCHAR', 'created_step': 'BIGINT', 'available_step': 'BIGINT'},
    'transactions': {'transaction_id': 'VARCHAR', 'sender_id': 'VARCHAR', 'receiver_id': 'VARCHAR',
                     'amount': 'DECIMAL(24,2)', 'step': 'BIGINT', 'available_step': 'BIGINT', 'type': 'VARCHAR'},
    'device_observations': {'observation_id': 'VARCHAR', 'account_id': 'VARCHAR', 'device_id': 'VARCHAR',
                            'step': 'BIGINT', 'available_step': 'BIGINT'},
    'groups': {'group_id': 'VARCHAR', 'scenario_family': 'VARCHAR', 'is_fraud': 'BOOLEAN',
               'start_step': 'BIGINT', 'end_step': 'BIGINT', 'label_available_step': 'BIGINT'},
    'group_memberships': {'group_id': 'VARCHAR', 'account_id': 'VARCHAR'},
    'ring_memberships': {'group_id': 'VARCHAR', 'account_id': 'VARCHAR'},
    'transaction_labels': {'transaction_id': 'VARCHAR', 'is_coordinated_fraud': 'BOOLEAN',
                           'group_id': 'VARCHAR', 'label_available_step': 'BIGINT'},
}
OBSERVED = ('accounts', 'devices', 'transactions', 'device_observations')
TRUTH = ('groups', 'group_memberships', 'ring_memberships', 'transaction_labels')


def checked_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name) or Path(name).is_reserved():
        raise ValueError('Use a non-reserved run name containing only letters, numbers, hyphens or underscores')
    return name


def validate_config(config):
    if config.get('generator_version') != VERSION:
        raise ValueError('Unsupported generator version')
    integer_keys = ('steps', 'warmup_steps', 'accounts_per_world', 'background_transactions_per_world',
                    'fraud_groups_per_world', 'benign_groups_per_world', 'group_size_min', 'group_size_max',
                    'episodes_min', 'episodes_max', 'max_device_delay_steps')
    if any(type(config.get(key)) is not int or config[key] < 1 for key in integer_keys):
        raise ValueError('Count and time settings must be positive integers')
    if not 3 <= config['group_size_min'] <= config['group_size_max'] <= 12:
        raise ValueError('Group size must be ordered within [3,12]')
    if not 2 <= config['episodes_min'] <= config['episodes_max'] <= 20:
        raise ValueError('Episode counts must be ordered within [2,20]')
    reserve = max(128, config['group_size_max'] * 7 + 50)
    if config['steps'] - reserve - config['warmup_steps'] <= config['episodes_max']:
        raise ValueError('Horizon must leave warmup, episode timing and discovery-delay space')
    if config['max_device_delay_steps'] > 24:
        raise ValueError('Device observation delay must not exceed 24 steps')
    groups = config['fraud_groups_per_world'] + config['benign_groups_per_world']
    if config['accounts_per_world'] < groups * config['group_size_max'] + 20:
        raise ValueError('Not enough accounts for disjoint planted groups and background hubs')
    if config['background_transactions_per_world'] < config['accounts_per_world']:
        raise ValueError('Background must be large enough to support persistent activity')
    for key in ('device_observation_probability', 'shared_group_device_probability'):
        if not isinstance(config.get(key), (int, float)) or not 0 < config[key] < 1:
            raise ValueError('Observation and device-sharing probabilities must lie strictly between 0 and 1')
    known, heldout = config['known_families'], config['heldout_families']
    if not known or not heldout or len(set(known)) != len(known) or len(set(heldout)) != len(heldout):
        raise ValueError('Family lists must be nonempty without duplicates')
    if set(known) & set(heldout) or not (set(known) | set(heldout)) <= FAMILIES:
        raise ValueError('Known and held-out families must be supported and disjoint')
    if min(config['fraud_groups_per_world'], config['benign_groups_per_world']) < max(len(known), len(heldout)):
        raise ValueError('Every family needs at least one fraudulent group and one benign lookalike')
    if set(config['world_seeds']) != set(SPLITS):
        raise ValueError('Expected train, validation, test_known and test_unseen seed lists')
    seeds = []
    for split in SPLITS:
        values = config['world_seeds'][split]
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
            raise ValueError('Each split needs a nonempty list of nonnegative integer seeds')
        seeds.extend(values)
    if len(seeds) != len(set(seeds)):
        raise ValueError('World seeds must be unique across all splits')


def opaque_id(seed, kind, index):
    value = hashlib.sha256(f'{VERSION}:{seed}:{kind}:{index}'.encode()).hexdigest()[:24]
    return f'{kind}_{value}'


def stream_rng(seed, stream):
    return random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{stream}'.encode()).digest(), 'big'))


def pattern_edges(family, members):
    if family == 'cycle':
        return list(zip(members, members[1:] + members[:1]))
    if family == 'fan_in':
        return [(account, members[0]) for account in members[1:]]
    if family == 'fan_out':
        return [(members[0], account) for account in members[1:]]
    if family == 'relay_chain':
        return list(zip(members, members[1:]))
    raise ValueError(f'Unknown pattern family: {family}')


def generate_world(config, split, seed):
    """Return a bounded world in memory; all serialized identifiers are label-neutral."""
    families = config['heldout_families'] if split == 'test_unseen' else config['known_families']
    rng = stream_rng(seed, 'population')
    background_rng = stream_rng(seed, 'background')
    group_rng = stream_rng(seed, 'groups')
    device_rng = stream_rng(seed, 'device-assignment')
    observation_rng = stream_rng(seed, 'device-observations')
    label_rng = stream_rng(seed, 'discovery-delay')
    account_ids = [opaque_id(seed, 'acct', i) for i in range(config['accounts_per_world'])]
    tables = {name: [] for name in SCHEMAS}
    tables['accounts'] = [(account, 1, 1) for account in account_ids]
    personal, shared = {}, {account: [] for account in account_ids}

    def device():
        identifier = opaque_id(seed, 'dev', len(tables['devices']))
        tables['devices'].append((identifier, device_rng.choice(['phone', 'phone', 'tablet']), 1, 1))
        return identifier

    for account in account_ids:
        personal[account] = [device()]
        if device_rng.random() < .2:
            personal[account].append(device())
    # Ordinary shared access, independent of planted group labels.
    for _ in range(max(2, len(account_ids) // 50)):
        identifier = device()
        for account in device_rng.sample(account_ids, device_rng.randint(2, 6)):
            shared[account].append(identifier)
    rng.shuffle(account_ids)
    hubs = account_ids[:10]
    peers = {account: background_rng.sample([a for a in account_ids if a != account], 5)
             for account in account_ids}
    events = []

    def add_event(sender, receiver, cents, step, group_id=None, fraud=False):
        identifier = opaque_id(seed, 'tx', len(events))
        events.append({'transaction_id': identifier, 'sender_id': sender, 'receiver_id': receiver,
                       'amount': f'{max(1, cents) // 100}.{max(1, cents) % 100:02d}',
                       'step': step, 'available_step': step, 'type': 'TRANSFER',
                       'group_id': group_id, 'is_coordinated_fraud': fraud})

    for _ in range(config['background_transactions_per_world']):
        sender = background_rng.choice(account_ids)
        route = background_rng.random()
        choices = peers[sender] if route < .65 else (hubs if route < .8 else account_ids)
        receiver = background_rng.choice(choices)
        while receiver == sender:
            receiver = background_rng.choice(account_ids)
        cents = int(100 * min(50000, max(10, background_rng.lognormvariate(7.3, 1.2))))
        add_event(sender, receiver, cents, background_rng.randint(2, config['steps']))
    # Interleave positive and benign templates, preserving balanced family coverage.
    templates = [(True, families[i % len(families)]) for i in range(config['fraud_groups_per_world'])]
    templates += [(False, families[i % len(families)]) for i in range(config['benign_groups_per_world'])]
    group_rng.shuffle(templates)
    member_pool = list(account_ids[10:])
    group_rng.shuffle(member_pool)
    offset = 0
    discovery = {}
    for i, (fraud, family) in enumerate(templates):
        group_id = opaque_id(seed, 'grp', i)
        size = group_rng.randint(config['group_size_min'], config['group_size_max'])
        members = member_pool[offset:offset + size]
        offset += size
        tables['group_memberships'].extend((group_id, account) for account in members)
        if fraud:
            tables['ring_memberships'].extend((group_id, account) for account in members)
        # Identical assignment mechanism for fraud and legitimate groups.
        if device_rng.random() < config['shared_group_device_probability']:
            identifier = device()
            for account in device_rng.sample(members, device_rng.randint(2, len(members))):
                shared[account].append(identifier)
        episode_count = group_rng.randint(config['episodes_min'], config['episodes_max'])
        reserve = max(128, config['group_size_max'] * 7 + 50)
        starts = sorted(group_rng.sample(range(config['warmup_steps'] + 1, config['steps'] - reserve), episode_count))
        times = []
        base = min(50000, max(50, group_rng.lognormvariate(7.7, 1.1)))
        # Overlapping distributions: no timing or amount value determines the class.
        variation = group_rng.uniform(.02, .15 if fraud else .25)
        max_gap = group_rng.choice([2, 3, 4, 5] if fraud else [2, 3, 4, 5, 6, 7])
        for start in starts:
            step = start
            episode_amount = base * group_rng.uniform(.7, 1.3)
            for sender, receiver in pattern_edges(family, members):
                step += group_rng.randint(1, max_gap)
                cents = max(1, round(100 * episode_amount * group_rng.uniform(1 - variation, 1 + variation)))
                add_event(sender, receiver, cents, step, group_id, fraud)
                times.append(step)
        discovered = max(times) + label_rng.randint(12, 48)
        discovery[group_id] = discovered
        tables['groups'].append((group_id, family, fraud, min(times), max(times), discovered))
    # Remove append-order clues: observed transactions are sorted by time then opaque ID.
    events.sort(key=lambda row: (row['step'], row['transaction_id']))
    observed_sessions = {}
    for event in events:
        tables['transactions'].append(tuple(event[name] for name in SCHEMAS['transactions']))
        group_id = event['group_id']
        label_time = discovery[group_id] if group_id else event['step'] + label_rng.randint(12, 48)
        tables['transaction_labels'].append((event['transaction_id'], event['is_coordinated_fraud'], group_id, label_time))
        if observation_rng.random() >= config['device_observation_probability']:
            continue
        sender = event['sender_id']
        candidates = shared[sender] if shared[sender] and observation_rng.random() < .5 else personal[sender]
        identifier = observation_rng.choice(candidates)
        key = (sender, identifier, event['step'])
        available = event['step'] + observation_rng.randint(0, config['max_device_delay_steps'])
        observed_sessions[key] = min(available, observed_sessions.get(key, available))
    for i, ((account, identifier, step), available) in enumerate(sorted(observed_sessions.items())):
        tables['device_observations'].append((opaque_id(seed, 'obs', i), account, identifier, step, available))
    return tables


def write_table(con, rows, name, destination, work):
    staging = work / (name + '.csv')
    with staging.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(SCHEMAS[name])
        writer.writerows(rows)
    schema = '{' + ','.join(f'{literal(k)}:{literal(v)}' for k, v in SCHEMAS[name].items()) + '}'
    con.execute(f"COPY (SELECT * FROM read_csv({literal(staging)}, header=true, columns={schema})) "
                f"TO {literal(destination)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    staging.unlink()


def visible_device_links(con, observed_directory, as_of_step):
    """A relationship exists in a snapshot only after its event and arrival time."""
    path = Path(observed_directory) / 'device_observations.parquet'
    return con.execute('''SELECT DISTINCT account_id, device_id FROM read_parquet(?)
        WHERE step < ? AND available_step < ? ORDER BY account_id, device_id''',
        [str(path), as_of_step, as_of_step]).fetchall()


def report_markdown(report):
    lines = ['# Phase 3: synthetic coordinated-activity benchmark', '',
             'Entirely generated scenario data, separate from PaySim. These are construction and validation counts, not detection results.', '',
             '| Split | Worlds | Accounts | Transactions | Fraud groups | Benign groups | Planted fraud events |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for split, item in report['splits'].items():
        lines.append(f"| {split} | {item['worlds']} | {item['accounts']:,} | {item['transactions']:,} | {item['fraud_groups']} | {item['benign_groups']} | {item['fraud_events']:,} |")
    lines += ['', 'Known planted families: ' + ', '.join(report['known_families']) + '.',
              'Held-out planted families: ' + ', '.join(report['heldout_families']) + '.', '',
              'Each world has independent seeds, account/device IDs, persistent background counterparties and mixed group/background activity. '
              'Splits are independent simulated populations, not consecutive periods of one financial system. '
              'Synthetic steps are relative ticks; no real calendar timestamps or actual customer records are supplied.', '',
              '## Validation', '', f"All {len(report['checks'])} structural checks passed across {len(report['worlds'])} worlds.", '',
              'Checks cover referential integrity, label separation, temporal availability, disjoint entities/seeds, '
              'known versus held-out planted templates, ring membership, repeated activity, and legitimate device sharing.', '',
              '## Limitations', '',
              '- Timing and amount distributions are designed and overlapping, not fitted to observed criminal behavior.',
              '- Background paths can incidentally resemble a held-out template; only explicit planted family labels are held out.',
              '- Some labels cannot be resolved from observable evidence. Device sharing and graph motifs do not establish guilt.',
              '- Group members are disjoint within each world, but background events connect groups and other accounts.',
              '- This is an event-pattern simulator, not a balance-conserving financial ledger.',
              '- No fraud model was trained or evaluated on this benchmark in Phase 3. Test ground truth was audited for construction correctness only.', '',
              'Observed account/device/transaction files and private evaluation truth use separate directories. '
              'Only observed files belong in graph/feature construction. Identity columns, paths, seed metadata and split names are not predictors.', '']
    return '\n'.join(lines)


def generate_benchmark(run_name, config_path=Path('config/phase3.json'), root=ROOT):
    root = root.resolve()
    checked_name(run_name)
    config = json.loads(inside_root(config_path, root).read_text(encoding='utf-8'))
    validate_config(config)
    output = inside_root(Path('data/synthetic') / run_name, root)
    reports = inside_root(Path('artifacts/phase3') / run_name, root)
    work = inside_root(Path('.cache/phase3') / run_name, root)
    if any(path.exists() for path in (output, reports, work)):
        raise ValueError('Run already exists; choose a new name to preserve existing artifacts')
    for path in (output, reports, work):
        path.mkdir(parents=True)
    manifest = {'status': 'running', 'generator_version': VERSION, 'config': config,
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'duckdb_version': duckdb.__version__,
                'python_version': platform.python_version(),
                'output_path': str(output.relative_to(root)), 'provenance': 'Fully synthetic GraphShield scenario benchmark; no PaySim rows used',
                'worlds': [], 'files': {}, 'final_test_model_evaluation': False}
    write_json(reports / 'manifest.json', manifest)
    start = time.monotonic()
    try:
        with duckdb.connect(config={'memory_limit': '512MB', 'threads': 2, 'temp_directory': str(work / 'spill')}) as con:
            for split in SPLITS:
                for seed in config['world_seeds'][split]:
                    world_id = opaque_id(seed, 'world', 0)
                    print(f'Generating {split}: {world_id}...', flush=True)
                    tables = generate_world(config, split, seed)
                    observed = output / 'observed' / split / world_id
                    truth = output / 'ground_truth' / split / world_id
                    observed.mkdir(parents=True)
                    truth.mkdir(parents=True)
                    for name, rows in tables.items():
                        destination = (observed if name in OBSERVED else truth) / (name + '.parquet')
                        write_table(con, rows, name, destination, work)
                        manifest['files'][str(destination.relative_to(output))] = sha256(destination)
                    manifest['worlds'].append({'world_id': world_id, 'split': split, 'seed': seed})
        write_json(reports / 'manifest.json', manifest)
        from .synthetic_audit import audit_benchmark
        report = audit_benchmark(output, manifest)
        write_json(reports / 'report.json', report)
        (reports / 'report.md').write_text(report_markdown(report), encoding='utf-8')
        manifest.update(status='complete', elapsed_seconds=round(time.monotonic() - start, 2))
        manifest['generator_code_sha256'] = sha256(Path(__file__))
        manifest['audit_code_sha256'] = sha256(Path(__file__).with_name('synthetic_audit.py'))
        write_json(reports / 'manifest.json', manifest)
        print(f'Phase 3 complete: {reports / "report.md"}', flush=True)
        return report
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        write_json(reports / 'manifest.json', manifest)
        raise
