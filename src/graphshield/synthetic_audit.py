"""Structural audit of generated truth, not a fraud-detection evaluation."""

from collections import Counter
from pathlib import Path

import duckdb

from .pipeline import literal, sha256
from .synthetic import OBSERVED, SCHEMAS, SPLITS, TRUTH, validate_config


def valid_pattern(family, members, edges):
    """Independently check planted graph shapes, without calling the generator."""
    members, edges = set(members), set(edges)
    if {a for edge in edges for a in edge} != members or any(a == b for a, b in edges):
        return False
    incoming, outgoing = Counter(b for a, b in edges), Counter(a for a, b in edges)
    n = len(members)
    if family == 'fan_in':
        return len(edges) == n - 1 and sorted(incoming.values()) == [n - 1] and max(outgoing.values()) == 1
    if family == 'fan_out':
        return len(edges) == n - 1 and sorted(outgoing.values()) == [n - 1] and max(incoming.values()) == 1
    if family == 'cycle':
        if len(edges) != n or any(incoming[a] != 1 or outgoing[a] != 1 for a in members):
            return False
        start = next(iter(members))
    elif family == 'relay_chain':
        starts = [a for a in members if incoming[a] == 0]
        if len(edges) != n - 1 or len(starts) != 1 or any(incoming[a] > 1 or outgoing[a] > 1 for a in members):
            return False
        start = starts[0]
    else:
        return False
    next_node = dict(edges)
    visited = set()
    current = start
    while current is not None and current not in visited:
        visited.add(current)
        current = next_node.get(current)
    return visited == members and (current == start if family == 'cycle' else current is None)


def audit_benchmark(output: Path, manifest, verify_hashes=True):
    output = output.resolve()
    config = manifest['config']
    validate_config(config)
    checks = {}
    failures = []

    def check(name, value, world='benchmark'):
        value = bool(value)
        checks[name] = checks.get(name, True) and value
        if not value:
            failures.append(f'{world}: {name}')

    listed_files = {str(path.relative_to(output)) for path in output.rglob('*.parquet')}
    check('parquet_inventory_matches_manifest', listed_files == set(manifest['files']))
    for relative, digest in manifest['files'].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError('Manifest path escapes the synthetic dataset directory')
        if verify_hashes:
            check('all_parquet_hashes_match', path.is_file() and sha256(path) == digest)
    declared = {(split, seed) for split, seeds in config['world_seeds'].items() for seed in seeds}
    observed_worlds = [(entry['split'], entry['seed']) for entry in manifest['worlds']]
    check('worlds_match_unique_configured_seeds', len(observed_worlds) == len(set(observed_worlds)) and set(observed_worlds) == declared)
    entities = {name: set() for name in ('accounts', 'devices', 'groups', 'transactions')}
    report = {'known_families': config['known_families'], 'heldout_families': config['heldout_families'],
              'worlds': [], 'splits': {}, 'checks': checks}
    shared_by_label = Counter()
    with duckdb.connect(config={'memory_limit': '512MB', 'threads': 2,
                               'temp_directory': str(output / '.audit_spill')}) as con:
        for entry in manifest['worlds']:
            split, world = entry['split'], entry['world_id']
            if Path(world).name != world or split not in SPLITS:
                raise ValueError('Invalid world partition')
            world_dir = output / 'observed' / split / world
            truth_dir = output / 'ground_truth' / split / world
            for name, schema in SCHEMAS.items():
                path = (world_dir if name in OBSERVED else truth_dir) / (name + '.parquet')
                con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({literal(path)})')
                actual = {row[0]: row[1] for row in con.execute(f'DESCRIBE {name}').fetchall()}
                check('exact_schemas_and_label_separation', actual == schema, world)
                required = [column for column in schema if not (name == 'transaction_labels' and column == 'group_id')]
                null_query = ' OR '.join(f'{column} IS NULL' for column in required)
                check('required_fields_non_null', con.execute(f'SELECT count(*) FROM {name} WHERE {null_query}').fetchone()[0] == 0, world)
            for name, key in [('accounts','account_id'), ('devices','device_id'), ('groups','group_id'),
                              ('transactions','transaction_id'), ('device_observations','observation_id'),
                              ('transaction_labels','transaction_id')]:
                size, distinct, nonnull = con.execute(f'SELECT count(*), count(DISTINCT {key}), count({key}) FROM {name}').fetchone()
                check('nonempty_unique_primary_keys', 0 < size == distinct == nonnull, world)
                if name in entities:
                    values = {row[0] for row in con.execute(f'SELECT {key} FROM {name}').fetchall()}
                    check('entities_disjoint_across_worlds_and_splits', not entities[name] & values, world)
                    entities[name].update(values)
            for name in ('group_memberships', 'ring_memberships'):
                size, unique = con.execute(f'SELECT count(*), count(DISTINCT (group_id, account_id)) FROM {name}').fetchone()
                check('unique_membership_pairs', size == unique, world)
            orphan_queries = [
                'SELECT count(*) FROM transactions t LEFT JOIN accounts a ON t.sender_id=a.account_id LEFT JOIN accounts b ON t.receiver_id=b.account_id WHERE a.account_id IS NULL OR b.account_id IS NULL',
                'SELECT count(*) FROM device_observations o LEFT JOIN accounts a USING(account_id) LEFT JOIN devices d USING(device_id) WHERE a.account_id IS NULL OR d.device_id IS NULL',
                'SELECT count(*) FROM group_memberships m LEFT JOIN accounts a USING(account_id) LEFT JOIN groups g USING(group_id) WHERE a.account_id IS NULL OR g.group_id IS NULL',
                'SELECT count(*) FROM transaction_labels l LEFT JOIN groups g USING(group_id) WHERE l.group_id IS NOT NULL AND g.group_id IS NULL',
            ]
            check('referential_integrity', all(con.execute(sql).fetchone()[0] == 0 for sql in orphan_queries), world)
            check('exactly_one_label_per_event', con.execute('''SELECT count(*) FROM transactions t
                FULL JOIN transaction_labels l USING(transaction_id)
                WHERE t.transaction_id IS NULL OR l.transaction_id IS NULL''').fetchone()[0] == 0, world)
            check('transaction_time_and_amount_valid', con.execute('''SELECT count(*) FROM transactions
                WHERE step < 1 OR step > ? OR available_step != step OR amount <= 0 OR sender_id=receiver_id
                   OR step IS NULL OR available_step IS NULL OR amount IS NULL OR type != 'TRANSFER' ''', [config['steps']]).fetchone()[0] == 0, world)
            check('device_observation_availability_valid', con.execute('''SELECT count(*) FROM device_observations
                WHERE step < 1 OR step > ? OR available_step < step OR available_step-step > ?
                OR step IS NULL OR available_step IS NULL''', [config['steps'], config['max_device_delay_steps']]).fetchone()[0] == 0, world)
            check('entities_exist_before_observations', con.execute('''SELECT count(*) FROM device_observations o
                JOIN accounts a USING(account_id) JOIN devices d USING(device_id)
                WHERE a.created_step > o.step OR a.available_step > o.available_step
                OR d.created_step > o.step OR d.available_step > o.available_step''').fetchone()[0] == 0 and
                con.execute('''SELECT count(*) FROM transactions t JOIN accounts a ON a.account_id=t.sender_id
                JOIN accounts b ON b.account_id=t.receiver_id WHERE a.created_step > t.step OR b.created_step > t.step
                OR a.available_step > t.available_step OR b.available_step > t.available_step''').fetchone()[0] == 0, world)
            check('labels_discovered_after_events', con.execute('''SELECT count(*) FROM transaction_labels l
                JOIN transactions t USING(transaction_id) WHERE l.label_available_step <= t.step OR l.label_available_step IS NULL''').fetchone()[0] == 0, world)
            check('planted_group_intervals_valid', con.execute('''SELECT count(*) FROM groups g JOIN (
                SELECT group_id, min(step) AS first, max(step) AS last FROM transactions
                JOIN transaction_labels USING(transaction_id) WHERE group_id IS NOT NULL GROUP BY group_id
                ) e USING(group_id) WHERE g.start_step != e.first OR g.end_step != e.last OR g.start_step <= ?
                OR g.label_available_step <= g.end_step OR g.label_available_step > ?''', [config['warmup_steps'], config['steps']]).fetchone()[0] == 0, world)
            check('labels_match_group_truth', con.execute('''SELECT count(*) FROM transaction_labels l
                LEFT JOIN groups g USING(group_id) WHERE (l.group_id IS NULL AND l.is_coordinated_fraud)
                OR (l.group_id IS NOT NULL AND (l.is_coordinated_fraud != g.is_fraud OR l.label_available_step != g.label_available_step))
                OR l.is_coordinated_fraud IS NULL''').fetchone()[0] == 0, world)
            check('all_planted_endpoints_are_members', con.execute('''SELECT count(*) FROM transactions t
                JOIN transaction_labels l USING(transaction_id)
                LEFT JOIN group_memberships a ON a.group_id=l.group_id AND a.account_id=t.sender_id
                LEFT JOIN group_memberships b ON b.group_id=l.group_id AND b.account_id=t.receiver_id
                WHERE l.group_id IS NOT NULL AND (a.account_id IS NULL OR b.account_id IS NULL)''').fetchone()[0] == 0, world)
            actual_rings = set(con.execute('SELECT * FROM ring_memberships').fetchall())
            expected_rings = set(con.execute('SELECT m.group_id, m.account_id FROM group_memberships m JOIN groups g USING(group_id) WHERE g.is_fraud').fetchall())
            check('ring_memberships_equal_positive_group_memberships', actual_rings == expected_rings, world)
            check('planted_groups_have_disjoint_members', con.execute('SELECT count(*) FROM (SELECT account_id FROM group_memberships GROUP BY account_id HAVING count(*)>1)').fetchone()[0] == 0, world)
            families = set(config['heldout_families'] if split == 'test_unseen' else config['known_families'])
            actual_families = set(con.execute('SELECT DISTINCT scenario_family, is_fraud FROM groups').fetchall())
            check('family_holdout_and_benign_coverage', actual_families == {(family, label) for family in families for label in (True, False)}, world)
            for group_id, family in con.execute('SELECT group_id, scenario_family FROM groups').fetchall():
                members = [row[0] for row in con.execute('SELECT account_id FROM group_memberships WHERE group_id=?', [group_id]).fetchall()]
                edges = con.execute('''SELECT DISTINCT sender_id, receiver_id FROM transactions
                    JOIN transaction_labels USING(transaction_id) WHERE group_id=?''', [group_id]).fetchall()
                check('planted_motif_shapes_correct', valid_pattern(family, members, edges), world)
                count = con.execute('SELECT count(*) FROM transaction_labels WHERE group_id=?', [group_id]).fetchone()[0]
                check('group_sizes_and_episode_counts_valid', config['group_size_min'] <= len(members) <= config['group_size_max'] and
                      count % len(edges) == 0 and config['episodes_min'] <= count // len(edges) <= config['episodes_max'], world)
            # Observed shared devices within labeled groups; supports hard-negative audit only.
            for fraud, count in con.execute('''SELECT g.is_fraud, count(*) FROM (
                SELECT m.group_id, o.device_id FROM group_memberships m JOIN device_observations o USING(account_id)
                GROUP BY m.group_id, o.device_id HAVING count(DISTINCT m.account_id) > 1
                ) s JOIN groups g USING(group_id) GROUP BY g.is_fraud''').fetchall():
                shared_by_label[bool(fraud)] += count
            counts = {name: con.execute(f'SELECT count(*) FROM {name}').fetchone()[0] for name in SCHEMAS}
            counts['fraud_groups'] = con.execute('SELECT count(*) FROM groups WHERE is_fraud').fetchone()[0]
            counts['benign_groups'] = counts['groups'] - counts['fraud_groups']
            counts['fraud_events'] = con.execute('SELECT count(*) FROM transaction_labels WHERE is_coordinated_fraud').fetchone()[0]
            counts['recurrent_accounts'] = con.execute('''SELECT count(*) FROM (
                SELECT account_id FROM (SELECT sender_id AS account_id, step FROM transactions UNION ALL SELECT receiver_id, step FROM transactions)
                GROUP BY account_id HAVING count(DISTINCT step) > 1)''').fetchone()[0]
            counts['background_events'] = con.execute('SELECT count(*) FROM transaction_labels WHERE group_id IS NULL').fetchone()[0]
            counts['delayed_device_observations'] = con.execute('SELECT count(*) FROM device_observations WHERE available_step > step').fetchone()[0]
            counts['members_with_background_activity'] = con.execute('''SELECT count(DISTINCT m.account_id) FROM group_memberships m
                JOIN (SELECT t.sender_id AS account_id FROM transactions t JOIN transaction_labels l USING(transaction_id) WHERE l.group_id IS NULL
                      UNION SELECT t.receiver_id FROM transactions t JOIN transaction_labels l USING(transaction_id) WHERE l.group_id IS NULL) b USING(account_id)''').fetchone()[0]
            counts['background_group_boundary_events'] = con.execute('''SELECT count(*) FROM transactions t
                JOIN transaction_labels l USING(transaction_id)
                LEFT JOIN group_memberships a ON a.account_id=t.sender_id
                LEFT JOIN group_memberships b ON b.account_id=t.receiver_id
                WHERE l.group_id IS NULL AND (a.group_id IS NOT NULL OR b.group_id IS NOT NULL)
                AND (a.group_id IS DISTINCT FROM b.group_id)''').fetchone()[0]
            check('configured_population_and_background_counts', counts['accounts'] == config['accounts_per_world'] and
                  counts['background_events'] == config['background_transactions_per_world'] and
                  counts['fraud_groups'] == config['fraud_groups_per_world'] and counts['benign_groups'] == config['benign_groups_per_world'], world)
            check('persistent_accounts_and_mixed_activity', counts['recurrent_accounts'] >= .8 * counts['accounts'] and
                  counts['members_with_background_activity'] >= .8 * counts['group_memberships'], world)
            check('background_connects_groups_to_outside_accounts', counts['background_group_boundary_events'] > 0, world)
            check('partial_and_delayed_device_coverage', 0 < counts['device_observations'] < counts['transactions'] and counts['delayed_device_observations'] > 0, world)
            report['worlds'].append({**entry, **counts})
            summary = report['splits'].setdefault(split, {'worlds': 0, **{key: 0 for key in counts}})
            summary['worlds'] += 1
            for key, count in counts.items():
                summary[key] += count
    check('shared_devices_observed_in_both_classes', shared_by_label[True] > 0 and shared_by_label[False] > 0)
    report['shared_group_device_pairs'] = {'fraud': shared_by_label[True], 'benign': shared_by_label[False]}
    if failures:
        raise ValueError('Synthetic benchmark audit failed: ' + '; '.join(failures[:12]))
    return report
