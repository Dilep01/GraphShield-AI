import json
from pathlib import Path

import duckdb
import pytest

from graphshield.graph_rules import components, select_alerts, snapshot, validate_config
from graphshield.pipeline import literal, sha256, write_json
from graphshield.ring_evaluation import evaluate, maximum_matching
from graphshield.rings import run_rules, source_file, verify_rules
from graphshield.synthetic import SCHEMAS


@pytest.fixture
def rule_config():
    return json.loads((Path(__file__).resolve().parents[1] / 'config/phase4.json').read_text())


def cycle(prefix='a', steps=(3, 5), count=5):
    return [(f'{prefix}-{step}-{i}', f'{prefix}{i}', f'{prefix}{(i+1)%count}', 100, step, step)
            for step in steps for i in range(count)]


def test_iterative_components_support_long_chain_and_cycle():
    edges = {str(i): {str(i+1)} for i in range(2000)}
    assert len(components(edges)) == 2001
    edges['2000'] = {'0'}
    assert len(components(edges)) == 1


def test_temporal_boundaries_future_and_late_arrivals(rule_config):
    base = cycle()
    obs = [('o1', 'a0', 'shared', 4, 4), ('o2', 'a1', 'shared', 4, 10)]
    original, _ = snapshot(base, obs, 10, rule_config)
    assert original[0]['features']['shared_device_count'] == 0
    changed = base + [('late', 'a0', 'a1', 1000000, 6, 10),
                      ('now', 'a1', 'a2', 9000000, 10, 10),
                      ('future', 'a2', 'a3', 100000, 50, 50)]
    assert snapshot(changed, obs + [('future-o', 'a2', 'shared', 10, 10)], 10, rule_config)[0] == original
    later, _ = snapshot(base, obs, 11, rule_config)
    assert later[0]['features']['shared_device_count'] == 1
    assert later[0]['device_observation_ids'] == ['o1', 'o2']
    config = dict(rule_config, window_steps=4)
    assert snapshot(base, [], 10, config)[0] == []


def test_observed_order_does_not_change_graph_or_candidates(rule_config):
    rows = cycle() + cycle('b')
    assert snapshot(rows, [], 10, rule_config) == snapshot(list(reversed(rows)), [], 10, rule_config)


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_repeated_star_detection(direction, rule_config):
    rows = []
    for step in (3, 5):
        for i in range(1, 5):
            u, v = ('a0', f'a{i}') if direction == 'out' else (f'a{i}', 'a0')
            rows.append((f'{step}-{i}', u, v, 100, step, step))
    cases, _ = snapshot(rows, [], 10, rule_config)
    assert len(cases) == 1
    assert cases[0]['reasons'] == [f'repeated_fan_{direction}']


def test_large_connected_graph_is_not_automatically_a_case(rule_config):
    cases, summary = snapshot(cycle(count=40), [], 10, rule_config)
    assert cases == []
    assert summary['largest_weak_component'] == 40
    assert summary['oversized_proposals_skipped']['repeated_strong_component'] == 1


def candidate(name, members, cutoff=10, score=.9):
    return {'candidate_id': name, 'members': members, 'cutoff_step': cutoff,
            'scores': {'graph_only': score, 'graph_device': score}, 'transaction_ids': [name]}


def group(name, members, fraud=True, start=1):
    return {'group_id': name, 'members': members, 'is_fraud': fraud,
            'start_step': start, 'end_step': 100, 'scenario_family': 'cycle'}


def test_budget_ties_and_duplicates_are_chronological(rule_config):
    cfg = dict(rule_config, alerts_per_snapshot=1)
    cases = [candidate('first', ['a','b','c','d','e']),
             candidate('second', ['v','w','x','y','z']),
             candidate('repeat', ['a','b','c','d','e'], 20),
             candidate('retry', ['v','w','x','y','z'], 20)]
    alerts, log = select_alerts(list(reversed(cases)), cfg, 'graph_only')
    assert [a['candidate_id'] for a in alerts] == ['first', 'retry']
    assert [r['decision'] for r in log] == ['alert','capacity','duplicate','alert']
    assert log[2]['duplicate_of'] == 'first'
    assert select_alerts(cases, cfg, 'graph_only') == (alerts, log)


def test_one_to_one_matching_counts_benign_and_duplicate_alerts(rule_config):
    groups = [group('fraud', ['a','b','c','d','e']), group('benign', ['v','w','x','y','z'], False)]
    cases = [candidate('one', groups[0]['members']), candidate('two', groups[0]['members'], 20),
             candidate('three', groups[1]['members'])]
    result = evaluate(cases, groups, {'one':'fraud', 'two':'fraud', 'three':'benign'}, rule_config)
    assert result['matched_groups'] == 1
    assert result['case_precision'] == pytest.approx(1 / 3)
    assert result['ring_recall'] == 1
    assert result['duplicate_positive_alerts'] == 1
    assert result['benign_groups_alerted'] == 1
    assert result['median_detection_delay_steps'] == 9


def test_matching_requires_purity_coverage_and_observed_planted_evidence(rule_config):
    truth = group('g', ['a','b','c','d','e'])
    for members, cutoff, evidence in [(['a','b'], 10, {'x':'g'}),
                                      (truth['members'] + list('uvwxyz'), 10, {'x':'g'}),
                                      (truth['members'], 10, {}),
                                      (truth['members'], 1, {'x':'g'})]:
        assert evaluate([candidate('x', members, cutoff)], [truth], evidence, rule_config)['matched_groups'] == 0
    assert evaluate([candidate('x', truth['members'][:4])], [truth], {'x':'g'}, rule_config)['matched_groups'] == 1


def test_matching_finds_augmenting_path_instead_of_greedy_undercount():
    assert maximum_matching({0:['A','B'], 1:['A']}) == {0:'B', 1:'A'}
    assert maximum_matching({}) == {}


def test_empty_evaluation_is_not_reported_as_perfect(rule_config):
    result = evaluate([], [group('g', ['a','b','c'])], {}, rule_config)
    assert result['case_precision'] is None
    assert result['ring_recall'] == 0
    assert result['median_detection_delay_steps'] is None


@pytest.mark.parametrize('key,value', [('window_steps',0), ('minimum_pair_events',1),
                                      ('minimum_score', float('nan')), ('duplicate_jaccard',0),
                                      ('alerts_per_snapshot',True), ('maximum_members',3)])
def test_invalid_rule_config(key, value, rule_config):
    rule_config[key] = value
    with pytest.raises(ValueError):
        validate_config(rule_config)


def fixture_source(project):
    data = project / 'data/synthetic/source'
    manifest = {'status':'complete', 'output_path':str(data.relative_to(project)),
                'config':{'warmup_steps':2, 'steps':30}, 'files':{},
                'worlds':[{'world_id':'world_train', 'split':'train'},
                          {'world_id':'world_validation', 'split':'validation'},
                          {'world_id':'missing_test', 'split':'test_known'},
                          {'world_id':'missing_unseen', 'split':'test_unseen'}]}
    for world in manifest['worlds'][:2]:
        prefix = world['world_id']
        tx = cycle(prefix + '_fraud') + cycle(prefix + '_benign')
        groups = []
        memberships = []
        labels = []
        for kind in ('fraud', 'benign'):
            gid = prefix + '_' + kind
            groups.append((gid, 'cycle', kind == 'fraud', 3, 5, 15))
            memberships.extend((gid, f'{gid}{i}') for i in range(5))
            labels.extend((r[0], kind == 'fraud', gid, 15) for r in tx if r[1].startswith(gid))
        tables = {'transactions':[r + ('TRANSFER',) for r in tx],
                  'device_observations':[(prefix+'o', prefix+'_fraud0', prefix+'d', 4, 9)],
                  'groups':groups, 'group_memberships':memberships, 'transaction_labels':labels}
        with duckdb.connect() as con:
            for table, rows in tables.items():
                kind = 'observed' if table in ('transactions','device_observations') else 'ground_truth'
                path = data / kind / world['split'] / prefix / f'{table}.parquet'
                path.parent.mkdir(parents=True, exist_ok=True)
                schema = ', '.join(f'{name} {dtype}' for name, dtype in SCHEMAS[table].items())
                con.execute(f'CREATE TABLE {table} ({schema})')
                con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in SCHEMAS[table])})", rows)
                con.execute(f'COPY {table} TO {literal(path)} (FORMAT PARQUET)')
                manifest['files'][str(path.relative_to(data))] = sha256(path)
    dest = project / 'artifacts/phase3/source'
    dest.mkdir(parents=True)
    write_json(dest / 'manifest.json', manifest)
    return data, manifest


def test_full_replay_with_absent_test_worlds_and_saved_artifact_verification(project, rule_config, monkeypatch):
    data, _ = fixture_source(project)
    write_json(project / 'config/phase4.json', rule_config)
    # A truth reader must only run after immutable candidate/alert outputs exist.
    import graphshield.rings as rings
    original = rings.read_truth
    def guarded(con, data, world):
        assert (project / 'artifacts/phase4/rules/candidates.jsonl').exists()
        assert (project / 'artifacts/phase4/rules/alerts.jsonl').exists()
        return original(con, data, world)
    monkeypatch.setattr(rings, 'read_truth', guarded)
    result = run_rules('source', 'rules', root=project)
    for split in ('train', 'validation'):
        for policy in ('graph_only', 'graph_device'):
            assert result['summary'][split][policy]['alerts'] == 2
            assert result['summary'][split][policy]['matched_groups'] == 1
            assert result['summary'][split][policy]['case_precision'] == .5
    assert all(verify_rules('rules', root=project).values())
    with pytest.raises(ValueError, match='already exists'):
        run_rules('source', 'rules', root=project)
    with pytest.raises(ValueError, match='development'):
        source_file(data, {'world_id':'missing_test','split':'test_known'}, 'observed','transactions')
    artifact = project / 'artifacts/phase4/rules/report.md'
    artifact.write_text('altered', encoding='utf-8')
    with pytest.raises(ValueError, match='verification failed'):
        verify_rules('rules', root=project)


def test_truth_changes_do_not_change_candidates_or_alerts(project, rule_config):
    data, manifest = fixture_source(project)
    write_json(project / 'config/phase4.json', rule_config)
    before = run_rules('source', 'before', root=project)
    # Flip all planted group classes, updating the source fingerprint for this fixture.
    for world in manifest['worlds'][:2]:
        path = data / 'ground_truth' / world['split'] / world['world_id'] / 'groups.parquet'
        with duckdb.connect() as con:
            con.execute(f'CREATE TABLE groups AS SELECT * FROM read_parquet({literal(path)})')
            con.execute('UPDATE groups SET is_fraud = FALSE')
            con.execute(f'COPY groups TO {literal(path)} (FORMAT PARQUET)')
        manifest['files'][str(path.relative_to(data))] = sha256(path)
    write_json(project / 'artifacts/phase3/source/manifest.json', manifest)
    after = run_rules('source', 'after', root=project)
    for file in ('candidates.jsonl','snapshots.jsonl','alerts.jsonl','decisions.jsonl'):
        assert (project / 'artifacts/phase4/before' / file).read_bytes() == (project / 'artifacts/phase4/after' / file).read_bytes()
    assert before['summary']['validation']['graph_only']['matched_groups'] == 1
    assert after['summary']['validation']['graph_only']['matched_groups'] == 0
