import copy
import json

import duckdb
import pytest

from graphshield.pipeline import literal
from graphshield.synthetic import (OBSERVED, SCHEMAS, generate_benchmark, generate_world, pattern_edges,
                                   validate_config, visible_device_links)
from graphshield.synthetic_audit import audit_benchmark, valid_pattern


@pytest.fixture
def synthetic_config():
    return {
        'generator_version': 'coordinated-v1', 'steps': 240, 'warmup_steps': 24,
        'accounts_per_world': 120, 'background_transactions_per_world': 1200,
        'fraud_groups_per_world': 3, 'benign_groups_per_world': 3,
        'group_size_min': 3, 'group_size_max': 5, 'episodes_min': 2, 'episodes_max': 3,
        'device_observation_probability': .7, 'max_device_delay_steps': 4,
        'shared_group_device_probability': .8,
        'known_families': ['cycle','fan_in','fan_out'], 'heldout_families': ['relay_chain'],
        'world_seeds': {'train':[1101], 'validation':[2201], 'test_known':[3301], 'test_unseen':[4401]},
    }


def test_generator_is_seed_deterministic_and_independent(synthetic_config):
    one = generate_world(synthetic_config, 'train', 1101)
    assert one == generate_world(synthetic_config, 'train', 1101)
    two = generate_world(synthetic_config, 'train', 1102)
    assert {row[0] for row in one['accounts']}.isdisjoint(row[0] for row in two['accounts'])
    # Changing another world's seed cannot change this world's generated events.
    changed = copy.deepcopy(synthetic_config)
    changed['world_seeds']['test_known'] = [9911]
    assert one == generate_world(changed, 'train', 1101)


@pytest.mark.parametrize('family', ['cycle', 'fan_in', 'fan_out', 'relay_chain'])
def test_planted_shapes_and_benign_shapes_have_same_topology(family):
    members = ['a','b','c','d','e']
    edges = pattern_edges(family, members)
    assert valid_pattern(family, members, edges)
    assert not valid_pattern(family, members, edges[:-1])


def test_disconnected_cycles_are_not_one_ring():
    assert not valid_pattern('cycle', ['a','b','c','d'], [('a','b'),('b','a'),('c','d'),('d','c')])


def test_labels_are_not_in_observed_schemas(synthetic_config):
    forbidden = {'group_id','is_fraud','is_coordinated_fraud','scenario_family','seed','split','label_available_step'}
    for name in OBSERVED:
        assert forbidden.isdisjoint(SCHEMAS[name])
    world = generate_world(synthetic_config, 'train', 1101)
    groups = {row[0]: row for row in world['groups']}
    assert {row[1] for row in groups.values()} == set(synthetic_config['known_families'])
    assert {row[2] for row in groups.values()} == {False, True}
    ring_members = {row[1] for row in world['ring_memberships']}
    transactions = {row[0]: row for row in world['transactions']}
    # A ring member's ordinary background activity is not relabeled as fraud.
    assert any(not label[1] and label[2] is None and transactions[label[0]][1] in ring_members
               for label in world['transaction_labels'])


@pytest.mark.parametrize('mutation', ['duplicate_seed','family_overlap','too_few_accounts','short_horizon','invalid_probability'])
def test_invalid_configs_rejected_before_generation(synthetic_config, mutation):
    if mutation == 'duplicate_seed':
        synthetic_config['world_seeds']['validation'] = [1101]
    elif mutation == 'family_overlap':
        synthetic_config['heldout_families'] = ['cycle']
    elif mutation == 'too_few_accounts':
        synthetic_config['accounts_per_world'] = 10
    elif mutation == 'short_horizon':
        synthetic_config['steps'] = 80
    else:
        synthetic_config['device_observation_probability'] = 1
    with pytest.raises(ValueError):
        validate_config(synthetic_config)


def test_snapshot_excludes_late_arrivals_and_future_observations(project):
    folder = project / 'observed'
    folder.mkdir()
    with duckdb.connect() as con:
        con.execute("CREATE TABLE o(observation_id VARCHAR, account_id VARCHAR, device_id VARCHAR, step BIGINT, available_step BIGINT)")
        con.execute("INSERT INTO o VALUES ('1','a','d',2,8),('2','b','e',4,4),('3','c','f',20,20),('4','b','e',5,5)")
        con.execute(f"COPY o TO {literal(folder / 'device_observations.parquet')} (FORMAT PARQUET)")
        assert visible_device_links(con, folder, 4) == []
        assert visible_device_links(con, folder, 6) == [('b','e')]
        assert visible_device_links(con, folder, 8) == [('b','e')]
        assert visible_device_links(con, folder, 9) == [('a','d'),('b','e')]


def test_full_benchmark_audit_reproduction_and_tamper_detection(project, synthetic_config):
    (project / 'config/phase3.json').write_text(json.dumps(synthetic_config))
    first = generate_benchmark('scenario-one', root=project)
    second = generate_benchmark('scenario-two', root=project)
    assert all(first['checks'].values())
    assert first == second
    manifests = [json.loads((project / 'artifacts/phase3' / name / 'manifest.json').read_text())
                 for name in ('scenario-one','scenario-two')]
    assert manifests[0]['files'] == manifests[1]['files']
    assert manifests[0]['status'] == 'complete'
    assert first['shared_group_device_pairs']['benign'] > 0
    with pytest.raises(ValueError, match='already exists'):
        generate_benchmark('scenario-one', root=project)
    output = project / manifests[0]['output_path']
    world = manifests[0]['worlds'][0]
    target = output / 'ground_truth' / world['split'] / world['world_id'] / 'groups.parquet'
    with duckdb.connect() as con:
        con.execute(f'CREATE TABLE bad AS SELECT * FROM read_parquet({literal(target)})')
        con.execute("UPDATE bad SET scenario_family='relay_chain'")
        con.execute(f'COPY bad TO {literal(target)} (FORMAT PARQUET)')
    with pytest.raises(ValueError, match='audit failed'):
        audit_benchmark(output, manifests[0])


def test_outside_project_run_path_rejected(project, synthetic_config):
    (project / 'config/phase3.json').write_text(json.dumps(synthetic_config))
    with pytest.raises(ValueError):
        generate_benchmark('../escape', root=project)
