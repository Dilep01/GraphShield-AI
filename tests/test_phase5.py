import copy
import json
from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from graphshield.learned_graph import (CASE_FEATURES, CaseNetwork, MeanLayer, embedding_features,
                                       historical_graph, member_indices, node2vec, transition_weights)
from graphshield.learned_rings import (MODELS, candidate_targets, choose_threshold, evaluate_models,
                                       fit_models, load_source, select_cases, train_rings,
                                       training_weights, validate_config, verify_learned)
from graphshield.pipeline import literal, sha256, write_json
from graphshield.rings import read_jsonl, read_truth, run_rules
from graphshield.synthetic import SCHEMAS
from test_phase4 import candidate, cycle, fixture_source, group


@pytest.fixture
def config5():
    config = json.loads((Path(__file__).resolve().parents[1] / 'config/phase5.json').read_text())
    config['threads'] = 2
    config['threshold_quantiles'] = 5
    config['lightgbm'].update(n_estimators=3, min_child_samples=1)
    config['node2vec'].update(dimensions=4, walks_per_node=1, walk_length=6, epochs=1)
    config['neural'].update(hidden_dimensions=4, epochs=2)
    torch.set_num_threads(2)
    return config


@pytest.fixture
def rules5():
    return json.loads((Path(__file__).resolve().parents[1] / 'config/phase4.json').read_text())


def test_graph_excludes_current_future_late_and_expired_observations():
    tx = cycle()
    obs = [('o1','a0','shared',4,4), ('o2','outside','shared',4,10)]
    before = historical_graph(tx, obs, 10, 168)
    after = historical_graph(tx + [('late','a0','bad',99,4,10),('now','bad','a1',100,10,10)],
                             obs + [('future','bad','shared',20,20)], 10, 168)
    assert all(np.array_equal(v, after[k]) for k, v in before.items())
    assert 'outside' not in before['nodes']
    visible = historical_graph(tx, obs, 11, 168)
    assert 'outside' in visible['nodes']
    assert historical_graph(tx, obs, 11, 5)['nodes'].size == 0


def test_device_projection_is_bounded_and_uses_arrival_time():
    obs = [(str(i), str(i), 'd', 2, 2) for i in range(5)]
    limited = historical_graph([], obs, 3, 10, device_limit=4)
    assert limited['edges'].shape == (2, 0)
    assert limited['oversized_devices_skipped'] == 1
    full = historical_graph([], obs, 3, 10, device_limit=5)
    assert full['edges'].shape == (2, 20)


def test_node2vec_transition_bias_is_second_order():
    adjacency = [{1}, {0,2,3}, {1}, {1}]
    weights = transition_weights(0, [0,1,2], adjacency, p=2, q=4)
    assert weights.tolist() == [.5,1,.25]


def test_node2vec_fit_is_deterministic_and_handles_isolates(config5):
    edges = np.asarray([[0,1,1,2],[1,0,2,1]], np.int64)
    first, log = node2vec(edges, 4, config5['node2vec'], 100)
    second, _ = node2vec(edges, 4, config5['node2vec'], 100)
    assert np.array_equal(first, second)
    assert log['pairs'] > 0 and np.isfinite(log['loss']).all()
    assert np.all(first[3] == 0)
    assert np.any(first[:3] != 0)


def test_embedding_summary_is_rotation_invariant_and_handles_unknowns():
    embeddings = np.asarray([[1.,0.],[1.,1.],[0.,1.]], np.float32)
    rotation = np.asarray([[0.,-1.],[1.,0.]], np.float32)
    members = member_indices([{'members':['a','b','missing']}, {'members':['c']}], ['a','b','c'])
    assert members.tolist() == [[0,1,-2],[2,-1,-1]]
    left = embedding_features(embeddings, members)
    assert np.allclose(left, embedding_features(embeddings @ rotation, members))
    assert left[0,4] == pytest.approx(1/3)
    assert left[1,4] == 0
    assert np.all(left[1,:4] == 0)


def test_mean_aggregation_and_no_edge_control():
    layer = MeanLayer(1,2)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(2))
        layer.linear.bias.zero_()
    x = torch.tensor([[1.],[3.],[5.]])
    edges = torch.tensor([[0,2],[1,1]])
    output = layer(x, edges)
    expected = torch.nn.functional.normalize(torch.tensor([[1.,0.],[3.,3.],[5.,0.]]), dim=1)
    assert torch.allclose(output, expected)
    assert torch.allclose(layer(x, edges, False), torch.tensor([[1.,0.],[1.,0.],[1.,0.]]))


def test_graphsage_is_node_permutation_invariant(config5):
    torch.manual_seed(1)
    model = CaseNetwork(4)
    x = torch.rand(4,8)
    edges = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    members = torch.tensor([[0,1,2],[1,2,3]])
    features = torch.rand(2,len(CASE_FEATURES))
    permutation = torch.tensor([2,0,3,1])
    inverse = torch.argsort(permutation)
    assert torch.allclose(model(x,edges,members,features),
                          model(x[permutation], inverse[edges], inverse[members], features), atol=1e-6)


def test_repeated_group_weight_and_class_balance():
    labels = np.asarray([1,1,1,1,0,0])
    weights = training_weights(labels, ['g','g','g','h','b','c'])
    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights[labels == 1].sum() == pytest.approx(weights[labels == 0].sum())
    with pytest.raises(ValueError):
        training_weights(np.ones(3, dtype=int), ['g','g','h'])


def test_targets_require_planted_historical_evidence(rules5):
    case = dict(candidate('x',list('abcde')), world_id='w')
    truth = {'w':([group('g',list('abcde'))], {})}
    assert candidate_targets([case], truth, rules5)[0].tolist() == [0]
    truth['w'][1]['x'] = 'g'
    assert candidate_targets([case], truth, rules5)[0].tolist() == [1]


def test_threshold_selects_review_tradeoff_on_tuning_cases(rules5):
    cases = [candidate('fraud',list('abcde')), candidate('benign',list('vwxyz'))]
    groups = [group('g',list('abcde')), group('b',list('vwxyz'),False)]
    best, _ = choose_threshold(cases, np.asarray([.8,.2]), groups, {'fraud':'g','benign':'b'}, rules5, 5)
    assert best['matched_groups'] == 1 and best['alerts'] == 1
    assert best['case_ring_f1'] == 1
    with pytest.raises(ValueError):
        select_cases(cases, np.asarray([np.nan,.5]), .5, rules5)


@pytest.mark.parametrize('section,key,value', [(None,'threads',0), ('node2vec','p',0),
                                               ('neural','epochs',True), ('node2vec','walk_length',1)])
def test_invalid_model_config(config5, section, key, value):
    (config5[section] if section else config5)[key] = value
    with pytest.raises(ValueError):
        validate_config(config5)


def source_with_comparison(project, rules5):
    data, manifest = fixture_source(project)
    original = manifest['worlds'][1]
    extra = {'world_id':'world_comparison', 'split':'validation'}
    for kind, tables in {'observed':('transactions','device_observations'),
                         'ground_truth':('groups','group_memberships','transaction_labels')}.items():
        for name in tables:
            source = data / kind / 'validation' / original['world_id'] / f'{name}.parquet'
            destination = data / kind / 'validation' / extra['world_id'] / f'{name}.parquet'
            destination.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect() as con:
                rows = con.execute(f'SELECT * FROM read_parquet({literal(source)})').fetchall()
                rows = [tuple(v.replace(original['world_id'],extra['world_id']) if isinstance(v,str) else v for v in row) for row in rows]
                con.execute(f"CREATE TABLE t ({', '.join(f'{k} {v}' for k,v in SCHEMAS[name].items())})")
                con.executemany(f"INSERT INTO t VALUES ({','.join('?' for _ in SCHEMAS[name])})", rows)
                con.execute(f'COPY t TO {literal(destination)} (FORMAT PARQUET)')
            manifest['files'][str(destination.relative_to(data))] = sha256(destination)
    manifest['worlds'].insert(2, extra)
    write_json(project / 'artifacts/phase3/source/manifest.json', manifest)
    write_json(project / 'config/phase4.json', rules5)
    run_rules('source','rules',root=project)


def test_end_to_end_saved_models_no_test_files_and_no_validation_fit(project, config5, rules5):
    source_with_comparison(project, rules5)
    write_json(project / 'config/phase5.json', config5)
    result = train_rings('rules','learned',root=project)
    assert result['tuning_world'] == 'world_validation'
    assert result['comparison_world'] == 'world_comparison'
    assert set(result['summary']['comparison']) == set(MODELS)
    assert all(verify_learned('learned',root=project).values())
    with pytest.raises(ValueError, match='already exists'):
        train_rings('rules','learned',root=project)
    output = project / 'artifacts/phase5/learned'
    _, _, _, data, worlds, cases = load_source('rules',project)
    with duckdb.connect() as con:
        truths = {w['world_id']:read_truth(con,data,w) for w in worlds}
    labels, clusters = candidate_targets(cases,truths,rules5)
    log = json.loads((output / 'graph_log.json').read_text())
    graphs = []
    embedding_x = np.zeros((len(cases),5),np.float32)
    for entry in log:
        with np.load(output / 'graphs' / entry['filename'], allow_pickle=False) as saved:
            graph = {k:saved[k] for k in saved.files}
        embedding_x[graph['indices']] = graph['embedding_x']
        graphs.append(dict(graph, split=entry['split'], world_id=entry['world_id']))
    changed = labels.copy()
    changed[[i for i,c in enumerate(cases) if c['split'] == 'validation']] ^= 1
    repeat = project / 'repeat-models'
    repeat.mkdir()
    predictions, _ = fit_models(graphs,cases,changed,clusters,embedding_x,config5,repeat)
    saved_predictions = read_jsonl(output / 'predictions.jsonl')
    for name in MODELS:
        assert np.allclose(predictions[name], [r['scores'][name] for r in saved_predictions], atol=1e-7)
    # Comparison outcomes cannot influence tuning thresholds or selected model.
    different = copy.deepcopy(truths)
    for g in different['world_comparison'][0]:
        g['is_fraud'] = False
    altered_labels, _ = candidate_targets(cases,different,rules5)
    altered, _, _ = evaluate_models(cases,predictions,different,worlds,altered_labels,config5,rules5)
    assert altered['thresholds'] == result['thresholds']
    assert altered['selected_on_tuning'] == result['selected_on_tuning']
    (output / 'models/graphsage.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        verify_learned('learned',root=project)
