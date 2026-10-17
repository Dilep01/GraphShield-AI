import json
import math

import duckdb
import numpy as np
import pytest

from graphshield.baselines import generate_features, predict_bundle, train_baselines, validation_cutoff
from graphshield.evaluation import choose_threshold, evaluate, paired_block_intervals, per_step_selection, reference_policies
from graphshield.features import FEATURES, build_features
from graphshield.pipeline import literal, run_pipeline


def source_parquet(con, path, future_amount=999):
    con.execute("""CREATE TABLE tx(transaction_id VARCHAR, source_row_number BIGINT, step BIGINT,
        type VARCHAR, amount DECIMAL(24,2), sender_id VARCHAR, receiver_id VARCHAR, split VARCHAR)""")
    con.executemany("INSERT INTO tx VALUES (?,?,?,?,?,?,?,?)", [
        ('a', 1, 1, 'TRANSFER', 10, 'A', 'B', 'train'),
        ('b', 2, 1, 'TRANSFER', 20, 'A', 'C', 'train'),
        ('c', 3, 2, 'TRANSFER', 5, 'B', 'A', 'train'),
        ('d', 4, 25, 'PAYMENT', 1, 'A', 'D', 'validation'),
        ('e', 5, 26, 'TRANSFER', 1, 'B', 'A', 'validation'),
        ('f', 6, 27, 'TRANSFER', future_amount, 'A', 'B', 'test'),
    ])
    con.execute(f"COPY tx TO {literal(path)} (FORMAT PARQUET)")


def test_features_exclude_same_step_and_test_and_replay_prior_validation(project):
    source, destination = project / 'events.parquet', project / 'features.parquet'
    with duckdb.connect() as con:
        source_parquet(con, source)
        counts = build_features(con, source, destination, 2, 26)
        assert counts == {'train': 3, 'validation': 2}
        rows = con.execute('SELECT transaction_id, sender_log_out_count, sender_log_in_count, sender_log_events_24steps, sender_has_history, receiver_log_out_count, receiver_log_in_count FROM features ORDER BY source_row_number').fetchall()
        assert rows[0][1:5] == (0, 0, 0, 0)
        assert rows[1][1:5] == (0, 0, 0, 0)  # same-step A activity is not history
        assert rows[2][1] == 0
        assert rows[2][2] == pytest.approx(math.log1p(1))  # B received once
        assert rows[2][5] == pytest.approx(math.log1p(2))  # A sent twice in prior step
        assert rows[3][1] == pytest.approx(math.log1p(2))
        assert rows[3][3] == pytest.approx(math.log1p(3))  # t=1 included at t=25
        assert rows[4][3] == pytest.approx(math.log1p(1))  # t=1 excluded at t=26
        assert rows[4][5] == pytest.approx(math.log1p(3))  # prior validation event updates A
        assert set(FEATURES).isdisjoint({'step','split','source_row_number','is_fraud'})
        assert con.execute("SELECT either_unseen_at_train_cutoff FROM features WHERE transaction_id='d'").fetchone()[0]


def test_future_test_amount_cannot_change_features(project):
    values = []
    for i, amount in enumerate((999, 1_000_000)):
        with duckdb.connect() as con:
            source = project / f'events{i}.parquet'
            source_parquet(con, source, amount)
            build_features(con, source, project / f'features{i}.parquet', 2, 26)
            values.append(con.execute('SELECT * FROM features ORDER BY source_row_number').fetchall())
    assert values[0] == values[1]


def test_validation_tuning_split_preserves_complete_steps():
    assert validation_cutoff([10, 10, 11, 12, 13], .5) == 11
    with pytest.raises(ValueError):
        validation_cutoff([1], .5)


def test_later_validation_changes_cannot_change_earlier_features(project):
    results = []
    for i, amount in enumerate((1, 9999)):
        with duckdb.connect() as con:
            source = project / f'validation{i}.parquet'
            source_parquet(con, source)
            con.execute("UPDATE tx SET amount = ? WHERE transaction_id = 'e'", [amount])
            changed = project / f'changed{i}.parquet'
            con.execute(f"COPY tx TO {literal(changed)} (FORMAT PARQUET)")
            build_features(con, changed, project / f'history{i}.parquet', 2, 26)
            results.append(con.execute('SELECT * FROM features WHERE step < 26 ORDER BY source_row_number').fetchall())
    assert results[0] == results[1]


def test_threshold_and_budget_have_different_correct_confusion_counts():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([.1, .9, .8, .7, .2, .6])
    steps = np.array([1, 1, 1, 2, 2, 2])
    source_rows = np.arange(6)
    threshold = choose_threshold(y, p)
    assert threshold == .6
    result = evaluate(y, p, steps, source_rows, threshold, 1, 2)
    assert result['threshold_policy']['tp'] == 3
    assert result['threshold_policy']['fp'] == 1
    assert result['per_step_budget']['alerts'] == 2
    assert result['per_step_budget']['tp'] == 2
    assert result['global_top_k']['tp'] == 1
    assert result['global_top_k']['fp'] == 1


def test_budget_ties_are_deterministic_and_capacity_not_exceeded():
    selected = per_step_selection(np.array([.5,.5,.5,.5]), np.array([1,1,2,2]), np.array([9,2,4,8]), 1)
    assert selected.tolist() == [False, True, True, False]


def test_random_budget_reference_accounts_for_small_time_buckets():
    result = reference_policies(np.array([1,0,0,0,1]), np.array([1,1,1,1,2]), 2, .1)
    random = result['random_within_step_budget']
    assert random['expected_true_positives'] == 1.5
    assert random['alerts'] == 3
    assert random['expected_precision'] == .5
    assert random['expected_recall'] == .75
    assert random['positives_in_those_steps'] == 1


def test_no_positive_subgroup_does_not_invent_recall_or_ap():
    result = evaluate(np.array([0,0]), np.array([.1,.2]), np.array([1,2]), np.array([1,2]), .5, 1, 1)
    assert result['average_precision'] is None
    assert result['threshold_policy']['recall'] is None


def test_paired_bootstrap_for_identical_models_has_zero_difference():
    y, p = np.array([0,1,0,1,0,1]), np.array([.1,.9,.2,.8,.3,.7])
    result = paired_block_intervals(y, {'a': p, 'b': p}, np.array([1,1,2,2,3,3]), 20, 42)
    assert result['difference']['interval'] == [0.0, 0.0]


def test_complete_training_pipeline_and_reload(project, write_csv):
    rows = []
    for step in range(1, 21):
        for i in range(20):
            fraud = i % 4 == 0
            rows.append([step, 'TRANSFER' if fraud else 'PAYMENT', '100.00' if fraud else '10.00',
                         f'A{i}', 100, 0, f'B{i}', 0, 100, int(fraud), 0])
    run_pipeline(write_csv(rows), 'config/phase1.json', 'fixture', root=project)
    config = {'seed': 42, 'memory_limit': '256MB', 'threads': 1,
              'validation_tuning_fraction': .5, 'alert_budget_per_step': 2, 'global_top_k': 5,
              'bootstrap_repeats': 5,
              'logistic_regression': {'C': 1, 'max_iter': 100, 'tol': .001},
              'lightgbm': {'n_estimators': 10, 'learning_rate': .1, 'num_leaves': 3,
                           'min_child_samples': 2, 'early_stopping_rounds': 3}}
    (project / 'config/phase2.json').write_text(json.dumps(config))
    generate_features('fixture', 'history', root=project)
    result = train_baselines('history', 'baseline', root=project)
    assert result['final_test_scored'] is False
    assert result['data']['train_rows'] == 200
    assert result['data']['validation_rows'] == 100
    for model in result['models'].values():
        assert model['assessment']['average_precision'] > .9
    path = project / 'artifacts/phase2/models/baseline'
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    with duckdb.connect() as con:
        con.read_parquet(str(path / 'validation_predictions.parquet')).create_view('p')
        assert con.execute('SELECT max(step), count(*) FROM p').fetchone() == (15, 100)
    import joblib
    bundle = joblib.load(path / 'logistic_regression.joblib')
    with pytest.raises(ValueError, match='Feature names/order'):
        predict_bundle(bundle, np.zeros((1, len(FEATURES))), list(reversed(FEATURES)))
