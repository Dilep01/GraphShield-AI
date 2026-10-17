"""Verify saved Phase 2 models reproduce scores and reported validation metrics."""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graphshield.baselines import checked_name, predict_bundle, read_complete  # noqa: E402
from graphshield.evaluation import choose_threshold, classification, per_step_selection, reference_policies  # noqa: E402
from graphshield.features import FEATURES  # noqa: E402
from graphshield.paths import inside_root  # noqa: E402
from graphshield.pipeline import literal, sha256, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    reports = inside_root(Path('artifacts/phase2/models') / checked_name(args.run_name))
    manifest = read_complete(reports / 'manifest.json')
    feature_dir = inside_root(Path('artifacts/phase2/features') / checked_name(manifest['feature_run']))
    features_manifest = read_complete(feature_dir / 'manifest.json')
    features = inside_root(features_manifest['output_path'])
    metrics = json.loads((reports / 'metrics.json').read_text())
    checks = {
        'model_artifact_hashes_match': all(sha256(inside_root(reports / name)) == digest for name, digest in manifest['files'].items()),
        'feature_manifest_hash_matches': sha256(feature_dir / 'manifest.json') == manifest['feature_manifest_sha256'],
        'feature_hash_matches': sha256(features) == features_manifest['sha256'] == manifest['feature_sha256'],
        'final_test_not_scored': manifest['final_test_scored'] is False and metrics['final_test_scored'] is False,
    }
    temp = inside_root('.cache/phase2/verification')
    temp.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={'memory_limit': '1GB', 'threads': 2, 'temp_directory': str(temp)}) as con:
        con.execute(f'CREATE VIEW features AS SELECT * FROM read_parquet({literal(features)})')
        checks['feature_file_excludes_test'] = con.execute("SELECT count(*) FROM features WHERE split NOT IN ('train','validation') OR step > ?", [features_manifest['validation_end_step']]).fetchone()[0] == 0
        predictions = con.read_parquet(str(reports / 'validation_predictions.parquet')).order('source_row_number').fetchnumpy()
        actual = con.execute('SELECT transaction_id, ' + ','.join('"'+f+'"' for f in FEATURES) + " FROM features WHERE split='validation' ORDER BY source_row_number").fetchnumpy()
        checks['prediction_rows_match_validation_ids'] = np.array_equal(predictions['transaction_id'], actual['transaction_id'])
        checks['prediction_period_excludes_test'] = bool(np.all(predictions['step'] <= features_manifest['validation_end_step']))
        checks['prediction_count_matches_validation'] = len(predictions['transaction_id']) == features_manifest['rows_by_split']['validation']
    x = np.column_stack([actual[f] for f in FEATURES]).astype(np.float32, copy=False)
    tune = predictions['section'] == 'tuning'
    assess = predictions['section'] == 'assessment'
    checks['validation_sections_match_saved_cutoff'] = bool(np.all(tune == (predictions['step'] <= metrics['data']['tuning_end_step'])) and np.all(assess == ~tune))
    checks['reference_policies_reproduce'] = reference_policies(
        predictions['is_fraud'][assess], predictions['step'][assess], metrics['config']['alert_budget_per_step'],
        metrics['data']['train_positives'] / metrics['data']['train_rows']) == metrics['reference_policies']
    for name in metrics['models']:
        bundle = joblib.load(reports / (name + '.joblib'))
        saved = predictions[name]
        recomputed = predict_bundle(bundle, x)
        checks[name + '_reload_predictions_match'] = bool(np.allclose(saved, recomputed, atol=1e-12, rtol=1e-12))
        threshold = choose_threshold(predictions['is_fraud'][tune], saved[tune])
        checks[name + '_threshold_reproduces'] = threshold == bundle['threshold']
        target, scores = predictions['is_fraud'][assess], saved[assess]
        assessment = metrics['models'][name]['assessment']
        checks[name + '_confusion_reproduces'] = classification(target, scores >= threshold) == assessment['threshold_policy']
        checks[name + '_ap_reproduces'] = bool(np.isclose(average_precision_score(target, scores), assessment['average_precision'], atol=1e-12))
        selected = per_step_selection(scores, predictions['step'][assess], predictions['source_row_number'][assess], metrics['config']['alert_budget_per_step'])
        checks[name + '_budget_policy_reproduces'] = classification(target, selected) == {k:v for k,v in assessment['per_step_budget'].items() if k != 'k'}
    write_json(reports / 'verification.json', checks)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit('Phase 2 verification failed')


if __name__ == '__main__':
    main()
