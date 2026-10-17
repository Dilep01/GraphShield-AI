"""Phase 2 artifact orchestration. The final test has no training or scoring path here."""

import csv
import gc
import importlib.metadata
import json
import math
import platform
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluation import choose_threshold, classification, evaluate, paired_block_intervals, per_step_selection, reference_policies
from .features import CONTRACT, FEATURES, build_features
from .paths import ROOT, inside_root
from .pipeline import literal, sha256, write_json


def checked_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or Path(name).is_reserved():
        raise ValueError("Use a non-reserved run name of 1-80 letters, numbers, hyphens or underscores")
    return name


def read_complete(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Input run is not complete: {path}")
    return manifest


def check_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Source artifact hash mismatch: {path}")


def new_directories(root, kind, name):
    checked_name(name)
    reports = inside_root(Path("artifacts/phase2") / kind / name, root)
    work = inside_root(Path(".cache/phase2") / kind / name, root)
    if reports.exists() or work.exists():
        raise ValueError("Run already exists; choose a new run name")
    for path in (reports, work):
        path.mkdir(parents=True)
    return reports, work


def start_manifest(config):
    return {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("duckdb", "numpy", "scikit-learn", "lightgbm", "joblib")}}


def generate_features(source_run, run_name, config_path=Path("config/phase2.json"), root=ROOT):
    checked_name(source_run)
    root = root.resolve()
    source_manifest_path = inside_root(Path("artifacts/phase1") / source_run / "manifest.json", root)
    source = read_complete(source_manifest_path)
    transactions = inside_root(Path("data/processed") / source_run / "transactions.parquet", root)
    check_hash(transactions, source["files"]["transactions.parquet"]["sha256"])
    config = json.loads(inside_root(config_path, root).read_text(encoding="utf-8"))
    checked_name(run_name)
    output = inside_root(Path("data/processed/phase2") / run_name, root)
    if output.exists():
        raise ValueError("Feature output already exists; use a new run name")
    reports, work = new_directories(root, "features", run_name)
    output.mkdir(parents=True)
    manifest = start_manifest(config)
    manifest.update(source_run=source_run, source_manifest_sha256=sha256(source_manifest_path),
                    source_files=source["files"], feature_contract=CONTRACT,
                    train_end_step=source["train_end_step"], validation_end_step=source["validation_end_step"])
    write_json(reports / "manifest.json", manifest)
    start = time.monotonic()
    database = work / "features.duckdb"
    try:
        with duckdb.connect(str(database), config={"memory_limit": config["memory_limit"],
                "threads": config["threads"], "temp_directory": str(work / "spill")}) as con:
            counts = build_features(con, transactions, output / "features.parquet",
                                    source["train_end_step"], source["validation_end_step"])
        manifest.update(status="complete", elapsed_seconds=round(time.monotonic() - start, 2),
                        rows_by_split=counts, output_path=str((output / "features.parquet").relative_to(root)),
                        sha256=sha256(output / "features.parquet"))
        write_json(reports / "manifest.json", manifest)
        write_json(reports / "feature_contract.json", CONTRACT)
        print(f"Feature run complete: {reports}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(reports / "manifest.json", manifest)
        raise
    finally:
        if database.exists():
            database.unlink()


def validation_cutoff(steps, fraction):
    steps = np.unique(steps)
    if not 0 < fraction < 1 or len(steps) < 2:
        raise ValueError("Validation tuning split requires at least two time steps and fraction in (0,1)")
    count = max(1, min(len(steps) - 1, math.floor(len(steps) * fraction)))
    return int(steps[count - 1])


def matrix(con, where):
    columns = ", ".join('"' + name + '"' for name in FEATURES)
    arrays = con.execute(f"SELECT {columns}, is_fraud FROM supervised WHERE {where} ORDER BY source_row_number").fetchnumpy()
    x = np.column_stack([arrays[name] for name in FEATURES]).astype(np.float32, copy=False)
    y = arrays["is_fraud"].astype(np.int8)
    if not len(y) or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite model input")
    return x, y


def predict_bundle(bundle, x, feature_names=FEATURES):
    if list(feature_names) != bundle["features"] or x.shape[1] != len(bundle["features"]):
        raise ValueError("Feature names/order must match the trained model contract")
    if not np.isfinite(x).all():
        raise ValueError("Model features must be finite")
    if bundle["kind"] == "lightgbm":
        return bundle["model"].predict(x, num_threads=bundle["threads"])
    return bundle["model"].predict_proba(x)[:, 1]


def report_markdown(report):
    lines = ["# Phase 2: historical-feature baselines", "",
             "PaySim synthetic-data validation results. No final-test scoring or real-world fraud claim.", "",
             f"Training rows: {report['data']['train_rows']:,}; positives: {report['data']['train_positives']:,}.",
             f"Validation tuning ends at step {report['data']['tuning_end_step']}. "
             "Later validation steps are used for comparison after thresholds and boosting iterations are selected.", "",
             "Both models use the same 20 features and all training rows. No resampling changes class prevalence.",
             "Class-balanced fitting is used, so scores are not assumed to be calibrated probabilities.", "",
             "| Model | Average precision | Precision | Recall | F1 | Alerts | False positives |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        metrics = model["assessment"]
        policy = metrics["threshold_policy"]
        lines.append(f"| {name} | {metrics['average_precision']:.6f} | {policy['precision']:.4f} | "
                     f"{policy['recall']:.4f} | {policy['f1']:.4f} | {policy['alerts']:,} | {policy['fp']:,} |")
    first = next(iter(report["models"].values()))["assessment"]
    lines += ["", f"Comparison rows: {first['rows']:,}; positives: {first['positives']:,}; "
              f"prevalence: {first['prevalence']:.4%} (constant-score AP reference).", "",
              "Average precision uses sklearn's non-interpolated definition, not trapezoidal PR area. "
              "Thresholds maximize F1 only on the earlier tuning period. Table metrics describe this threshold "
              "policy; the separate budget policy below does not apply that threshold.", "",
              "## Fixed investigation capacity", "",
              f"At most {report['config']['alert_budget_per_step']} transactions per observed time step, "
              "ranked by score, with source-row order breaking ties. This policy fills available slots, "
              "even if scores are low. It is transaction-level capacity, not deduplicated investigation cases.", ""]
    for name, model in report["models"].items():
        budget = model["assessment"]["per_step_budget"]
        lines.append(f"- {name}: {budget['alerts']:,} alerts; precision {budget['precision']:.4f}; recall {budget['recall']:.4f}.")
    if 'reference_policies' in report:
        ref = report['reference_policies']
        random = ref['random_within_step_budget']
        lines += ['', f"Uniform random selection within each step has expected precision {random['expected_precision']:.4f} "
                  f"and expected recall {random['expected_recall']:.4f} at the same budget. "
                  f"{random['positives_in_those_steps']:,} positives fall in {random['steps_with_rows_at_or_below_budget']} "
                  'steps where every transaction fits within the budget. High capacity-based recall must be read against this reference.',
                  '', f"A constant score equal to training prevalence has comparison Brier score {ref['constant_training_prevalence_brier']:.6f}. "
                  'Compare this with each model\'s Brier score in metrics.json; ranking quality does not establish probability calibration.']
    intervals = report["paired_intervals"]
    lines += ["", "## Uncertainty and selection", "", json.dumps(intervals, indent=2), "",
              f"Development candidate selected by tuning-period average precision: `{report['selected_model']}`.",
              "Selection is not based on the later comparison period. The selected model has not been deployed.", "",
              "Time-step bootstrap intervals are conditional on this fitted model and dataset. "
              "They omit training-seed uncertainty and may understate uncertainty from cross-step dependence.", "",
              "## Diagnostics and artifacts", "",
              "metrics.json contains calibration bins, Brier/log-loss diagnostics, fixed global top-K results, "
              "and seen/unseen-account subgroup metrics. Subgroup budget results use the full comparison "
              "queue's selections, not a fresh budget for each subgroup.", "",
              "validation_predictions.parquet includes both validation sections and labels for reproducible "
              "evaluation. No final-test predictions were created. Features use unlabeled earlier validation "
              "events during replay; the model and training scaler remain fixed.", "",
              "Model joblib files, LightGBM text export, feature contract, manifest hashes and model "
              "contributions are saved beside this report. Contributions are associations, not causal proof.", ""]
    return "\n".join(lines)


def train_baselines(feature_run, run_name, config_path=Path("config/phase2.json"), root=ROOT):
    root = root.resolve()
    checked_name(feature_run)
    feature_manifest_path = inside_root(Path("artifacts/phase2/features") / feature_run / "manifest.json", root)
    feature_manifest = read_complete(feature_manifest_path)
    if feature_manifest["feature_contract"] != CONTRACT:
        raise ValueError("Feature contract differs from the implemented model contract")
    features = inside_root(feature_manifest["output_path"], root)
    check_hash(features, feature_manifest["sha256"])
    source_run = checked_name(feature_manifest["source_run"])
    labels = inside_root(Path("data/processed") / source_run / "labels.parquet", root)
    check_hash(labels, feature_manifest["source_files"]["labels.parquet"]["sha256"])
    config = json.loads(inside_root(config_path, root).read_text(encoding="utf-8"))
    reports, work = new_directories(root, "models", run_name)
    manifest = start_manifest(config)
    manifest.update(feature_run=feature_run, feature_manifest_sha256=sha256(feature_manifest_path),
                    feature_sha256=feature_manifest["sha256"], final_test_scored=False)
    write_json(reports / "manifest.json", manifest)
    start = time.monotonic()
    try:
        with duckdb.connect(config={"memory_limit": config["memory_limit"], "threads": config["threads"],
                                    "temp_directory": str(work / "spill")}) as con:
            con.execute(f"CREATE VIEW f AS SELECT * FROM read_parquet({literal(features)})")
            if con.execute("SELECT count(*) FROM f WHERE split NOT IN ('train','validation')").fetchone()[0]:
                raise ValueError("Training input unexpectedly contains final-test rows")
            con.execute(f"""CREATE VIEW supervised AS SELECT f.*, l.is_fraud
                FROM f JOIN read_parquet({literal(labels)}) l USING(transaction_id)""")
            for split in ("train", "validation"):
                count = con.execute("SELECT count(*) FROM supervised WHERE split = ?", [split]).fetchone()[0]
                if count != feature_manifest["rows_by_split"][split]:
                    raise ValueError("Supervised label join changed row count")
            print("Loading identical training features for both baselines...", flush=True)
            x_train, y_train = matrix(con, "split = 'train'")
            x_val, y_val = matrix(con, "split = 'validation'")
            metadata = con.execute("""SELECT transaction_id, source_row_number, step,
                either_unseen_at_train_cutoff FROM supervised
                WHERE split = 'validation' ORDER BY source_row_number""").fetchnumpy()
        tuning_end = validation_cutoff(metadata["step"], config["validation_tuning_fraction"])
        tune = metadata["step"] <= tuning_end
        assessment = ~tune
        for name, target in (("training", y_train), ("validation tuning", y_val[tune]), ("validation assessment", y_val[assessment])):
            if len(np.unique(target)) != 2:
                raise ValueError(f"{name} must contain both classes; do not report unsupported metrics")
        results, scores, fit_info = {}, {}, {}
        bundles = {}
        with threadpool_limits(limits=config["threads"]):
            print(f"Fitting Logistic Regression on {len(y_train):,} training rows...", flush=True)
            lr = make_pipeline(StandardScaler(), LogisticRegression(
                **config["logistic_regression"], class_weight="balanced", solver="lbfgs", random_state=config["seed"]))
            model_start = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                lr.fit(x_train, y_train)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise ValueError("Logistic Regression did not converge: " + convergence[0])
            bundles["logistic_regression"] = {"kind": "logistic_regression", "model": lr,
                                              "features": FEATURES, "threads": config["threads"]}
            fit_info["logistic_regression"] = {"fit_seconds": round(time.monotonic() - model_start, 2),
                                               "iterations": int(lr[-1].n_iter_[0]), "converged": True}
            scores["logistic_regression"] = np.asarray(predict_bundle(bundles["logistic_regression"], x_val), dtype=np.float64)
            print("Fitting LightGBM; early stopping uses the tuning section only...", flush=True)
            params = dict(config["lightgbm"])
            patience = params.pop("early_stopping_rounds")
            gbm = lgb.LGBMClassifier(**params, class_weight="balanced", random_state=config["seed"],
                                    n_jobs=config["threads"], deterministic=True, force_col_wise=True,
                                    verbosity=-1, metric="average_precision")
            model_start = time.monotonic()
            gbm.fit(x_train, y_train, feature_name=FEATURES,
                    eval_X=x_val[tune], eval_y=y_val[tune], eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
            # Save only the chosen iteration, ensuring in-memory and reloaded scoring agree.
            gbm.booster_.save_model(str(reports / "lightgbm.txt"), num_iteration=gbm.best_iteration_)
            booster = lgb.Booster(model_file=str(reports / "lightgbm.txt"))
            bundles["lightgbm"] = {"kind": "lightgbm", "model": booster, "features": FEATURES,
                                   "threads": config["threads"]}
            fit_info["lightgbm"] = {"fit_seconds": round(time.monotonic() - model_start, 2),
                                    "best_iteration": gbm.best_iteration_}
            scores["lightgbm"] = predict_bundle(bundles["lightgbm"], x_val)
        train_rows, train_positives = len(y_train), int(y_train.sum())
        del x_train, y_train, gbm
        gc.collect()
        for name, predictions in scores.items():
            threshold = choose_threshold(y_val[tune], predictions[tune])
            parts = {}
            for section, mask in (("tuning", tune), ("assessment", assessment)):
                parts[section] = evaluate(y_val[mask], predictions[mask], metadata["step"][mask],
                    metadata["source_row_number"][mask], threshold,
                    config["alert_budget_per_step"], config["global_top_k"])
            budget_selected = per_step_selection(predictions[assessment], metadata["step"][assessment],
                metadata["source_row_number"][assessment], config["alert_budget_per_step"])
            unseen = metadata["either_unseen_at_train_cutoff"][assessment]
            cohorts = {}
            for cohort, mask in (("either_account_unseen", unseen), ("both_accounts_seen", ~unseen)):
                target, score = y_val[assessment][mask], predictions[assessment][mask]
                cohorts[cohort] = {"rows": len(target), "positives": int(target.sum()),
                    "average_precision": float(average_precision_score(target, score)) if target.sum() else None,
                    "threshold_policy": classification(target, score >= threshold),
                    "full_queue_budget_policy": classification(target, budget_selected[mask])}
            parts.update(fit=fit_info[name], assessment_cohorts=cohorts)
            results[name] = parts
            bundles[name].update(threshold=threshold, feature_contract=CONTRACT)
            joblib.dump(bundles[name], reports / (name + ".joblib"), compress=3)
            reloaded = joblib.load(reports / (name + ".joblib"))
            if not np.allclose(predict_bundle(reloaded, x_val), predictions, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Saved model predictions changed: {name}")
        print("Computing paired time-step bootstrap intervals...", flush=True)
        intervals = paired_block_intervals(y_val[assessment], {name: p[assessment] for name, p in scores.items()},
                                          metadata["step"][assessment], config["bootstrap_repeats"], config["seed"])
        selected = max(results, key=lambda name: results[name]["tuning"]["average_precision"])
        report = {"dataset": "Original PaySim synthetic data", "config": config,
                  "data": {"train_rows": train_rows, "train_positives": train_positives,
                           "tuning_end_step": tuning_end, "validation_rows": len(y_val),
                           "assessment_min_step": int(metadata["step"][assessment].min()),
                           "assessment_max_step": int(metadata["step"][assessment].max())},
                  "models": results, "paired_intervals": intervals, "selected_model": selected,
                  "final_test_scored": False, "feature_contract": CONTRACT,
                  "reference_policies": reference_policies(y_val[assessment], metadata['step'][assessment],
                      config['alert_budget_per_step'], train_positives / train_rows)}
        write_json(reports / "metrics.json", report)
        (reports / "report.md").write_text(report_markdown(report), encoding="utf-8")
        write_json(reports / "model_contributions.json", {
            "logistic_standardized_coefficients": dict(zip(FEATURES, lr[-1].coef_[0].tolist())),
            "lightgbm_training_split_gain": dict(zip(FEATURES, booster.feature_importance(importance_type="gain").tolist())),
            "interpretation": "Training-model associations only; not causal explanations or ring evidence"})
        # Stream a contained CSV staging file; DuckDB cannot scan structured NumPy records.
        prediction_csv = work / 'validation_predictions.csv'
        prediction_columns = ["transaction_id", "source_row_number", "step", "section", "is_fraud", "either_unseen_at_train_cutoff", "logistic_regression", "lightgbm"]
        with prediction_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(prediction_columns)
            writer.writerows(zip(metadata['transaction_id'], metadata['source_row_number'], metadata['step'],
                np.where(tune, 'tuning', 'assessment'), y_val.astype(bool), metadata['either_unseen_at_train_cutoff'],
                scores['logistic_regression'], scores['lightgbm']))
        with duckdb.connect(config={"temp_directory": str(work / "export")}) as con:
            types = ['VARCHAR', 'BIGINT', 'BIGINT', 'VARCHAR', 'BOOLEAN', 'BOOLEAN', 'DOUBLE', 'DOUBLE']
            schema = '{' + ','.join(f'{literal(name)}:{literal(dtype)}' for name, dtype in zip(prediction_columns, types)) + '}'
            con.execute(f"COPY (SELECT * FROM read_csv({literal(prediction_csv)}, header=true, columns={schema})) TO {literal(reports / 'validation_predictions.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        prediction_csv.unlink()
        write_json(reports / "feature_contract.json", CONTRACT)
        manifest.update(status="complete", selected_model=selected,
                        elapsed_seconds=round(time.monotonic() - start, 2),
                        files={path.name: sha256(path) for path in reports.iterdir() if path.name != "manifest.json"})
        write_json(reports / "manifest.json", manifest)
        print(f"Baseline run complete: {reports / 'report.md'}", flush=True)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - start, 2))
        write_json(reports / "manifest.json", manifest)
        raise
