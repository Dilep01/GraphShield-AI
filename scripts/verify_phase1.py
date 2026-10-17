"""Verify completed artifacts against hashes, keys and temporal split contracts."""

import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphshield.paths import inside_root  # noqa: E402
from graphshield.pipeline import sha256, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    reports = inside_root(Path("artifacts/phase1") / args.run_name)
    output = inside_root(Path("data/processed") / args.run_name)
    manifest = json.loads((reports / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs may be verified or consumed")
    results = {}
    results["source_hash_matches"] = sha256(inside_root(manifest["source"]["path"])) == manifest["source"]["sha256"]
    for name, expected in manifest["files"].items():
        results[name + "_hash_matches"] = sha256(inside_root(output / name)) == expected["sha256"]
    temp = inside_root(".cache/verification")
    temp.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"memory_limit": "1GB", "threads": 2, "temp_directory": str(temp)}) as con:
        con.read_parquet(str(output / "transactions.parquet")).create_view("tx")
        con.read_parquet(str(output / "labels.parquet")).create_view("labels")
        expected_rows = json.loads((reports / "validation.json").read_text())["rows"]
        results["rows_preserved"] = con.execute("SELECT count(*) FROM tx").fetchone()[0] == expected_rows
        results["unique_transaction_ids"] = con.execute("SELECT count(DISTINCT transaction_id) FROM tx").fetchone()[0] == expected_rows
        results["unique_label_ids"] = con.execute("SELECT count(DISTINCT transaction_id) FROM labels").fetchone()[0] == expected_rows
        results["labels_one_to_one"] = con.execute("SELECT count(*) FROM tx JOIN labels USING(transaction_id)").fetchone()[0] == expected_rows
        results["no_time_bucket_crosses_splits"] = con.execute("SELECT count(*) FROM (SELECT step FROM tx GROUP BY step HAVING count(DISTINCT split) > 1)").fetchone()[0] == 0
        results["split_boundaries_match_manifest"] = con.execute("""SELECT count(*) FROM tx WHERE
            split != CASE WHEN step <= ? THEN 'train' WHEN step <= ? THEN 'validation' ELSE 'test' END""",
            [manifest["train_end_step"], manifest["validation_end_step"]]).fetchone()[0] == 0
        columns = {row[0] for row in con.execute("DESCRIBE tx").fetchall()}
        results["forbidden_features_absent"] = columns.isdisjoint({"is_fraud", "isFraud", "isFlaggedFraud", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"})
        results["amount_is_exact_decimal"] = dict((row[0], row[1]) for row in con.execute("DESCRIBE tx").fetchall())["amount"] == "DECIMAL(24,2)"
    write_json(reports / "verification.json", results)
    print(json.dumps(results, indent=2))
    if not all(results.values()):
        raise SystemExit("Verification failed")


if __name__ == "__main__":
    main()
