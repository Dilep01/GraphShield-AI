"""Phase 1: bounded-memory ingestion, validation, temporal split and graph audit."""

import csv
import hashlib
import json
import math
import re
import time
from array import array
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import duckdb

from .paths import ROOT, inside_root

HEADERS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


def valid_money(value: str) -> bool:
    """Accept exact nonnegative cents, including the source's scientific notation."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", value):
        return False
    try:
        amount = Decimal(value)
        return (amount.is_finite() and 0 <= amount < Decimal('1e22')
                and amount == amount.quantize(Decimal('0.01')))
    except (InvalidOperation, ValueError):
        return False


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def records(connection, query: str) -> list[dict]:
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_boundaries(steps: list[int], train_fraction: float, validation_fraction: float):
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive with train + validation < 1")
    steps = sorted(set(steps))
    if len(steps) < 3:
        raise ValueError("At least three distinct time steps are required")
    train_count = max(1, math.floor(len(steps) * train_fraction))
    validation_count = max(1, math.floor(len(steps) * validation_fraction))
    if train_count + validation_count >= len(steps):
        raise ValueError("Split fractions leave no final test period")
    return steps[train_count - 1], steps[train_count + validation_count - 1]


def history_before(connection, parquet_path: Path, as_of_step: int):
    """Exclude the entire current time bucket; never join ground-truth labels."""
    return connection.execute(
        "SELECT * FROM read_parquet(?) WHERE step < ? ORDER BY source_row_number",
        [str(parquet_path), as_of_step],
    ).fetchall()


def component_summary(node_count: int, edge_batches):
    """Exact weak components with compact arrays; edge directions ignored here only."""
    parent = array("q", range(node_count))
    sizes = array("q", [1]) * node_count

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for batch in edge_batches:
        for left, right in batch:
            left, right = find(left), find(right)
            if left == right:
                continue
            if sizes[left] < sizes[right]:
                left, right = right, left
            parent[right] = left
            sizes[left] += sizes[right]
    histogram = Counter(sizes[node] for node in range(node_count) if parent[node] == node)
    largest = max(histogram, default=0)
    return {
        "count": sum(histogram.values()), "largest_component_accounts": largest,
        "largest_component_fraction": largest / node_count if node_count else 0,
        "component_size_histogram": dict(sorted(histogram.items())),
        "definition": "Weak connectivity on training-only directed transaction events; not fraud rings",
    }


def graph_report(con, compute_components: bool):
    con.execute("CREATE TABLE train AS SELECT * FROM ledger WHERE split = 'train'")
    con.execute("""CREATE TABLE node_activity AS
        SELECT account_id, sum(outgoing)::BIGINT AS outgoing,
               sum(incoming)::BIGINT AS incoming
        FROM (SELECT sender_id AS account_id, count(*) AS outgoing, 0 AS incoming
              FROM train GROUP BY sender_id
              UNION ALL
              SELECT receiver_id, 0, count(*) FROM train GROUP BY receiver_id)
        GROUP BY account_id""")
    activity = records(con, """SELECT count(*) AS accounts,
        count(*) FILTER (WHERE outgoing > 0) AS senders,
        count(*) FILTER (WHERE incoming > 0) AS receivers,
        count(*) FILTER (WHERE outgoing > 0 AND incoming > 0) AS both_roles,
        count(*) FILTER (WHERE outgoing > 1) AS repeat_senders,
        count(*) FILTER (WHERE outgoing + incoming = 1) AS single_event_accounts,
        max(outgoing + incoming) AS max_incident_events
        FROM node_activity""")[0]
    activity["both_roles_fraction"] = activity["both_roles"] / activity["accounts"]
    activity["single_event_fraction"] = activity["single_event_accounts"] / activity["accounts"]
    pairs = records(con, """SELECT count(*) AS distinct_directed_pairs,
        count(*) FILTER (WHERE n > 1) AS repeated_directed_pairs,
        max(n) AS max_events_per_pair
        FROM (SELECT sender_id, receiver_id, count(*) AS n FROM train GROUP BY ALL)""")[0]
    # Group by account and step first: no unsupported ordering inside one step.
    history = records(con, """WITH buckets AS (
        SELECT sender_id, step, count(*) AS n FROM train GROUP BY sender_id, step),
        first_steps AS (SELECT sender_id, min(step) AS first_step FROM buckets GROUP BY sender_id)
        SELECT sum(n)::BIGINT AS transactions,
               sum(CASE WHEN step > first_step THEN n ELSE 0 END)::BIGINT
                   AS transactions_with_prior_sender_step
        FROM buckets JOIN first_steps USING (sender_id)""")[0]
    cold_start = records(con, """SELECT split, count(*) AS transactions,
        count(*) FILTER (WHERE s.account_id IS NULL) AS unseen_sender,
        count(*) FILTER (WHERE r.account_id IS NULL) AS unseen_receiver,
        count(*) FILTER (WHERE s.account_id IS NULL OR r.account_id IS NULL) AS either_unseen
        FROM ledger l
        LEFT JOIN node_activity s ON l.sender_id = s.account_id
        LEFT JOIN node_activity r ON l.receiver_id = r.account_id
        WHERE split = 'validation' GROUP BY split""")
    result = {
        "scope": "Training topology only. Validation cold-start is relative to the frozen training graph.",
        "accounts": activity, "pairs": pairs, "sender_history": history,
        "validation_cold_start": cold_start,
        "type_counts": records(con, "SELECT type, count(*) AS transactions FROM train GROUP BY type ORDER BY type"),
        "self_transfers": con.execute("SELECT count(*) FROM train WHERE sender_id = receiver_id").fetchone()[0],
        "cautions": [
            "Public CSV account identifiers are treated as observed IDs, not verified customer identities.",
            "A connected component is not evidence of collusion.",
            "Graph results include all transaction types; transfer-only analysis belongs to a separate experiment.",
            "Validation novel-account rates are not final-test results.",
        ],
    }
    if compute_components:
        print("Computing exact training-graph weak components...", flush=True)
        con.execute("CREATE TABLE node_index AS SELECT account_id, row_number() OVER () - 1 AS id FROM node_activity")
        cursor = con.execute("""SELECT a.id, b.id FROM train t
            JOIN node_index a ON t.sender_id = a.account_id
            JOIN node_index b ON t.receiver_id = b.account_id""")
        result["weak_components"] = component_summary(
            activity["accounts"], iter(lambda: cursor.fetchmany(50000), [])
        )
    else:
        result["weak_components"] = {"status": "not computed by configuration"}
    return result


def render_report(report):
    quality, graph = report["quality"], report["graph"]
    activity = graph["accounts"]
    lines = [
        "# GraphShield AI: Phase 1 data and graph suitability", "",
        "Original PaySim synthetic transactions. No trained model or real-world fraud result.", "",
        f"- Source rows retained: {quality['rows']:,}",
        f"- Exact duplicate rows beyond first occurrence: {quality['exact_duplicate_extra_rows']:,} (retained)",
        f"- Zero-amount transactions: {quality['zero_amount_rows']:,} (retained)",
        f"- Source SHA-256: `{report['source']['sha256']}`", "",
        "## Temporal split", "",
        "Splits use ordered distinct time steps, not shuffled rows. No time step crosses a split.",
        "Final-test labels and graph diagnostics are not summarized. Split row counts are inventory only.", "",
        "| Split | First step | Last step | Rows |", "|---|---:|---:|---:|",
    ]
    for item in report["splits"]:
        lines.append(f"| {item['split']} | {item['min_step']} | {item['max_step']} | {item['rows']:,} |")
    labels = report["training_label_summary"]
    lines += ["", f"Training positives: {labels['positives']:,} / {labels['rows']:,} "
              f"({labels['positives'] / labels['rows']:.4%}).", "",
              "## Training graph", "",
              f"- Observed account IDs: {activity['accounts']:,}",
              f"- IDs appearing as both sender and receiver: {activity['both_roles']:,} ({activity['both_roles_fraction']:.2%})",
              f"- Accounts with only one incident event: {activity['single_event_accounts']:,} ({activity['single_event_fraction']:.2%})",
              f"- Senders with more than one outgoing event: {activity['repeat_senders']:,}",
              f"- Repeated directed account pairs: {graph['pairs']['repeated_directed_pairs']:,}"]
    hist = graph["sender_history"]
    lines.append(f"- Transactions whose sender had an earlier outgoing time step: {hist['transactions_with_prior_sender_step']:,} / {hist['transactions']:,}")
    components = graph["weak_components"]
    if "count" in components:
        lines += [f"- Weak components: {components['count']:,}",
                  f"- Largest weak component: {components['largest_component_accounts']:,} accounts ({components['largest_component_fraction']:.4%})"]
    for row in graph["validation_cold_start"]:
        lines += ["", f"Validation transactions with at least one account absent from the training graph: "
                  f"{row['either_unseen']:,} / {row['transactions']:,} ({row['either_unseen'] / row['transactions']:.2%})."]
    lines += ["", "## Interpretation and next gate", "",
              "These statistics measure data suitability, not model quality. Inspect account recurrence, "
              "role overlap and cold-start rates before selecting graph experiments. A large component "
              "may arise from normal high-volume receivers; it does not establish a ring.", "",
              "Keep the original transaction benchmark separate from generated coordinated-activity scenarios. "
              "Choose features and tune models using training and validation periods. Leave the final test sealed.", "",
              "## Data contracts", "",
              "- `transactions.parquet` has no fraud labels, balances, device links, or invented dates.",
              "- `labels.parquet` stores original transaction labels separately; discovery time is unknown.",
              "- Stable transaction IDs combine the source-file fingerprint with its 1-based data-row number.",
              "- Historical features for step t must use steps strictly less than t.",
              "- Identical source rows remain separate events; source data is never deduplicated silently.", ""]
    return "\n".join(lines)


def run_pipeline(csv_path: Path, config_path: Path, run_name: str, root: Path = ROOT):
    root = root.resolve()
    csv_path = inside_root(csv_path, root)
    config_path = inside_root(config_path, root)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_name):
        raise ValueError("Run name must contain 1-80 letters, numbers, underscores or hyphens")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        if next(csv.reader(stream), None) != HEADERS:
            raise ValueError(f"Expected the original PaySim CSV headers in order: {HEADERS}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = inside_root(Path("data/processed") / run_name, root)
    reports = inside_root(Path("artifacts/phase1") / run_name, root)
    work = inside_root(Path(".cache/runs") / run_name, root)
    if any(path.exists() for path in (output, reports, work)):
        raise ValueError("Run already exists. Choose a new run name; existing artifacts are never overwritten.")
    for path in (output, reports, work):
        path.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"status": "running", "run_name": run_name,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "duckdb_version": duckdb.__version__}
    write_json(reports / "manifest.json", manifest)
    database = work / "ingest.duckdb"
    con = None
    try:
        fingerprint = sha256(csv_path)
        manifest["source"] = {"path": str(csv_path.relative_to(root)), "sha256": fingerprint,
                              "bytes": csv_path.stat().st_size, "dataset_id": config["dataset_id"]}
        con = duckdb.connect(str(database), config={
            "memory_limit": config["memory_limit"], "threads": config["threads"],
            "temp_directory": str(work / "spill"), "preserve_insertion_order": True,
            "enable_external_access": True,
        })
        print("Reading original CSV with strict parsing...", flush=True)
        con.execute(f"""CREATE TABLE raw AS SELECT row_number() OVER () AS source_row_number, *
            FROM read_csv({literal(csv_path)}, header=true, all_varchar=true,
                          parallel=false, strict_mode=true, null_padding=false)""")
        missing = ", ".join(f'count(*) FILTER (WHERE "{name}" IS NULL OR trim("{name}") = \'\') AS "{name}"' for name in HEADERS)
        nulls = records(con, f"SELECT {missing} FROM raw")[0]
        checks = records(con, """SELECT count(*) AS rows,
            count(*) FILTER (WHERE NOT regexp_full_match(step, '[0-9]+') OR try_cast(step AS BIGINT) IS NULL) AS invalid_step,
            count(*) FILTER (WHERE try_cast(step AS BIGINT) < 0) AS negative_step,
            count(*) FILTER (WHERE regexp_full_match(amount, '[0-9]+(\\.[0-9]{1,2})?')
                            AND try_cast(amount AS DECIMAL(24,2)) IS NULL) AS invalid_amount,
            count(*) FILTER (WHERE try_cast(amount AS DOUBLE) = 0) AS zero_amount_rows,
            count(*) FILTER (WHERE isFraud NOT IN ('0', '1')) AS invalid_label,
            count(*) FILTER (WHERE isFlaggedFraud NOT IN ('0', '1')) AS invalid_rule_flag,
            count(*) FILTER (WHERE type NOT IN ('CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER')) AS invalid_type
            FROM raw""")[0]
        checks["missing_by_column"] = nulls
        unusual = con.execute("""SELECT amount, count(*) FROM raw
            WHERE amount IS NOT NULL AND NOT regexp_full_match(amount, '[0-9]+(\\.[0-9]{1,2})?')
            GROUP BY amount""")
        for batch in iter(lambda: unusual.fetchmany(10000), []):
            checks["invalid_amount"] += sum(count for amount, count in batch if not valid_money(amount))
        write_json(reports / "validation.json", checks)
        invalid = sum(checks[key] for key in ("invalid_step", "negative_step", "invalid_amount", "invalid_label", "invalid_rule_flag", "invalid_type"))
        if not checks["rows"] or invalid or any(nulls.values()):
            raise ValueError("Data validation failed; inspect validation.json. No cleaned Parquet published.")
        steps = [row[0] for row in con.execute("SELECT DISTINCT cast(step AS BIGINT) AS s FROM raw ORDER BY s").fetchall()]
        train_end, validation_end = split_boundaries(steps, config["train_fraction"], config["validation_fraction"])
        columns = ", ".join(f'"{name}"' for name in HEADERS)
        checks["exact_duplicate_extra_rows"] = con.execute(
            f"SELECT coalesce(sum(n - 1), 0)::BIGINT FROM (SELECT count(*) AS n FROM raw GROUP BY {columns} HAVING count(*) > 1)"
        ).fetchone()[0]
        write_json(reports / "validation.json", checks)
        con.execute(f"""CREATE TABLE ledger AS SELECT
            {literal(fingerprint[:16] + ':')} || source_row_number::VARCHAR AS transaction_id,
            source_row_number, cast(step AS BIGINT) AS step, type,
            cast(amount AS DECIMAL(24,2)) AS amount, nameOrig AS sender_id, nameDest AS receiver_id,
            CASE WHEN cast(step AS BIGINT) <= {train_end} THEN 'train'
                 WHEN cast(step AS BIGINT) <= {validation_end} THEN 'validation' ELSE 'test' END AS split
            FROM raw""")
        print("Writing label-separated Parquet data...", flush=True)
        con.execute(f"COPY ledger TO {literal(output / 'transactions.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.execute(f"""COPY (SELECT {literal(fingerprint[:16] + ':')} || source_row_number::VARCHAR AS transaction_id,
            cast(isFraud AS BOOLEAN) AS is_fraud, NULL::BIGINT AS label_available_step,
            'paysim_simulator' AS label_source FROM raw)
            TO {literal(output / 'labels.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        split_rows = records(con, """SELECT split, min(step) AS min_step, max(step) AS max_step,
            count(*) AS rows FROM ledger GROUP BY split ORDER BY min_step""")
        label_summary = records(con, f"""SELECT count(*) AS rows, sum(cast(isFraud AS BIGINT))::BIGINT AS positives
            FROM raw WHERE cast(step AS BIGINT) <= {train_end}""")[0]
        report = {"source": manifest["source"], "quality": checks, "splits": split_rows,
                  "training_label_summary": label_summary, "graph": graph_report(con, config["compute_components"])}
        write_json(reports / "report.json", report)
        (reports / "report.md").write_text(render_report(report), encoding="utf-8")
        manifest.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 2),
                        train_end_step=train_end, validation_end_step=validation_end,
                        files={name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                               for name in ("transactions.parquet", "labels.parquet")})
        write_json(reports / "manifest.json", manifest)
        print(f"Phase 1 complete: {reports / 'report.md'}", flush=True)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(reports / "manifest.json", manifest)
        raise
    finally:
        if con is not None:
            con.close()
        # Only the exact disposable database created by this run is removed.
        if database.exists():
            database.unlink()
