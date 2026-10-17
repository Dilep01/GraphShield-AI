"""Label-free features from strictly earlier time buckets, including validation replay."""

from pathlib import Path

from .pipeline import TYPES, literal

CURRENT_FEATURES = ["log_amount", *["type_" + kind.lower() for kind in TYPES]]
HISTORY_FEATURES = [
    f"{role}_{name}" for role in ("sender", "receiver") for name in (
        "log_out_count", "log_in_count", "log_out_amount", "log_in_amount",
        "log_events_24steps", "log_steps_since_seen", "has_history",
    )
]
FEATURES = CURRENT_FEATURES + HISTORY_FEATURES
CONTRACT = {
    "version": "historical-v1", "features": FEATURES,
    "history_cutoff": "step < current step; all events in current step excluded",
    "validation_mode": "Replay earlier validation events as unlabeled history; models stay fixed",
    "rolling_window": "[step - 24, step); time-step units, not invented calendar timestamps",
    "transforms": "ln(1+x) for amounts, counts and gaps; fixed one-hot types; float32 inputs",
    "cold_start": "Zero history aggregates/gap and has_history=0; current attributes still scored",
    "excluded": ["account IDs", "transaction IDs", "source row numbers", "split", "absolute step",
                 "balance columns", "fraud labels", "isFlaggedFraud", "final-test events"],
}


def build_features(con, transactions: Path, destination: Path, train_end: int, validation_end: int):
    """Disk-backed SQL avoids holding millions of Python account dictionaries."""
    con.execute(f"""CREATE TABLE events AS SELECT * FROM read_parquet({literal(transactions)})
        WHERE split IN ('train','validation') AND step <= {int(validation_end)}""")
    mismatch = con.execute(f"""SELECT count(*) FROM events
        WHERE (split = 'train' AND step > {int(train_end)})
           OR (split = 'validation' AND step <= {int(train_end)})""").fetchone()[0]
    if mismatch:
        raise ValueError("Source split values disagree with the Phase 1 manifest")
    print("Aggregating account activity by complete time steps...", flush=True)
    con.execute("""CREATE TABLE buckets AS
        SELECT account_id, step, sum(out_n)::BIGINT AS out_n, sum(in_n)::BIGINT AS in_n,
               sum(out_amount) AS out_amount, sum(in_amount) AS in_amount
        FROM (
            SELECT sender_id AS account_id, step, 1 AS out_n, 0 AS in_n,
                   amount AS out_amount, 0::DECIMAL(24,2) AS in_amount FROM events
            UNION ALL
            SELECT receiver_id, step, 0, 1, 0::DECIMAL(24,2), amount FROM events
        ) GROUP BY account_id, step""")
    con.execute("""CREATE TABLE histories AS SELECT account_id, step,
        coalesce(sum(out_n) OVER past, 0) AS prior_out_n,
        coalesce(sum(in_n) OVER past, 0) AS prior_in_n,
        coalesce(sum(out_amount) OVER past, 0) AS prior_out_amount,
        coalesce(sum(in_amount) OVER past, 0) AS prior_in_amount,
        coalesce(sum(out_n + in_n) OVER recent, 0) AS prior_recent_n,
        max(step) OVER past AS last_step
        FROM buckets
        WINDOW past AS (PARTITION BY account_id ORDER BY step RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
               recent AS (PARTITION BY account_id ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING)""")
    con.execute("CREATE TABLE first_seen AS SELECT account_id, min(step) AS first_step FROM buckets GROUP BY account_id")
    expressions = ["ln(1 + e.amount::DOUBLE)::FLOAT AS log_amount"]
    expressions += [f"(e.type = {literal(kind)})::FLOAT AS type_{kind.lower()}" for kind in TYPES]
    for role, alias in (("sender", "s"), ("receiver", "r")):
        for name, column in (("log_out_count", "prior_out_n"), ("log_in_count", "prior_in_n"),
                             ("log_out_amount", "prior_out_amount"), ("log_in_amount", "prior_in_amount"),
                             ("log_events_24steps", "prior_recent_n")):
            expressions.append(f"ln(1 + {alias}.{column}::DOUBLE)::FLOAT AS {role}_{name}")
        expressions += [
            f"ln(1 + coalesce(e.step - {alias}.last_step, 0))::FLOAT AS {role}_log_steps_since_seen",
            f"({alias}.last_step IS NOT NULL)::FLOAT AS {role}_has_history",
        ]
    query = f"""SELECT e.transaction_id, e.source_row_number, e.step, e.split,
        (fs.first_step > {int(train_end)} OR fr.first_step > {int(train_end)}) AS either_unseen_at_train_cutoff,
        {', '.join(expressions)}
        FROM events e JOIN histories s ON e.sender_id = s.account_id AND e.step = s.step
        JOIN histories r ON e.receiver_id = r.account_id AND e.step = r.step
        JOIN first_seen fs ON e.sender_id = fs.account_id
        JOIN first_seen fr ON e.receiver_id = fr.account_id"""
    print("Writing historical feature Parquet (train and validation only)...", flush=True)
    con.execute(f"COPY ({query}) TO {literal(destination)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"CREATE VIEW features AS SELECT * FROM read_parquet({literal(destination)})")
    counts = con.execute("SELECT split, count(*) FROM features GROUP BY split ORDER BY split").fetchall()
    if sum(row[1] for row in counts) != con.execute("SELECT count(*) FROM events").fetchone()[0]:
        raise ValueError("Feature joins changed transaction cardinality")
    invalid = " OR ".join(f'"{name}" IS NULL OR NOT isfinite("{name}")' for name in FEATURES)
    if con.execute(f"SELECT count(*) FROM features WHERE {invalid}").fetchone()[0]:
        raise ValueError("Nonfinite or missing features")
    return dict(counts)
