"""Final-test replay of the frozen Phase 2 transformations, without labels."""
from .features import FEATURES
from .pipeline import TYPES, literal


def build_final_features(con, transactions, destination, train_end, validation_end):
    # Full history is needed, but only final-test rows are exported for scoring.
    con.execute(f'CREATE TABLE events AS SELECT * FROM read_parquet({literal(transactions)})')
    bad = con.execute('''SELECT count(*) FROM events WHERE split NOT IN ('train','validation','test')
        OR (split='train' AND step>?) OR (split='validation' AND (step<=? OR step>?))
        OR (split='test' AND step<=?)''',[train_end,train_end,validation_end,validation_end]).fetchone()[0]
    if bad:
        raise ValueError('Source split boundaries disagree with the frozen manifest')
    con.execute('''CREATE TABLE buckets AS SELECT account_id,step,sum(out_n)::BIGINT AS out_n,
        sum(in_n)::BIGINT AS in_n,sum(out_amount) AS out_amount,sum(in_amount) AS in_amount FROM (
        SELECT sender_id AS account_id,step,1 AS out_n,0 AS in_n,amount AS out_amount,
            0::DECIMAL(24,2) AS in_amount FROM events UNION ALL
        SELECT receiver_id,step,0,1,0::DECIMAL(24,2),amount FROM events) GROUP BY account_id,step''')
    con.execute('''CREATE TABLE histories AS SELECT account_id,step,
        coalesce(sum(out_n) OVER past,0) AS prior_out_n,coalesce(sum(in_n) OVER past,0) AS prior_in_n,
        coalesce(sum(out_amount) OVER past,0) AS prior_out_amount,
        coalesce(sum(in_amount) OVER past,0) AS prior_in_amount,
        coalesce(sum(out_n+in_n) OVER recent,0) AS prior_recent_n,max(step) OVER past AS last_step
        FROM buckets WINDOW past AS (PARTITION BY account_id ORDER BY step RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        recent AS (PARTITION BY account_id ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING)''')
    con.execute('CREATE TABLE first_seen AS SELECT account_id,min(step) AS first_step FROM buckets GROUP BY account_id')
    expressions=['ln(1+e.amount::DOUBLE)::FLOAT AS log_amount']
    expressions += [f'(e.type={literal(kind)})::FLOAT AS type_{kind.lower()}' for kind in TYPES]
    for role,alias in (('sender','s'),('receiver','r')):
        for name,column in (('log_out_count','prior_out_n'),('log_in_count','prior_in_n'),
                            ('log_out_amount','prior_out_amount'),('log_in_amount','prior_in_amount'),
                            ('log_events_24steps','prior_recent_n')):
            expressions.append(f'ln(1+{alias}.{column}::DOUBLE)::FLOAT AS {role}_{name}')
        expressions += [f'ln(1+coalesce(e.step-{alias}.last_step,0))::FLOAT AS {role}_log_steps_since_seen',
                        f'({alias}.last_step IS NOT NULL)::FLOAT AS {role}_has_history']
    query=f'''SELECT e.transaction_id,e.source_row_number,e.step,e.split,
        (fs.first_step>{int(train_end)} OR fr.first_step>{int(train_end)}) AS either_unseen_at_train_cutoff,
        {','.join(expressions)} FROM events e
        JOIN histories s ON e.sender_id=s.account_id AND e.step=s.step
        JOIN histories r ON e.receiver_id=r.account_id AND e.step=r.step
        JOIN first_seen fs ON e.sender_id=fs.account_id JOIN first_seen fr ON e.receiver_id=fr.account_id
        WHERE e.split='test' '''
    con.execute(f'COPY ({query}) TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    con.execute(f'CREATE VIEW features AS SELECT * FROM read_parquet({literal(destination)})')
    count=con.execute('SELECT count(*) FROM features').fetchone()[0]
    if count != con.execute("SELECT count(*) FROM events WHERE split='test'").fetchone()[0] or not count:
        raise ValueError('Final feature cardinality mismatch or empty test')
    invalid=' OR '.join(f'{name} IS NULL OR NOT isfinite({name})' for name in FEATURES)
    if con.execute('SELECT count(*) FROM features WHERE '+invalid).fetchone()[0]:
        raise ValueError('Invalid final features')
    return count
