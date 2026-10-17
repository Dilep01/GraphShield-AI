# Build phases

Each phase ends with runnable artifacts, checks, and a recorded result. Features below
are planned until the status explicitly says complete. All planned phases are now
complete for version 0.7.0. The decisions below are historical records of each gate.

| Phase | Work | Acceptance evidence | Status |
|---|---|---|---|
| 0 | Scope, benchmark boundaries, time rules, schema, resource limits | PROJECT_SPEC.md and config/phase1.json | Complete |
| 1 | Original download, validation, Parquet, chronological splits, graph suitability | Successful full-data manifest, report and tests | Complete: paysim-v2 |
| 2 | Historical features, Logistic Regression and LightGBM | Reproducible training/validation comparison, cold-start fallback, saved feature contracts | Complete: historical-v1 / baselines-v1 |
| 3 | Coordinated scenarios and legitimate lookalikes | Isolated synthetic ground truth, seeds, group and pattern holdouts | Complete: coordinated-v1 |
| 4 | Historical graph features and candidate-group rules | Candidate cases and baseline ring metrics | Complete: rules-v1 |
| 5 | Node2Vec, GraphSAGE and controlled ablations | Learned ring comparisons and runtime; original transaction baseline remains separate | Complete: learned-v1 |
| 6 | API, batch replay, investigation dashboard | Traceable cases and idempotent ingestion demonstration | Complete: serving-v1 |
| 7 | Monitoring, packaging, final evaluation, documentation | Clean-environment reproduction, final-test results and limitations | Complete: final-v1 / 0.7.0 |

Tests accompany every phase. GAT, Identity nodes, streaming, and a graph database are
optional extensions after the first complete release.

Phase 2 starts from Phase 1 findings. If account recurrence or graph connectivity is weak,
document that result and adjust the experiment scope without altering source identities.
Do not generate relationships to improve the original PaySim benchmark's results.

## Phase 1 decision

The original public CSV contains 6,362,620 transactions. In the training period,
94.83% of observed account IDs have one incident event, no directed account pair repeats,
and only 1,596 of 8,566,556 IDs appear in both sender and receiver roles. The largest weak
component contains 119 accounts. Against the frozen training graph, 99.84% of validation
transactions have at least one unseen account.

This file provides limited evidence for persistent account behavior and coordinated
multi-hop groups. Do not rewrite IDs or add label-derived relationships to repair that
limitation. Phase 2 will establish an honest transaction baseline and account-history
availability. The generated benchmark in Phase 3 will provide persistent accounts,
explicit coordinated scenarios, benign lookalikes and separate membership ground truth.
GNN experiments on PaySim will be treated as limited cold-start experiments rather than
the primary evidence of ring detection.

The 60/20/20 time-step split produced 6,010,937 / 221,567 / 130,116 rows. This is not
a 60/20/20 row split; transaction volume is concentrated in earlier steps. Keep the
frozen split and report this distribution when evaluating future models.

Evidence: `artifacts/phase1/paysim-v2/report.md` and `manifest.json`.

## Phase 2 decision

LightGBM is the development ranking candidate, selected on the earlier validation
tuning section. On the later section it obtained AP 0.3787 versus 0.1526 for Logistic
Regression. Its tuning-selected threshold yielded 544 true positives and 1,602 false
positives out of 153,534 transactions with 780 positives. No final-test predictions exist.

The separate 20-alerts-per-step policy obtained 75.26% recall and 45.93% precision.
Uniform random selection within the same steps has expected recall 40.99%, partly
because 278 positives occur in steps where all transactions fit under the budget.

Both models have poor probability calibration under balanced-class fitting. Preserve
the outputs as risk scores. Before an API exposes probabilities, a later experiment
must validate calibration using development data without fitting to final-test labels.
The both-accounts-seen subgroup contains only one positive in the comparison section,
so it cannot support a reliable seen-versus-unseen performance conclusion.

Phase 3 follows this decision with a separate generated benchmark; its completed result
is recorded below.

## Phase 3 decision

Generated nine independent networks with 13,500 persistent accounts and 235,288
transactions, including 108 planted fraudulent groups and 162 legitimate groups.
Known planted templates are cycle, fan_in and fan_out; the relay_chain template is
reserved for test_unseen. Test_known uses unseen groups/seeds from known families.
These worlds have disjoint entities and local relative time, not a chronological
continuation of the PaySim dataset.

Observed records and evaluation truth are stored separately. Both classes have shared
devices and similar motifs. Background activity connects group members to outsiders;
candidate components cannot be assumed to be ground-truth rings. All 42 tests and 30
saved-artifact checks passed at the Phase 3 gate. Detection was evaluated in Phase 4 below.

Phase 4 followed with historical graphs, explicit features, candidate rules, duplicate
handling and predefined matching. Both synthetic test splits remain reserved.

## Phase 4 decision

The rules-v1 run built 135 snapshots and 2,802 candidate observations across five
development worlds. Both fixed policies matched 24/24 validation fraud groups across
150 alerts: 16% case precision, with 35/36 legitimate groups also alerted. Graph-only
and graph-plus-device priority had equal headline metrics, but different alerts and
delays. Preserve both as baselines; neither result establishes reliable fraud classification.

All 61 tests and 16 full-replay artifact checks passed. Candidate construction and
ranking use only historical observations; evaluation truth is loaded after predictions
are saved. No PaySim or synthetic final-test evaluation was performed.

Phase 5 followed this decision: first fit a supervised ranking baseline using the explicit graph features,
then compare Node2Vec and GraphSAGE with controlled information, review budgets and
runtime measurements. Reducing false alerts is the main open question. Preserve
group/seed separation, handle independent-world embeddings explicitly and report
candidate-coverage limits. Reserve both synthetic test splits for the final protocol.

Evidence: `docs/PHASE4_METHOD.md`, `docs/PHASE4_RESULTS.md`, and `artifacts/phase4/rules-v1/`.

## Phase 5 decision

Trained five learned rankers and compared them with both fixed rule scores using the
same historical candidates, review budget and group matching. The tuning network chose
Node2Vec plus LightGBM (F1 0.6286). On the separate development comparison network,
it matched 7/12 fraud groups with 17 alerts: 41.18% precision and 58.33% recall.

GraphSAGE had higher comparison F1 (0.6207), matching 9/12 groups with 17 alerts, but
did not win on tuning. Preserve the frozen selection and report the disagreement.
All budget-only variants recovered every group with roughly 16% precision. Threshold
abstention trades fewer alerts for missed groups; tuned rules are also included as controls.

The CPU run took 455.68 seconds, including 403.90 seconds for historical graphs and
snapshot-local Node2Vec. All 76 tests and 20 artifact checks passed. Both synthetic test
splits and PaySim final test remain reserved. Original transaction models were not
retrained; the limited recurring PaySim graph remains a documented benchmark constraint.

Phase 6 followed with a versioned API, batch replay and investigation dashboard. Keep observed
evidence, heuristic/model scores and retrospective benchmark truth distinct. Make
policy choice explicit, retain the tuning-selected experimental model and avoid silently
promoting GraphSAGE based on this single comparison network. Streaming and production
operation are not established by these offline results.

Evidence: `docs/PHASE5_METHOD.md`, `docs/PHASE5_RESULTS.md`, and `artifacts/phase5/learned-v1/`.

## Phase 6 decision

The local FastAPI app serves an investigation dashboard, observed-event ingestion,
read-only scoring, resumable replay and separate reviewer records. A static frontend
replaces the planned Streamlit app so the API and UI use one process and local address.
The frozen tuning-selected Node2Vec model remains the default. Its demonstration
database contains 17 comparison-network cases across 27 completed snapshots.

All 88 tests passed. Fourteen comparison-network replays (seven models, two policies)
matched the Phase 5 alert selections and scores. Browser checks covered evidence,
contributions, reviewer notes, status filtering, alternate-model replay and phone-width
layout. No new model selection or final-test evaluation was performed.

Phase 7 followed with monitoring, packaging, clean-environment reproduction and the
frozen final evaluation protocol. The local single-worker demonstration does not establish
production readiness or real-world fraud performance.

Evidence: `docs/PHASE6_METHOD.md`, `docs/PHASE6_RESULTS.md`, and `artifacts/phase6/`.

## Phase 7 decision

Completed the frozen final evaluation without selecting new thresholds or promoting
models after viewing test results. The transaction LightGBM reached AP 0.5378 on
130,116 held-out PaySim rows. The selected Node2Vec ranker matched 19/24 known-family
groups at 38% precision. Tuned repeated-transfer rules matched 22/24 at 39.29% precision.
Every ranker missed the 24 withheld relay-chain groups because candidate coverage
for that family was zero. This is a documented evaluation failure, not a solved feature.

Operational request counters, rotating logs and replay health checks are implemented.
The version 0.7.0 wheel and synthetic development demo reproduced 27 snapshots and
17 cases in a fresh offline environment. The installed-package suite passed 96 tests;
seven final verification checks passed. The subsequent CSS-only fix was verified at
615- and 375-pixel widths and included in the final wheel. Source/package checksums,
run instructions and browser evidence complete the planned release.

No planned implementation phase remains open. Any relay-chain improvement needs a
new experiment and fresh held-out worlds because these final labels have been used.
Production authentication, streaming, deployment and real-data validation remain
optional work outside this local prototype.

Evidence: `docs/PHASE7_RESULTS.md`, `docs/RUNBOOK.md`, `docs/MODEL_CARD.md`, and
`artifacts/phase7/manifest.json`.
