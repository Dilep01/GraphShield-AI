# GraphShield AI project specification

Status: planned Phases 0–7 complete as the version 0.7.0 local research prototype.
Final evaluation and limits are recorded in PHASE7_RESULTS.md; launch and recovery
instructions are in RUNBOOK.md. Earlier design statements below retain their scope.

GraphShield AI is a graph-based fraud investigation prototype. It ranks transaction risk and prioritizes connected groups for investigation using historical transaction activity and
explicitly simulated relationships. Graph methods are compared against tabular and rule baselines, and the dashboard
shows the evidence behind candidate cases. Account involvement is visible; there is
no separately validated account-level fraud probability.

## Scope and labels

There are two independent benchmarks:

1. **Original PaySim transaction benchmark.** Target is the simulator's original
   transaction fraud label. Original account IDs and transaction events are preserved.
2. **Generated coordinated-activity benchmark.** Targets are explicit generated ring
   membership and coordinated event labels. Generated relationships and transactions
   are kept separate from original PaySim records. Results measure recovery of designed
   synthetic scenarios, not demonstrated real-world collusion detection.

An account risk score initially means an investigation priority. It is not a calibrated
probability of guilt. A receiver of a fraudulent transaction is not automatically a
colluder. Communities and connected components are candidate-generation aids.

## Time contract

- Keep the original `step` rather than inventing event timestamps or calendar dates.
- For scoring step t, historical events and graph edges must have step strictly below t.
- Score a complete bucket before updating historical state with that bucket's events.
- Current submitted amount, type, and endpoints may enter the transaction head.
- Do not infer an order among transactions in one step from their CSV row numbers.
- Discovery time is not supplied by this CSV. Labels have a null availability time;
  they cannot be used as historically known-fraud features.
- Future relationship `last_seen` values and final graph topology are forbidden as
  past prediction inputs. Generated observations will carry availability times.
- Validation may replay earlier validation events as they become history. Model fitting
  and tuning must not consume final-test outcomes. Report static and replay settings
  separately when both are implemented.

## Split contract

The initial configuration uses the earliest 60% of distinct observed steps for training,
the next 20% for validation, and the remainder for the final test. Percentages refer to
steps, not transaction counts. Splits never divide one step. This choice is frozen before
model fitting; boundaries, source fingerprint, and counts are saved in each manifest.

Phase 1 validates schema and inventories row counts across the file. Topology diagnostics
and label prevalence use training data only; the validation cold-start audit uses IDs,
not labels. It does not print final-test label counts, prevalence, or topology.

Final-test selection and threshold selection remain separate. Later experiments must
report positive counts and confidence intervals, not just a headline score.

## Data contract

`transactions.parquet`:

| Field | Meaning |
|---|---|
| transaction_id | Source SHA-256 prefix plus one-based source data-row number |
| source_row_number | Source traceability; never a predictive feature |
| step | Original simulator time bucket |
| type | Original transaction category |
| amount | Exact nonnegative decimal amount, scale two |
| sender_id / receiver_id | Original observed identifiers |
| split | train, validation, or test; never a predictive feature |

`labels.parquet` contains transaction_id, is_fraud, label_available_step (null), and
label_source (`paysim_simulator`). These labels are joined explicitly for supervised
training or evaluation, never implicitly during historical feature construction.

All four balance fields and `isFlaggedFraud` are excluded from model-input Parquet.
They remain in the immutable original CSV. Duplicate source rows and zero amounts are
reported and retained; neither condition alone proves an invalid event. Unknown types,
missing fields, invalid labels, invalid steps, nonfinite/negative amounts and fractional
cents fail validation. Scientific notation such as `1.0E7` is valid exact currency.

Phase 3 now supplies synthetic accounts, devices, transactions, device_observations,
groups, group_memberships, ring_memberships and transaction_labels in a separate dataset.
See PHASE3_METHOD.md for its independent-world split and observation-availability contracts.
Phase 4 exports historical graph snapshots, candidate features/evidence, alert cases and
selection decisions as JSONL, with retrospective metrics stored separately.
Phase 6 serving tables: observed transactions/device observations, ingestion receipts,
replays, snapshots, immutable cases and separate reviews. Ground truth
remains separate from case predictions and user feedback. Identity nodes are deferred
until a concrete shared-identity scenario requires them.

## Models and experiments

Implemented: Logistic Regression and LightGBM on the original transaction benchmark;
activity/explicit-graph LightGBM, Node2Vec plus LightGBM, GraphSAGE and a no-message neural
control on generated candidate groups. Phase 5 uses an account graph with device-derived
edges and PyTorch directly. Typed heterogeneous account-device GraphSAGE and GAT remain
optional extensions.

Compare models on the same permitted information, split, and investigation budget.
Include historical account aggregates in the strong tabular baseline. GraphSAGE need
not win for this project to succeed. Standard GraphSAGE does not consume arbitrary edge
amount/time attributes automatically; the feature and message design must specify how
they are used. Node2Vec needs an unseen-node fallback and a paired embedding/classifier
version. Never train embeddings on future edges to improve validation coverage.

Planned transaction metrics: average precision (explicit metric definition), precision,
recall, F1, precision at top K, recall at a fixed alert budget, calibration diagnostics,
and performance on unseen-account transactions. Thresholds are selected on validation.
Training imbalance handling must not change evaluation prevalence.

Phase 4 ring metrics: case precision under a fixed review budget, ring recovery under a predefined one-to-one
matching rule, member precision/recall, duplicate cases, and detection delay. Large
clusters containing many known rings must not receive multiple unqualified successes.
Compare motif rules, graph-feature ranking, and GNN-informed ranking separately.

Use paired resampling of appropriate time blocks or generated groups for model
differences. Generated benchmark splits hold out groups and seeds, with an additional
test for withheld pattern families. Include legitimate shared devices and transaction
patterns as hard negatives.

## Delivery architecture

Current: Python, DuckDB, Parquet, strictly historical account features, trained Logistic
Regression and LightGBM baselines, a separate synthetic coordinated-activity generator
and structural auditor, bounded graph candidate rules, explicit group features,
budgeted case replay, command-line pipeline and JSON/JSONL/Markdown reports.
Phase 5 adds CPU PyTorch, snapshot-local Node2Vec, learned group ranking, inductive
GraphSAGE weights, controlled ablations and reproducible checkpoint evaluation.
Phase 6 adds FastAPI, SQLite and a static investigation dashboard served by the same
local process, replacing the planned separate Streamlit frontend. PyTorch Geometric is optional; the current
mean aggregation and biased-walk models do not require compiled graph extensions.

Phase 4 rules and metrics are defined in PHASE4_METHOD.md. They use only synthetic
development worlds, with observed events strictly preceding each snapshot. Scores
remain heuristic priorities; high recovery of planted structures is not proof of fraud.

Phase 5 fits models on three completed archival worlds. It uses the first validation
world to select alert thresholds/model and the second for development comparison.
All graph inputs are historical; training labels are retrospective and fully adjudicated.
This is not online label-discovery/retraining simulation. See PHASE5_METHOD.md for the
shared information boundary, model controls and whole-world duplicate policy.

Phase 2 now divides the existing validation window into an earlier tuning section and
a later comparison section, preserving the Phase 1 final-test boundary. See PHASE2_METHOD.md
for the exact fitting and threshold-selection contract and PHASE2_RESULTS.md for measured
results. Outputs remain risk scores because current calibration diagnostics are poor.

Batch replay is the initial serving mode. Scoring does not insert events. A separate
idempotent ingestion operation updates history. Responses include bundle/model version,
graph snapshot, feature cutoff, and an explicit null fallback reason when no fallback occurs. Evidence views distinguish
observed events, triggered rules, and model contributions.

No graph database, streaming broker, identity resolution, automatic retraining, or GAT
is required for the first release. Public hosting and cloud infrastructure are deferred.

## Final release boundary

Phase 7 adds bounded request/error/latency counters, rotating logs, durable replay
health checks, an installed-wheel launcher, offline packaging and sealed final
predictions. Monitoring is operational; validated statistical drift detection and
live fraud-accuracy measurement are not implemented. The final labels have now been
used and cannot serve as untouched evidence for future model changes.

The selected ring ranker recovered 19/24 known-family groups but 0/24 withheld
relay-chain groups. Candidate coverage is the limiting stage on that family. The
tuned transfer-rule baseline also outperformed the selected learned ranker on known
case precision and recall. Keep these failures visible in any demonstration or report.

## Resource and filesystem limits

All project-generated files, downloads, virtual environments, temporary data, and caches
belong inside this repository. Use the provided PowerShell scripts. Raw CSV inputs must
also be inside the project. Path resolution rejects parent traversal and resolved paths
outside the repository. The runtime reads its installed Python standard library.

Initial environment: Windows, Python 3.11.9, approximately 15.3 GiB physical RAM reported
by the OS and 12 logical processors visible to the process. Phase 1 caps DuckDB at 1 GB
and four threads. Python component arrays add memory outside DuckDB's cap. GPU support
has not been assessed and is not required for Phase 1.

## Acceptance gates

- Raw bytes and download origin are fingerprinted; original CSV remains unchanged.
- Valid input produces exact-decimal Parquet with separate labels and deterministic IDs.
- Validation failures cannot produce apparently successful artifacts.
- Temporal boundaries, training graph structure, and validation cold-start rates are reported.
- Tests exercise malformed data, row preservation, deterministic ingestion, component
  correctness, path containment, and exclusion of current/future history.
- Each completed run has a manifest with source/output hashes and pinned dependencies.
- Later claims distinguish source data, generated data, model metrics, and final case policy.

## Sources

- PaySim dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
- Creator's simulator repository: https://github.com/EdgarLopezPhD/PaySim
- PyG heterogeneous models: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
- PyG SAGEConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html

The simulator repository's software license does not by itself establish the dataset's
redistribution terms. Raw downloads are excluded from Git; reproduction uses the source URL.
