# Phase 7: final evaluation and local release

Release: 0.7.0. This phase finishes the planned local research prototype. It does not
establish production readiness or performance on real financial records.

## Frozen evaluation

`config/phase7.json` names the existing PaySim and generated benchmark runs. The
`freeze` command exports the previously selected models, thresholds, budgets, source
manifest checksums and implementation checksums to `protocol.json`. The selected
transaction model remains LightGBM; the selected group ranker remains Node2Vec plus
LightGBM. No parameters, scalers, weights, candidate rules or thresholds are fitted
using final labels.

Prediction generation opens only observed final events. PaySim final features replay
all earlier training, validation and final-test buckets, excluding every event in the
current bucket. The 20 input transformations are unchanged from Phase 2; a fixture
test checks exact agreement with that implementation. Only final-test rows are exported.

The four synthetic final worlds use the same 27 cutoffs, 168-step history, arrival
filter, candidate rules and duplicate policy as development. Snapshot-local Node2Vec
embeddings are fitted on visible edges only. The two neural models use frozen training
scalers and weights. All seven rankers run under both policies without model selection.

All predictions, candidates, alert decisions and embeddings are checksummed in
`prediction_seal.json` before the evaluator opens held-out truth. The evaluator refuses
missing or changed sealed inputs. Results are stored separately from predictions and
the dashboard. Historical phase manifests are preserved as records of those phases.

The final labels have now been consumed. Further model changes require a new benchmark
for another untouched evaluation. Replaying this run verifies reproducibility; it does
not create a new independent test.

## Metrics and uncertainty

PaySim reports average precision, positive counts, threshold confusion counts, a fixed
20-alerts-per-step budget, top-100 selection, cold-start cohorts and calibration
diagnostics. Thresholds and budgets come from Phase 2. A 200-repeat paired bootstrap
resamples complete time steps for AP intervals. This omits cross-step dependence and
training uncertainty. The random within-step budget reference is an analytical
expectation, not measured model performance.

Synthetic group metrics preserve the Phase 4 one-to-one matching rule: member precision,
member recall and a planted evidence event are required. Candidate coverage is measured
before ranking. Known-family and withheld-family results remain separate. Whole-world
paired bootstrap ranges use 2,000 resamples, but only two independent worlds exist per
split. These are descriptive ranges, not reliable population confidence or significance
evidence. A [0, 0] range on the withheld family does not establish zero risk in other data.

## Monitoring

`GET /api/v1/monitoring` exposes bounded counters by HTTP method and route template:
request counts, 4xx/5xx counts, mean/max latency and latency buckets. Counts reset when
the process restarts. Route templates omit account IDs, case IDs, bodies, query strings
and reviewer notes. The launcher writes rotating request logs under
`artifacts/phase7/runtime/requests.log`, limited to 1 MB plus three backups.

Durable workflow checks report case statuses, active/completed/failed replay jobs,
database size and per-model/version saved-case score histograms. A running or pending
job with no active worker is flagged for explicit resume. The dashboard's Local system
health panel refreshes these checks on demand. These are operational observations;
there is no claim of live accuracy monitoring or a validated statistical drift detector.

## Packaging and verification

The release contains a Python wheel, a Windows CPython 3.11 dependency wheelhouse and
a synthetic development-data archive. The data archive contains the serving bundle,
its referenced observed development networks and embedding caches, and the clean demo
database. It excludes PaySim raw data, final-test data and QA reviewer notes.

An independent virtual environment under `.cache/reproduction/env` installed pinned
dependencies offline and the built wheel without editable source imports. The
reproduction script verified its installed import path, extracted the demo into a new
project-local workspace, reproduced all 27 snapshots and 17 default cases with exact
scores, and checked API routes and packaged web assets. The full suite then ran against
the installed package with the source-path override disabled.

The final verifier reloads both transaction models and all seven group rankers,
reconstructs all final graph snapshots, checks candidate/alert selections, and recomputes
headline metrics and coverage. Cached Node2Vec embeddings are checksummed, not retrained
by this verifier. Float32 Logistic Regression scores use a 1e-6 absolute tolerance;
LightGBM uses 1e-12. Saved final predictions remain unchanged.

See [RUNBOOK.md](RUNBOOK.md) for launch, installation, recovery and reproduction commands.
