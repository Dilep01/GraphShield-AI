# Phase 2 methodology

This phase builds transaction baselines on original synthetic PaySim records. It does
not generate devices or rings and does not evaluate collusion detection.

## Historical features

For each account, aggregate outgoing and incoming events into complete time buckets.
Use only buckets strictly earlier than the transaction's step. Both endpoints receive
the same kinds of account-history features:

| Features | Calculation |
|---|---|
| Current log amount | ln(1 + submitted amount) |
| Five type indicators | Fixed CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER vocabulary |
| Prior outgoing/incoming counts | ln(1 + all prior events in that direction) |
| Prior outgoing/incoming amounts | ln(1 + sum of prior amounts in that direction) |
| Recent incident events | ln(1 + incoming + outgoing count in steps [t-24, t)) |
| Time since prior event | ln(1 + t - most recent earlier step) |
| Has history | 1 if an event exists at an earlier step, otherwise 0 |

There are 20 numeric model inputs. Account identifiers are lookup keys only. Raw IDs,
source row numbers, absolute step, split, balances, labels and original rule flags are
not predictors. Historical amounts are aggregated as decimals before log transformation;
model inputs use float32.

New accounts have zero history features and `has_history=0`. Zero gap for an unseen
account is therefore distinguishable from genuine history. Current amount and type
still enter the model. A self-transfer contributes to both incoming and outgoing counts;
the incident count is a count of endpoint participations in that case.

Validation replays earlier validation transactions as unlabeled history, after their
bucket has completed. No model or scaler is refitted during this replay. A separate
metadata flag records whether either account was absent at the training cutoff; it is
used for diagnostics only and is not a model feature.

## Training and validation

The existing Phase 1 time split stays unchanged. All training rows are used, with no
negative undersampling or oversampling. Logistic Regression uses training-fitted
standardization and balanced class weights. LightGBM receives the identical inputs
and balanced class weights, with deterministic CPU settings and four worker threads.

Divide the existing validation period into earlier and later halves of its observed
steps. The earlier section is used for LightGBM early stopping (average precision),
each model's maximum-F1 threshold, and model selection by average precision. The later
section reports the comparison after those choices. It remains validation, not an
independent final test. Future development must not relabel it as a final holdout.

Configurable settings are frozen in config/phase2.json and copied into manifests.
Logistic Regression convergence warnings fail the run rather than silently publishing
a nonconverged model. Both classes must exist in training and each validation section.

## Evaluation

Average precision is sklearn's non-interpolated precision-recall summary, not trapezoidal
area. Also record precision, recall, F1 and confusion counts at the tuning-selected
threshold; Brier score, log loss and equal-width reliability bins; a global top-100 queue;
and a separate budget policy selecting at most 20 transactions per observed time step.

The budget policy fills min(20, available transactions) slots, independent of the F1
threshold. Ties use source row order for deterministic ranking, not as a prediction
feature. The policy operates on transactions, not deduplicated accounts or cases.

Also compute the exact expected yield of uniform random selection within each time
step under the same capacity. When a step has fewer than 20 events, every event is
selected even without a useful ranking. Report how many positives fall in those steps
so budget recall is not mistaken for pure model discrimination.

Seen/unseen-account subgroup budget metrics inherit selections from the full comparison
queue. They do not grant each subgroup a fresh alert allowance. Empty/no-positive
subgroups have null AP/recall where those metrics are undefined.

Class weighting can distort probabilities. Scores are not claimed to be calibrated;
calibration is diagnosed rather than adjusted using the comparison period. The constant
score average-precision reference equals prevalence in the evaluated section. The
constant-score Brier reference uses training prevalence, without estimating it from
comparison outcomes.

Compute 200 paired bootstrap replicates of complete observed comparison steps and
percentile 95% intervals for AP and the between-model AP difference. This assumes
exchangeable step blocks. It excludes training-seed variation and may understate
uncertainty if dependence persists across consecutive steps.

## Artifacts and reproduction

```powershell
.\scripts\bootstrap.ps1
.\scripts\test.ps1
.\.venv\Scripts\python.exe -B -m graphshield.cli features --source-run paysim-v2 --run-name historical-reproduction
.\.venv\Scripts\python.exe -B -m graphshield.cli train --feature-run historical-reproduction --run-name baselines-reproduction
.\.venv\Scripts\python.exe -B scripts/verify_phase2.py --run-name baselines-reproduction
```

Feature files are under data/processed/phase2. Feature manifests, model bundles,
LightGBM text export, predictions, metrics and report are under artifacts/phase2.
Models preserve the exact feature order, tuning threshold and feature contract.
The verifier checks artifact hashes, reload predictions and recomputed metrics.
Only runs marked `complete` are valid inputs. Run names never overwrite prior artifacts.

Feature generation, model fitting, prediction export and tests keep caches and temporary
work inside this project. The final-test rows are excluded before history construction,
and no final-test predictions or performance metrics are produced in this phase.
