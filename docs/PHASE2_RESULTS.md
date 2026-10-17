# Phase 2 completion record

Completed on 2026-10-14. Feature run: `historical-v1`. Model run: `baselines-v1`.
All artifacts, environments, caches and temporary work for this phase stayed inside
`C:\Users\dilip\Project\graphshield_aI`.

## What is implemented

Twenty model inputs combine submitted transaction amount/type with historical sender
and receiver activity. Histories exclude the current step and future steps. Validation
uses earlier validation events as unlabeled history, while fitted models stay fixed.
New accounts have explicit no-history indicators and zero history aggregates.

Both models used all 6,010,937 training rows, including 5,007 fraud labels, with balanced
class weights. Logistic Regression converged in 40 iterations. LightGBM selected 81
boosting iterations using the earlier validation tuning section.

The feature pipeline processed training and validation rows only. The 130,116 final-test
transactions were not featurized, predicted, or used for model selection in this phase.

## Validation comparison

Training covers steps 1-445. The original validation window is divided into tuning
steps 446-519 and comparison steps 520-593. Maximum-F1 thresholds and the development
model selection use tuning only. The later comparison is still validation, not a final test.

The comparison section contains 153,534 transactions, including 780 fraud labels.
The constant-score average-precision reference is its 0.5080% prevalence.

| Model | Average precision | Precision | Recall | F1 | True positives | False positives |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.152604 | 23.13% | 29.36% | 0.2588 | 229 | 761 |
| LightGBM | 0.378687 | 25.35% | 69.74% | 0.3718 | 544 | 1,602 |

These confusion counts use each model's threshold selected on the earlier tuning
section: approximately 0.92929 for Logistic Regression and 0.93850 for LightGBM.
Thresholds are tied to the saved class-weighted models; they are not general fraud
probability cutoffs.

LightGBM was selected by tuning-section average precision, before considering the
later comparison results. It is a development ranking candidate, not a deployed service.

## Investigation capacity

The separate budget policy selects at most 20 transactions per observed step, regardless
of the F1 threshold. It selected 1,278 transactions for each model in the comparison period.

| Ranking | Precision | Recall | True positives | False positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 40.14% | 65.77% | 513 | 765 |
| LightGBM | 45.93% | 75.26% | 587 | 691 |

Uniform random selection within each step at that capacity has **expected** precision
25.02% and recall 40.99%. This is an analytical reference, not an observed model result.
There are 24 comparison steps with at most 20 transactions; these steps contain 278
positives that any capacity-filling policy would select. Report this reference alongside
budget recall to avoid attributing all detections to model discrimination.

## Limits and uncertainty

- Paired bootstrap over 200 resamples of complete comparison time steps gives AP
  intervals of approximately [0.0896, 0.2469] for Logistic Regression and
  [0.3039, 0.4740] for LightGBM. The AP-difference interval is [0.1899, 0.2513].
  These intervals omit training-seed uncertainty and assume exchangeable time blocks;
  dependence across adjacent steps can make them optimistic.
- Brier scores are 0.177888 and 0.098540 respectively. A constant score equal to
  training prevalence has comparison Brier score 0.005073. Both models rank better
  than constant scores but are poorly calibrated under balanced-class fitting.
  Present outputs as risk scores, not fraud probabilities.
- Almost all comparison transactions involve an account absent from the frozen
  training graph. The both-accounts-seen subgroup has only 261 rows and one positive.
  Its metrics cannot support a reliable subgroup performance claim.
- These results concern individual synthetic PaySim transactions. They do not establish
  ring detection, real customer identity continuity, financial loss prevented, or live
  deployment performance.

## Verification and saved files

All 27 automated tests pass. All 20 Phase 2 artifact checks pass, including saved-model
reload predictions, reproduced thresholds, confusion counts, AP, budget policies,
random-policy references, hashes and exclusion of final-test events.

Feature manifest and contract:
`artifacts/phase2/features/historical-v1/`.

Historical feature Parquet:
`data/processed/phase2/historical-v1/features.parquet`.

Models, LightGBM text export, validation predictions, metrics, coefficients/gains,
hash manifest and verification:
`artifacts/phase2/models/baselines-v1/`.

Reproduction commands and feature definitions are in [PHASE2_METHOD.md](PHASE2_METHOD.md).
The next phase is the separate synthetic coordinated-activity generator.
