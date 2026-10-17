# Phase 7 completion record

Release completion: 17 October 2026. The planned seven-phase local research prototype
is implemented. All results below use synthetic data.

## Final benchmark results

The frozen final run is `artifacts/phase7/final-v1`. It scored 130,116 held-out PaySim
transactions and 108 snapshots across four independent generated final networks.
There were 1,532 group-candidate observations. Predictions were sealed before reading
the corresponding held-out labels. Prediction generation took 237.52 seconds, including
36.49 seconds for PaySim feature construction and scoring, on this Windows CPU setup.

PaySim final test contains 1,664 positives. The model selections and thresholds are
unchanged from development.

| Transaction model | Average precision | Threshold precision | Threshold recall | Threshold alerts |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3073 | 46.92% | 29.33% | 1,040 |
| LightGBM, selected | 0.5378 | 44.14% | 69.47% | 2,619 |

At the separate 20-alerts-per-step budget, LightGBM produced 2,116 alerts with 1,394 true
positives and 722 false positives: 65.88% precision and 83.77% recall. The analytical
random-selection reference at the same budget has expected recall 64.00%; 916 positives
occur in 90 steps where every transaction fits within the budget.

The paired whole-step bootstrap AP interval is [0.4824, 0.6019] for LightGBM and
[0.2494, 0.3962] for Logistic Regression. The paired AP difference interval is
[0.2031, 0.2540]. These intervals are conditional on the fitted models and this dataset,
and omit dependence across adjacent steps. They do not establish real-world accuracy
or calibrated fraud probabilities.

Known-family final networks contain 24 planted fraud groups and 36 legitimate groups:

| Group ranker, frozen tuned threshold | Alerts | Matched groups | Case precision | Ring recall |
|---|---:|---:|---:|---:|
| Activity LightGBM | 30 | 13/24 | 43.33% | 54.17% |
| Graph-feature LightGBM | 48 | 17/24 | 35.42% | 70.83% |
| Node2Vec + LightGBM, selected | 50 | 19/24 | 38.00% | 79.17% |
| Neural control without messages | 47 | 16/24 | 34.04% | 66.67% |
| GraphSAGE | 33 | 14/24 | 42.42% | 58.33% |
| Repeated-transfer rules | 56 | 22/24 | 39.29% | 91.67% |
| Transfer + device rules | 35 | 12/24 | 34.29% | 50.00% |

The selected Node2Vec model did not outperform the tuned repeated-transfer rules on
case precision or recall. It alerted on 26/36 legitimate groups. GraphSAGE is not
promoted after viewing test results. Under budget-only selection, all seven models
recovered 24/24 known-family groups across 152 alerts, with 15.79% case precision and
all 36 legitimate groups alerted.

**The withheld relay-chain family was not detected.** Candidate coverage was 0/24
fraud groups, versus 24/24 for known families. All tuned policies abstained; precision
is undefined with zero alerts and recall is zero. Budget-only policies produced 151
alerts each and recovered no withheld-family group. There were background candidates,
but none satisfied the frozen group-matching conditions for the relay chains.
The candidate generator is therefore the limiting stage; changing only the ranker
cannot recover groups that never appear as matching candidates.

Only two worlds exist per ring split. Selected-model known-family bootstrap ranges
were 34.62–41.67% case precision and 75.00–83.33% recall. They are descriptive resampling
ranges, not reliable population confidence intervals. The full per-world, per-family,
delay, member-quality and calibration results are in `results.json`.

## Release and verification

- Added request/error/latency counters, rotating local request logs, replay recovery
  checks and a dashboard health panel. Monitoring does not infer fraud accuracy.
- Built version 0.7.0 as an installable wheel, with pinned offline Windows dependencies
  and a separate synthetic development demo archive.
- The fresh installed environment reproduced 27 snapshots and 17 default cases with
  exact candidate and score agreement. API and packaged assets passed smoke checks.
- **96 tests passed** against the installed wheel in 47.19 seconds. There is one upstream
  Starlette HTTPX TestClient deprecation warning. Test report: `artifacts/phase7/clean-tests.xml`.
- **Seven final artifact verification checks passed**, covering all PaySim scores,
  108 reconstructed graph snapshots, seven rankers, alert policies, metrics and coverage.
  Logistic Regression's largest repeated-score difference was 2.98e-8 with unchanged
  threshold decisions; LightGBM's difference was zero. No predictions were rewritten.

The initial verifier incorrectly demanded float64-level agreement from float32 Logistic
Regression output. Its tolerance was corrected to 1e-6 after checking the numerical
error and unchanged threshold decisions. The frozen models, protocol, predictions and
final metrics were not changed. This verifier correction is recorded for transparency.

Browser checks cover the installed health panel and its operational counts. A narrow
615-pixel window exposed overflow in the model-selection controls; the corrected
layout stacks those controls below 760 pixels. Final checks at 615 and 375 pixels
showed no horizontal document overflow; the controls and health panel were visually
inspected, with no browser console warnings/errors. The QA session recorded one
unmatched 404 request and zero server errors; workflow health reported no failures
or inactive jobs. The final browser record and screenshots
are saved under `artifacts/phase7/`.

The earlier full-suite and replay reproduction checks precede this CSS-only correction;
the correction is checked in the browser and included in the final rebuilt wheel.
No backend or model code changed after those tests.

## Completion boundary

This is a completed local prototype with documented evaluation failures. Public hosting,
multi-user authentication, production data controls, streaming, automatic retraining,
validated drift detection and demonstrated real-world performance are not implemented.
Future work should address relay-chain candidate coverage using a new development and
test protocol. The current final labels are consumed and cannot serve as untouched
evidence for that next model revision.

Evidence under `artifacts/phase7/`: `final-v1/protocol.json`,
`final-v1/prediction_seal.json`, `final-v1/results.json`, `final-v1/verification.json`,
`reproduction/report.json`, `browser-qa.json`, `clean-tests.xml`, and `manifest.json`.
The source archive, demo archive, wheel and 42 dependency wheels are checksummed in
`release/manifest.json`. See RUNBOOK.md for the verification command.
