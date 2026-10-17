# Phase 6 completion record

Completed 15 October 2026. Scope: local API, batch replay and investigation dashboard.
All generated files, databases, dependencies, caches and browser evidence are inside
`C:\Users\dilip\Project\graphshield_aI`.

## Delivered

- FastAPI endpoints for observed-event ingestion, historical scoring, replay jobs,
  case evidence and reviewer notes; machine-readable OpenAPI schema.
- SQLite receipts and atomic checkpoints for safe retries and interrupted replay.
  Conflicting event IDs or stale reviewer versions are rejected.
- Versioned `serving-v1` model bundle with the frozen tuning-selected Node2Vec model.
  Seven rankers and both thresholded and budget-only policies are available.
- Local dashboard with account/device graph, historical transfer table, activity
  timeline, feature contributions for LightGBM, model provenance and review history.
- A Windows launcher and a default database with 17 cases, 116 distinct case accounts
  and 27 completed snapshots from the synthetic development comparison network.
  The default database has no QA review notes.

## Verification evidence

The full suite passed: **88 tests**, including 12 new Phase 6 tests, in 83.08 seconds.
The saved report is `artifacts/phase6/tests.xml`. The installed environment passed
`pip check`; exact dependencies are recorded in `requirements-lock.txt`.
There is one upstream deprecation warning about Starlette's HTTPX TestClient adapter.
API tests required Windows loopback access because the restricted execution sandbox
stalled while creating the event loop's internal socket pair.

`scripts/verify_phase6.py` completed **14 replays / 378 snapshots** in 133.84 seconds.
For every model and policy, the served alert candidate IDs exactly matched Phase 5.
The maximum saved-case score difference was zero. Temporal evidence checks, LightGBM
contribution-sum checks and unchanged completed-job retries also passed.

| Model | Tuned-threshold cases | Budget-only cases |
|---|---:|---:|
| Activity LightGBM | 12 | 74 |
| Graph features + LightGBM | 19 | 75 |
| Node2Vec + LightGBM | 17 | 75 |
| Neural control without messages | 27 | 75 |
| GraphSAGE | 17 | 76 |
| Repeated-transfer rules | 27 | 75 |
| Transfer + device rules | 13 | 75 |

Evidence: `artifacts/phase6/verification/serving-v1/report.json` and its separate
`service.sqlite`. These counts verify serving parity on the comparison network;
they are not new fraud-detection metrics or a reason to change the selected model.

Browser verification used `artifacts/phase6/qa.sqlite`, a copy of the clean demo.
The local page loaded 17 Node2Vec cases. Checks covered the graph and timeline,
historical transfers/device evidence, model contributions, saving one review note,
the corresponding status/history update, filtering to that reviewed case, an empty
alternate-model queue, and a full GraphSAGE replay producing 17 cases. Browser logs
contained no warnings or errors during these checks. Desktop and 390-pixel viewport
layouts were inspected; the phone-width document had no horizontal overflow.

Screenshots: `artifacts/phase6/dashboard.jpg`, `review-qa.jpg` and
`dashboard-mobile.jpg`. The screenshots show the QA database, including its one
test review. The temporary verification server was stopped after the checks.

## Limits and next phase

The API operates on generated development events and ranks candidate groups. It does
not expose the separate PaySim transaction classifier, establish calibrated fraud
probabilities or support production multi-user operation. Reviewer decisions remain
separate from benchmark truth and training. Node2Vec may need snapshot-local fitting
for new demo graphs; cached replay time is not a general live-scoring latency claim.

Both synthetic final-test splits and the PaySim final test remain unscored.
Phase 7 remains: monitoring, packaging, clean-environment reproduction, final evaluation
under the frozen protocol and final documentation.

Launch instructions and API contracts are in [PHASE6_METHOD.md](PHASE6_METHOD.md).
