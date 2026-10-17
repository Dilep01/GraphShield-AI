# Phase 4 completion record

Completed on 2026-10-15 in `C:\Users\dilip\Project\graphshield_aI`.
Run: `rules-v1`, using synthetic source `coordinated-v1`.

## Delivered

- Rolling historical transfer graphs and account-device evidence, with both event-time
  and arrival-time filters. Current and future time buckets are excluded.
- Candidate rules for repeated strongly connected groups, fan-in and fan-out patterns.
  Sixteen explicit graph/activity/device features are saved per candidate.
- Two fixed policies: graph-only priority and graph-plus-device priority. Both use
  the same candidates, a minimum score and a three-alert limit per world per snapshot.
- Chronological duplicate suppression, immutable alert membership/evidence and a
  decision log for every candidate under each policy.
- Retrospective one-to-one group matching, per-world and pooled metrics, member quality,
  detection delay, benign-group alert rate and duplicate-positive counts.
- Command-line replay, source/output fingerprints, regression tests and full artifact replay.

The run processed five development networks, with 135 snapshots and 2,802 candidate
observations. Repeated observations of a member set at different cutoffs are included
in that candidate count. It took 12.28 seconds on this machine; this is a single local
run, not a latency benchmark or a memory measurement. Output artifacts occupy about
7.4 MB. No new packages were required.

## Measured results

These are fully synthetic development results. Each positive group can earn at most
one successful case. Matching requires >=80% member precision and >=80% member recall,
plus a planted transaction present in the alert's historical evidence.

| Split | Policy | Alerts | Matched groups | Case precision | Ring recall | Benign groups alerted |
|---|---|---:|---:|---:|---:|---:|
| Training | Graph only | 223 | 35 / 36 | 15.70% | 97.22% | 53 / 54 |
| Training | Graph + device | 223 | 35 / 36 | 15.70% | 97.22% | 54 / 54 |
| Validation | Graph only | 150 | 24 / 24 | 16.00% | 100.00% | 35 / 36 |
| Validation | Graph + device | 150 | 24 / 24 | 16.00% | 100.00% | 35 / 36 |

Both validation policies therefore produced 126 unmatched alerts. Graph-only had one
additional positive-group alert beyond the unique matches; graph-plus-device had three.
These duplicates count against case precision, even when their overlap was too small
for the member-Jaccard suppression rule to remove them.

For matched validation cases, mean member precision/recall were 96.90% / 99.54% for
graph-only and 97.96% / 99.02% for graph-plus-device. Median assigned detection delay
was 101 and 105.5 relative simulation steps, respectively. Delay is conditional on
matched groups and the maximum-cardinality assignment; it is not a streaming latency.
Each policy matched all eight validation groups in each known family: cycle, fan-in,
and fan-out. There are only two validation worlds, so these results do not establish
performance across a broad population of networks.

Before scoring, capacity and duplicate suppression, candidates covered all 36 training
and 24 validation fraud groups. This is an oracle candidate-coverage ceiling. It does
not mean all those groups can be identified as fraud from observed information. The
largest historical weak component contained all 1,500 accounts in a world, confirming
that simply treating a connected component as a ring would be unhelpful here.

## Interpretation and next-phase decision

The rules recover planted structures but discriminate poorly between fraud, legitimate
lookalikes and background activity. Nearly every legitimate validation group generated
an alert. Shared-device scoring did not improve the headline recall or case precision
under this configuration. The two policies retain different individual alerts, delays
and duplicate counts; equal headline results do not mean their predictions are identical.

Keep these results as the fixed baseline. Do not claim 100% fraud accuracy, calibrated
probabilities or proof of collusion. No constants were changed in response to this run,
and no policy was selected as a validation winner. The minimum score excluded no
candidates in this run; capacity and duplicate suppression controlled alert volume.

Phase 5 should first establish a supervised graph-feature ranking baseline, then compare
Node2Vec and GraphSAGE under the same world/time boundaries and review budget. The main
question is whether learned ranking reduces false alerts at a useful recovery level.
Candidate coverage, unseen-account handling, repeated-case weighting, model size and
runtime must remain visible. Independent-world embeddings need a valid alignment or
inductive fallback; fitting embeddings on future or test edges would invalidate the
comparison. Do not assume a GNN will improve these results.

## Verification and boundaries

The full automated suite passed: **61 tests** in 44.11 seconds, including 19 Phase 4
checks. New tests cover current/future/late-record exclusion, event-order invariance,
repeated stars, oversized connected groups, capacity, duplicate suppression, one-to-one
matching, member/evidence eligibility, empty outputs and invalid configurations.

Integration tests run with both test-world directories absent. They verify that candidate
and alert files exist before evaluation truth is loaded, and that changing truth does
not change any candidates, graph features or alerts. Saved-artifact tampering is rejected.
The completed run also passed **16 artifact checks**, including full snapshot, alert,
decision and metric reproduction, evidence timestamps, source hashes and output hashes.

Both synthetic test splits and the original PaySim final test remain unevaluated.
Phase 4 opens only the selected synthetic training/validation data files. It performs
no API serving, live ingestion, trained ring classification, GNN inference or dashboard
work. All new code, artifacts, caches and test workspaces are inside the project folder.

Evidence:

- `artifacts/phase4/rules-v1/manifest.json`
- `artifacts/phase4/rules-v1/report.md` and `evaluation.json`
- `artifacts/phase4/rules-v1/verification.json`
- `artifacts/phase4/rules-v1/candidates.jsonl`, `alerts.jsonl`, `decisions.jsonl`, `snapshots.jsonl`
- `docs/PHASE4_METHOD.md`
