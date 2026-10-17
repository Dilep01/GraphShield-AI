# Phase 5 completion record

Completed on 2026-10-15 in `C:\Users\dilip\Project\graphshield_aI`.
Run: `learned-v1`, using frozen candidate run `rules-v1` and synthetic source `coordinated-v1`.

## Delivered

Five trained ranking variants: activity-only LightGBM, explicit-graph-feature LightGBM,
Node2Vec plus LightGBM, a neural model without neighbor messages, and a two-layer mean
GraphSAGE case ranker. The two Phase 4 rule scores are also evaluated under the same
Phase 5 policies. Model files, feature contracts, embeddings, predictions, thresholds,
case evidence, decision logs and measured results are saved.

The models score the same 2,802 historical candidate observations. Local embeddings and
neural inputs were constructed for the 127 snapshots containing candidates; the eight
empty snapshots require no case scores. Each snapshot uses only its permitted historical
events and arrivals. Node2Vec is fitted separately per snapshot; only within-case cosine
summaries enter its downstream model, avoiding raw unaligned embedding coordinates.
GraphSAGE uses shared learned weights on new account identities. It is an account graph
with shared-device projection, not a heterogeneous account/device model.

## Development comparison

Three independent networks train the models. One validation network selects thresholds
and the model; the other measures the frozen policies. Both networks were examined in
Phase 4, so this is a development comparison, not a previously unseen final test.

The tuning network selected **Node2Vec plus LightGBM**, with case/ring F1 0.6286:
11 of 12 groups matched across 23 alerts. Its fixed threshold is 0.8901825803662127.
The comparison network produced the following results with each model's tuning-selected
threshold and the same maximum of three alerts per snapshot:

| Model | Alerts | Matched fraud groups | Case precision | Ring recall | Case/ring F1 | Benign groups alerted |
|---|---:|---:|---:|---:|---:|---:|
| Activity LightGBM | 12 | 7 / 12 | 58.33% | 58.33% | 0.5833 | 4 / 18 |
| Explicit graph LightGBM | 19 | 9 / 12 | 47.37% | 75.00% | 0.5806 | 8 / 18 |
| Node2Vec + LightGBM | 17 | 7 / 12 | 41.18% | 58.33% | 0.4828 | 8 / 18 |
| Neural model without messages | 27 | 8 / 12 | 29.63% | 66.67% | 0.4103 | 15 / 18 |
| GraphSAGE | 17 | 9 / 12 | 52.94% | 75.00% | 0.6207 | 5 / 18 |
| Graph-only rule | 27 | 10 / 12 | 37.04% | 83.33% | 0.5128 | 16 / 18 |
| Graph + device rule | 13 | 6 / 12 | 46.15% | 50.00% | 0.4800 | 6 / 18 |

Case precision counts a planted fraudulent group at most once. Unmatched alerts include
legitimate activity, incomplete/impure candidates and extra cases for an already matched
group. Case/ring F1 is `2 * matched_groups / (alerts + planted_fraud_groups)`.

GraphSAGE had the highest comparison F1, but it was not the tuning winner. Its threshold
was 0.9480562210083008, selected before comparison. Do not change the recorded selection
to GraphSAGE after seeing these results. The activity model had the highest precision,
with lower recall. Even a tuned rule score recovered 10 groups, but required 27 alerts.
There is no single method that dominates every metric in this run.

With the score floor removed, all seven methods recovered 12/12 groups but emitted
74-76 alerts, leaving case precision around 16%. Thresholds are therefore doing much
of the work in reducing review volume, and the reduction comes with missed groups.
The fair threshold-tuned rule comparisons above matter; comparing only against Phase 4's
untuned 16% precision would overstate the learned models' advantage.

The selected Node2Vec policy produced 10 unmatched alerts and missed five groups on the
comparison world. GraphSAGE produced eight unmatched alerts and missed three groups.
Their median assigned detection delays were 113 and 121 relative simulation steps,
respectively, conditional on matched groups. These are not real-time serving latencies.

## What the experiment supports

GraphSAGE's comparison candidate AP was 0.5702, versus 0.4790 for the no-message neural
control, 0.4775 for explicit graph LightGBM and 0.4462 for Node2Vec plus LightGBM. Comparison
candidate prevalence was 25.48%. AP counts repeated snapshot observations and must not
be described as precision over unique rings or interchangeable with the alert-policy result.

Node2Vec did not give a consistent improvement over the simpler explicit-feature model:
it won the tuning selection but had lower AP and case/ring F1 on comparison. GraphSAGE
performed better than the neural control on this one network, but there are only 12
positive comparison groups and one model seed. No reliable population-level or production
claim follows. GraphSAGE's training AP was 0.9514 versus 0.4658 on tuning and 0.5702 on
comparison, showing a substantial training/development gap.

Amount CV had the largest LightGBM gain importance in both graph and Node2Vec models.
The generated fraud/benign groups have overlapping but different amount-variation
distributions, so the models may be exploiting that simulation assumption. Feature gain
is not causal evidence, and these results do not establish detection of real collusion.

All scores remain uncalibrated investigation priorities. Training uses retrospective
labels from completed independent archives, not a simulation of investigators discovering
labels online. The full protocol and limitations are in PHASE5_METHOD.md.

## Runtime and saved outputs

The single local CPU run took **455.68 seconds** (about 7.6 minutes). It used two compute
threads with PyTorch 2.14.1+cpu on Windows/Python 3.11.9. Approximately 403.90 seconds
were spent building historical graphs, fitting local Node2Vec embeddings and saving
those artifacts. No GPU was used; GPU hardware capability was not established.

| Model | Classifier fit | Score all 2,802 candidates |
|---|---:|---:|
| Activity LightGBM | 0.091 s | 0.015 s |
| Explicit graph LightGBM | 0.083 s | 0.009 s |
| Node2Vec + LightGBM | 0.092 s | 0.008 s |
| Neural model without messages | 16.55 s | 0.214 s |
| GraphSAGE | 26.45 s | 0.543 s |

LightGBM scoring starts from prepared features; Node2Vec's substantial embedding cost
is excluded from its classifier timing. Neural scoring includes the graph forward pass
on saved inputs. These timings are not an end-to-end service benchmark. The neural
models each have 5,249 parameters. Saved neural checkpoints are approximately 24 KB;
LightGBM joblib files are approximately 138 KB each. Full run artifacts occupy about
39 MB, including 127 snapshot graph/embedding files and all model/policy case logs.
Peak process memory was not measured.

## Verification and next step

- **76 automated tests passed** in 77.52 seconds, including 15 Phase 5 tests.
- **20 saved-artifact checks passed**, including every historical graph reconstruction,
  model reload/prediction reproduction, training-only scalers, threshold/model selection,
  case decisions, metrics, hashes, capacity limits and one-to-one group matching.
- Tests explicitly change validation labels and verify that fitted predictions do not
  change; changing comparison outcomes cannot change thresholds or the selected model.
- Tests cover time/arrival exclusion, biased walks, deterministic embedding fitting,
  rotation-invariant summaries, unknown-vector handling, node permutation invariance,
  mean aggregation, repeated-group weighting and artifact corruption.
- Integration fixtures contain no test-world data directories. The final-test datasets
  remain unevaluated. The original PaySim pipeline and trained models were not changed.
- Dependency checks passed. Downloads, the environment, caches, test workspaces and
  new artifacts are all contained inside the project folder.

Phase 6 can now add the API, batch replay and investigation dashboard around these
versioned artifacts. Preserve `node2vec_lgb` as the tuning-selected experimental policy
and make model/policy selection explicit in the investigation workflow. Keep GraphSAGE
as a comparison model. Do not imply that the selected model is production-approved or
silently promote a model using the reserved final tests. Batch replay is appropriate
for the current snapshot-local Node2Vec cost.

Evidence: `artifacts/phase5/learned-v1/manifest.json`, `evaluation.json`, `timings.json`,
`verification.json`, `report.md`, `feature_contract.json`, `models/`, and `graphs/`.
Reproduce with `scripts/train_phase5.ps1 -RunName learned-reproduction`.
