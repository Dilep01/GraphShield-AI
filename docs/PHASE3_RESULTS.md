# Phase 3 completion record

Completed on 2026-10-14. Generated run: `coordinated-v1`.

All new source, tests, configuration, staging files, caches and generated data are
inside `C:\Users\dilip\Project\graphshield_aI`. This phase requires no additional
dependencies and does not read or alter PaySim data or the trained Phase 2 models.

## Generated benchmark

| Partition | Networks | Accounts | Transactions | Fraud groups | Legitimate groups |
|---|---:|---:|---:|---:|---:|
| Training | 3 | 4,500 | 78,363 | 36 | 54 |
| Validation | 2 | 3,000 | 52,253 | 24 | 36 |
| Known-family test | 2 | 3,000 | 52,424 | 24 | 36 |
| Unseen-family test | 2 | 3,000 | 52,248 | 24 | 36 |
| Total | 9 | 13,500 | 235,288 | 108 | 162 |

There are 4,063 planted fraudulent transaction events. A ring member's ordinary
background transactions retain negative event labels; account membership and event
fraud are different targets.

All 13,500 generated accounts participate in more than one time step. The benchmark
contains 16,653 generated devices and 163,408 observed device sessions. Of these
observations, 130,362 arrive after their event step. Snapshot access checks both event
and arrival time, so late observations cannot enter earlier graphs.

Shared-device relationships occur within both positive and legitimate groups: 71
fraud-group/device pairs and 98 benign-group/device pairs have observations from more
than one member. These are group/device pairs, not counts of distinct groups.
There are 55,440 ordinary background events crossing planted group boundaries.

## What the labels mean

Training and validation contain planted circular transfers, collection into a central
account and distribution from a central account. Legitimate groups use the same
templates and overlapping timing/amount distributions. Shared devices alone do not
determine a label.

The relay-chain planted family is reserved for the unseen-family test. The known-family
test contains familiar template families with new seeds and groups. Background events
can incidentally create similar motifs; the holdout applies to deliberately planted
scenario families, not the absence of every matching path in a graph.

Each network is an independent population with a local 720-step timeline. These splits
are not a chronological train/test division of one network. Histories are persistent
within a network, while account/device identities are disjoint across networks.

This dataset measures recovery of designed simulated activity. It does not provide
evidence of actual collusion, customer identity, real financial loss, or production
fraud-detection quality. It does not model balances or enforce financial solvency.
Some fraudulent and legitimate examples can be observationally indistinguishable.

## Separation and verification

Observed records and ground truth live in separate directories with exact schemas.
Observed tables exclude labels, group IDs, scenario names, seeds and split columns.
Ground truth includes positive and legitimate memberships, event labels, active periods
and simulated discovery times. Model pipelines must not read truth into graph features.

All 42 automated tests pass. They cover original phases plus deterministic generation,
seed independence, motif correctness, corrupted-truth rejection, path containment,
observation-arrival cutoffs, and repeated use of existing run names. A small benchmark
generated twice produced byte-identical Parquet files.

All 30 full-benchmark verification checks pass, including 27 structural checks, report
reproduction and generator/auditor source hashes. All 72 Parquet files match the saved
manifest. The original full generation and structural audit took 31.67 seconds on this
machine; this is one development observation, not a throughput guarantee.

No model was trained or scored on this benchmark during Phase 3. Test labels were
audited only to verify construction. Future rule/model tuning must use training and
validation; both test splits remain reserved for later model evaluation.

## Saved artifacts

- `data/synthetic/coordinated-v1/observed/`: observable accounts, devices, transactions
  and device observations, partitioned by split and network.
- `data/synthetic/coordinated-v1/ground_truth/`: scenario and membership truth, with
  separately stored event labels.
- `artifacts/phase3/coordinated-v1/`: manifest, audit report and verification results.
- `config/phase3.json`: the frozen default generator settings and independent seeds.

See [PHASE3_METHOD.md](PHASE3_METHOD.md) for reproduction commands and schema details.
Next is Phase 4: historical graphs, candidate-group rules and ring evaluation.
