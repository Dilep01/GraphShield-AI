# Phase 3 methodology

This benchmark is entirely generated. It does not augment, relabel, or otherwise read
PaySim. Its ground truth describes the scenarios deliberately planted by this generator;
it does not establish real-world criminal intent from observable transaction patterns.

## Split design

The default configuration creates nine independent simulated networks, called worlds.
Each world has 1,500 persistent accounts, 720 relative time steps, 25,000 background
transactions, 12 fraudulent groups and 18 legitimate groups. Each planted group has
5-9 members and 4-8 repeated episodes. The first 96 steps are background warmup.

| Partition | Independent seeds | Planted families |
|---|---|---|
| train | 1101, 1102, 1103 | cycle, fan_in, fan_out |
| validation | 2201, 2202 | cycle, fan_in, fan_out |
| test_known | 3301, 3302 | cycle, fan_in, fan_out |
| test_unseen | 4401, 4402 | relay_chain |

Accounts, devices, groups and events are disjoint across worlds. The same relative step
numbers repeat across worlds; these are independent populations, not consecutive
calendar periods. Account histories remain available within a world, even when its
identities were absent from training. Future evaluation must retain that distinction.

Separate seeded random streams control population, ordinary transactions, planted
groups, device assignment, observation loss/delay, and discovery delay. IDs are opaque
hashes with entity-type prefixes and do not encode a fraud class or scenario family.
IDs, folder names, seeds and world/split metadata are never model predictors.

Known and held-out families refer to explicit planted templates. Background events may
incidentally form chains or other motifs. The holdout does not mean every mathematical
chain is absent from the training graph.

## Ordinary and planted behavior

Background accounts preferentially transact with five recurring counterparties, with
additional normal high-volume receivers and occasional other recipients. Background
activity connects planted members to outside accounts. Groups cannot be recovered
simply by treating every connected component as a fraud ring.

The three development templates are circular transfers, collection from multiple
accounts into a central account, and distribution from a central account. The unseen
test template passes funds along a relay chain. Each template is generated for both
fraudulent and legitimate groups. Examples of legitimate interpretations include
settlement loops, merchant collection, business disbursement and intermediary settlement.
Those names are descriptions of the simulation, not verified real transaction purposes.

Both classes draw base amounts from the same distribution. Fraud scenarios tend to have
less amount variation and shorter gaps, but the parameter ranges overlap. Device sharing
uses the same assignment mechanism for both classes. All transactions use type TRANSFER,
so transaction type cannot separate generated positive and negative events.

No deterministic label rule is exposed as an observed field. There is intentional
ambiguity: some benign and fraudulent sequences may be observationally indistinguishable.
Any later performance measures how well a model recognizes this designed distribution.

Members belong to one planted group within a world, but they also have ordinary events.
Only planted fraudulent events get a positive transaction label. An ordinary transaction
involving a ring member stays negative. Group membership and transaction-level targets
must not be silently interchanged.

This is an event-pattern simulator, not a balance-conserving ledger. There are no supplied
account balances, verified identities, actual customer records, real devices or currencies.
Amounts are positive two-decimal units, and time is represented by synthetic steps.

## Device observations and point-in-time access

Every account has one personal device and some have a second. Ordinary shared access and
planted-group sharing add further relationships. Approximately 70% of transaction-sender
sessions are sampled for observation before duplicate account/device/step sessions are
collapsed. The final observation count is therefore not exactly 70% of transaction count.

Observed sessions have an event step and an arrival step delayed by 0-4 steps. A graph
snapshot at t uses only observations whose event step < t **and** available step < t.
Repeated sightings are deduplicated to account-device links only at query time. Future
`last_seen` values are never stored as historical account features.

All generated account and device metadata is created/available at step 1. Device-use
edges become known only through observations. Observation arrival may extend beyond
the final event horizon; those edges remain unavailable to earlier snapshots.

Group discovery time is 12-48 steps after the final planted group event. Background
event labels are available 12-48 steps after the event. These are simulated availability
assumptions, not estimates of real fraud-investigation delays. Labels are private
evaluation records and are not inputs to graph construction.

## Files and separation

`data/synthetic/<run>/observed/<split>/<world>/` contains only:

- `accounts.parquet`: account_id, created_step, available_step.
- `devices.parquet`: device_id, device_type, created_step, available_step.
- `transactions.parquet`: transaction_id, sender_id, receiver_id, amount, step,
  available_step, type.
- `device_observations.parquet`: observation_id, account_id, device_id, step,
  available_step.

`data/synthetic/<run>/ground_truth/<split>/<world>/` contains:

- `groups.parquet`: family, fraud/benign label, active interval and discovery time.
- `group_memberships.parquet`: membership for both fraudulent and legitimate groups.
- `ring_memberships.parquet`: exactly the positive-group subset of those memberships.
- `transaction_labels.parquet`: transaction target, optional planted group association,
  and simulated label availability.

Ground truth is separated by directory and schema, not encrypted or protected by access
control. Feature-building code must explicitly restrict itself to observed files.
Generated test truth is audited now for construction correctness; it must remain outside
future training and tuning. No model has been evaluated on either synthetic test split.

## Checks and reproduction

The generation command fails if the structural audit fails. Checks include exact schemas,
non-null required values, unique IDs, foreign keys, label/event alignment, entity and seed
separation, group topology, repeated episodes, legitimate shared devices, observation
delays, recurrent activity, and connections between group members and outside accounts.

The audit verifies each planted motif independently of the generator's edge-construction
function. Tests also reject two disconnected cycles masquerading as one ring, verify
same-seed byte-identical Parquet output on a small benchmark, and detect corrupted truth.

Run names are immutable: existing output directories are not overwritten. The manifest
stores configuration, Python/DuckDB versions, source-code hashes, seeds and every Parquet
hash. All caches, staging files and outputs stay inside the project.

```powershell
.\scripts\test.ps1
.\.venv\Scripts\python.exe -B -m graphshield.cli generate-synthetic --run-name coordinated-reproduction
.\.venv\Scripts\python.exe -B scripts/verify_phase3.py --run-name coordinated-reproduction
```

Phase 4 will build historical graphs, candidate-generation rules and ring-matching
metrics. It must evaluate false alerts on legitimate groups as well as recovery of
planted fraud, with fixed definitions for matching and investigation capacity.
