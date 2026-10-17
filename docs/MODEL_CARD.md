# GraphShield AI model card

Release 0.7.0 is a local investigation research prototype. It ranks candidate groups
for human review and retains observed evidence. Scores are priorities, not calibrated
probabilities of fraud, proof of collusion, or decisions about people.

## Data and intended use

The original PaySim synthetic transaction benchmark and a separately generated network
benchmark are independent. PaySim contains 6,362,620 source transactions with a frozen
time-step split. Original IDs are preserved. Its sparse recurring account activity
limits persistent-ring research; synthetic relationships were not added to repair it.

The generated benchmark supplies independent networks with persistent accounts, shared
devices, coordinated groups, legitimate lookalikes and delayed observations. Training,
tuning, comparison and final worlds have separate identities and seeds. Cycle, fan-in
and fan-out families appear in development; relay chains are withheld for the unseen
family test. Labels are retrospective simulator truth, not real investigations.

Appropriate use: demonstrations, reproducible comparisons and studying historical
graph candidate generation. The application should not be used to block transactions,
accuse account holders or make operational financial decisions.

## Models and selection

The transaction pipeline compares Logistic Regression and LightGBM on 20 historical
features. LightGBM was selected on the validation tuning period. The group pipeline
compares three LightGBM representations, GraphSAGE, a neural control without messages,
and two rule scores on shared graph-generated candidates. Node2Vec plus LightGBM was
selected on the tuning network. Both choices remain frozen after final evaluation.

GraphSAGE uses an account graph with device-derived edges, not a heterogeneous identity
graph. Node2Vec fits snapshot-local embeddings and classifies coordinate-invariant
within-group cosine summaries. It is not an inductive embedding lookup service.
All snapshots exclude current/future events and not-yet-available observations.

## Final evidence and failures

- PaySim LightGBM: final AP 0.5378 on 130,116 rows with 1,664 positives. The frozen
  threshold produced 44.14% precision and 69.47% recall.
- Selected group model on known families: 50 alerts, 19/24 groups recovered,
  38.00% case precision, 79.17% recall and 26/36 legitimate groups alerted.
- Tuned transfer rules had higher known-family case precision and recall than the
  selected learned ranker. The experiment does not establish learned-graph superiority.
- Withheld relay chains: 0/24 matching candidate coverage and 0/24 recovered groups.
  All tuned models abstained. This is a structural generalization failure.

Only two final worlds exist per ring split. Results do not establish performance on
other simulators or real institutions. Class-balanced fitting does not yield validated
fraud probabilities. The dashboard's feature contributions explain model associations,
not causal evidence. Review notes remain separate from training and benchmark truth.

## Operation and changes

The supported deployment is one local worker with SQLite and trusted local model
artifacts. There is no authenticated reviewer identity, RBAC or public deployment.
Request monitoring measures operational behavior, not fraud accuracy or drift.

The final test is now consumed. Any revised candidate generator, model, threshold or
selection rule requires explicitly labeled follow-up development and new untouched
evaluation data. Preserve this release and its original results for comparison.

Detailed evidence: [Phase 7 results](PHASE7_RESULTS.md) and
[evaluation protocol](PHASE7_METHOD.md).
