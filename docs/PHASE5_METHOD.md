# Phase 5: learned ranking on historical candidate groups

This experiment builds on the frozen `rules-v1` candidate set. It compares five learned
rankers and two rule scores on the separate, fully synthetic coordinated-activity
benchmark. It does not retrain the original PaySim transaction models. Their Phase 2
results remain separate; the sparse PaySim identity history does not justify constructing
artificial relationships for a graph-model comparison.

The settings in `config/phase5.json` are fixed before the full run. Models use the three
training worlds only. The first validation world in the source manifest tunes alert
thresholds and selects a model; the second compares the frozen choices. Both validation
worlds were already inspected during Phase 4, so this is a development comparison, not
an untouched test. `test_known`, `test_unseen` and the PaySim final test remain reserved.
The implementation requires exactly two validation worlds and rejects test-world inputs.

## Shared inputs and supervision

Every method uses the same Phase 4 candidates, membership, evidence, 168-step history,
strict `step < cutoff` and `available_step < cutoff` rules. Model scores cannot generate
new candidates, so candidate coverage bounds every model's recall. No labels, scenario
families, world seeds, group IDs, account IDs or absolute cutoff values enter a classifier.
IDs identify graph nodes, preserve evidence and break ties only.

Each historical graph has account nodes with eight observed features: log1p inbound and
outbound transaction counts, amounts, distinct counterparties, observed-device count and
active transaction steps. Build undirected, unweighted edges from historical transfers
and shared-device co-use. Device observations must pass the same event/arrival filters.
Devices observed for more than 64 accounts are omitted from the projection and counted.
Direction and amounts enter the node/candidate features; message passing does not consume
edge amounts or timestamps directly. This is an account graph with device-derived edges,
not a typed account-device heterogeneous GNN.

Labels describe fully adjudicated archival training worlds: a candidate is positive if
it passes Phase 4's member precision, member recall and planted-evidence matching conditions
for at least one fraudulent group. This supervision is retrospective and can be discovered
after a candidate's historical cutoff. Training therefore represents fitting on a completed
archive and transferring to independent worlds; it does not simulate online label discovery
or retraining before a group is investigated. No training label is added to node features.

To reduce repeated-group dominance, training observations whose members overlap a planted
group by at least 50% share a cluster. Other observations share a cluster across repeated
exact member sets. Each observation initially receives inverse cluster-size weight;
weights are then normalized to equal positive/negative total mass. This is not perfectly
equal weight per group after class balancing when a cluster contains both labels. Validation
prevalence is unchanged. Candidate AP counts all repeated snapshot observations and is a
secondary ranking diagnostic; unique-ring metrics are the main comparison.

## Models and controls

| Method | Classifier input / computation |
|---|---|
| `activity_lgb` | Six candidate aggregates: member count, internal/boundary event counts, internal amount, amount CV, active steps |
| `graph_lgb` | All 16 explicit Phase 4 candidate features |
| `node2vec_lgb` | Those 16 features plus four within-candidate embedding cosine summaries and missing-member fraction |
| `mlp` | Candidate mean/max pooling of node representations plus 16 case features; both neighbor inputs forced to zero |
| `graphsage` | Two learned full-neighbor mean aggregation layers, candidate mean/max pooling and the same case features |
| `graph_only`, `graph_device` | Existing Phase 4 rule scores, with the same Phase 5 selection policies as learned methods |

The activity model still relies on graph-generated candidates. It is a non-topological
candidate-feature control, not an independent transaction classifier or candidate generator.
The MLP/SAGE pair has identical parameter dimensions, initialization seed, pooling, readout
and training schedule. The MLP's neighbor-channel parameters receive no data signal. The
other methods receive different representations of the same permitted historical records;
this is not a claim that their information compression or model capacity is identical.

LightGBM uses 150 trees, seven leaves, learning rate 0.04, minimum child size 15 and L2
regularization 1.0. There is no hyperparameter search or validation-based early stopping.

The Node2Vec implementation uses second-order biased walks (`p=1`, `q=2`), two walks per
active node, length 12, context radius two, 16 dimensions, three negatives and two skip-gram
training epochs. Negative sampling uses context frequency raised to 0.75. Isolated nodes
receive zero vectors. It fits a separate unsupervised embedding at each snapshot, including
validation snapshots, using only edges visible at that cutoff. It never trains on future
edges or shares validation labels with an embedding objective.

Independent embeddings do not have aligned coordinates. Consequently the downstream
classifier receives only pairwise cosine mean, standard deviation, minimum and maximum,
plus missing-member fraction. These summaries are invariant to an orthogonal change of
coordinates; independently trained embeddings can still differ because of optimization
noise and limited walk/epoch budgets. Missing and isolated vectors are excluded from
pair statistics; empty pair statistics are zero and missing fraction remains explicit.
The benchmark candidates all have local observed history; unseen global IDs still require
snapshot-local embedding work. This is not an inductive Node2Vec lookup service.

GraphSAGE concatenates self and mean-neighbor representations, then applies a learned
linear map, ReLU and L2 normalization at each of two layers (hidden size 32). Candidate
mean/max pooling is concatenated with standardized case features and passed to a small
readout network. All neighbors are used because the generated snapshots fit in memory;
this implementation does not implement sampled minibatch neighborhoods. Model weights
are shared across training worlds and applied unchanged to new validation identities.
The MLP uses the same architecture with neighbor means fixed to zero.

Both neural variants use 30 fixed epochs, Adam learning rate 0.003, weight decay 0.001,
and weighted binary cross-entropy. Training graph order is seeded and shuffled each epoch.
Node/case standardization is fitted on training observations only, including repeated
training snapshots. All runs use deterministic CPU operations and two compute threads.
Class-balanced outputs are uncalibrated investigation scores, not validated probabilities.

## Policies and evaluation

Two policies are reported for every method:

- **Budget only:** no score floor, at most three alerts per world/snapshot and the existing
  whole-world member-Jaccard suppression at 0.8.
- **Tuned threshold:** same budget/suppression, plus a threshold selected on the tuning
  world. The grid contains 25 score quantiles, zero and a value above the maximum score
  so abstaining from all alerts is possible.

The threshold objective is `2 * uniquely_matched_groups / (alerts + planted_fraud_groups)`.
This is the harmonic mean of case precision and ring recall, called case/ring F1 here.
Ties prefer greater precision, then a higher threshold. Model selection uses tuning
case/ring F1, then precision, then the fixed model order in code. These choices are frozen
before reading comparison metrics. A high threshold can reduce both false alerts and
recovery; all results must report both. Budget-only policies share a maximum capacity,
but different early alerts can change later suppression and total alert counts.

Group matching, legitimate lookalikes, duplicate-positive penalties and delay definitions
are unchanged from Phase 4. Report pooled training diagnostics, tuning results, comparison
results and individual worlds. Candidate AP, member quality and runtime supplement ring
precision/recall. One comparison network has only 12 fraudulent groups; no confidence
interval or real-world accuracy claim is justified from this experiment. Choosing a model
after looking at comparison results would require a new, explicitly labeled experiment.

## Outputs, checks and reproduction

Install with `scripts/bootstrap.ps1`; the lock includes the official CPU PyTorch index.
The tested environment is Windows/Python 3.11.9 with CPU PyTorch. GPU capability is not
required or established by this run. Set caches inside the project:

```powershell
$env:TEMP = Join-Path $PWD '.cache/tmp'
$env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $PWD '.cache/torch'
.\.venv\Scripts\python.exe -B -m graphshield.cli train-rings `
  --rule-run rules-v1 --run-name learned-reproduction
.\.venv\Scripts\python.exe -B .\scripts\verify_phase5.py --run-name learned-reproduction
.\scripts\test.ps1
```

All output is under `artifacts/phase5/<run>/`. Immutable source runs are not overwritten.
Each NPZ stores observed nodes, features, edges, candidate membership indices and local
embeddings. Separate files contain retrospective targets, predictions, decisions, alerts,
threshold searches, metrics, feature contracts, model checkpoints, loss histories and
timings. LightGBM models have joblib and text exports; neural checkpoints contain weights
only. Source/input/output hashes and package versions are recorded.

The verifier checks hashes before loading local model artifacts, reconstructs every
historical graph, checks embedding summaries and training-only scalers, reloads all five
models, reproduces scores, threshold searches, selection, cases and metrics. It does not
retrain every Node2Vec embedding; embeddings are hashed and a small deterministic fitting
test covers the implementation. Tests also exercise future/late data, device projection
limits, biased walks, rotation/permutation invariance, unseen-vector fallback, repeated
group weights, budget selection, absent test directories and validation-label independence
of fitted predictions. Files outside the project are not used for generated output.

Graph construction and unsupervised embedding runtime are reported separately from
classifier fitting/scoring. Neural scoring timings include passing saved graphs through
the model; LightGBM scoring timings begin with prepared features. These are different
cost boundaries, not directly comparable end-to-end serving latencies. DuckDB is capped
at 512 MB; Python arrays and PyTorch tensors use additional memory. The runner holds the
development graph set in memory and is not designed for a full PaySim-sized GNN.

Algorithm references: [Node2Vec paper](https://arxiv.org/abs/1607.00653),
[GraphSAGE paper](https://arxiv.org/abs/1706.02216), and
[official PyTorch installation guidance](https://pytorch.org/get-started/locally/).
