# Phase 4: historical graph rules

This phase implements an observed-only investigation baseline on the separate generated
coordinated-activity benchmark. It does not change PaySim or its Phase 2 models. All
constants and both policies in `config/phase4.json` are specified before the first full
development run. There is no threshold search, supervised fitting or validation winner
selection in this phase. These rules know the generator's development motif families;
results measure performance on those scenarios, not discovery of unknown real crimes.

## Time and graph contract

Replay the three training worlds and two validation worlds independently. Snapshots
run every 24 steps from 97 through 721. Step 721 lets events through step 720 become
visible, but does not admit device observations arriving at or after 721.

At cutoff t, include events with `max(1, t-168) <= step < t` and `available_step < t`.
The same filter applies to transfers and device observations. A delayed observation
whose event falls outside the rolling window is excluded even after it arrives.
No ordering is inferred inside a time bucket. Entity existence comes from observed
events; account/device metadata, absolute IDs, seeds and labels are not predictors.
The runner reads full development event files for offline replay and filters at every
snapshot. It is not a streaming-memory implementation. DuckDB has a 512 MB limit;
Python holds development rows and candidate evidence in memory outside that limit.

Directed account edges aggregate transfer counts and amounts. A second graph retains
non-self edges with at least two events in the window. Device links are undirected
account-device observations deduplicated for feature counts, with source observation
IDs retained for evidence. Duplicate transfer IDs are excluded by the source benchmark's
validated contract, not collapsed here.

## Candidates and explicit features

Candidates are bounded strongly connected components of the repeated-transfer graph,
or the complete repeated inbound/outbound neighborhood of one account including itself.
Accept 5-12 members. Equal member sets within a snapshot are one candidate with multiple
reason codes. Oversized proposals are skipped and counted; there is no label-aware
trimming. Strong components may contain multiple cycles and are not proofs of collusion.
Weak components are measured for context but never automatically become cases.

Every candidate saves its exact member set, cutoff/window, reasons, internal transaction
IDs, shared-device IDs, observation IDs and 16 explicit features. Features include
directed density, repeated-event fraction, event support, internal/boundary activity,
amount variation, shared-device coverage, repeated degrees and active time steps.
Boundary activity uses all observed transfers touching a member, counted once per event.
Evidence IDs cover internal transfers and shared devices; boundary-derived features can
be reconstructed from the hashed source and saved cutoff.

The graph-only priority is:

`0.35 * repeated_event_fraction + 0.25 * event_support + 0.25 * amount_similarity + 0.15 * internal_event_share`

Event support is `min(1, internal_events / (2 * members))`; amount similarity is
`1 / (1 + population_amount_CV)`. The device policy uses the same candidates and features:

`(graph_only_score + 0.15 * shared_device_member_fraction) / 1.15`

This normalization changes the scale at the common threshold. This is a comparison of
two specified policies, not an isolated causal estimate of a device feature's benefit.
Neither score is a fraud probability. Legitimate groups also share devices and repeat
similar transfers. Amounts are converted to floating point for features; raw money remains
exact in the source. IDs only identify entities, preserve evidence and break score ties.

## Review budget and duplicates

For each policy separately, accept scores >=0.55 and emit at most three new alerts per
world per snapshot, ordered by descending score and stable candidate ID. Suppress a
candidate if its member Jaccard overlap with any already emitted alert is >=0.8. This
includes alerts already emitted in the current snapshot. Suppression lasts for the
whole world; cases do not reopen, merge hindsight membership or silently update evidence.
Cases blocked only by capacity can be emitted at a later snapshot if still present.
Every decision, including the prior case responsible for suppression, is exported.

Whole-world suppression can miss later campaigns involving the same accounts and can
suppress a later, more complete candidate after an incomplete earlier alert. This is
an explicit baseline tradeoff; candidate coverage is reported before suppression.

## Evaluation contract

All observed-only candidates and alerts are saved before the evaluator loads truth.
The runner verifies source file hashes, including development truth hashes, before
execution. Hash verification does not expose label values to detection. Test files
are never opened, even for checksum verification. The source manifest contains all
worlds, but this runner explicitly allows only `train` and `validation` inputs.

A candidate/group pair is eligible when all conditions hold:

- At least 80% of candidate members belong to the group (member precision).
- At least 80% of the group's members occur in the candidate (member recall).
- The alert follows the group's start and contains at least one of that group's planted
  transaction IDs as historical evidence. Background-only overlap cannot earn credit.

These thresholds are saved config values. A deterministic maximum-cardinality bipartite
matching assigns each alert and each positive group at most once. Augmenting paths avoid
greedy undercounting. This assignment maximizes group count, not the speed of detection.
Delay is measured from group start to the assigned alert; it need not be the earliest
possible alert under another equally large matching. Truth is retrospective and may be
discovered after the alert. Label-discovery times are never features.

Case precision = uniquely matched fraudulent groups / emitted alerts; ring recall =
matched fraudulent groups / planted fraudulent groups. Repeated positive matches that
survive suppression count as unmatched alerts and are also reported separately. A
legitimate lookalike is not a true positive. Benign-group alert rate uses the same member
and evidence eligibility against legitimate groups. Unmatched alerts also include
background and impure/incomplete candidates, so they are broader than benign-group hits.

Report mean matched-member precision/recall, assigned detection delay, per-family
recall and each world's results. Split-level count metrics pool world numerators and
denominators; delay/member metrics pool matched cases. No uncertainty interval is
claimed from only two validation networks. Candidate coverage uses every unfiltered
candidate as an oracle upper bound; it is not a deployable alert policy.

## Reproduction

Run from the project folder with its existing environment:

```powershell
.\.venv\Scripts\python.exe -B -m graphshield.cli detect-rings `
  --source-run coordinated-v1 --run-name rules-reproduction
.\.venv\Scripts\python.exe -B .\scripts\verify_phase4.py --run-name rules-reproduction
.\scripts\test.ps1
```

Use a new run name. Completed artifacts are under `artifacts/phase4/<run>/`: manifest,
snapshots, candidates, alerts, decisions, evaluation and report. JSONL records retain
nested features and evidence for later model comparisons and investigation views.
The verifier replays every development snapshot and selection decision, reproduces
all metrics, checks evidence timestamps/member endpoints, checks capacity and one-to-one
matches, and verifies source/output hashes. No external services or new packages are needed.
