# GraphShield AI

Historical transaction analysis and graph-based fraud investigation research prototype.
The planned Phases 1–7 are complete as a local prototype, version 0.7.0.
All evaluation data is synthetic; real-world fraud performance is not established.

The project includes transaction baselines, a separate coordinated-activity benchmark,
graph rules, five learned group rankers, a local API, resumable replay, an investigation
dashboard and operational monitoring. The installed-package suite recorded **96 passing
tests**. A fresh offline installation reproduced 27 snapshots and 17 default cases with
exact score agreement. The final CSS-only correction was then checked in the browser.

Start with the [run guide](docs/RUNBOOK.md), [final results](docs/PHASE7_RESULTS.md)
and [model card](docs/MODEL_CARD.md). Release files and checksums are under
`artifacts/phase7/release/`; completion evidence is in `artifacts/phase7/manifest.json`.

## Final evaluation

All 6,362,620 original PaySim rows passed validation. On 130,116 held-out transactions,
LightGBM reached 0.5378 average precision versus 0.3073 for Logistic Regression.
At the frozen threshold, LightGBM precision was 44.14% and recall was 69.47%.
Under the separate 20-alerts-per-step policy, precision was 65.88% and recall was 83.77%.
The outputs remain risk scores, not calibrated fraud probabilities.

The generated benchmark supplies persistent accounts and explicit synthetic group
truth separately from PaySim. On the known-family final networks, the selected
Node2Vec + LightGBM ranker recovered **19/24 groups**, with **38% case precision**.
The tuned repeated-transfer baseline recovered 22/24 at 39.29% precision. The learned
ranker did not outperform that baseline on these two measures.

**None of the 24 withheld relay-chain groups was detected.** Candidate coverage was
zero for that family, so a different ranking model alone cannot solve this failure.
The final labels have now been consumed. Improvements require a separate experiment
and fresh evaluation data; these results do not establish real-world collusion detection.

Earlier phase reports preserve development results and the decisions made before
final evaluation. Their statements about reserved final tests describe that earlier stage.

## Current capabilities

- Download the original public PaySim ZIP and retain its CSV unchanged.
- Record dataset and output SHA-256 hashes for reproducibility.
- Validate schema, missing values, labels, transaction types, time steps and exact amounts.
- Export transactions and ground-truth labels into separate compressed Parquet files.
- Split complete chronological time steps into train, validation and final test.
- Audit training-account reuse, role overlap, repeated pairs and exact weak components.
- Measure validation cold-start against the frozen training graph.
- Build 20 strictly historical model features with explicit no-history defaults.
- Train Logistic Regression and LightGBM using separate validation tuning and comparison periods.
- Generate persistent account/device networks with explicit fraud and legitimate scenarios.
- Audit isolated ground truth, delayed observations, seed/group separation and a held-out pattern family.
- Build historical graphs, 16 explicit candidate features and traceable transaction/device evidence.
- Replay two graph-rule policies with review budgets, duplicate suppression and one-to-one ring evaluation.
- Train and compare five learned case rankers using historical observations and separate tuning/comparison networks.
- Save local Node2Vec embeddings, inductive GraphSAGE weights, model ablations and reproducible case decisions.
- Serve seven frozen ranking models through a versioned local API with separate ingestion and scoring.
- Resume batch replay without duplicate cases and reject conflicting event retries.
- Inspect account connections, transfers, model contributions and separate reviewer notes in a local dashboard.
- Monitor request errors/latency, replay recovery status and saved-case counts locally.
- Verify packaged demo files and sealed final evaluation artifacts using recorded hashes.
- Keep all project outputs, environments, caches and temporary data inside this folder.

## Run the investigation app

From this project folder in PowerShell:

```powershell
.\scripts\start_app.ps1
```

Open `http://127.0.0.1:8765`. The default comparison-network replay contains 17
cases from the tuning-selected Node2Vec model. Choose a case to inspect its graph,
historical evidence and model details, then record a review. Select another model
and run batch replay to populate its queue. Reviews never become training labels.
The app runs locally with one worker; it has no multi-user login or public deployment.
Stop it with Ctrl+C. See the [run guide](docs/RUNBOOK.md) for installation, monitoring,
recovery and verification, and [Phase 6 method](docs/PHASE6_METHOD.md) for API contracts.

## Setup on Windows

Open PowerShell in `C:\Users\dilip\Project\graphshield_aI`.
An isolated `.venv` is already present on this machine. For another checkout, create a
venv with a directly installed Python 3.11-3.13 executable (avoid Windows installer aliases):

```powershell
New-Item -ItemType Directory -Force -Path '.cache/tmp', '.cache/pip' | Out-Null
$env:TEMP = Join-Path $PWD '.cache/tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $PWD '.cache/pip'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Replace this interpreter path only if your Python installation is different.
& 'C:\Users\dilip\AppData\Local\Programs\Python\Python311\python.exe' -B -m venv .venv
.\scripts\bootstrap.ps1
```

`requirements-lock.txt` records the tested dependencies. The bootstrap installs the
project in editable mode without writing dependencies into the global interpreter.
The lock includes the official CPU PyTorch package index for Phase 5; the recorded
environment uses Python 3.11.9 on Windows. A GPU is not required.

## Reproduce Phase 1

```powershell
.\.venv\Scripts\python.exe -B .\scripts\download_paysim.py
.\.venv\Scripts\python.exe -B -m graphshield.cli ingest `
  --input data/raw/PS_20174392719_1491204439457_log.csv `
  --run-name paysim-reproduction
.\.venv\Scripts\python.exe -B .\scripts\verify_phase1.py --run-name paysim-reproduction
.\scripts\test.ps1
```

Use a new run name on each execution. Existing run directories are not overwritten.
Only consume outputs from a run whose `manifest.json` says `complete`.

Output locations:

- `data/raw/`: original ZIP, CSV and download manifest.
- `data/processed/<run>/transactions.parquet`: events without fraud flags or balances.
- `data/processed/<run>/labels.parquet`: original ground truth, with unknown discovery time.
- `artifacts/phase1/<run>/`: validation, graph report and run manifest.
- `.cache/`: contained installation caches, test workspaces and temporary database work.

The original dataset has no supplied transaction identifier in the expected schema;
IDs combine its file fingerprint with the source data-row number. Equal source rows
remain distinct records. File order is not assumed to establish within-step event order.

The graph audit includes all transaction types and measures weak connectivity, not
fraud-ring membership. Phase 1 deliberately omitted final-test outcomes; the completed final evaluation is
reported separately in Phase 7.

## Project documents

- [Specification and data contracts](docs/PROJECT_SPEC.md)
- [Build roadmap](docs/ROADMAP.md)
- [Local run guide and offline installation](docs/RUNBOOK.md)
- [Model card and limitations](docs/MODEL_CARD.md)
- [Phase 7 final evaluation method](docs/PHASE7_METHOD.md)
- [Phase 7 completion record](docs/PHASE7_RESULTS.md)
- [Phase 2 methodology and reproduction commands](docs/PHASE2_METHOD.md)
- [Phase 2 verified results and limitations](docs/PHASE2_RESULTS.md)
- [Phase 3 generator methodology](docs/PHASE3_METHOD.md)
- [Phase 3 verified results](docs/PHASE3_RESULTS.md)
- [Phase 4 graph-rule methodology and reproduction commands](docs/PHASE4_METHOD.md)
- [Phase 4 verified results and limitations](docs/PHASE4_RESULTS.md)
- [Phase 5 model protocol and reproduction commands](docs/PHASE5_METHOD.md)
- [Phase 5 verified results, model selection and tradeoffs](docs/PHASE5_RESULTS.md)
- [Phase 6 application guide and API contracts](docs/PHASE6_METHOD.md)
- [Phase 6 verification and browser results](docs/PHASE6_RESULTS.md)

The original transaction benchmark and generated coordinated-activity benchmark remain
separate. A GNN improving on a strong tabular model is a question to test, not a promised result.
