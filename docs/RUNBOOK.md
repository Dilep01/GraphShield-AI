# GraphShield AI local run guide

Version 0.7.0, completed 17 October 2026. Run commands from the project folder in
PowerShell. All generated files must remain inside this folder. This is a synthetic
research demonstration with one local worker; it has no multi-user authentication.

## Start the existing project

```powershell
Set-Location 'C:\Users\dilip\Project\graphshield_aI'
.\scripts\start_app.ps1
```

Open http://127.0.0.1:8765. The script prepares the bundle if needed and resumes the
default development replay before starting the server. The initial Node2Vec queue
contains 17 cases. Stop with Ctrl+C. If the port is occupied, use
`.\scripts\start_app.ps1 -Port 8766` and open the corresponding address.

Select a case, inspect Connections, Transfers and Model notes, then use Review to
save a decision and note. These reviewer records are separate from benchmark labels.
Changing the network, model or policy changes the displayed queue; Run batch replay
populates that combination. Empty queues can mean that combination has not run yet.

## Check health and recover a replay

Expand Local system health near the bottom of the page and select Refresh system
health. Request counters reset on restart. Case and replay counts cover the entire
database, across models and policies, so they can exceed the selected queue's counts.
The panel does not measure fraud accuracy, probability calibration or statistical drift.

If a replay is interrupted, restart the application, select the same network, model
and policy, and select Run batch replay again. Committed snapshots are retained and
completed jobs return their existing results. Restart alone does not resume every
job. Review a failed job's error before retrying; do not delete the database to clear it.

Operational details are available at `/api/v1/monitoring`, `/api/v1/replays` and
`/api/v1/health`. Request logs are stored in `artifacts/phase7/runtime/requests.log`,
with a 1 MB limit and three backups. Request bodies and reviewer notes are not logged.
Schema details and ingestion retry contracts are in [the application guide](PHASE6_METHOD.md).

Stop the server before backing up SQLite. Preserve the database and any adjacent
`-wal` and `-shm` files together in a new project-local backup folder. Keep the serving
bundle and its matching observations/embedding caches with the backup. Restore into
a separate project-local folder first. Never overwrite the only copy of reviewer work.

## Install the packaged demo offline

The release folder is `artifacts/phase7/release`. Its wheelhouse targets Windows x64
and CPython 3.11; other platforms require their own dependency packages. Use a new
environment and demo directory. The following commands use this project's existing
Python to create the environment. On another computer, substitute a directly installed
Python 3.11 executable for the first interpreter.

```powershell
New-Item -ItemType Directory -Force '.cache/tmp', '.cache/pip', '.cache/torch' | Out-Null
$env:TEMP = Join-Path $PWD '.cache/tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $PWD '.cache/pip'
$env:TORCH_HOME = Join-Path $PWD '.cache/torch'
$env:PYTHONDONTWRITEBYTECODE = '1'
$releaseDir = Join-Path $PWD 'artifacts/phase7/release'
& '.venv/Scripts/python.exe' -B scripts/finalize_release.py --verify --release-only
& '.venv/Scripts/python.exe' -B -m venv .cache/demo-env
& '.cache/demo-env/Scripts/python.exe' -B -m pip install --no-index --no-compile --find-links "$releaseDir/wheelhouse" -r "$releaseDir/requirements-lock.txt"
& '.cache/demo-env/Scripts/python.exe' -B -m pip install --no-index --no-deps --no-compile "$releaseDir/graphshield_ai-0.7.0-py3-none-any.whl"
Expand-Archive -LiteralPath "$releaseDir/graphshield-demo-data-v0.7.0.zip" -DestinationPath artifacts/phase7/demo-install
$env:GRAPHSHIELD_ROOT = Join-Path $PWD 'artifacts/phase7/demo-install'
& '.cache/demo-env/Scripts/python.exe' -I -B -m graphshield.launch
# After stopping the server:
Remove-Item Env:GRAPHSHIELD_ROOT
```

Check each command succeeds before continuing. Do not reuse an existing destination
for extraction. The demo includes trusted locally generated checkpoints; do not load
replacement serialized model files from unknown sources. This archive contains only
generated development observations and the clean demo database, not raw PaySim data,
final-test networks or QA reviewer notes. The source ZIP contains code/configuration/
documentation/tests; it is not a copy of the full training datasets and checkpoints.
On another computer, extract the source ZIP into the chosen project root and keep
the complete release folder at `artifacts/phase7/release` beneath it. The copied
RUNBOOK.md is also present at `docs/RUNBOOK.md`, where its document links resolve.

## Verify the existing artifacts

```powershell
.\.venv\Scripts\python.exe -B scripts/finalize_release.py --verify
.\.venv\Scripts\python.exe -B scripts/final_evaluation.py verify --run-name final-v1
```

The first command checks packaged files and recorded completion evidence. The second
checks the frozen final protocol, sealed predictions and result checksum. For a full
model/metric reproduction check, the existing project has all required benchmark data:

```powershell
.\.venv\Scripts\python.exe -B scripts/verify_final.py --run-name final-v1
```

This last command rewrites `verification.json` with its findings; changes in numeric
diagnostics can change that file's checksum. Preserve the release's original evidence
before a new experiment. The recorded seven-check verification covers 130,116 PaySim
rows and 108 graph snapshots. It uses saved, checksummed embeddings rather than
retraining Node2Vec. Final labels have already been consumed; this is reproduction,
not a fresh independent evaluation. See [Phase 7 method](PHASE7_METHOD.md).

## Reproduction and tests

The completed clean installation lives in `.cache/reproduction/env`. Its replay report
is `artifacts/phase7/reproduction/report.json`; it reproduced 27 snapshots and 17 cases
with exact scores. The full installed-package suite recorded 96 passing tests in
`artifacts/phase7/clean-tests.xml`. That suite preceded the final CSS-only layout fix,
which was subsequently checked in the browser and included in the rebuilt wheel.

For source-development tests, run `.\scripts\test.ps1`. To exercise the installed
package instead, disable pytest's source-path override:

```powershell
.\.cache\reproduction\env\Scripts\python.exe -I -B -m pytest -o 'pythonpath=' --basetemp .cache/tests/release-check -q --tb=short
```

`scripts/reproduce_release.py --project <absolute-project-path>` is intended for a
fresh project-local reproduction environment with the wheel installed. It refuses
an existing `artifacts/phase7/reproduction/workspace` to protect the recorded run.
Do not delete that evidence merely to rerun it. Earlier phase methods document dataset
and training reproduction; the release demo alone cannot retrain every benchmark.

## Known limits and subsequent work

The selected ranker recovered 19/24 known-family groups at 38% case precision and
0/24 withheld relay-chain groups. The tuned transfer-rule baseline recovered 22/24
known groups. Candidate generation missed the withheld family before ranking.
Improving that stage requires a separate experiment and a fresh held-out benchmark.
Public deployment, authentication, streaming and real-data validation are outside
this release. See the [model card](MODEL_CARD.md) and [final results](PHASE7_RESULTS.md).
