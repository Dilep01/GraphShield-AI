# Phase 1 completion record

Completed on 2026-10-14. The successful dataset run is `paysim-v2`.

## Verified outputs

- Original public PaySim CSV: 6,362,620 rows, 493,534,783 bytes.
- Zero missing fields, invalid steps, invalid categories, invalid labels or invalid amounts.
- Zero exact duplicate source rows; 16 zero-amount rows retained.
- Transactions and labels are separate Zstandard-compressed Parquet files.
- Stable IDs are unique and labels join one-to-one across the full dataset.
- Source and output SHA-256 hashes match their manifests.
- Whole time steps stay within one chronological split.
- Balances and fraud flags are absent from transaction-input columns.
- All 17 current automated tests passed; all 11 full-data artifact checks passed.

Commands used:

```powershell
.\scripts\test.ps1 --tb=short
.\.venv\Scripts\python.exe -B -m graphshield.cli ingest --input data/raw/PS_20174392719_1491204439457_log.csv --run-name paysim-v2
.\.venv\Scripts\python.exe -B scripts/verify_phase1.py --run-name paysim-v2
.\.venv\Scripts\python.exe -B -m pip check
```

The successful ingestion and graph audit took 71.58 seconds on this machine. This is
one observed development run, not a throughput or production latency guarantee.

## Findings that affect the next phase

The training graph has 8,566,556 observed account IDs. 94.83% participate in only one
event; only 1,596 appear in both sender and receiver roles. No directed account pair
repeats. There are 2,555,619 weak components, and the largest has 119 accounts.

Of 221,567 validation transactions, 221,209 have at least one account absent from the
training graph. Sender history and learned node-ID embeddings therefore need explicit
cold-start handling. These are topology findings, not measured model performance.

The 60/20/20 split of observed steps produces unequal row counts:
6,010,937 training, 221,567 validation, and 130,116 final-test transactions. Final-test
outcomes have not been summarized or used for model selection. The source's original
fraud labels have only been validated and saved separately; no classifier is trained yet.

Decision: retain PaySim as the original transaction benchmark. Use the separate,
explicitly generated benchmark to evaluate persistent accounts and coordinated groups.
Do not infer customer identity continuity beyond what the observed IDs support.

## Development corrections and environment note

The first full run (`paysim-v1`) failed validation because the initial parser rejected
scientific-notation amounts. PaySim contains values such as `1.0E7`. Exact decimal
validation now accepts these without accepting fractional cents; a regression test
covers this case. The failed run's manifest is retained; no Parquet was published by it.

The initial default `python --version` check invoked Windows' Python install manager,
which reported updating itself and wrote an installer log outside the project before
failing to locate a runtime. No global installation was requested. Subsequent commands
used the existing Python 3.11 executable directly. All GraphShield source, data, virtual
environment, dependency caches, test workspaces and pipeline outputs are inside this
project. The supplied scripts redirect temporary and pip-cache paths accordingly.

Detailed evidence: `artifacts/phase1/paysim-v2/report.md`, `validation.json`,
`manifest.json` and `verification.json`.
