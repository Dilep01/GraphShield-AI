# Phase 6: local investigation application

The FastAPI application serves both a JSON API and a static HTML/CSS/JavaScript
dashboard. This replaces the planned separate Streamlit frontend: one process and
one local address run the complete investigation flow, without a frontend build or CDN.
The application is a synthetic research demonstration, bound to `127.0.0.1`.

## Start and use

From the project folder, run:

```powershell
.\scripts\start_app.ps1
```

Open `http://127.0.0.1:8765`. The launcher prepares the serving bundle if absent,
resumes the default comparison-network replay, then starts one application worker.
Close the server with Ctrl+C. Use `-Port 8766` if the default port is occupied.
The direct equivalent after bundle/replay preparation is:

```powershell
& '.venv/Scripts/python.exe' -B scripts/serve.py
```

The default is the tuning-selected Node2Vec plus LightGBM model with its frozen
threshold. Network, model and policy selectors allow explicit development comparisons.
Select a case to inspect its account graph, historical transfers, device observations,
time window, model details and review history. Review status is workflow state; closing
or escalating a case does not establish a fraud label.

The default database is `artifacts/phase6/demo.sqlite`. All caches and generated
artifacts remain inside this project. Bundle files, databases and model checkpoints
are local trusted artifacts. Never load checkpoints obtained from an untrusted source.

## Evidence and model boundaries

- The serving bundle pins model files, training implementation hashes, configuration,
  tuning thresholds, development inputs and the default model selection.
- Only the three training and two validation worlds enter the replay catalog.
  Bundle preparation reads existing development evaluation summaries to export model
  selection and thresholds. Serving never opens raw benchmark truth or final-test files.
- A snapshot includes only `max(1, cutoff - 168) <= step < cutoff` and
  `available_step < cutoff`. Scoring reads a consistent database revision and does not
  insert events, create cases or save reviewer decisions.
- Node2Vec reuses a Phase 5 embedding only after checking its checksum and exact
  historical graph nodes, features and edges. A different visible graph is fitted
  locally with the frozen configuration. This can be slower than cached replay.
- Scores are investigation priorities. LightGBM feature contributions explain the
  model's raw margin; they do not establish guilt or calibrated fraud probability.
  GraphSAGE and the neural control explicitly report that contributions are unavailable.
  No model is silently substituted if inference fails.
- The ring API does not expose the separate Phase 2 PaySim transaction classifier.

## Ingestion, replay and review

SQLite stores observed events, ingestion receipts, replay checkpoints, frozen cases
and append-only reviewer records. Amounts are normalized exact decimal strings.
One ingestion request commits atomically. An identical receipt retries safely;
conflicting receipt content or an event ID with changed values returns HTTP 409.
The public ingestion endpoint accepts separately created demo networks. Bundled
development networks are protected from supplied event writes.

Replay progressively ingests observed arrivals and scores 27 snapshot cutoffs, from
97 through 721. At most three new cases are saved per snapshot. The frozen threshold
and whole-world Jaccard duplicate suppression match Phase 5. Each world/model/policy/
bundle combination has one stable replay ID. A completed retry returns the same job.
An interrupted job resumes when requested again, skipping committed checkpoints.
Jobs are not automatically resumed when the process restarts.

Case creation and its checkpoint share one database transaction. Revision checks reject
history changes during scoring. New arrivals cannot claim availability earlier than a
committed watermark; genuinely later arrivals can refer to older event steps.
Cases preserve evidence, score, model, policy, bundle fingerprint and snapshot ID.
Review writes use an expected version to prevent stale updates and a request ID to
prevent duplicate submissions. Review records never modify evidence, retrain a model
or update benchmark labels. The local app does not authenticate reviewer identity.

## API surface

| Method and path | Purpose |
|---|---|
| GET `/api/v1/health` | Readiness and bundle fingerprint |
| GET `/api/v1/catalog` | Development networks, models and policies |
| POST `/api/v1/worlds` | Create a separate `demo_...` network |
| POST `/api/v1/ingest` | Atomically ingest observed demo events |
| POST `/api/v1/score` | Return candidates for a historical cutoff without writes |
| POST `/api/v1/replays` | Start or resume a bundled development replay |
| GET `/api/v1/replays` and `/{replay_id}` | Replay progress or failure |
| GET `/api/v1/cases` and `/{case_id}` | Filtered case queue and frozen evidence |
| POST `/api/v1/cases/{case_id}/reviews` | Save a separate review decision and note |
| GET `/api/v1/openapi.json` | Machine-readable request schemas |

The local boundary checks Host and browser Origin, serves only local assets and uses
a restrictive content policy. These controls are not multi-user authentication or a
production security assessment. One worker, SQLite and bounded benchmark inputs are
the supported setup. Public hosting, RBAC, streaming, arbitrary-data load testing,
retention controls and production monitoring remain outside Phase 6.

## Verification

`tests/test_phase6.py` exercises retry and conflict handling, atomic rollback, schema
validation, current/future/late-event exclusion, read-only scoring, interrupted replay,
duplicate suppression, review concurrency, bundle checksums and local API boundaries.

```powershell
.\scripts\test.ps1
& '.venv/Scripts/python.exe' -B scripts/verify_phase6.py --run-name serving-v2-check
```

Verification needs a new output name each time. It uses its own SQLite database and
compares all seven models under both policies against saved Phase 5 comparison-network
alerts and scores. It checks every saved case's temporal evidence, contribution sums
where available, completed snapshot counts and unchanged retries. This is serving
parity verification, not a new fraud-detection evaluation. Final tests stay reserved.

FastAPI lifecycle and testing follow its official [lifespan documentation](https://fastapi.tiangolo.com/advanced/events/)
and [testing documentation](https://fastapi.tiangolo.com/tutorial/testing/).
