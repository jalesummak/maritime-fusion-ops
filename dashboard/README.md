# Maritime Fusion Operations Dashboard

This Streamlit application turns the notebook outputs into an explainable maritime-anomaly investigation system.

## Run

From the project directory:

```powershell
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The application looks for checkpoint files in `project_checkpoint/`. Set `CHECKPOINT_DIR` when the files are stored elsewhere.

An explicit `CHECKPOINT_DIR` is always respected, including for an empty directory. Without it, a legacy home-directory checkpoint can also be discovered. Prefer an explicit directory for reproducible runs. CSV files with the same basename are preferred when available; load pickle files only from trusted sources.

On a clean clone, model charts and headline totals are **recorded historical summaries**. The Data & Audit tab reports actual loaded row counts. Missing case details and maps are not reconstructed automatically. See [the data ledger](../data/README.md) for the missing original news export and other limits.

PostgreSQL is optional. Set `DATABASE_URL` to persist analyst feedback in the database. Without it, feedback is written to `project_checkpoint/analyst_feedback.csv`.

## Expected checkpoint files

Core files:

- `satellite_clean.pkl`
- `news_clean.pkl`
- one 3,289-row thermal-event checkpoint
- `event_news_matches_v2.pkl` (preferred); `event_news_matches.pkl` is a legacy fallback

For CSV equivalents, use the same basenames. Event files are checked in this order: `thermal_events_verified`, `thermal_events_v2`, `thermal_anomaly_v4`, `thermal_events_final`. Compatible schemas are those used by the original research dashboard; the teaching notebook outputs do not reproduce every advanced field automatically.

Advanced evidence files, when available:

- `final_priority_maritime_review.pkl`
- `sentinel_manual_review.pkl`
- `sar_event_summary.pkl`
- `event_control_comparison.pkl`

The application remains usable when optional evidence files are absent and reports their status in the Data & Audit page.

## Current NASA collection

In PowerShell, from the repository root:

```powershell
$env:NASA_FIRMS_MAP_KEY = Read-Host "Paste only your NASA FIRMS MAP key"
$env:CHECKPOINT_DIR = Join-Path (Get-Location) "project_checkpoint"
python dashboard/live_ingest.py
python -m streamlit run dashboard/app.py
```

The commands use the same Python environment. If `python` opens the Microsoft Store, use the full path of your installed Python interpreter instead.

For recurring collection run `python dashboard/live_ingest.py --loop --interval-minutes 30` in a separate terminal. For a specific available UTC day use `--date YYYY-MM-DD`. The application does not itself start a collector or load a local `.env` file; Compose supplies those variables in the Docker setup.

The default pull requests today's UTC date. Late publication for previous days needs an explicit backfill. New-row suppression uses the local detection history. Review flags use confidence and FRP thresholds, not the historical Random Forest. No flags can be a valid result. Failed one-shot collection exits with a nonzero status.

## Docker deployment package

Create the private Docker environment file:

```powershell
Copy-Item dashboard\.env.docker.example dashboard\.env
notepad dashboard\.env
```

Set `CHECKPOINT_HOST_DIR` and a new private `NASA_FIRMS_MAP_KEY`, then run:

```powershell
docker compose --env-file dashboard\.env -f dashboard\docker-compose.yml up -d --build
```

The compose project starts two services:

- `maritime-dashboard`: Streamlit application on port 8501.
- `nasa-firms-collector`: pulls NOAA-20 and NOAA-21 FIRMS data every 30 minutes.

Local Docker access remains `http://localhost:8501`. To provide a non-local URL, deploy the same Compose project on a server and place HTTPS authentication or a reverse proxy in front of port 8501.
