from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]


def resolve_checkpoint_dir() -> Path:
    configured = os.getenv("CHECKPOINT_DIR", "").strip()
    if configured:
        result = Path(configured).expanduser()
        result.mkdir(parents=True, exist_ok=True)
        return result
    candidates = [
        Path(configured) if configured else None,
        ROOT / "project_checkpoint",
        Path.cwd() / "project_checkpoint",
        Path.home() / "project_checkpoint",
    ]
    existing = next(
        (candidate for candidate in candidates if candidate is not None and candidate.exists()),
        None,
    )
    result = existing or ROOT / "project_checkpoint"
    result.mkdir(parents=True, exist_ok=True)
    return result


CHECKPOINT_DIR = resolve_checkpoint_dir()
MAP_KEY = os.getenv("NASA_FIRMS_MAP_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]

REGIONS = {
    "gaza_eastern_med_box": "33,30,36,34",
    "red_sea_yemen_box": "40,12,50,20",
    "south_ukraine_black_sea_box": "29,45,38,49",
}

MEDIUM_FRP = float(os.getenv("NASA_ALERT_FRP_MEDIUM", "10"))
HIGH_FRP = float(os.getenv("NASA_ALERT_FRP_HIGH", "20"))


def fetch_region_source(region: str, bbox: str, source: str, query_date: str | None = None) -> pd.DataFrame:
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{source}/{bbox}/1"
    )
    if query_date:
        url += f"/{query_date}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text" not in content_type and "csv" not in content_type:
        raise RuntimeError(f"Unexpected FIRMS response type: {content_type}")

    text = response.text.strip()
    if not text or text.lower().startswith("error"):
        raise RuntimeError(f"FIRMS returned no usable CSV for {region} / {source}")

    data = pd.read_csv(io.StringIO(text))
    data["region"] = region
    data["firms_source"] = source
    data["ingested_at_utc"] = datetime.now(timezone.utc)
    return data


def normalize_detections(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["acq_date"] = pd.to_datetime(result["acq_date"], errors="coerce")
    result["acq_time_text"] = (
        pd.to_numeric(result["acq_time"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )
    result["timestamp"] = pd.to_datetime(
        result["acq_date"].dt.strftime("%Y-%m-%d")
        + " "
        + result["acq_time_text"].str[:2]
        + ":"
        + result["acq_time_text"].str[2:],
        utc=True,
        errors="coerce",
    )

    confidence = result["confidence"].astype(str).str.lower().str.strip()
    result = result[confidence.isin(["n", "h", "nominal", "high"])].copy()

    result["frp"] = pd.to_numeric(result["frp"], errors="coerce").fillna(0)
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    result = result.dropna(subset=["timestamp", "latitude", "longitude"])

    identity_text = (
        result["firms_source"].astype(str)
        + "|"
        + result["region"].astype(str)
        + "|"
        + result["timestamp"].astype(str)
        + "|"
        + result["latitude"].round(5).astype(str)
        + "|"
        + result["longitude"].round(5).astype(str)
    )
    result["detection_id"] = identity_text.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )

    high_confidence = confidence.loc[result.index].isin(["h", "high"])
    result["alert_priority"] = np.select(
        [
            high_confidence & result["frp"].ge(HIGH_FRP),
            high_confidence | result["frp"].ge(MEDIUM_FRP),
        ],
        ["high_review", "medium_review"],
        default="monitor",
    )
    result["alert_reason"] = np.select(
        [
            high_confidence & result["frp"].ge(HIGH_FRP),
            high_confidence,
            result["frp"].ge(MEDIUM_FRP),
        ],
        [
            "high confidence and elevated FRP",
            "high confidence detection",
            "elevated FRP detection",
        ],
        default="nominal detection",
    )

    return result.drop(columns=["acq_time_text"], errors="ignore")


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        if not path.with_suffix(".csv").exists():
            return pd.DataFrame()
        history = pd.read_csv(path.with_suffix(".csv"))
    else:
        try:
            history = pd.read_pickle(path)
        except (NotImplementedError, ValueError, TypeError, ModuleNotFoundError, AttributeError):
            # CSV exports preserve history across pandas versions used by Docker and Windows.
            history = pd.read_csv(path.with_suffix(".csv"))
    for column in ("timestamp", "ingested_at_utc"):
        if column in history.columns:
            history[column] = pd.to_datetime(history[column], utc=True, errors="coerce")
    return history


def persist_to_database(new_rows: pd.DataFrame, new_alerts: pd.DataFrame) -> None:
    if not DATABASE_URL:
        return
    from sqlalchemy import create_engine

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    if not new_rows.empty:
        new_rows.to_sql("live_firms_detections", engine, if_exists="append", index=False)
    if not new_alerts.empty:
        new_alerts.to_sql("live_review_alerts", engine, if_exists="append", index=False)


def run_once(query_date: str | None = None) -> dict[str, int | str]:
    if not MAP_KEY:
        raise RuntimeError(
            "NASA_FIRMS_MAP_KEY is missing. Set it in the terminal before running this script."
        )

    query_date = query_date or datetime.now(timezone.utc).date().isoformat()
    datetime.strptime(query_date, "%Y-%m-%d")
    frames = []
    failures = []
    for region, bbox in REGIONS.items():
        for source in SOURCES:
            try:
                frames.append(fetch_region_source(region, bbox, source, query_date))
            except Exception as error:
                failures.append(f"{region}/{source}: {str(error).replace(MAP_KEY, '[REDACTED]')}")

    if not frames:
        raise RuntimeError("All FIRMS requests failed: " + " | ".join(failures))

    raw = pd.concat(frames, ignore_index=True)
    current = normalize_detections(raw)
    current = current[current["timestamp"].dt.strftime("%Y-%m-%d").eq(query_date)]
    current = current.drop_duplicates("detection_id").copy()
    detections_path = CHECKPOINT_DIR / "live_firms_detections.pkl"
    alerts_path = CHECKPOINT_DIR / "live_review_alerts.pkl"

    history = load_history(detections_path)
    known_ids = set(history.get("detection_id", pd.Series(dtype=str)).astype(str))
    new_rows = current[~current["detection_id"].astype(str).isin(known_ids)].copy()

    combined = pd.concat([history, new_rows], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates("detection_id", keep="last")
        combined = combined.sort_values("timestamp", ascending=False).reset_index(drop=True)
    combined.to_pickle(detections_path)
    combined.to_csv(CHECKPOINT_DIR / "live_firms_detections.csv", index=False)

    new_alerts = new_rows[new_rows["alert_priority"].isin(["high_review", "medium_review"])].copy()
    previous_alerts = load_history(alerts_path)
    all_alerts = pd.concat([previous_alerts, new_alerts], ignore_index=True)
    if not all_alerts.empty:
        all_alerts = all_alerts.drop_duplicates("detection_id", keep="last")
        all_alerts = all_alerts.sort_values("timestamp", ascending=False).reset_index(drop=True)
    all_alerts.to_pickle(alerts_path)
    all_alerts.to_csv(CHECKPOINT_DIR / "live_review_alerts.csv", index=False)

    status = pd.DataFrame(
        [
            {
                "last_success_utc": datetime.now(timezone.utc),
                "observation_date_utc": query_date,
                "raw_records_returned": int(raw.shape[0]),
                "records_returned": int(current.shape[0]),
                "new_detections": int(new_rows.shape[0]),
                "new_review_alerts": int(new_alerts.shape[0]),
                "request_failures": int(len(failures)),
                "failure_details": " | ".join(failures),
            }
        ]
    )
    status.to_pickle(CHECKPOINT_DIR / "live_pipeline_status.pkl")
    status.to_csv(CHECKPOINT_DIR / "live_pipeline_status.csv", index=False)

    persist_to_database(new_rows, new_alerts)

    return {
        "observation_date_utc": query_date,
        "raw_records_returned": int(raw.shape[0]),
        "records_returned": int(current.shape[0]),
        "new_detections": int(new_rows.shape[0]),
        "new_review_alerts": int(new_alerts.shape[0]),
        "request_failures": int(len(failures)),
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NASA FIRMS near-real-time ingestion")
    parser.add_argument("--loop", action="store_true", help="Keep running at a fixed interval")
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--date", help="Observation day in UTC (YYYY-MM-DD); default: today")
    args = parser.parse_args()

    while True:
        try:
            summary = run_once(args.date)
            print(datetime.now(timezone.utc).isoformat(), summary, flush=True)
        except Exception as error:
            print(datetime.now(timezone.utc).isoformat(), "ERROR", str(error), file=sys.stderr, flush=True)
            if not args.loop:
                raise SystemExit(1)

        if not args.loop:
            break
        time.sleep(max(args.interval_minutes, 15) * 60)


if __name__ == "__main__":
    main()
