from __future__ import annotations

import os
from html import escape
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


st.set_page_config(
    page_title="Maritime Fusion Operations",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parents[1]


def resolve_checkpoint_dir() -> Path:
    configured = os.getenv("CHECKPOINT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
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
    return existing or ROOT / "project_checkpoint"


CHECKPOINT_DIR = resolve_checkpoint_dir()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


REGION_NAMES = {
    "gaza_eastern_med_box": "Gaza / Eastern Mediterranean",
    "red_sea_yemen_box": "Red Sea / Yemen",
    "south_ukraine_black_sea_box": "Southern Ukraine / Black Sea",
}

CLASS_NAMES = {
    "ordinary_vessel_or_port_context": "Ordinary vessel / port context",
    "unconfirmed_maritime_anomaly": "Unconfirmed maritime anomaly",
    "probable_maritime_association": "Probable maritime association",
    "externally_confirmed_incident": "Externally confirmed incident",
    "known_persistent_infrastructure": "Known persistent infrastructure",
}

COLORS = {
    "navy": "#0B1F33",
    "blue": "#2F6BFF",
    "cyan": "#32B8D8",
    "green": "#2AA876",
    "amber": "#F4A261",
    "red": "#E45756",
    "purple": "#7C5CFC",
    "gray": "#A9B4C2",
}

pio.templates["maritime"] = go.layout.Template(layout=dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    font=dict(family="Arial, sans-serif", color="#17324A"),
    colorway=["#2F6BFF", "#16A6AD", "#F4A261", "#7C5CFC"],
    hoverlabel=dict(bgcolor="#0B1F33", font_color="white"),
    xaxis=dict(gridcolor="#EAF0F6", zeroline=False),
    yaxis=dict(gridcolor="#EAF0F6", zeroline=False),
    transition=dict(duration=450, easing="cubic-in-out"),
))
pio.templates.default = "maritime"


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #F3F6FA, #EAF2FB, #F2F6FB, #E7F2F7);
        background-size: 400% 400%;
        animation: bgFlow 26s ease infinite;
    }
    @keyframes bgFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    [data-testid="stSidebar"] { background: #0B1F33; }
    [data-testid="stSidebar"] * { color: #F4F8FC; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .ops-header {
        padding: 1.15rem 1.35rem;
        border-radius: 14px;
        background: linear-gradient(100deg, #0B1F33, #173D60, #176A7E, #0F3B5C, #0B1F33);
        background-size: 320% 100%;
        animation: headerFlow 16s ease-in-out infinite;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 26px rgba(11, 31, 51, 0.16);
    }
    @keyframes headerFlow {
        0% { background-position: 0% 0; }
        50% { background-position: 100% 0; }
        100% { background-position: 0% 0; }
    }
    .ops-kicker {
        color: #83DDF2;
        letter-spacing: .16em;
        font-size: .72rem;
        display: flex;
        align-items: center;
        gap: .55rem;
    }
    .live-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #2AA876;
        box-shadow: 0 0 0 0 rgba(42, 168, 118, 0.65);
        animation: livePulse 1.8s ease-out infinite;
    }
    @keyframes livePulse {
        0% { box-shadow: 0 0 0 0 rgba(42,168,118,.65); }
        70% { box-shadow: 0 0 0 10px rgba(42,168,118,0); }
        100% { box-shadow: 0 0 0 0 rgba(42,168,118,0); }
    }
    .ops-title { font-size: 1.75rem; font-weight: 700; margin: .15rem 0; }
    .ops-subtitle { color: #D3E4F1; font-size: .92rem; }
    .finding {
        padding: 1rem 1.1rem;
        border-left: 5px solid #2F6BFF;
        background: white;
        border-radius: 8px;
        color: #17324A;
        box-shadow: 0 3px 12px rgba(11,31,51,.06);
    }
    .caution {
        padding: .85rem 1rem;
        border-left: 5px solid #F4A261;
        background: #FFF8EF;
        border-radius: 8px;
        color: #4B3520;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #DFE7EF;
        border-top: 3px solid #2F6BFF;
        padding: .8rem 1rem;
        border-radius: 12px;
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(11, 31, 51, .12);
        border-color: #2F6BFF;
    }
    div[data-testid="stMetricValue"] { color: #0B1F33; }
    .status-good { color: #188760; font-weight: 700; animation: softPulse 2.6s ease-in-out infinite; }
    .status-warn { color: #B86D13; font-weight: 700; animation: softPulse 2.6s ease-in-out infinite; }
    @keyframes softPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .55; }
    }
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"] {
        color: #38516A !important;
        font-weight: 650 !important;
        transition: color .18s ease;
    }
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"]:hover {
        color: #E45756 !important;
    }
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #D63F3F !important;
        font-weight: 750 !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4 {
        color: #0B1F33 !important;
    }
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] .stMarkdown p {
        color: #263F57;
    }
    .ops-header { position: relative; overflow: hidden; padding: 2rem; border-radius: 22px; }
    .ops-header::after { content: ''; position: absolute; right: -60px; top: -160px;
        width: 420px; height: 420px; border: 1px solid #83ddf240; border-radius: 50%;
        box-shadow: 0 0 0 50px #83ddf208, 0 0 0 100px #83ddf208; pointer-events: none; }
    .ops-title { font-size: clamp(1.5rem, 2.4vw, 2.5rem); letter-spacing: -.035em; }
    .ops-status { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 22px; }
    .ops-chip { padding: 8px 14px; border-radius: 24px; border: 1px solid #dbe6ef;
        background: #fff; color: #38516A; font-size: .8rem; }
    .ops-chip b { color: #0B1F33; }
    [data-testid="stMetric"], [data-testid="stPlotlyChart"] { animation: reveal .5s ease both; }
    [data-testid="stPlotlyChart"] { border-radius: 16px; overflow: hidden; border: 1px solid #E1E9F2; }
    [data-baseweb="tab-list"] { gap: 8px; padding: 8px; background: #e6edf5; border-radius: 14px; }
    button[data-baseweb="tab"] { padding: 10px 18px; border-radius: 9px; }
    button[data-baseweb="tab"][aria-selected="true"] { background: white; box-shadow: 0 3px 10px #0b1f3310; }
    .stButton > button { transition: transform .2s ease, box-shadow .2s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 16px #2f6bff20; }
    @keyframes reveal { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_existing(names: list[str]) -> Path | None:
    for name in names:
        candidate = CHECKPOINT_DIR / name
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False, ttl=60)
def read_pickle_candidates(names: list[str]) -> tuple[pd.DataFrame, str]:
    path = first_existing([candidate for name in names for candidate in (str(Path(name).with_suffix(".csv")), name)])
    if path is None:
        return pd.DataFrame(), "missing"
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path), path.name
        return pd.read_pickle(path), path.name
    except Exception as error:
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path), csv_path.name
            except Exception:
                pass
        return pd.DataFrame(), f"error: {error}"


@st.cache_data(show_spinner=False)
def read_database_table(table_name: str) -> pd.DataFrame:
    if not DATABASE_URL:
        return pd.DataFrame()
    try:
        from sqlalchemy import create_engine

        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        return pd.read_sql_table(table_name, engine)
    except Exception:
        return pd.DataFrame()


def prefer_checkpoint_or_database(
    checkpoint_names: list[str],
    database_table: str,
) -> tuple[pd.DataFrame, str]:
    checkpoint, source = read_pickle_candidates(checkpoint_names)
    if not checkpoint.empty:
        return checkpoint, f"checkpoint: {source}"
    database = read_database_table(database_table)
    if not database.empty:
        return database, f"PostgreSQL: {database_table}"
    return pd.DataFrame(), "missing"


satellite, satellite_source = prefer_checkpoint_or_database(
    ["satellite_clean.pkl", "df_satellite_clean.pkl"],
    "satellite_detections",
)
news, news_source = prefer_checkpoint_or_database(
    ["news_clean.pkl", "df_news_clean.pkl"],
    "news_articles",
)
events, events_source = prefer_checkpoint_or_database(
    [
        "thermal_events_verified.pkl",
        "thermal_events_v2.pkl",
        "thermal_anomaly_v4.pkl",
        "thermal_events_final.pkl",
    ],
    "thermal_events",
)
matches, matches_source = prefer_checkpoint_or_database(
    ["event_news_matches.pkl"],
    "event_news_matches",
)
priority, priority_source = read_pickle_candidates(
    ["final_priority_maritime_review.pkl"]
)
sentinel_review, sentinel_source = read_pickle_candidates(
    ["sentinel_manual_review.pkl"]
)
sar_summary, sar_source = read_pickle_candidates(
    ["sar_event_summary.pkl"]
)
control_comparison, control_source = read_pickle_candidates(
    ["event_control_comparison.pkl"]
)
live_detections, live_detection_source = read_pickle_candidates(
    ["live_firms_detections.pkl"]
)
live_alerts, live_alert_source = read_pickle_candidates(
    ["live_review_alerts.pkl"]
)
live_status, live_status_source = read_pickle_candidates(
    ["live_pipeline_status.pkl"]
)


def normalize_datetime(frame: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in candidates:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    return result


events = normalize_datetime(events, ["event_date", "start_time", "end_time"])
news = normalize_datetime(news, ["publication_date", "gdelt_seen_at"])

if not events.empty:
    if "region" in events.columns:
        events["region_name"] = events["region"].map(REGION_NAMES).fillna(events["region"])
    if "event_date" in events.columns:
        events["month"] = events["event_date"].dt.strftime("%Y-%m")
    if "has_supported_match" not in events.columns:
        events["has_supported_match"] = False
    events["association"] = np.where(
        events["has_supported_match"].fillna(False),
        "Source-supported",
        "No source support",
    )

if not priority.empty:
    if "region" in priority.columns:
        priority["region_name"] = priority["region"].map(REGION_NAMES).fillna(priority["region"])
    if "final_maritime_class" in priority.columns:
        priority["class_name"] = priority["final_maritime_class"].map(CLASS_NAMES).fillna(
            priority["final_maritime_class"]
        )


def count_rows(frame: pd.DataFrame, known_default: int = 0) -> int:
    return int(frame.shape[0]) if not frame.empty else known_default


def evidence_is_present(value) -> bool:
    """Missing and free-text review statuses must not become positive evidence."""
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "visible", "confirmed"}


def _play_button(fig: go.Figure, frame_ms: int = 700, transition_ms: int = 350) -> None:
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶ Animate",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": transition_ms, "easing": "cubic-in-out"},
                            },
                        ],
                    }
                ],
            }
        ]
    )


@st.cache_data(show_spinner=False)
def _autoplay_components():
    return """
    <script>
    (function () {
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      var attempts = 0;
      var timer = window.setInterval(function () {
        attempts += 1;
        if (attempts > 20) { window.clearInterval(timer); return; }
        var doc = window.parent.document;
        if (!doc) return;
        var plots = doc.querySelectorAll('.js-plotly-plot');
        var found = false;
        plots.forEach(function (p) {
          var btn = p.querySelector('.updatemenu-button');
          if (btn && !p.getAttribute('data-autoplay')) {
            p.setAttribute('data-autoplay', '1');
            btn.click();
            found = true;
          }
        });
      }, 400);
    })();
    </script>
    """


def _auto_play_animations():
    st.components.v1.html(_autoplay_components(), height=0)


def _growth_frames(fig: go.Figure, steps: int = 9) -> None:
    base = [trace.to_plotly_json() for trace in fig.data]
    for trace in base:
        for axis in ("x", "y", "text"):
            if isinstance(trace.get(axis), np.ndarray):
                trace[axis] = trace[axis].tolist()
    frames = []
    for step in range(steps + 1):
        frac = step / steps
        traces = []
        for trace in base:
            scaled = dict(trace)
            for axis in ("x", "y"):
                values = scaled.get(axis)
                if isinstance(values, list):
                    scaled[axis] = [
                        value * frac if isinstance(value, (int, float)) else value
                        for value in values
                    ]
            if step < steps and "text" in scaled and isinstance(scaled["text"], list):
                scaled["text"] = [""] * len(scaled["text"])
            traces.append(scaled)
        frames.append(go.Frame(data=traces, name=f"step-{step}"))
    fig.frames = frames


def _draw_frames(fig: go.Figure, steps: int = 10) -> None:
    base = [trace.to_plotly_json() for trace in fig.data]
    for trace in base:
        for axis in ("x", "y"):
            if isinstance(trace.get(axis), np.ndarray):
                trace[axis] = trace[axis].tolist()
    frames = []
    for step in range(1, steps + 1):
        frac = step / steps
        traces = []
        for trace in base:
            partial = dict(trace)
            for axis in ("x", "y"):
                values = partial.get(axis)
                if isinstance(values, list) and values:
                    keep = max(1, round(len(values) * frac))
                    partial[axis] = list(values[:keep]) + [None] * (len(values) - keep)
            traces.append(partial)
        frames.append(go.Frame(data=traces, name=f"step-{step}"))
    fig.frames = frames


def _geo_grow_frames(fig: go.Figure, codes, months) -> None:
    base = fig.data[0].to_plotly_json()
    latitudes = list(base.get("lat", []))
    longitudes = list(base.get("lon", []))
    frames = []
    for index in range(len(months)):
        keep = [code <= index and code >= 0 for code in codes]
        staged = dict(base)
        staged["lat"] = [point if kept else None for point, kept in zip(latitudes, keep)]
        staged["lon"] = [point if kept else None for point, kept in zip(longitudes, keep)]
        frames.append(go.Frame(data=[staged], name=str(months[index])))
    fig.frames = frames
    fig.update_layout(
        sliders=[
            {
                "active": len(months) - 1,
                "currentvalue": {"prefix": "Month: "},
                "pad": {"t": 40},
                "steps": [
                    {
                        "label": str(month),
                        "method": "animate",
                        "args": [
                            [str(month)],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
                        ],
                    }
                    for month in months
                ],
            }
        ]
    )
    _play_button(fig, frame_ms=800, transition_ms=500)


clean_detection_count = count_rows(satellite, 11525)
news_count = count_rows(news, 3589)
event_count = count_rows(events, 3289)
match_count = count_rows(matches, 5677)
water_count = 132
priority_count = count_rows(priority, 7)


with st.sidebar:
    st.markdown("## Maritime Fusion Ops")
    st.caption("Explainable anomaly triage")
    operating_mode = st.radio(
        "Operating mode",
        ["Historical replay", "Connected database"],
        index=0,
    )
    selected_regions = st.multiselect(
        "Study regions",
        list(REGION_NAMES.values()),
        default=list(REGION_NAMES.values()),
    )
    association_filter = st.multiselect(
        "Association status",
        ["Source-supported", "No source support"],
        default=["Source-supported", "No source support"],
    )
    st.divider()
    auto_refresh = st.selectbox(
        "Auto refresh",
        ["Off", "30s", "60s", "120s"],
        index=0,
        help="Live ingestion çıktılarını bir süre döngüsüyle yeniden okur.",
    )
    auto_refresh_seconds = {"Off": 0, "30s": 30, "60s": 60, "120s": 120}[auto_refresh]
    auto_play = st.checkbox("Auto-play animations", value=False)
    reduce_motion = st.checkbox("Reduce motion", value=False)
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Checkpoint: {CHECKPOINT_DIR}")
    if operating_mode == "Connected database":
        if DATABASE_URL:
            st.markdown('<span class="status-good">● PostgreSQL configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">● DATABASE_URL is not configured</span>', unsafe_allow_html=True)


if reduce_motion:
    auto_play = False
    st.markdown("<style>*,*::before,*::after{animation:none!important;transition:none!important}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="ops-header">
      <div class="ops-kicker"><span class="live-dot"></span>MARITIME FUSION OPERATIONS</div>
      <div class="ops-title">Thermal Anomaly & Conflict-Evidence Monitor</div>
      <div class="ops-subtitle">NASA FIRMS · multilingual news · UCDP · UKMTO · AIS · SAR · Sentinel · analyst feedback</div>
    </div>
    """,
    unsafe_allow_html=True,
)


filtered_events = events.copy()
if not filtered_events.empty and "region_name" in filtered_events.columns:
    filtered_events = filtered_events[filtered_events["region_name"].isin(selected_regions)]
if not filtered_events.empty and "association" in filtered_events.columns:
    filtered_events = filtered_events[filtered_events["association"].isin(association_filter)]


last_pull = pd.to_datetime(live_status.iloc[-1].get("last_success_utc"), utc=True, errors="coerce") if not live_status.empty else pd.NaT
age_minutes = (pd.Timestamp.now(tz="UTC") - last_pull).total_seconds() / 60 if pd.notna(last_pull) else None
feed_label = "No successful pull recorded" if age_minutes is None else ("Feed updated recently" if age_minutes <= 60 else "Stored feed · refresh overdue")
last_label = last_pull.strftime("%d %b %Y, %H:%M UTC") if pd.notna(last_pull) else "Unavailable"
st.markdown(
    f'<div class="ops-status"><span class="ops-chip"><b>{escape(operating_mode)}</b></span>'
    f'<span class="ops-chip">{escape(feed_label)}</span>'
    f'<span class="ops-chip">Last pull <b>{last_label}</b></span>'
    f'<span class="ops-chip"><b>{len(selected_regions)}</b> regions selected</span></div>',
    unsafe_allow_html=True,
)

live_tab, overview_tab, assignment_tab, model_tab, maritime_tab, audit_tab = st.tabs(
    [
        "Live NASA Feed",
        "Command Center",
        "Assignment Coverage",
        "Model & Validation",
        "Maritime Investigation",
        "Data & Audit",
    ]
)


with live_tab:
    st.subheader("NASA FIRMS near-real-time review feed")
    if live_status.empty:
        st.info(
            "The live collector has not run yet. Start dashboard/live_ingest.py with your private NASA FIRMS MAP_KEY."
        )
    else:
        status_row = live_status.iloc[-1]
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Last successful pull", str(status_row.get("last_success_utc", "—"))[:19] + " UTC")
        s2.metric("Current NASA snapshot", f"{int(status_row.get('records_returned', 0)):,}")
        s3.metric("New since previous pull", f"{int(status_row.get('new_detections', 0)):,}")
        s4.metric("Stored detections", f"{live_detections.shape[0]:,}")
        s5.metric("Alert history", f"{live_alerts.shape[0]:,}")
        st.caption(
            f"This pull created {int(status_row.get('new_review_alerts', 0)):,} new review alerts. "
            "Previously stored detections are deduplicated and remain in the history tables."
        )

    if live_detections.empty:
        st.warning("No live FIRMS detections have been stored yet.")
    else:
        live_detections = normalize_datetime(live_detections, ["timestamp", "ingested_at_utc"])
        live_detections["region_name"] = live_detections["region"].map(REGION_NAMES).fillna(live_detections["region"])
        date_scope = st.selectbox("Observation period", ["Today (UTC)", "Last 7 days (UTC)", "All stored history"])
        utc_today = pd.Timestamp.now(tz="UTC").floor("D")
        if date_scope == "Today (UTC)":
            live_detections = live_detections[live_detections["timestamp"].ge(utc_today) & live_detections["timestamp"].lt(utc_today + pd.Timedelta(days=1))].copy()
        elif date_scope == "Last 7 days (UTC)":
            live_detections = live_detections[live_detections["timestamp"].ge(utc_today - pd.Timedelta(days=6)) & live_detections["timestamp"].lt(utc_today + pd.Timedelta(days=1))].copy()
        st.caption(f"{date_scope} · Today is {utc_today:%d %b %Y}. Counts reflect observations currently published by NASA; the day is not complete.")
        live_detections = live_detections[live_detections["region_name"].isin(selected_regions)].copy()
        live_detections = live_detections.sort_values("timestamp", ascending=False)
        live_controls, live_summary = st.columns([2, 1])
        with live_controls:
            priority_options = st.multiselect("Show review priorities", ["high_review", "medium_review", "monitor"], default=["high_review", "medium_review", "monitor"])
        live_detections = live_detections[live_detections["alert_priority"].isin(priority_options)]
        with live_summary:
            st.metric("Detections in this view", f"{live_detections.shape[0]:,}")
        if live_detections.empty:
            st.info("No detections match these filters. Select another region or priority.")
        replay = st.toggle("Play observations over time", value=False, help="Replays stored NASA observation times. This is not a live satellite video.")
        map_data = live_detections.sort_values("timestamp").copy()
        map_data["observation_hour"] = map_data["timestamp"].dt.strftime("%Y-%m-%d %H:00 UTC")
        map_data = map_data.dropna(subset=["observation_hour"])
        map_options = {"animation_frame": "observation_hour"} if replay and not map_data.empty else {}
        live_map = px.scatter_geo(
            map_data,
            lat="latitude",
            lon="longitude",
            color="alert_priority",
            size=np.maximum(pd.to_numeric(map_data["frp"], errors="coerce").fillna(0), 1),
            hover_name="region_name",
            hover_data=["timestamp", "frp", "confidence", "firms_source", "alert_reason"],
            color_discrete_map={
                "high_review": COLORS["red"],
                "medium_review": COLORS["amber"],
                "monitor": COLORS["gray"],
            },
            projection="natural earth",
            **map_options,
        )
        live_map.update_geos(lataxis_range=[10, 51], lonaxis_range=[27, 52], showcountries=True, showcoastlines=True, showland=True, landcolor="#E4ECF3", showocean=True, oceancolor="#D9F0F7")
        live_map.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), legend_title="Review priority")
        st.plotly_chart(live_map, use_container_width=True)
        st.caption("Source: NASA FIRMS · NOAA-20 / NOAA-21 VIIRS. Play replays actual stored observation hours; it does not create new detections.")

        activity = live_detections.assign(hour=live_detections["timestamp"].dt.floor("h")).groupby(["hour", "alert_priority"]).size().reset_index(name="detections")
        activity_chart = px.bar(activity, x="hour", y="detections", color="alert_priority", color_discrete_map={"high_review": COLORS["red"], "medium_review": COLORS["amber"], "monitor": COLORS["gray"]}, title="Observed activity · hourly satellite detections")
        activity_chart.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Observation time (UTC)", yaxis_title="Detections", legend_title="Priority")
        st.plotly_chart(activity_chart, use_container_width=True)
        st.download_button("Download this filtered view", live_detections.to_csv(index=False).encode("utf-8-sig"), "filtered_firms_detections.csv", "text/csv")

        st.markdown("#### Latest detections")
        live_columns = [
            column
            for column in [
                "timestamp",
                "region_name",
                "latitude",
                "longitude",
                "frp",
                "confidence",
                "firms_source",
                "alert_priority",
                "alert_reason",
            ]
            if column in live_detections.columns
        ]
        st.dataframe(
            live_detections[live_columns].head(100),
            use_container_width=True,
            hide_index=True,
        )

    if not live_alerts.empty:
        st.markdown("#### Review alert history")
        alert_columns = [
            column
            for column in [
                "timestamp",
                "region",
                "latitude",
                "longitude",
                "frp",
                "confidence",
                "alert_priority",
                "alert_reason",
            ]
            if column in live_alerts.columns
        ]
        st.dataframe(live_alerts[alert_columns].head(100), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="caution"><b>Alert meaning</b><br>
        These are new thermal detections selected for review using confidence and FRP thresholds. They are not conflict alerts and require the same independent evidence workflow used in the historical study.</div>
        """,
        unsafe_allow_html=True,
    )


with overview_tab:
    st.caption("Historical study, September 2025–February 2026. When research files are absent, KPI cards retain the recorded study totals; they do not represent loaded records. Data & Audit shows actual loaded row counts. Live NASA observations are in the first tab.")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clean detections", f"{clean_detection_count:,}", "704 low-confidence removed")
    k2.metric("News articles", f"{news_count:,}", "multilingual GDELT")
    k3.metric("Thermal events", f"{event_count:,}", "3-day / 7 km grouping")
    k4.metric("Events over water", f"{water_count:,}", "4.0% of events")
    k5.metric("Priority reviews", f"{priority_count:,}", "4 remained unexplained")

    map_col, rate_col = st.columns([1.65, 1])

    with map_col:
        st.subheader("Multi-region thermal-event map")
        if not filtered_events.empty and {"latitude", "longitude"}.issubset(filtered_events.columns):
            map_data = filtered_events.copy()
            map_data["marker_size"] = (
                pd.to_numeric(map_data.get("total_frp", 1), errors="coerce")
                .fillna(1)
                .rank(pct=True)
                .mul(16)
                .add(3)
            )
            fig_map = px.scatter_geo(
                map_data,
                lat="latitude",
                lon="longitude",
                color="association",
                size="marker_size",
                hover_name="event_id" if "event_id" in map_data.columns else None,
                hover_data={
                    column: True
                    for column in ["region_name", "event_date", "total_frp", "max_brightness"]
                    if column in map_data.columns
                },
                color_discrete_map={
                    "Source-supported": COLORS["red"],
                    "No source support": COLORS["gray"],
                },
                projection="natural earth",
            )
            fig_map.update_geos(
                showcountries=True,
                showcoastlines=True,
                fitbounds="locations",
                bgcolor="rgba(0,0,0,0)",
            )
            fig_map.update_layout(height=510, margin=dict(l=0, r=0, t=5, b=0), legend_title="Evidence")
            if "month" in map_data.columns and map_data["month"].notna().any():
                months = sorted(map_data["month"].dropna().unique().tolist())
                codes = pd.Categorical(map_data["month"], categories=months, ordered=True).codes
                _geo_grow_frames(fig_map, codes, months)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Thermal-event coordinates are not available in the loaded checkpoint.")

    with rate_col:
        st.subheader("Source-supported rate")
        if not filtered_events.empty and "region_name" in filtered_events.columns:
            regional = (
                filtered_events.groupby("region_name", as_index=False)
                .agg(total_events=("event_id", "nunique"), supported_events=("has_supported_match", "sum"))
            )
            regional["supported_rate"] = regional["supported_events"] / regional["total_events"] * 100
            fig_rate = px.bar(
                regional.sort_values("supported_rate"),
                x="supported_rate",
                y="region_name",
                orientation="h",
                text=regional.sort_values("supported_rate")["supported_rate"].map(lambda value: f"{value:.1f}%"),
                color="supported_rate",
                color_continuous_scale=["#DCE5EF", COLORS["blue"]],
            )
            fig_rate.update_layout(
                height=300,
                xaxis_title="Supported events (%)",
                yaxis_title="",
                coloraxis_showscale=False,
                margin=dict(l=0, r=15, t=30, b=0),
            )
            _growth_frames(fig_rate, steps=8)
            _play_button(fig_rate, frame_ms=110, transition_ms=260)
            st.plotly_chart(fig_rate, use_container_width=True)

        st.markdown(
            """
            <div class="finding"><b>Decision finding</b><br>
            Satellite anomalies provide candidate detections. Region, port context and independent evidence provide stronger prioritisation signals than absolute thermal intensity alone.</div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Monthly event volume and news-support rate")
    if not filtered_events.empty and {"month", "region_name"}.issubset(filtered_events.columns):
        monthly = (
            filtered_events.groupby(["month", "region_name"], as_index=False)
            .agg(events=("event_id", "nunique"), supported=("has_supported_match", "sum"))
        )
        monthly["support_rate"] = monthly["supported"] / monthly["events"] * 100
        trend_left, trend_right = st.columns(2)
        with trend_left:
            fig_events = px.line(monthly, x="month", y="events", color="region_name", markers=True)
            fig_events.update_layout(height=310, xaxis_title="Month", yaxis_title="Thermal events", legend_title="Region")
            _draw_frames(fig_events, steps=12)
            _play_button(fig_events, frame_ms=420, transition_ms=200)
            st.plotly_chart(fig_events, use_container_width=True)
        with trend_right:
            fig_support = px.line(monthly, x="month", y="support_rate", color="region_name", markers=True)
            fig_support.update_layout(height=310, xaxis_title="Month", yaxis_title="Source-supported events (%)", legend_title="Region")
            _draw_frames(fig_support, steps=12)
            _play_button(fig_support, frame_ms=420, transition_ms=200)
            st.plotly_chart(fig_support, use_container_width=True)


with assignment_tab:
    st.subheader("What the assignment required — and what was delivered")
    st.caption("Historical project checklist; not a certification of the assignment or of end-to-end reproducibility. The teaching notebooks and advanced research extensions have different scopes. See the repository data-provenance notes.")
    coverage = pd.DataFrame(
        [
            ["Task 1", "≥5,000 thermal detections", f"{clean_detection_count:,} clean detections", "Completed"],
            ["Task 1", "≥1,000 conflict-news articles", f"{news_count:,} multilingual articles", "Completed"],
            ["Task 1", "At least two news sources", "Multiple publishers discovered through GDELT", "Completed"],
            ["Task 2", "Spatial-temporal event clustering", f"{event_count:,} V2 thermal events", "Completed"],
            ["Task 2", "Temporal and spatial visualisations", "Monthly trends, regional maps and hotspot review", "Completed"],
            ["Task 3", "Thermal-news matching", f"{match_count:,} event-news candidate pairs", "Completed"],
            ["Task 3", "Hypothesis testing", "Mann-Whitney and independent evidence comparisons", "Completed"],
            ["Task 3", "Compare classifiers", "LR/RF research; Decision Tree in teaching code", "Scope documented"],
            ["Task 4", "Four-panel dashboard", "Interactive multi-page operations dashboard", "Exceeded"],
            ["Database", "PostgreSQL in Docker", "Four relational tables with PK/FK constraints", "Completed"],
        ],
        columns=["Section", "Requirement", "Delivered", "Status"],
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("Work completed beyond the assignment")
    beyond = pd.DataFrame(
        [
            ["Independent conflict validation", "UCDP finalized GED and 2026 candidate events"],
            ["Official maritime verification", "UKMTO incident audit and accident comparison"],
            ["True maritime scope", "Land-water mask and port-distance enrichment"],
            ["Persistent-source control", "Known offshore infrastructure and recurring hotspots"],
            ["Remote-sensing context", "Sentinel-1 SAR and Sentinel-2 optical review"],
            ["Vessel context", "AIS proximity, identity and control-day comparison"],
            ["Causality protection", "Association was separated from confirmed causation"],
            ["Human-in-the-loop", "Analyst review, feedback and auditable decisions"],
        ],
        columns=["Extension", "What it added"],
    )
    st.dataframe(beyond, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="finding"><b>Research answer</b><br>
        Thermal anomalies and conflict reporting show meaningful spatial-temporal association, but the available evidence does not support automatic conflict causation. The strongest defensible product is an explainable analyst-triage system.</div>
        """,
        unsafe_allow_html=True,
    )


with model_tab:
    st.subheader("Model journey: strong single-month result, weaker multi-month stability")
    st.caption("Recorded research results, not scores recalculated from the live feed. PR-AUC here means average precision. The repeated February evaluations are exploratory.")

    holdout = pd.DataFrame(
        {
            "Model": ["Context Logistic Regression", "Context Random Forest"],
            "Accuracy": [0.868, 0.941],
            "Precision": [0.269, 0.464],
            "Recall": [0.875, 0.812],
            "F1": [0.412, 0.591],
            "ROC-AUC": [0.922, 0.953],
            "PR-AUC": [0.459, 0.663],
        }
    )
    pooled = pd.DataFrame(
        {
            "Model": [
                "Historical-rate lookup",
                "Geography-only Random Forest",
                "Satellite + context Random Forest",
                "Relative anomaly + context Random Forest",
            ],
            "Accuracy": [0.877, 0.881, 0.875, 0.870],
            "Precision": [0.285, 0.304, 0.275, 0.256],
            "Recall": [0.431, 0.460, 0.416, 0.387],
            "F1": [0.343, 0.366, 0.331, 0.308],
            "ROC-AUC": [0.785, 0.797, 0.805, 0.805],
            "PR-AUC": [0.284, 0.312, 0.284, 0.289],
        }
    )

    holdout_col, pooled_col = st.columns(2)
    with holdout_col:
        st.markdown("#### February 2026 holdout")
        holdout_long = holdout.melt(id_vars="Model", value_vars=["F1", "ROC-AUC", "PR-AUC"], var_name="Metric", value_name="Score")
        fig_holdout = px.bar(holdout_long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f")
        fig_holdout.update_layout(height=360, yaxis_range=[0, 1], legend_title="", margin=dict(t=40))
        _growth_frames(fig_holdout, steps=8)
        _play_button(fig_holdout, frame_ms=110, transition_ms=260)
        st.plotly_chart(fig_holdout, use_container_width=True)
    with pooled_col:
        st.markdown("#### Multi-month pooled backtest")
        pooled_long = pooled.melt(id_vars="Model", value_vars=["F1", "PR-AUC"], var_name="Metric", value_name="Score")
        fig_pooled = px.bar(pooled_long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f")
        fig_pooled.update_layout(height=360, yaxis_range=[0, 0.7], legend_title="", margin=dict(t=40))
        _growth_frames(fig_pooled, steps=8)
        _play_button(fig_pooled, frame_ms=110, transition_ms=260)
        st.plotly_chart(fig_pooled, use_container_width=True)

    confusion_col, importance_col = st.columns(2)
    with confusion_col:
        st.markdown("#### February Random Forest confusion matrix")
        confusion = np.array([[273, 15], [3, 13]])
        fig_confusion = px.imshow(
            confusion,
            text_auto=True,
            x=["Predicted no support", "Predicted support"],
            y=["Actual no support", "Actual support"],
            color_continuous_scale=["#EDF2F7", COLORS["blue"]],
        )
        fig_confusion.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_confusion, use_container_width=True)
    with importance_col:
        st.markdown("#### Permutation importance")
        importance = pd.DataFrame(
            {
                "Feature group": ["Region", "Port zone", "Nearest-port distance", "Max brightness", "Total FRP", "Day ratio"],
                "Importance": [0.436, 0.299, 0.172, 0.022, 0.004, 0.004],
            }
        )
        fig_importance = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature group",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#DCE5EF", COLORS["purple"]],
            text_auto=".3f",
        )
        fig_importance.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=40))
        _growth_frames(fig_importance, steps=8)
        _play_button(fig_importance, frame_ms=110, transition_ms=260)
        st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown(
        """
        <div class="caution"><b>Model governance conclusion</b><br>
        The February result is based on only 16 news-supported events. In the pooled backtest, geography-only prioritisation outperformed the tested satellite-plus-context classifier. These are retrospective research models. The live collector uses confidence and FRP rules; it does not run this classifier or confirm conflict.</div>
        """,
        unsafe_allow_html=True,
    )


with maritime_tab:
    st.subheader("From regional monitoring to maritime investigation")
    st.caption("Recorded historical selection funnel: 132 centroid-based water candidates, seven reviewed cases. This is not a measured workload reduction or a recall estimate. Map and case details require local research data.")
    funnel = go.Figure(
        go.Funnel(
            y=["All thermal events", "Events over water", "Priority transient reviews", "Still unexplained"],
            x=[event_count, water_count, priority_count, 4],
            textinfo="value+percent initial",
            marker={"color": [COLORS["navy"], COLORS["blue"], COLORS["amber"], COLORS["red"]]},
        )
    )
    funnel.update_layout(height=360, margin=dict(l=15, r=15, t=40, b=15))
    _growth_frames(funnel, steps=12)
    _play_button(funnel, frame_ms=140, transition_ms=220)
    st.plotly_chart(funnel, use_container_width=True)

    if priority.empty:
        st.warning("Final priority review checkpoint is not available. Save final_priority_maritime_review.pkl and refresh.")
    else:
        final_counts = priority["class_name"].value_counts().rename_axis("Final class").reset_index(name="Events")
        class_col, selector_col = st.columns([1, 1.4])
        with class_col:
            fig_classes = px.bar(
                final_counts,
                x="Final class",
                y="Events",
                color="Final class",
                text_auto=True,
                color_discrete_map={
                    "Ordinary vessel / port context": COLORS["blue"],
                    "Unconfirmed maritime anomaly": COLORS["amber"],
                    "Probable maritime association": COLORS["purple"],
                    "Externally confirmed incident": COLORS["red"],
                    "Known persistent infrastructure": COLORS["green"],
                },
            )
            fig_classes.update_layout(height=330, showlegend=False, xaxis_title="", yaxis_title="Priority events", margin=dict(t=40))
            _growth_frames(fig_classes, steps=8)
            _play_button(fig_classes, frame_ms=110, transition_ms=260)
            st.plotly_chart(fig_classes, use_container_width=True)
        with selector_col:
            event_ids = priority["event_id"].astype(str).tolist()
            case_labels = priority.set_index(priority["event_id"].astype(str)).apply(lambda row: f"#{row.get('investigation_rank', '—')} · {str(row.get('event_date', ''))[:10]} · {row.get('nearest_port_name', 'Unknown port')}", axis=1).to_dict()
            selected_event = st.selectbox("Select an investigation", event_ids, format_func=lambda event_id: case_labels.get(event_id, event_id))
            case = priority.loc[priority["event_id"].astype(str).eq(selected_event)].iloc[0]
            st.metric("Final classification", case.get("class_name", "Not available"))
            nearest_port_distance = pd.to_numeric(
                pd.Series([case.get("nearest_port_km", np.nan)]),
                errors="coerce",
            ).iloc[0]
            nearest_port_display = (
                f"{nearest_port_distance:.1f} km"
                if pd.notna(nearest_port_distance)
                else "—"
            )
            details = {
                "Investigation rank": case.get("investigation_rank", "—"),
                "Region": case.get("region_name", case.get("region", "—")),
                "Date": str(case.get("event_date", "—"))[:10],
                "Total FRP": case.get("total_frp", "—"),
                "Nearest port": case.get("nearest_port_name", "—"),
                "Nearest-port distance": nearest_port_display,
                "Closest AIS vessel": case.get("causal_ship_name", "No close AIS candidate"),
                "Sentinel review": case.get("sentinel_sentinel_review_status", "No direct visual evidence"),
            }
            st.dataframe(pd.DataFrame(details.items(), columns=["Evidence field", "Value"]).astype(str), use_container_width=True, hide_index=True)

        evidence_columns = {
            "News": "verified_has_supported_match",
            "UCDP": "verified_has_finalized_ucdp_association",
            "UKMTO": "verified_has_ukmto_conflict_day_context",
            "Close AIS": "causal_ship_name",
            "Direct Sentinel": "sentinel_sentinel_target_visible",
            "Causality confirmed": "causality_established",
        }
        evidence_values = []
        for label, column in evidence_columns.items():
            value = case.get(column, False)
            if label == "Close AIS":
                present = pd.notna(value) and str(value).strip().lower() not in {"", "nan", "none"}
            else:
                present = evidence_is_present(value)
            evidence_values.append([label, int(present)])
        evidence_frame = pd.DataFrame(evidence_values, columns=["Evidence", "Available"])
        fig_evidence = px.bar(
            evidence_frame,
            x="Evidence",
            y="Available",
            color="Available",
            color_continuous_scale=["#E5EAF0", COLORS["green"]],
            text=evidence_frame["Available"].map({0: "Not established", 1: "Available"}),
        )
        fig_evidence.update_layout(height=280, yaxis=dict(range=[0, 1.25], tickvals=[0, 1]), coloraxis_showscale=False, margin=dict(t=40))
        _growth_frames(fig_evidence, steps=8)
        _play_button(fig_evidence, frame_ms=110, transition_ms=260)
        st.plotly_chart(fig_evidence, use_container_width=True)

        st.markdown("#### Analyst feedback")
        feedback_col, notes_col = st.columns([1, 2])
        with feedback_col:
            analyst_label = st.radio("Decision", ["Confirmed", "False alarm", "Uncertain"], index=2, horizontal=True)
        with notes_col:
            analyst_notes = st.text_input("Review notes", placeholder="State the evidence supporting the decision")

        if st.button("Save analyst decision", type="primary"):
            feedback_row = pd.DataFrame(
                [
                    {
                        "event_id": selected_event,
                        "analyst_label": analyst_label,
                        "analyst_notes": analyst_notes,
                        "reviewed_at_utc": datetime.now(timezone.utc),
                    }
                ]
            )
            if DATABASE_URL:
                try:
                    from sqlalchemy import create_engine

                    feedback_row.to_sql(
                        "analyst_feedback",
                        create_engine(DATABASE_URL, pool_pre_ping=True),
                        if_exists="append",
                        index=False,
                    )
                    st.success("Analyst decision saved to PostgreSQL.")
                except Exception as error:
                    st.error(f"Database write failed: {error}")
            else:
                feedback_path = CHECKPOINT_DIR / "analyst_feedback.csv"
                feedback_path.parent.mkdir(parents=True, exist_ok=True)
                feedback_row.to_csv(
                    feedback_path,
                    mode="a",
                    header=not feedback_path.exists(),
                    index=False,
                )
                st.success(f"Analyst decision saved to {feedback_path.name}.")


with audit_tab:
    st.subheader("Data lineage, storage and reproducibility")
    source_status = pd.DataFrame(
        [
            ["NASA satellite detections", satellite_source, count_rows(satellite)],
            ["Multilingual news", news_source, count_rows(news)],
            ["V2 thermal events", events_source, count_rows(events)],
            ["Event-news matches", matches_source, count_rows(matches)],
            ["Final maritime review", priority_source, count_rows(priority)],
            ["Sentinel manual review", sentinel_source, count_rows(sentinel_review)],
            ["SAR event summary", sar_source, count_rows(sar_summary)],
            ["AIS control-day comparison", control_source, count_rows(control_comparison)],
        ],
        columns=["Dataset", "Loaded from", "Rows"],
    )
    st.dataframe(source_status, use_container_width=True, hide_index=True)

    st.subheader("End-to-end analytical pipeline")
    pipeline = pd.DataFrame(
        [
            [1, "Collect", "NASA FIRMS + GDELT", "Raw satellite and multilingual news"],
            [2, "Clean", "Confidence and schema checks", "11,525 reliable detections"],
            [3, "Cluster", "3-day windows + DBSCAN", "3,289 candidate thermal events"],
            [4, "Associate", "Time + location + text", "Source-supported event candidates"],
            [5, "Model", "Logistic Regression + Random Forest", "Risk ranking, not causal declaration"],
            [6, "Validate", "UCDP + UKMTO", "Independent conflict and maritime context"],
            [7, "Constrain", "Water mask + ports + persistent sites", "132 true water events"],
            [8, "Investigate", "AIS + SAR + Sentinel + controls", "7 priority investigations"],
            [9, "Decide", "Human analyst feedback", "3 ordinary context, 4 unconfirmed"],
            [10, "Persist", "PostgreSQL + checkpoints", "Auditable and reproducible outputs"],
        ],
        columns=["Stage", "Action", "Method", "Output"],
    )
    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="caution"><b>Operational boundary</b><br>
        This application supports analyst prioritisation. Production deployment requires scheduled NASA ingestion, a commercially licensed AIS feed, monitored data-quality checks, authenticated users and prospective performance evaluation.</div>
        """,
        unsafe_allow_html=True,
    )


st.caption(
    "Association and model scores indicate investigation priority. "
    "They do not independently prove conflict causation or vessel responsibility."
)


@st.fragment(run_every=auto_refresh_seconds or None)
def refresh_clock():
    now = time.monotonic()
    last_refresh = st.session_state.setdefault("dashboard_refresh_clock", now)
    if auto_refresh_seconds and now - last_refresh >= auto_refresh_seconds:
        st.session_state["dashboard_refresh_clock"] = now
        st.cache_data.clear()
        st.rerun()


refresh_clock()


if auto_play and not st.session_state.get("_autoplayed"):
    _auto_play_animations()
    st.session_state["_autoplayed"] = True
