from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "maritime_conflict_monitoring_clean.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Maritime Conflict Situation Monitoring

## Satellite Thermal Anomalies, Multilingual News and Independent Event Evidence

**Research question:** Can satellite-detected thermal events be associated with conflict-related reporting in time and space, and can the resulting evidence help an analyst prioritise maritime investigations?

This is a teaching reconstruction of the workflow, not the original experiment log. NASA and selected reference datasets are requested from their sources. The news stage requires a separately supplied GDELT export. The original BigQuery SQL and untouched news export are not included; a local CSV recovered from a cleaned checkpoint is not raw data. Read `data/README.md` before running. No checkpoint is loaded by this notebook.

The historical numbers in the conclusion are recorded results, not assertions about this notebook's rerun. Simplified matching rules and model settings can produce different results. Multi-month backtests and the seven-case investigation are documented extensions rather than fully reproduced here.

The system produces investigation candidates. A spatial or temporal association does not prove that armed conflict caused a thermal anomaly.
"""),
    md(r"""
## 1. Project workflow

1. Define the study period and three geographic boxes.
2. Download NASA FIRMS thermal detections.
3. Load the GDELT BigQuery news export and document its origin.
4. Clean and audit both datasets.
5. Convert individual satellite pixels into multi-day thermal events.
6. create multilingual text indicators and match events with news.
7. add port-distance context from the World Port Index.
8. perform exploratory and statistical analysis.
9. compare classification models with a future-month holdout.
10. validate associations against independent UCDP evidence.
11. separate land and water events for maritime review.
12. save reproducible outputs and state the limitations.
"""),
    md("## 2. Libraries and reproducibility settings"),
    code(r"""
from pathlib import Path
import getpass as getpass_module
from urllib.request import urlretrieve
import html
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import BallTree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
"""),
    code(r"""
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 100)
RANDOM_STATE = 42
EARTH_RADIUS_KM = 6371.0088
PROJECT_START = pd.Timestamp("2025-09-01", tz="UTC")
PROJECT_END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = Path("outputs")
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""),
    md("## 3. Study regions"),
    code(r"""
regions = pd.DataFrame({"region": ["gaza_eastern_med_box", "red_sea_yemen_box", "south_ukraine_black_sea_box"], "region_name": ["Gaza / Eastern Mediterranean", "Red Sea / Yemen", "Southern Ukraine / Black Sea"], "west": [33, 40, 29], "south": [30, 12, 45], "east": [36, 50, 38], "north": [34, 20, 49]})
regions["bbox"] = regions[["west", "south", "east", "north"]].astype(str).agg(",".join, axis=1)
regions
"""),
    md(r"""
The boxes deliberately include both land and sea. They are study windows rather than political boundaries. A land–water mask is applied later before maritime interpretation.
"""),
    md("## 4. NASA FIRMS data collection"),
    md(r"""
NASA FIRMS is the primary dataset. Each row returned by the Area API is an individual VIIRS thermal detection. It is not yet an incident or a conflict event.

The study uses the NOAA-20 VIIRS standard product and five-day requests from 1 September 2025 to 28 February 2026. The API key is requested securely at run time and is never written into the notebook.
"""),
    code(r"""
map_key = getpass_module.getpass("NASA FIRMS MAP key: ").strip()
assert map_key, "NASA FIRMS MAP key cannot be empty."
"""),
    code(r"""
request_starts = pd.date_range(PROJECT_START.floor("D"), PROJECT_END.floor("D"), freq="5D")
request_windows = pd.DataFrame({"start_date": request_starts})
request_windows["day_range"] = ((PROJECT_END.floor("D") - request_windows["start_date"]).dt.days + 1).clip(upper=5)
collection_plan = regions.merge(request_windows, how="cross")
collection_plan["url"] = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/" + map_key + "/VIIRS_NOAA20_SP/" + collection_plan["bbox"] + "/" + collection_plan["day_range"].astype(str) + "/" + collection_plan["start_date"].dt.strftime("%Y-%m-%d")
collection_plan[["region", "start_date", "day_range"]].head()
"""),
    code(r"""
satellite_parts = collection_plan["url"].map(pd.read_csv)
df_satellite_raw = pd.concat(satellite_parts.tolist(), keys=collection_plan["region"], names=["region", "source_row"]).reset_index(level="region").reset_index(drop=True)
pd.Series({"raw_detections": df_satellite_raw.shape[0], "columns": df_satellite_raw.shape[1], "first_date": df_satellite_raw["acq_date"].min(), "last_date": df_satellite_raw["acq_date"].max()})
"""),
    md("### 4.1 Clean and audit the satellite data"),
    code(r"""
df_satellite_clean = df_satellite_raw.copy()
df_satellite_clean["acq_date"] = pd.to_datetime(df_satellite_clean["acq_date"], utc=True, errors="coerce")
df_satellite_clean["acq_time"] = df_satellite_clean["acq_time"].astype(str).str.zfill(4)
df_satellite_clean["timestamp"] = pd.to_datetime(df_satellite_clean["acq_date"].dt.strftime("%Y-%m-%d") + " " + df_satellite_clean["acq_time"], format="%Y-%m-%d %H%M", utc=True, errors="coerce")
df_satellite_clean = df_satellite_clean[df_satellite_clean["confidence"].isin(["n", "h"]) & df_satellite_clean["timestamp"].notna()].drop_duplicates().reset_index(drop=True)
"""),
    code(r"""
satellite_audit = pd.Series({"raw_rows": df_satellite_raw.shape[0], "clean_rows": df_satellite_clean.shape[0], "removed_rows": df_satellite_raw.shape[0] - df_satellite_clean.shape[0], "missing_timestamps": df_satellite_clean["timestamp"].isna().sum(), "minimum_frp": df_satellite_clean["frp"].min(), "regions": df_satellite_clean["region"].nunique()})
satellite_audit
"""),
    code(r"""
df_satellite_clean.groupby("region").agg(detections=("frp", "size"), first_detection=("timestamp", "min"), last_detection=("timestamp", "max"), median_frp=("frp", "median")).round(2)
"""),
    md("## 5. GDELT multilingual news collection"),
    md(r"""
The historical news sample was discovered through **GDELT in Google BigQuery**. The recorded sampling design aimed at up to 200 articles per region-month. The exact original query and untouched export have not been recovered for this release, so the historical sample cannot yet be recreated exactly. GDELT is the discovery platform; `publisher` and `url` identify the reporting sources. Do not relabel a checkpoint-derived CSV as an original export.

Export the BigQuery result as `data/raw/gdelt_news.csv`. Required columns:

`region`, `publication_date`, `gdelt_seen_at`, `title_original`, `publisher`, `url`, `language`, `location_mentions`.

The balanced news table supports comparable modelling coverage, but it must not be interpreted as the complete volume of media reporting.
"""),
    code(r"""
news_path = RAW_DIR / "gdelt_news.csv"
assert news_path.exists(), "Export the GDELT BigQuery result to data/raw/gdelt_news.csv before continuing."
df_news_raw = pd.read_csv(news_path)
required_news_columns = {"region", "publication_date", "gdelt_seen_at", "title_original", "publisher", "url", "language", "location_mentions"}
assert required_news_columns.issubset(df_news_raw.columns), f"Missing columns: {required_news_columns.difference(df_news_raw.columns)}"
df_news_raw.shape
"""),
    md("### 5.1 Clean and audit the news data"),
    code(r"""
df_news_clean = df_news_raw.copy()
df_news_clean["publication_date"] = pd.to_datetime(df_news_clean["publication_date"], utc=True, errors="coerce")
df_news_clean["gdelt_seen_at"] = pd.to_datetime(df_news_clean["gdelt_seen_at"], utc=True, errors="coerce")
df_news_clean["title_original"] = df_news_clean["title_original"].astype("string")
df_news_clean["publisher"] = df_news_clean["publisher"].astype("string")
df_news_clean["url"] = df_news_clean["url"].astype("string")
df_news_clean["language"] = df_news_clean["language"].astype("string").fillna("unknown")
df_news_clean["location_mentions"] = df_news_clean["location_mentions"].astype("string").fillna("")
df_news_clean = df_news_clean.dropna(subset=["title_original", "publication_date", "url"]).drop_duplicates("url").reset_index(drop=True)
df_news_clean["title_original"] = df_news_clean["title_original"].astype(str).map(html.unescape).map(html.unescape).str.replace(r"\s+", " ", regex=True).str.strip()
df_news_clean["publisher"] = df_news_clean["publisher"].fillna("unknown").str.replace("\\", "", regex=False).str.lower().str.strip()
df_news_clean["news_id"] = "N" + (df_news_clean.index + 1).astype(str).str.zfill(5)
df_news_clean["publication_day"] = df_news_clean["publication_date"].dt.floor("D")
"""),
    code(r"""
news_audit = pd.Series({"raw_rows": df_news_raw.shape[0], "clean_articles": df_news_clean.shape[0], "unique_publishers": df_news_clean["publisher"].nunique(), "languages": df_news_clean["language"].nunique(), "regions": df_news_clean["region"].nunique(), "first_article": df_news_clean["publication_date"].min(), "last_article": df_news_clean["publication_date"].max()})
news_audit
"""),
    code(r"""
df_news_clean.groupby("region").agg(articles=("news_id", "nunique"), publishers=("publisher", "nunique"), languages=("language", "nunique"))
"""),
    md("## 6. Multilingual headline indicators"),
    code(r"""
title_nlp = df_news_clean[["news_id", "region", "title_original", "language"]].copy()
title_nlp["title_normalized"] = title_nlp["title_original"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
conflict_pattern = r"war|conflict|attack|strike|missile|drone|shelling|explosion|війна|война|атак|удар|ракет|дрон|обстр|قصف|هجوم|صاروخ|غارة|füze|saldırı|çatışma|krieg|angriff|rakete|מלחמה|תקיפה|טיל|guerra|ataque|misil|战争|袭击|导弹"
maritime_pattern = r"ship|vessel|tanker|port|maritime|navy|red sea|black sea|gulf of aden|кораб|судн|порт|морськ|بحر|سفينة|ناقلة|ميناء|gemi|tanker|liman|deniz|schiff|hafen|ספינה|נמל|buque|puerto|船|港口"
title_nlp["direct_conflict_title"] = title_nlp["title_normalized"].str.contains(conflict_pattern, regex=True, na=False)
title_nlp["maritime_title"] = title_nlp["title_normalized"].str.contains(maritime_pattern, regex=True, na=False)
title_nlp.groupby("region")[["direct_conflict_title", "maritime_title"]].agg(["sum", "mean"]).round(3)
"""),
    md("## 7. Convert satellite pixels into candidate thermal events"),
    md(r"""
Nearby detections are grouped within fixed three-day windows using DBSCAN and a seven-kilometre Haversine radius. `min_samples=1` retains isolated detections. The resulting rows are candidate thermal events, not confirmed incidents.
"""),
    code(r"""
thermal_work = df_satellite_clean.copy()
thermal_work["window_id"] = ((thermal_work["timestamp"] - PROJECT_START).dt.total_seconds() // (3 * 24 * 60 * 60)).astype(int)
thermal_work["window_start"] = PROJECT_START + pd.to_timedelta(thermal_work["window_id"] * 3, unit="D")
thermal_clustered = thermal_work.groupby(["region", "window_start"], group_keys=False).apply(lambda frame: frame.assign(cluster=DBSCAN(eps=7 / EARTH_RADIUS_KM, min_samples=1, metric="haversine").fit_predict(np.radians(frame[["latitude", "longitude"]])))).reset_index(drop=True)
thermal_clustered["event_id"] = thermal_clustered["region"] + "_" + thermal_clustered["window_start"].dt.strftime("%Y%m%d") + "_" + thermal_clustered["cluster"].astype(str)
"""),
    code(r"""
thermal_events = thermal_clustered.groupby(["region", "event_id"], as_index=False).agg(window_start=("window_start", "first"), start_time=("timestamp", "min"), end_time=("timestamp", "max"), latitude=("latitude", "mean"), longitude=("longitude", "mean"), pixel_count=("frp", "size"), total_frp=("frp", "sum"), max_frp=("frp", "max"), max_brightness=("bright_ti4", "max"), day_detections=("daynight", lambda values: values.eq("D").sum()), night_detections=("daynight", lambda values: values.eq("N").sum()))
thermal_events["event_date"] = thermal_events["start_time"].dt.floor("D")
thermal_events["duration_hours"] = (thermal_events["end_time"] - thermal_events["start_time"]).dt.total_seconds().div(3600)
thermal_events["day_ratio"] = thermal_events["day_detections"].div(thermal_events["pixel_count"])
thermal_events = thermal_events[["event_id", "region", "window_start", "event_date", "start_time", "end_time", "duration_hours", "latitude", "longitude", "pixel_count", "total_frp", "max_frp", "max_brightness", "day_detections", "night_detections", "day_ratio"]]
pd.Series({"satellite_detections_retained": thermal_clustered.shape[0], "candidate_thermal_events": thermal_events.shape[0]})
"""),
    md("## 8. Spatial, temporal and textual event–news matching"),
    code(r"""
news_locations = df_news_clean[["news_id", "region", "location_mentions"]].assign(location_record=lambda frame: frame["location_mentions"].str.split(";")).explode("location_record")
location_parts = news_locations["location_record"].str.split("#", expand=True)
news_locations["location_type"] = pd.to_numeric(location_parts[0], errors="coerce")
news_locations["location_name"] = location_parts[1]
news_locations["location_latitude"] = pd.to_numeric(location_parts[5], errors="coerce")
news_locations["location_longitude"] = pd.to_numeric(location_parts[6], errors="coerce")
news_locations = news_locations.dropna(subset=["location_latitude", "location_longitude"])
news_locations = news_locations.merge(regions[["region", "west", "south", "east", "north"]], on="region", how="left")
news_locations = news_locations[news_locations["location_longitude"].between(news_locations["west"], news_locations["east"]) & news_locations["location_latitude"].between(news_locations["south"], news_locations["north"]) & news_locations["location_type"].ne(1)].drop_duplicates(["news_id", "location_latitude", "location_longitude"])
"""),
    code(r"""
event_days = thermal_events[["event_id", "region", "event_date"]].assign(match_day=lambda frame: frame["event_date"].map(lambda day: pd.date_range(day, day + pd.Timedelta(days=7), freq="D"))).explode("match_day")
candidate_matches = event_days.merge(df_news_clean, left_on=["region", "match_day"], right_on=["region", "publication_day"], how="inner")
candidate_matches["days_after_event"] = (candidate_matches["publication_day"] - candidate_matches["event_date"]).dt.days
spatial_matches = candidate_matches.merge(thermal_events[["event_id", "latitude", "longitude"]], on="event_id", how="left").merge(news_locations[["news_id", "region", "location_name", "location_latitude", "location_longitude"]], on=["news_id", "region"], how="inner")
"""),
    code(r"""
event_lat, event_lon = np.radians(spatial_matches["latitude"]), np.radians(spatial_matches["longitude"])
news_lat, news_lon = np.radians(spatial_matches["location_latitude"]), np.radians(spatial_matches["location_longitude"])
haversine_value = np.sin((news_lat - event_lat) / 2) ** 2 + np.cos(event_lat) * np.cos(news_lat) * np.sin((news_lon - event_lon) / 2) ** 2
spatial_matches["distance_km"] = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(haversine_value.clip(0, 1)))
event_news_matches = spatial_matches[spatial_matches["distance_km"].le(50)].sort_values("distance_km").drop_duplicates(["event_id", "news_id"]).merge(title_nlp[["news_id", "direct_conflict_title", "maritime_title"]], on="news_id", how="left")
"""),
    code(r"""
event_news_matches["distance_points"] = np.select([event_news_matches["distance_km"].le(10), event_news_matches["distance_km"].le(25), event_news_matches["distance_km"].le(50)], [3, 2, 1], default=0)
event_news_matches["time_points"] = np.select([event_news_matches["days_after_event"].le(1), event_news_matches["days_after_event"].le(3), event_news_matches["days_after_event"].le(7)], [3, 2, 1], default=0)
event_news_matches["text_points"] = event_news_matches["direct_conflict_title"].fillna(False).astype(int) * 2
event_news_matches["maritime_points"] = event_news_matches["maritime_title"].fillna(False).astype(int)
event_news_matches["match_score"] = event_news_matches[["distance_points", "time_points", "text_points", "maritime_points"]].sum(axis=1)
event_news_matches["match_confidence"] = np.select([event_news_matches["match_score"].ge(7), event_news_matches["match_score"].ge(5)], ["high", "medium"], default="contextual")
event_news_matches["match_confidence"].value_counts()
"""),
    code(r"""
event_evidence = event_news_matches.groupby("event_id", as_index=False).agg(news_count=("news_id", "nunique"), publisher_count=("publisher", "nunique"), high_article_count=("match_confidence", lambda values: values.eq("high").sum()), medium_article_count=("match_confidence", lambda values: values.eq("medium").sum()), contextual_article_count=("match_confidence", lambda values: values.eq("contextual").sum()), direct_conflict_news=("direct_conflict_title", "sum"), maritime_news=("maritime_title", "sum"), nearest_news_km=("distance_km", "min"), earliest_news_days=("days_after_event", "min"))
supported_pairs = event_news_matches[event_news_matches["match_confidence"].isin(["high", "medium"])]
supported_sources = supported_pairs.groupby("event_id", as_index=False).agg(supported_news_count=("news_id", "nunique"), supported_publisher_count=("publisher", "nunique"))
thermal_events_final = thermal_events.merge(event_evidence, on="event_id", how="left").merge(supported_sources, on="event_id", how="left")
evidence_count_columns = ["news_count", "publisher_count", "high_article_count", "medium_article_count", "contextual_article_count", "direct_conflict_news", "maritime_news", "supported_news_count", "supported_publisher_count"]
thermal_events_final[evidence_count_columns] = thermal_events_final[evidence_count_columns].fillna(0).astype(int)
thermal_events_final["event_evidence"] = np.select([thermal_events_final["high_article_count"].ge(1) & thermal_events_final["supported_publisher_count"].ge(2), thermal_events_final["high_article_count"].ge(1) | (thermal_events_final["supported_news_count"].ge(2) & thermal_events_final["supported_publisher_count"].ge(2)), thermal_events_final["news_count"].ge(1)], ["high", "medium", "contextual"], default="no_match")
thermal_events_final["has_supported_match"] = thermal_events_final["event_evidence"].isin(["high", "medium"])
thermal_events_final["event_evidence"].value_counts()
"""),
    md("## 9. World Port Index integration"),
    code(r"""
wpi_url = "https://gis.unocha.org/server/rest/services/Hosted/global_world_seaport_index_202511/FeatureServer/0/query"
wpi_offsets = pd.Series([0, 2000], name="offset")
wpi_pages = wpi_offsets.map(lambda offset: requests.get(wpi_url, params={"where": "1=1", "outFields": "*", "returnGeometry": "true", "outSR": 4326, "f": "json", "resultRecordCount": 2000, "resultOffset": int(offset)}, timeout=60).json()["features"])
ports_raw = pd.json_normalize(wpi_pages.explode().dropna().tolist())
ports = ports_raw[["attributes.port_name", "attributes.country", "attributes.latitude", "attributes.longitude"]].copy()
ports.columns = ["port_name", "port_country", "port_latitude", "port_longitude"]
ports[["port_latitude", "port_longitude"]] = ports[["port_latitude", "port_longitude"]].apply(pd.to_numeric, errors="coerce")
ports = ports.dropna(subset=["port_latitude", "port_longitude"]).drop_duplicates().reset_index(drop=True)
ports.shape
"""),
    code(r"""
port_tree = BallTree(np.radians(ports[["port_latitude", "port_longitude"]]), metric="haversine")
nearest_distance, nearest_index = port_tree.query(np.radians(thermal_events_final[["latitude", "longitude"]]), k=1)
thermal_events_final["nearest_port_km"] = nearest_distance[:, 0] * EARTH_RADIUS_KM
thermal_events_final["nearest_port_name"] = ports.iloc[nearest_index[:, 0]]["port_name"].to_numpy()
thermal_events_final["nearest_port_country"] = ports.iloc[nearest_index[:, 0]]["port_country"].to_numpy()
thermal_events_final["port_zone"] = pd.cut(thermal_events_final["nearest_port_km"], bins=[0, 10, 25, 75, np.inf], labels=["immediate_port_area", "port_vicinity", "coastal_or_port_approach", "inland_or_distant"], include_lowest=True)
thermal_events_final["model_month"] = thermal_events_final["event_date"].dt.strftime("%Y-%m")
thermal_events_final[["event_id", "nearest_port_name", "nearest_port_km", "port_zone"]].head()
"""),
    md("## 10. Exploratory and statistical analysis"),
    code(r"""
regional_summary = thermal_events_final.groupby("region").agg(total_events=("event_id", "size"), supported_events=("has_supported_match", "sum"), median_frp=("total_frp", "median"), median_port_distance_km=("nearest_port_km", "median"))
regional_summary["supported_rate"] = regional_summary["supported_events"].div(regional_summary["total_events"])
regional_summary.round(3)
"""),
    code(r"""
monthly_summary = thermal_events_final.groupby(["region", "model_month"], as_index=False).agg(total_events=("event_id", "size"), supported_events=("has_supported_match", "sum"), median_frp=("total_frp", "median"))
monthly_summary["supported_rate"] = monthly_summary["supported_events"].div(monthly_summary["total_events"])
monthly_summary.round(3)
"""),
    code(r"""
matched_brightness = thermal_events_final.loc[thermal_events_final["has_supported_match"], "max_brightness"]
unmatched_brightness = thermal_events_final.loc[~thermal_events_final["has_supported_match"], "max_brightness"]
u_statistic, p_value = mannwhitneyu(matched_brightness, unmatched_brightness, alternative="greater")
rank_biserial = 2 * u_statistic / (matched_brightness.size * unmatched_brightness.size) - 1
pd.Series({"U statistic": u_statistic, "p-value": p_value, "rank-biserial effect": rank_biserial}).round(4)
"""),
    md(r"""
The Mann–Whitney result tests association, not causation. With a large sample, a very small p-value can coexist with a small practical effect, so the rank-biserial effect must be reported beside the p-value.
"""),
    md("## 11. Time-based classification"),
    md(r"""
September 2025–January 2026 form the training period; February 2026 is an unseen future test month. The target is **source-supported news association**, not verified conflict occurrence. Logistic Regression and Decision Tree meet the assignment requirement; Random Forest is included as an additional ensemble benchmark.
"""),
    code(r"""
train_data = thermal_events_final[thermal_events_final["model_month"].ne("2026-02")].copy()
test_data = thermal_events_final[thermal_events_final["model_month"].eq("2026-02")].copy()
numeric_features = ["pixel_count", "total_frp", "max_frp", "max_brightness", "duration_hours", "day_ratio", "nearest_port_km"]
categorical_features = ["region", "port_zone"]
model_features = numeric_features + categorical_features
X_train, X_test = train_data[model_features], test_data[model_features]
y_train, y_test = train_data["has_supported_match"].astype(int), test_data["has_supported_match"].astype(int)
preprocessor = ColumnTransformer([("numeric", StandardScaler(), numeric_features), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)])
"""),
    code(r"""
logistic_model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))])
tree_model = Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE))])
forest_model = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1))])
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)
"""),
    code(r"""
def evaluate_model(name, model):
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {"Model": name, "Accuracy": accuracy_score(y_test, prediction), "Precision": precision_score(y_test, prediction, zero_division=0), "Recall": recall_score(y_test, prediction, zero_division=0), "F1": f1_score(y_test, prediction, zero_division=0), "ROC-AUC": roc_auc_score(y_test, probability), "PR-AUC": average_precision_score(y_test, probability)}

model_results = pd.DataFrame([evaluate_model("Logistic Regression", logistic_model), evaluate_model("Decision Tree", tree_model), evaluate_model("Random Forest", forest_model)]).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
model_results.round(3)
"""),
    code(r"""
forest_prediction = forest_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, forest_prediction).ravel()
pd.Series({"true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp, "test_positive_rate": y_test.mean()}).round(3)
"""),
    md("## 12. Independent validation with UCDP"),
    md(r"""
UCDP supplies separately coded event evidence. Its underlying reporting can overlap the news sample, so evidence sources are not guaranteed to be statistically independent. Finalized 2025 records and provisional 2026 candidate records are kept separate. Proximity does not establish the cause of a heat source.
"""),
    code(r"""
ucdp_dir = EXTERNAL_DIR / "ucdp"
ucdp_dir.mkdir(parents=True, exist_ok=True)
urlretrieve("https://ucdp.uu.se/downloads/ged/ged261-csv.zip", ucdp_dir / "ged261-csv.zip")
urlretrieve("https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_01_26_06.csv", ucdp_dir / "GEDEvent_v26_01_26_06.csv")
df_ucdp_2025_raw = pd.read_csv(ucdp_dir / "ged261-csv.zip", compression="zip", low_memory=False)
df_ucdp_2026_raw = pd.read_csv(ucdp_dir / "GEDEvent_v26_01_26_06.csv", low_memory=False)
pd.Series({"finalized_rows": df_ucdp_2025_raw.shape[0], "candidate_2026_rows": df_ucdp_2026_raw.shape[0]})
"""),
    code(r"""
ucdp_columns = ["id", "code_status", "type_of_violence", "conflict_name", "dyad_name", "side_a", "side_b", "number_of_sources", "where_prec", "where_coordinates", "where_description", "adm_1", "adm_2", "country", "event_clarity", "date_prec", "date_start", "date_end", "latitude", "longitude", "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown", "best", "high", "low"]
ucdp_2025 = df_ucdp_2025_raw[ucdp_columns].assign(release_status="finalized_ged")
ucdp_2026 = df_ucdp_2026_raw[ucdp_columns].assign(release_status="candidate_2026")
df_ucdp = pd.concat([ucdp_2025, ucdp_2026], ignore_index=True)
df_ucdp[["date_start", "date_end"]] = df_ucdp[["date_start", "date_end"]].apply(pd.to_datetime, utc=True)
df_ucdp[["latitude", "longitude"]] = df_ucdp[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
df_ucdp = df_ucdp[df_ucdp["date_start"].le(PROJECT_END) & df_ucdp["date_end"].ge(PROJECT_START)].copy()
df_ucdp = df_ucdp.merge(regions[["region", "west", "south", "east", "north"]], how="cross")
df_ucdp_study = df_ucdp[df_ucdp["longitude"].between(df_ucdp["west"], df_ucdp["east"]) & df_ucdp["latitude"].between(df_ucdp["south"], df_ucdp["north"])].drop(columns=["west", "south", "east", "north"]).rename(columns={"id": "ucdp_event_id"})
"""),
    code(r"""
high_finalized = df_ucdp_study["release_status"].eq("finalized_ged") & df_ucdp_study["code_status"].eq("Clear") & df_ucdp_study["date_prec"].eq(1) & df_ucdp_study["where_prec"].eq(1) & df_ucdp_study["event_clarity"].eq(1)
strong_finalized = df_ucdp_study["release_status"].eq("finalized_ged") & df_ucdp_study["code_status"].eq("Clear") & df_ucdp_study["date_prec"].le(2) & df_ucdp_study["where_prec"].le(2) & df_ucdp_study["event_clarity"].eq(1)
provisional_2026 = df_ucdp_study["release_status"].eq("candidate_2026") & df_ucdp_study["code_status"].eq("Clear") & df_ucdp_study["date_prec"].le(2) & df_ucdp_study["where_prec"].le(2) & df_ucdp_study["event_clarity"].eq(1)
df_ucdp_study["ucdp_evidence_quality"] = np.select([high_finalized, strong_finalized, provisional_2026], ["high_finalized", "strong_finalized", "provisional_2026"], default="excluded")
df_ucdp_matchable = df_ucdp_study[df_ucdp_study["ucdp_evidence_quality"].ne("excluded")].copy()
df_ucdp_matchable["match_day"] = df_ucdp_matchable.apply(lambda row: pd.date_range(row["date_start"] - pd.Timedelta(days=3), row["date_end"] + pd.Timedelta(days=3), freq="D"), axis=1)
ucdp_match_days = df_ucdp_matchable.explode("match_day", ignore_index=True)
df_ucdp_study["ucdp_evidence_quality"].value_counts()
"""),
    code(r"""
thermal_ucdp = thermal_events_final.merge(ucdp_match_days, left_on=["region", "event_date"], right_on=["region", "match_day"], how="inner", suffixes=("_thermal", "_ucdp"))
thermal_lat, thermal_lon = np.radians(thermal_ucdp["latitude_thermal"]), np.radians(thermal_ucdp["longitude_thermal"])
ucdp_lat, ucdp_lon = np.radians(thermal_ucdp["latitude_ucdp"]), np.radians(thermal_ucdp["longitude_ucdp"])
thermal_ucdp["ucdp_distance_km"] = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt((np.sin((ucdp_lat - thermal_lat) / 2) ** 2 + np.cos(thermal_lat) * np.cos(ucdp_lat) * np.sin((ucdp_lon - thermal_lon) / 2) ** 2).clip(0, 1)))
thermal_ucdp["days_from_ucdp_interval"] = (thermal_ucdp["date_start"] - thermal_ucdp["event_date"]).dt.days.clip(lower=0) + (thermal_ucdp["event_date"] - thermal_ucdp["date_end"]).dt.days.clip(lower=0)
thermal_ucdp = thermal_ucdp[thermal_ucdp["ucdp_distance_km"].le(50)].copy()
thermal_ucdp["ucdp_association_level"] = np.select([thermal_ucdp["ucdp_evidence_quality"].eq("high_finalized") & thermal_ucdp["days_from_ucdp_interval"].le(1) & thermal_ucdp["ucdp_distance_km"].le(15), thermal_ucdp["ucdp_evidence_quality"].isin(["high_finalized", "strong_finalized"]) & thermal_ucdp["days_from_ucdp_interval"].le(1) & thermal_ucdp["ucdp_distance_km"].le(25), thermal_ucdp["ucdp_evidence_quality"].eq("provisional_2026") & thermal_ucdp["days_from_ucdp_interval"].le(1) & thermal_ucdp["ucdp_distance_km"].le(25)], ["high_finalized_association", "supported_finalized_proximity", "provisional_2026_proximity"], default="contextual_proximity")
thermal_ucdp[["event_id", "ucdp_event_id", "ucdp_association_level"]].drop_duplicates().groupby("ucdp_association_level").agg(match_pairs=("ucdp_event_id", "size"), thermal_events=("event_id", "nunique"), ucdp_events=("ucdp_event_id", "nunique"))
"""),
    code(r"""
association_priority = {"high_finalized_association": 4, "supported_finalized_proximity": 3, "provisional_2026_proximity": 2, "contextual_proximity": 1}
best_ucdp = thermal_ucdp.assign(association_priority=lambda frame: frame["ucdp_association_level"].map(association_priority)).sort_values(["event_id", "association_priority", "ucdp_distance_km"], ascending=[True, False, True]).drop_duplicates("event_id")
thermal_events_verified = thermal_events_final.merge(best_ucdp[["event_id", "ucdp_event_id", "ucdp_association_level", "ucdp_distance_km", "release_status", "conflict_name"]], on="event_id", how="left")
thermal_events_verified["ucdp_association_level"] = thermal_events_verified["ucdp_association_level"].fillna("no_ucdp_proximity")
thermal_events_verified["has_finalized_ucdp_association"] = thermal_events_verified["ucdp_association_level"].isin(["high_finalized_association", "supported_finalized_proximity"])
thermal_events_verified["has_provisional_ucdp_association"] = thermal_events_verified["ucdp_association_level"].eq("provisional_2026_proximity")
pd.crosstab(thermal_events_verified["has_supported_match"], thermal_events_verified["has_finalized_ucdp_association"], margins=True)
"""),
    md("## 13. Maritime land–water separation"),
    code(r"""
import geopandas as gpd
natural_earth_dir = EXTERNAL_DIR / "natural_earth"
natural_earth_dir.mkdir(parents=True, exist_ok=True)
land_zip = natural_earth_dir / "ne_10m_land.zip"
urlretrieve("https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip", land_zip)
land_polygons = gpd.read_file(land_zip)
thermal_events_geo = gpd.GeoDataFrame(thermal_events_verified.copy(), geometry=gpd.points_from_xy(thermal_events_verified["longitude"], thermal_events_verified["latitude"]), crs="EPSG:4326")
land_geometry = land_polygons.geometry.union_all()
thermal_events_geo["is_land"] = thermal_events_geo.geometry.within(land_geometry)
thermal_events_geo["surface_type"] = np.where(thermal_events_geo["is_land"], "land", "water")
thermal_events_geo["maritime_location_class"] = np.select([~thermal_events_geo["is_land"] & thermal_events_geo["nearest_port_km"].le(50), ~thermal_events_geo["is_land"] & thermal_events_geo["nearest_port_km"].gt(50), thermal_events_geo["is_land"] & thermal_events_geo["nearest_port_km"].le(25)], ["water_near_port", "offshore_water", "land_near_port"], default="inland_land")
pd.crosstab(thermal_events_geo["maritime_location_class"], thermal_events_geo["region"], margins=True)
"""),
    md(r"""
Only water-based events should enter the core maritime investigation queue. AIS, Sentinel-1/2, SAR and official UKMTO/JMIC records were used as analyst-review extensions for the final shortlist. Those credentialed or manually audited investigations are kept separate from the reproducible assignment pipeline so that proximity is not mistaken for causal attribution.
"""),
    md("## 14. Final dashboard"),
    code(r"""
region_names = regions.set_index("region")["region_name"].to_dict()
dashboard_events = thermal_events_final.assign(region_name=lambda frame: frame["region"].map(region_names), support_status=lambda frame: frame["has_supported_match"].map({True: "Source-supported", False: "No source support"}))
dashboard_monthly = monthly_summary.assign(region_name=lambda frame: frame["region"].map(region_names))
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
sns.scatterplot(data=dashboard_events, x="longitude", y="latitude", hue="support_status", palette={"No source support": "lightgray", "Source-supported": "crimson"}, s=16, alpha=.65, ax=axes[0, 0])
sns.lineplot(data=dashboard_monthly, x="model_month", y="total_events", hue="region_name", marker="o", ax=axes[0, 1])
sns.barplot(data=regional_summary.reset_index().assign(region_name=lambda frame: frame["region"].map(region_names)), x="supported_rate", y="region_name", hue="region_name", legend=False, ax=axes[1, 0])
sns.barplot(data=model_results.melt(id_vars="Model", value_vars=["Accuracy", "F1", "PR-AUC"], var_name="Metric", value_name="Score"), x="Metric", y="Score", hue="Model", ax=axes[1, 1])
axes[0, 0].set_title("Candidate Thermal Events and News Evidence")
axes[0, 1].set_title("Monthly Candidate Thermal Events")
axes[1, 0].set_title("Regional Source-Supported Rate")
axes[1, 1].set_title("Future-Month Model Performance")
fig.suptitle("Satellite Thermal Anomaly and Conflict-Evidence Monitoring", fontsize=18, fontweight="bold")
fig.text(.5, .01, "Associations prioritise analyst review; they do not confirm conflict causation.", ha="center")
fig.tight_layout(rect=[0, .03, 1, .96])
fig.savefig(OUTPUT_DIR / "dashboard.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("## 15. Save derived outputs"),
    md(r"""
These files are outputs of the reproducible pipeline. They make later dashboard or database work faster, but they are not the starting point of this notebook.
"""),
    code(r"""
df_satellite_clean.to_csv(OUTPUT_DIR / "satellite_detections_clean.csv", index=False)
df_news_clean.to_csv(OUTPUT_DIR / "news_articles_clean.csv", index=False)
thermal_events_geo.drop(columns="geometry").to_csv(OUTPUT_DIR / "thermal_events_verified.csv", index=False)
event_news_matches.to_csv(OUTPUT_DIR / "event_news_matches.csv", index=False)
model_results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
pd.Series({"satellite_detections": df_satellite_clean.shape[0], "news_articles": df_news_clean.shape[0], "thermal_events": thermal_events_final.shape[0], "event_news_pairs": event_news_matches.shape[0], "water_events": thermal_events_geo["surface_type"].eq("water").sum()})
"""),
    md(r"""
## 16. Conclusion

The original exploratory study recorded 11,525 cleaned FIRMS detections, 3,289 multi-day candidate thermal events and 3,589 news articles. These are historical findings; compare the actual outputs above when running this teaching reconstruction. Complete reproduction of the original study requires the missing query/export and original experiment configuration.

Independent UCDP comparison showed that news-supported events were more likely to occur near coded conflict events, supporting the existence of a meaningful association. The modelling experiments also showed that geography and port context were more stable predictors than absolute thermal intensity. The system is therefore best interpreted as an explainable analyst-triage tool.

It does not autonomously identify conflict and it does not establish causality. Unconfirmed means that additional evidence is required; it does not mean that an event was peaceful or irrelevant.

### Main limitations

- GDELT coverage reflects media availability and a balanced sampling design.
- UCDP finalized and provisional releases have different evidence status.
- Broad study boxes contain persistent industrial, agricultural and natural heat sources.
- Spatial and temporal proximity can generate plausible but incorrect associations.
- The final test month contains few positive events, so uncertainty remains large.

### Next development steps

- extend official-event history to 12–24 months;
- add verified negative classes such as refinery heat, gas flaring, wildfires and vessel accidents;
- integrate licensed AIS and higher-resolution Sentinel features;
- evaluate repeated out-of-time and out-of-region backtests;
- deploy the pipeline with PostgreSQL/PostGIS, scheduled FIRMS ingestion and analyst feedback.
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

nbf.write(nb, OUT)
print(OUT)
