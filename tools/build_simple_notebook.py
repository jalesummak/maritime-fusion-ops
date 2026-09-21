from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "maritime_conflict_monitoring_simple.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# Maritime Conflict Situation Monitoring

## Satellite Thermal Events and Conflict-News Associations

**Question:** Can thermal events observed by satellite be associated with conflict-related news by date and location?

This teaching reconstruction calls NASA directly and expects a separately supplied GDELT news export. It does not load checkpoints. The original BigQuery query and untouched export are missing from this release; see `data/README.md`. A CSV recovered from already-cleaned data is not raw input. Historical totals in the conclusion are recorded findings, not guaranteed outputs of these simplified rules.
"""),
    md("""
## Project flow

1. Collect NASA FIRMS thermal detections.
2. Load the GDELT BigQuery news export.
3. Clean both datasets.
4. Group nearby satellite detections into thermal events.
5. Match events and news by region, date, location and headline evidence.
6. Add nearest-port context.
7. Perform EDA and a statistical test.
8. Train and compare classification models.
9. Present the final dashboard and conclusion.
"""),
    md("## 1. Libraries"),
    code("""
from pathlib import Path
import getpass as getpass_module
import html
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import BallTree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
"""),
    md("## 2. Project settings"),
    code('START_DATE = pd.Timestamp("2025-09-01", tz="UTC")'),
    code('END_DATE = pd.Timestamp("2026-02-28", tz="UTC")'),
    code('EARTH_RADIUS_KM = 6371.0088'),
    code('PROJECT_DIR = Path.cwd()'),
    code('DATA_DIR = PROJECT_DIR / "data" / "raw"'),
    code('OUTPUT_DIR = PROJECT_DIR / "outputs"'),
    code('DATA_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)'),
    md("## 3. Study regions"),
    code('regions = pd.DataFrame({"region":["gaza_eastern_med_box","red_sea_yemen_box","south_ukraine_black_sea_box"],"region_name":["Gaza / Eastern Mediterranean","Red Sea / Yemen","Southern Ukraine / Black Sea"],"west":[33,40,29],"south":[30,12,45],"east":[36,50,38],"north":[34,20,49]})'),
    code('regions["bbox"] = regions[["west","south","east","north"]].astype(str).agg(",".join, axis=1)'),
    code('regions'),
    md("""
These rectangles are study areas. They include land and water and are not administrative borders.
"""),
    md("## 4. Collect NASA FIRMS data"),
    code('map_key = getpass_module.getpass("NASA FIRMS MAP key: ").strip()'),
    code('request_plan = pd.DataFrame({"start_date":pd.date_range(START_DATE, END_DATE, freq="5D")})'),
    code('request_plan["day_range"] = ((END_DATE.floor("D") - request_plan["start_date"]).dt.days + 1).clip(upper=5)'),
    code('collection_plan = regions.merge(request_plan, how="cross")'),
    code('collection_plan["url"] = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/" + map_key + "/VIIRS_NOAA20_SP/" + collection_plan["bbox"] + "/" + collection_plan["day_range"].astype(str) + "/" + collection_plan["start_date"].dt.strftime("%Y-%m-%d")'),
    code('satellite_parts = collection_plan["url"].map(pd.read_csv)'),
    code('df_satellite_raw = pd.concat(satellite_parts.tolist(), keys=collection_plan["region"].tolist(), names=["region","source_row"]).reset_index(level="region").reset_index(drop=True)'),
    code('df_satellite_raw.shape'),
    md("## 5. Clean NASA FIRMS data"),
    code('df_satellite_clean = df_satellite_raw.copy()'),
    code('df_satellite_clean["acq_date"] = pd.to_datetime(df_satellite_clean["acq_date"], utc=True, errors="coerce")'),
    code('df_satellite_clean["acq_time"] = df_satellite_clean["acq_time"].astype(str).str.zfill(4)'),
    code('df_satellite_clean["timestamp"] = pd.to_datetime(df_satellite_clean["acq_date"].dt.strftime("%Y-%m-%d") + " " + df_satellite_clean["acq_time"], format="%Y-%m-%d %H%M", utc=True, errors="coerce")'),
    code('df_satellite_clean = df_satellite_clean[df_satellite_clean["confidence"].isin(["n","h"]) & df_satellite_clean["timestamp"].notna()].drop_duplicates().reset_index(drop=True)'),
    code('pd.Series({"raw_detections":df_satellite_raw.shape[0],"clean_detections":df_satellite_clean.shape[0],"removed_detections":df_satellite_raw.shape[0]-df_satellite_clean.shape[0]})'),
    code('df_satellite_clean.groupby("region").agg(detections=("frp","size"),median_frp=("frp","median"),maximum_frp=("frp","max")).round(2)'),
    md("""
Low-confidence observations and invalid timestamps are removed. High FRP values are retained because they may represent real extreme heat rather than data errors.
"""),
    md("## 6. Load the GDELT BigQuery news export"),
    md("""
The historical sample came from GDELT through BigQuery. This step requires your documented compatible export at `data/raw/gdelt_news.csv`. The original SQL and untouched historical export are not included. Preserve the publisher and URL fields, record your query and retrieval date, and report your own rerun results.
"""),
    code('df_news_raw = pd.read_csv(DATA_DIR / "gdelt_news.csv")'),
    code('df_news_raw.shape'),
    md("## 7. Clean news data"),
    code('df_news_clean = df_news_raw.copy()'),
    code('df_news_clean["publication_date"] = pd.to_datetime(df_news_clean["publication_date"], utc=True, errors="coerce")'),
    code('df_news_clean["title_original"] = df_news_clean["title_original"].astype("string")'),
    code('df_news_clean["publisher"] = df_news_clean["publisher"].astype("string").fillna("unknown")'),
    code('df_news_clean["location_mentions"] = df_news_clean["location_mentions"].astype("string").fillna("")'),
    code('df_news_clean = df_news_clean.dropna(subset=["title_original","publication_date","url"]).drop_duplicates("url").reset_index(drop=True)'),
    code('df_news_clean["title_original"] = df_news_clean["title_original"].astype(str).map(html.unescape).str.replace(r"\\s+"," ",regex=True).str.strip()'),
    code('df_news_clean["news_id"] = "N" + (df_news_clean.index + 1).astype(str).str.zfill(5)'),
    code('df_news_clean["publication_day"] = df_news_clean["publication_date"].dt.floor("D")'),
    code('pd.Series({"raw_articles":df_news_raw.shape[0],"clean_articles":df_news_clean.shape[0],"publishers":df_news_clean["publisher"].nunique(),"languages":df_news_clean["language"].nunique()})'),
    code('df_news_clean.groupby("region").agg(articles=("news_id","nunique"),publishers=("publisher","nunique"))'),
    md("## 8. Simple multilingual headline analysis"),
    code('title_nlp = df_news_clean[["news_id","title_original"]].copy()'),
    code('title_nlp["title_normalized"] = title_nlp["title_original"].str.casefold()'),
    code('conflict_words = r"war|conflict|attack|strike|missile|drone|shelling|explosion|війна|война|атак|удар|ракет|дрон|قصف|هجوم|صاروخ|غارة|saldırı|çatışma|krieg|angriff|מלחמה|תקיפה|guerra|ataque|战争|袭击"'),
    code('maritime_words = r"ship|vessel|tanker|port|maritime|navy|red sea|black sea|кораб|судн|порт|بحر|سفينة|ميناء|gemi|liman|deniz|schiff|hafen|ספינה|נמל|buque|puerto|船|港口"'),
    code('title_nlp["direct_conflict_title"] = title_nlp["title_normalized"].str.contains(conflict_words, regex=True, na=False)'),
    code('title_nlp["maritime_title"] = title_nlp["title_normalized"].str.contains(maritime_words, regex=True, na=False)'),
    code('title_nlp[["direct_conflict_title","maritime_title"]].sum()'),
    md("## 9. Create thermal events with DBSCAN"),
    md("""
DBSCAN uses a seven-kilometre neighbourhood within fixed three-day windows. Chains of neighbours can create a cluster wider than seven kilometres; window boundaries can split a continuous event. These groups are candidate thermal events, not conflict incidents.
"""),
    code('thermal_work = df_satellite_clean.copy()'),
    code('thermal_work["window_id"] = ((thermal_work["timestamp"] - START_DATE).dt.total_seconds() // 259200).astype(int)'),
    code('thermal_work["window_start"] = START_DATE + pd.to_timedelta(thermal_work["window_id"] * 3, unit="D")'),
    code('thermal_clustered = thermal_work.groupby(["region","window_start"],group_keys=False).apply(lambda data:data.assign(cluster=DBSCAN(eps=7/EARTH_RADIUS_KM,min_samples=1,metric="haversine").fit_predict(np.radians(data[["latitude","longitude"]])))).reset_index(drop=True)'),
    code('thermal_clustered["event_id"] = thermal_clustered["region"] + "_" + thermal_clustered["window_start"].dt.strftime("%Y%m%d") + "_" + thermal_clustered["cluster"].astype(str)'),
    code('thermal_events = thermal_clustered.groupby(["region","event_id"],as_index=False).agg(start_time=("timestamp","min"),end_time=("timestamp","max"),latitude=("latitude","mean"),longitude=("longitude","mean"),pixel_count=("frp","size"),total_frp=("frp","sum"),max_frp=("frp","max"),max_brightness=("bright_ti4","max"),day_detections=("daynight",lambda values:values.eq("D").sum()))'),
    code('thermal_events["event_date"] = thermal_events["start_time"].dt.floor("D")'),
    code('thermal_events["duration_hours"] = (thermal_events["end_time"] - thermal_events["start_time"]).dt.total_seconds().div(3600)'),
    code('thermal_events["day_ratio"] = thermal_events["day_detections"].div(thermal_events["pixel_count"])'),
    code('pd.Series({"clean_detections":df_satellite_clean.shape[0],"thermal_events":thermal_events.shape[0]})'),
    md("## 10. Prepare news locations"),
    code('news_locations = df_news_clean[["news_id","region","location_mentions"]].assign(location_record=lambda data:data["location_mentions"].str.split(";")).explode("location_record")'),
    code('location_parts = news_locations["location_record"].str.split("#",expand=True)'),
    code('news_locations["location_type"] = pd.to_numeric(location_parts[0],errors="coerce")'),
    code('news_locations["location_name"] = location_parts[1]'),
    code('news_locations["location_latitude"] = pd.to_numeric(location_parts[5],errors="coerce")'),
    code('news_locations["location_longitude"] = pd.to_numeric(location_parts[6],errors="coerce")'),
    code('news_locations = news_locations.dropna(subset=["location_latitude","location_longitude"]).merge(regions[["region","west","south","east","north"]],on="region",how="left")'),
    code('news_locations = news_locations[news_locations["location_longitude"].between(news_locations["west"],news_locations["east"]) & news_locations["location_latitude"].between(news_locations["south"],news_locations["north"]) & news_locations["location_type"].ne(1)].drop_duplicates(["news_id","location_latitude","location_longitude"])'),
    code('pd.Series({"location_records":news_locations.shape[0],"articles_with_location":news_locations["news_id"].nunique()})'),
    md("## 11. Match thermal events and news"),
    code('event_days = thermal_events[["event_id","region","event_date"]].assign(match_day=lambda data:data["event_date"].map(lambda day:pd.date_range(day,day+pd.Timedelta(days=7),freq="D"))).explode("match_day")'),
    code('candidate_matches = event_days.merge(df_news_clean,left_on=["region","match_day"],right_on=["region","publication_day"],how="inner")'),
    code('candidate_matches["days_after_event"] = (candidate_matches["publication_day"] - candidate_matches["event_date"]).dt.days'),
    code('spatial_matches = candidate_matches.merge(thermal_events[["event_id","latitude","longitude"]],on="event_id").merge(news_locations[["news_id","region","location_name","location_latitude","location_longitude"]],on=["news_id","region"])'),
    code('distance_value = np.sin((np.radians(spatial_matches["location_latitude"])-np.radians(spatial_matches["latitude"]))/2)**2 + np.cos(np.radians(spatial_matches["latitude"]))*np.cos(np.radians(spatial_matches["location_latitude"]))*np.sin((np.radians(spatial_matches["location_longitude"])-np.radians(spatial_matches["longitude"]))/2)**2'),
    code('spatial_matches["distance_km"] = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(distance_value.clip(0,1)))'),
    code('event_news_matches = spatial_matches[spatial_matches["distance_km"].le(50)].sort_values("distance_km").drop_duplicates(["event_id","news_id"]).merge(title_nlp[["news_id","direct_conflict_title","maritime_title"]],on="news_id",how="left")'),
    code('event_news_matches["distance_points"] = np.select([event_news_matches["distance_km"].le(10),event_news_matches["distance_km"].le(25)],[3,2],default=1)'),
    code('event_news_matches["time_points"] = np.select([event_news_matches["days_after_event"].le(1),event_news_matches["days_after_event"].le(3)],[3,2],default=1)'),
    code('event_news_matches["match_score"] = event_news_matches["distance_points"] + event_news_matches["time_points"] + event_news_matches["direct_conflict_title"].astype(int)*2 + event_news_matches["maritime_title"].astype(int)'),
    code('event_news_matches["match_confidence"] = np.select([event_news_matches["match_score"].ge(7),event_news_matches["match_score"].ge(5)],["high","medium"],default="contextual")'),
    code('event_news_matches["match_confidence"].value_counts()'),
    md("## 12. Create the event-level target"),
    code('event_summary = event_news_matches.groupby("event_id",as_index=False).agg(news_count=("news_id","nunique"),publisher_count=("publisher","nunique"),high_count=("match_confidence",lambda values:values.eq("high").sum()),medium_count=("match_confidence",lambda values:values.eq("medium").sum()),nearest_news_km=("distance_km","min"))'),
    code('supported_summary = event_news_matches[event_news_matches["match_confidence"].isin(["high","medium"])].groupby("event_id",as_index=False).agg(supported_news=("news_id","nunique"),supported_publishers=("publisher","nunique"))'),
    code('thermal_events_final = thermal_events.merge(event_summary,on="event_id",how="left").merge(supported_summary,on="event_id",how="left")'),
    code('thermal_events_final[["news_count","publisher_count","high_count","medium_count","supported_news","supported_publishers"]] = thermal_events_final[["news_count","publisher_count","high_count","medium_count","supported_news","supported_publishers"]].fillna(0).astype(int)'),
    code('thermal_events_final["has_supported_match"] = thermal_events_final["high_count"].ge(1) | (thermal_events_final["supported_news"].ge(2) & thermal_events_final["supported_publishers"].ge(2))'),
    code('thermal_events_final.groupby("region")["has_supported_match"].agg(["count","sum","mean"]).round(3)'),
    md("## 13. Add World Port Index information"),
    code('wpi_url = "https://gis.unocha.org/server/rest/services/Hosted/global_world_seaport_index_202511/FeatureServer/0/query"'),
    code('wpi_pages = pd.Series([0,2000]).map(lambda offset:requests.get(wpi_url,params={"where":"1=1","outFields":"*","returnGeometry":"true","outSR":4326,"f":"json","resultRecordCount":2000,"resultOffset":int(offset)},timeout=60).json()["features"])'),
    code('ports_raw = pd.json_normalize(wpi_pages.explode().dropna().tolist())'),
    code('ports = ports_raw[["attributes.port_name","attributes.country","attributes.latitude","attributes.longitude"]].copy()'),
    code('ports.columns = ["port_name","port_country","port_latitude","port_longitude"]'),
    code('ports[["port_latitude","port_longitude"]] = ports[["port_latitude","port_longitude"]].apply(pd.to_numeric,errors="coerce")'),
    code('ports = ports.dropna(subset=["port_latitude","port_longitude"]).drop_duplicates().reset_index(drop=True)'),
    code('port_tree = BallTree(np.radians(ports[["port_latitude","port_longitude"]]),metric="haversine")'),
    code('nearest_distance, nearest_index = port_tree.query(np.radians(thermal_events_final[["latitude","longitude"]]),k=1)'),
    code('thermal_events_final["nearest_port_km"] = nearest_distance[:,0] * EARTH_RADIUS_KM'),
    code('thermal_events_final["nearest_port_name"] = ports.iloc[nearest_index[:,0]]["port_name"].to_numpy()'),
    code('thermal_events_final["port_zone"] = pd.cut(thermal_events_final["nearest_port_km"],bins=[0,10,25,75,np.inf],labels=["0-10 km","10-25 km","25-75 km","75+ km"],include_lowest=True)'),
    code('thermal_events_final[["event_id","nearest_port_name","nearest_port_km","port_zone"]].head()'),
    md("## 14. EDA and statistical test"),
    code('thermal_events_final["month"] = thermal_events_final["event_date"].dt.strftime("%Y-%m")'),
    code('monthly_summary = thermal_events_final.groupby(["region","month"],as_index=False).agg(total_events=("event_id","size"),supported_events=("has_supported_match","sum"),median_frp=("total_frp","median"))'),
    code('monthly_summary["supported_rate"] = monthly_summary["supported_events"] / monthly_summary["total_events"]'),
    code('monthly_summary.round(3)'),
    code('matched_brightness = thermal_events_final.loc[thermal_events_final["has_supported_match"],"max_brightness"]'),
    code('unmatched_brightness = thermal_events_final.loc[~thermal_events_final["has_supported_match"],"max_brightness"]'),
    code('u_statistic, p_value = mannwhitneyu(matched_brightness,unmatched_brightness,alternative="greater")'),
    code('rank_biserial = 2*u_statistic/(matched_brightness.size*unmatched_brightness.size)-1'),
    code('pd.Series({"U statistic":u_statistic,"p-value":p_value,"rank-biserial effect":rank_biserial}).round(4)'),
    md("""
The test measures association. Statistical significance does not prove that conflict caused the thermal event, and a small effect size limits practical interpretation.
"""),
    md("## 15. Classification models"),
    md("""
September–January is used for training and February is used as a future test month. Logistic Regression and Decision Tree satisfy the assignment requirement. Random Forest is included as an additional model. The target is source-supported news association.
"""),
    code('train_data = thermal_events_final[thermal_events_final["month"].ne("2026-02")].copy()'),
    code('test_data = thermal_events_final[thermal_events_final["month"].eq("2026-02")].copy()'),
    code('numeric_features = ["pixel_count","total_frp","max_frp","max_brightness","duration_hours","day_ratio","nearest_port_km"]'),
    code('categorical_features = ["region","port_zone"]'),
    code('model_features = numeric_features + categorical_features'),
    code('X_train = train_data[model_features]; X_test = test_data[model_features]'),
    code('y_train = train_data["has_supported_match"].astype(int); y_test = test_data["has_supported_match"].astype(int)'),
    code('preprocessor = ColumnTransformer([("numeric",StandardScaler(),numeric_features),("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features)])'),
    code('logistic_model = Pipeline([("preprocessor",preprocessor),("model",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42))])'),
    code('tree_model = Pipeline([("preprocessor",preprocessor),("model",DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,class_weight="balanced",random_state=42))])'),
    code('forest_model = Pipeline([("preprocessor",preprocessor),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=5,class_weight="balanced_subsample",random_state=42,n_jobs=-1))])'),
    code('logistic_model.fit(X_train,y_train); tree_model.fit(X_train,y_train); forest_model.fit(X_train,y_train)'),
    code('logistic_prediction = logistic_model.predict(X_test); tree_prediction = tree_model.predict(X_test); forest_prediction = forest_model.predict(X_test)'),
    code('logistic_probability = logistic_model.predict_proba(X_test)[:,1]; tree_probability = tree_model.predict_proba(X_test)[:,1]; forest_probability = forest_model.predict_proba(X_test)[:,1]'),
    code('model_results = pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest"],"Accuracy":[accuracy_score(y_test,logistic_prediction),accuracy_score(y_test,tree_prediction),accuracy_score(y_test,forest_prediction)],"Precision":[precision_score(y_test,logistic_prediction),precision_score(y_test,tree_prediction),precision_score(y_test,forest_prediction)],"Recall":[recall_score(y_test,logistic_prediction),recall_score(y_test,tree_prediction),recall_score(y_test,forest_prediction)],"F1":[f1_score(y_test,logistic_prediction),f1_score(y_test,tree_prediction),f1_score(y_test,forest_prediction)],"ROC-AUC":[roc_auc_score(y_test,logistic_probability),roc_auc_score(y_test,tree_probability),roc_auc_score(y_test,forest_probability)],"PR-AUC":[average_precision_score(y_test,logistic_probability),average_precision_score(y_test,tree_probability),average_precision_score(y_test,forest_probability)]})'),
    code('model_results.round(3).sort_values("PR-AUC",ascending=False)'),
    md("## 16. Final dashboard"),
    code('region_names = regions.set_index("region")["region_name"].to_dict()'),
    code('plot_events = thermal_events_final.assign(status=thermal_events_final["has_supported_match"].map({True:"Source-supported",False:"No source support"}))'),
    code('plot_monthly = monthly_summary.assign(region_name=monthly_summary["region"].map(region_names))'),
    code('plot_regions = thermal_events_final.groupby("region",as_index=False).agg(total=("event_id","size"),supported=("has_supported_match","sum"))'),
    code('plot_regions["supported_rate"] = plot_regions["supported"] / plot_regions["total"]'),
    code('plot_regions["region_name"] = plot_regions["region"].map(region_names)'),
    code('figure, axes = plt.subplots(2,2,figsize=(16,10))'),
    code('sns.scatterplot(data=plot_events,x="longitude",y="latitude",hue="status",palette={"No source support":"lightgray","Source-supported":"crimson"},s=16,ax=axes[0,0])'),
    code('sns.lineplot(data=plot_monthly,x="month",y="total_events",hue="region_name",marker="o",ax=axes[0,1])'),
    code('sns.barplot(data=plot_regions,x="supported_rate",y="region_name",hue="region_name",legend=False,ax=axes[1,0])'),
    code('sns.barplot(data=model_results.melt(id_vars="Model",value_vars=["Accuracy","F1","PR-AUC"],var_name="Metric",value_name="Score"),x="Metric",y="Score",hue="Model",ax=axes[1,1])'),
    code('axes[0,0].set_title("Thermal Events and News Evidence"); axes[0,1].set_title("Monthly Thermal Events"); axes[1,0].set_title("Regional Supported Rate"); axes[1,1].set_title("Model Performance")'),
    code('figure.suptitle("Satellite Thermal Events and Conflict-News Monitoring",fontsize=18,fontweight="bold")'),
    code('figure.tight_layout()'),
    code('figure.savefig(OUTPUT_DIR/"dashboard.png",dpi=300,bbox_inches="tight")'),
    code('plt.show()'),
    md("## 17. Save final tables"),
    code('df_satellite_clean.to_csv(OUTPUT_DIR/"satellite_clean.csv",index=False)'),
    code('df_news_clean.to_csv(OUTPUT_DIR/"news_clean.csv",index=False)'),
    code('thermal_events_final.to_csv(OUTPUT_DIR/"thermal_events_final.csv",index=False)'),
    code('event_news_matches.to_csv(OUTPUT_DIR/"event_news_matches.csv",index=False)'),
    code('model_results.to_csv(OUTPUT_DIR/"model_results.csv",index=False)'),
    md("""
## 18. Recorded historical conclusion (separate from this rerun)

The study combined more than 11,000 cleaned satellite detections with 3,589 multilingual news articles and produced 3,289 candidate thermal events. Spatial, temporal and headline evidence identified a smaller source-supported group for analyst review.

The analysis found a relationship between news-supported thermal events and conflict evidence, but it did not establish causation. Model performance was influenced strongly by region and port context, while satellite intensity alone was not sufficient for reliable conflict identification.

The extended investigation later added UCDP, UKMTO/JMIC, a land–water mask, Sentinel imagery and AIS/SAR context. That work reduced the maritime review queue to seven priority events: three were consistent with ordinary vessel or port traffic and four remained unexplained. No event was causally confirmed.

The final product is an analyst-prioritisation system, not an autonomous conflict-decision system.
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook["metadata"] = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
nbf.write(notebook, OUTPUT)
print(OUTPUT)
