# Data origins and reproducibility ledger

The project combines satellite observations with reporting and contextual evidence. Each row type means something different: a pixel observation, a clustered event, an article, a reference incident, an association pair or an analyst decision.

## Source inventory

| Source | Acquisition in the research | Use | Included in this repository |
|---|---|---|---|
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) | Area API; historical NOAA-20 standard processing | Thermal observations | Acquisition code; no historical extract |
| [GDELT](https://gdeltproject.org/data.html) | Google BigQuery export | Multilingual article metadata and location mentions | Expected schema and limitation below |
| World Port Index | UN OCHA hosted ArcGIS service, URL in notebooks | Nearest-port distance | Retrieval code |
| [UCDP](https://ucdp.uu.se/downloads/) | GED v26.1 and 2026 Candidate Events download | Separately coded event evidence | Selected retrieval and association code |
| Natural Earth | 1:10m land polygons | Approximate centroid land/water classification | Retrieval code in extended notebook |
| UKMTO / JMIC | Manual review of five incident notices | Maritime incident-date context and one accident comparison | Aggregate findings; no complete notice archive |
| Copernicus Sentinel-1/2 | Catalog queries and imagery previews | Manual imagery context for seven cases | Documented research scope; images not redistributed |
| [Global Fishing Watch](https://globalfishingwatch.org/our-apis/documentation/) | Credentialed API queries | AIS/SAR and non-event-day context | Documented research scope; extracts not redistributed |

Historical period: 2025-09-01 through 2026-02-28. Current collection is a separate stream using NOAA-20 and NOAA-21 near-real-time products. FIRMS supports dated area queries; publication latency and product revisions affect availability.

## GDELT: a known gap, not an invented raw file

The recorded study began with a 3,600-row news sample and retained 3,589 articles after cleaning. The sampling design was up to 200 records per region-month across three regions and six months.

The exact original BigQuery SQL and untouched export are **not available in this release**. During local recovery a CSV named `data/raw/gdelt_news.csv` was recreated from `news_clean.pkl`. That file contains already-cleaned data and is excluded from Git. Its directory name does not make it raw.

The notebooks expect a compatible, separately obtained CSV at `data/raw/gdelt_news.csv`. Required fields:

```text
region
publication_date
gdelt_seen_at
title_original
publisher
url
language
location_mentions
```

Location mentions use the GDELT field structure parsed in the notebooks. Validate your export's structure before using it. Preserve the actual SQL, filters, sampling choices, download date and dataset version alongside a new export. A new export should be reported as a **new run**, not as an exact recreation of the original sample.

Do not fabricate a query and call it the original. Full historical reproducibility remains open until the acquisition query, raw input and original experiment configurations are recovered.

## Joining the data

- UTC dates align thermal detections, article publication timestamps and reference-event dates.
- Spatial clustering produces `event_id`; cleaned news has `news_id`.
- Article location mentions create candidate locations, which may differ from the event described in the headline.
- Candidate pairs are filtered/scored using time, geographic distance and multilingual headline indicators.
- Aggregation returns one row per thermal event. Pair counts are not unique-event counts.
- Ports, land masks and later source reviews add context; they do not independently establish the physical cause.

The local historical database contained four related tables: satellite detections (11,525), news articles (3,589), thermal events (3,289) and event–news matches (5,677). Primary/foreign-key checks were part of the original workflow. The repository does not include a dump or reconstruction of that historical database.

## Evidence status and availability

Keep finalized 2025 UCDP records separate from provisional 2026 records. Source coding quality and time/location precision determine whether proximity is usable. UCDP and the article sample may rely on overlapping underlying reporting.

A missing report is not a verified negative. AIS absence is not proof that no vessel was present. A catalog match is not a visual confirmation. UKMTO incident-date proximity alone does not identify a thermal source.

Raw and derived source records stay local by default. Article content, imagery and vessel-data access remain subject to their providers' terms. No data license is granted by this repository.
