# Methods, design decisions and limitations

## Detection is not an incident

NASA FIRMS supplies thermal detections. The study grouped observations using geographic DBSCAN inside fixed three-day windows. The V1 daily grouping yielded 5,144 groups; V2 yielded 3,289, retaining 11,525 cleaned detections.

The V2 neighbourhood radius was seven kilometres with a haversine metric. Neighbour chains can make a cluster larger than seven kilometres; fixed time boundaries can split a continuous source. These are modelling choices, not ground truth.

FRP is a power-related measurement per observation. Summing FRP across pixels and times is an aggregate feature, not a physical estimate of total released energy.

## Articles, locations and weak labels

GDELT supplied article metadata. Basic multilingual headline indicators contributed conflict and maritime terms; they are not a trained multilingual understanding model. A detected word does not establish the article's topic or factual truth.

The matching workflow used article time and extracted location mentions to find candidate event associations. It kept stronger support separate from contextual proximity. A headline about one location can mention another nearby location, producing an incorrect association.

The historical event target `has_supported_match` identifies news support under those rules. It does not encode verified conflict. Classifying it is a retrospective news-association experiment.

The public teaching notebooks simplify scoring and model settings. Their results must be evaluated on their own outputs rather than assigned the recorded historical scores.

## Model evaluation changed the conclusion

Initial class-stratified work was extended to forward temporal splits. Logistic Regression and Random Forest were compared; a Decision Tree is present in the teaching reconstruction. Later work compared context-only RF, historical-rate lookup, absolute thermal features and relative site-history anomalies.

The geography-only baseline had the best pooled AP among these tested configurations. Single-month improvements did not reliably persist. Correlated features and one test-set importance calculation cannot prove leakage by themselves or prove that all possible satellite representations have no value.

Repeated use of February for model comparison limits its status as a pristine final holdout. Before deployment, freeze feature construction and model selection on development periods, then test on untouched later periods and held-out geographies. Match alert budgets, report month-level variability and use uncertainty estimation that respects site/time dependence.

Satellite availability must also be evaluated against an independent incident inventory: the recall of the candidate-generation stage was not measured here.

## Independent coding, potentially shared reporting

UCDP offers separately coded events with quality, time and location precision fields. Finalized and provisional releases were separated. Proximity to a UCDP event adds evidence but is not a physical attribution; shared underlying reporting limits source independence.

The small UKMTO set included incident-date windows and an accident comparison. Conflict-window thermal counts did not exceed the background daily average in the reported comparison. The set is too small and geographically imprecise for a validated maritime detector.

## Water, vessels and imagery

A Natural Earth land mask and port-distance context narrowed maritime scope. The seven selected cases were compared with Sentinel-1/2 previews and GFW AIS/SAR context. Acquisition times can differ from thermal observations by hours or days. Coloured SAR previews are not direct fire or damage labels.

AIS proximity can be ordinary background traffic. Non-event-day controls help challenge a proposed link; absent AIS does not establish an empty sea. Four unresolved candidates remain unresolved.

## Built versus planned

| Component | Current status |
|---|---|
| Historical data cleaning, association and model experiments | Performed locally; aggregate results documented |
| Exact replay of every original experiment | Incomplete; original news query/export and full configurations missing |
| Current NASA collector | Implemented; date-scoped retrieval, deduplication and review rules |
| Streamlit review and feedback | Implemented for supplied local evidence; feedback stored locally or in PostgreSQL |
| Docker services | Packaged; runs on a machine where Docker is active |
| Autonomous live news/UCDP/AIS/Sentinel fusion | Not implemented |
| Trained RF serving on the live feed | Not implemented |
| Automatic learning from analyst feedback | Not implemented |
| Authenticated public service, monitoring and SLA | Not deployed |
| Causal conflict attribution or forecasting | Not validated |

## Engineering limits before a sustained deployment

The collector currently queries the specified UTC day, defaulting to today. It does not automatically backfill late records from preceding days. Local files are the duplicate ledger; database writes are optional appends, not a transactional exactly-once pipeline. Concurrent collectors, crash recovery, atomic file replacement and database uniqueness/upserts require further work.

FIRMS API revision handling, ingestion latency and blind periods require explicit monitoring. Runtime credentials must stay outside source control. Load only trusted local pickle files; portable CSV inputs are preferred.

The next milestone is stronger measurement and traceability, followed by public-service hardening if a deployment is requested.
