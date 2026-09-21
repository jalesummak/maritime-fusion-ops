# Recorded research findings

These tables transcribe the historical exploratory results reported during the project. They are not recomputed by the dashboard, and the simplified teaching notebooks are not a complete replay of the original experiment.

## February 2026: one encouraging month

Target: the project's **news-support rule**, not independently confirmed conflict.

| Metric | Context Random Forest |
|---|---:|
| Test events | 304 |
| News-supported events | 16 |
| True positives / false positives | 13 / 15 |
| False negatives / true negatives | 3 / 273 |
| Accuracy | 0.941 |
| Precision | 0.464 |
| Recall | 0.812 |
| F1 | 0.591 |
| ROC-AUC | 0.953 |
| Average precision (reported as PR-AUC) | 0.663 |

Here a false positive means a flag without the news-support label. It is not proof that the physical event was harmless. With only 16 positive labels, one event changes recall by 6.25 percentage points.

A model that predicts no support for every February record obtains 288/304 = 94.7% accuracy but finds no positive labels. This is why accuracy alone is a poor headline.

## Stronger baseline and temporal comparisons

Five forward test months: October 2025–February 2026. The pooled set contains 1,836 events, including 137 news-supported events. The baseline comparisons used matched alert budgets within each month. These comparisons were developed through repeated experimentation and are exploratory.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Historical-rate lookup | 0.877 | 0.285 | 0.431 | 0.343 | 0.785 | 0.284 |
| Geography-only Random Forest | 0.881 | 0.304 | 0.460 | 0.366 | 0.797 | 0.312 |
| Absolute satellite + context RF | 0.875 | 0.275 | 0.416 | 0.331 | 0.805 | 0.284 |
| Relative anomaly + context RF | 0.870 | 0.256 | 0.387 | 0.308 | 0.805 | 0.289 |

Pooled average precision is computed on concatenated out-of-time scores; it is not the average of monthly AP. Score comparability across fitted models and month-specific prevalence also affect a pooled ranking.

| Test month | Positive labels | Geography RF AP | Absolute satellite + context AP | Relative anomaly + context AP |
|---|---:|---:|---:|---:|
| 2025-10 | 55 | 0.233 | 0.184 | 0.245 |
| 2025-11 | 34 | 0.355 | 0.374 | 0.316 |
| 2025-12 | 16 | 0.359 | 0.406 | 0.328 |
| 2026-01 | 16 | 0.292 | 0.240 | 0.278 |
| 2026-02 | 16 | 0.585 | 0.663 | 0.732 |

The anomaly representation improved February AP but did not improve pooled performance. The observed differences have not been established as statistically significant. Regional dependence, recurring locations and reporting coverage require further controlled evaluation.

## External evidence comparison

For September–December 2025, 117 of 262 news-supported thermal events had finalized UCDP proximity (44.7%). Among events without news support, 430 of 2,502 had such proximity (17.2%). The recorded risk ratio was about 2.60.

This supports an association **within these matching rules and this sample**. It does not demonstrate cause, forecast skill or independent corroboration by every underlying report. Spatially repeated events also complicate naive independence assumptions.

The final 3,289-event evidence partition was:

| Category | Events |
|---|---:|
| Unconfirmed | 2,478 |
| Finalized UCDP proximity only | 430 |
| News support only | 168 |
| News + finalized UCDP proximity | 117 |
| UKMTO conflict incident-day context only | 56 |
| UKMTO accident-day context only | 16 |
| Provisional UCDP proximity only | 15 |
| News + provisional UCDP proximity | 9 |

These mutually exclusive final classes were assigned by a precedence rule. They must not be interpreted as eight independently verified physical event types.

## Maritime review

Approximate centroid classification found 132 water candidates: 104 offshore and 28 near ports. There were also 403 land events near ports and 2,754 inland events. A coarse land mask is not definitive coastline evidence.

Seven selected maritime cases received further Sentinel, AIS/SAR and public-source review:

| Final review interpretation | Cases |
|---|---:|
| Compatible with ordinary vessel or port context | 3 |
| Unconfirmed maritime anomaly | 4 |
| Causally confirmed conflict incident | 0 |

Ordinary context is a plausible alternative explanation, not a verified negative label. The selected seven do not support an estimate of conflict detection recall for all 132 water candidates. Nothing in this review establishes that a named nearby vessel caused a thermal observation.

## Current-feed demonstration

A one-shot collection on 21 September 2026 recorded 33 retained observations for that UTC date, no new review flags and no request failures. This is a point-in-time execution record, not a continuing availability claim. Subsequent pulls and the dashboard may differ.

The current feed applies confidence/FRP rules to detections. It does not implement the retrospective event model, maritime filtering or automatic multi-source causal verification.

## Defensible conclusion

The project built and exercised a multi-source research and analyst-review workflow. News associations overlap coded conflict proximity, while tested thermal intensity features do not consistently outperform geography-only prioritisation. It also demonstrates current NASA ingestion and a feedback interface. Operational conflict detection, causal attribution and measured workload reduction remain unvalidated.
