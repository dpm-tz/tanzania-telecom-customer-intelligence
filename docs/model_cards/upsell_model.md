# Model card: Large-bundle upsell propensity

## Overview
| | |
|---|---|
| Task | Binary classification: will the customer buy a **large (>= 10 GB) data bundle in the next 30 days**? |
| Trained in | `notebooks/60_train_upsell_model.py` |
| Scored in | `notebooks/70_score_and_recommend.py` |
| Candidates | Logistic Regression (baseline), Random Forest (challenger) |
| Champion selection | Highest AUC-PR on a random 20 % hold-out |
| Saved to | Volume `tz_ml.artifacts/models/upsell/versions/<version>`, pointer in `champion.json` |

## Intended use
Prioritise a sales/CRM large-bundle campaign towards customers most likely to convert, so a limited number of SMS or
call slots reach the best candidates first. Complements, and is deliberately kept separate from, the churn model:
a customer can be low churn-risk and still be a strong upsell candidate.

## Population and label
* **Population:** customers who did **not** already buy a large bundle in the 30 days before the snapshot
  (`eligible_for_upsell = 1` in `tz_ml.labels`). Offering an upsell to someone who just bought one is not useful.
* **Label:** bought a large bundle (>= `LARGE_DATA_MB` = 10,240 MB) at any point in the 30 days after the snapshot.
* **Features:** the same `ML_FEATURES` list used for churn (data usage is naturally the strongest signal here).
* **Split:** random 80/20 by eligible customer.

## Metrics (tiny-scale rehearsal — re-run for your own numbers)
| Metric | Typical value |
|---|---|
| Base rate (eligible customers who convert) | ~15-20 % |
| AUC-PR | ~0.40-0.50 |
| Lift in top 15 % of scores | ~2-3x random |

This task is harder than churn (weaker signal, more heterogeneous population), which is reflected in the lower
AUC-PR. This is expected and typical of propensity-to-buy problems.

## How the "High propensity" list is built
Rank-based: top 15 % of upsell scores among eligible customers = **High** (`UPSELL_HIGH_SHARE` in `00_config.py`).
In `tz_ml.retention_actions`, upsell is only surfaced as a recommended action for customers who are **not** already
a churn or retention priority (i.e. `risk_band = 'Low'`), so the two campaigns never compete for the same customer
in the default recommendation logic.

## Limitations
* **Weaker signal than churn.** Whether someone buys a bigger bundle depends on many things outside these features
  (a new phone, a new job, a change in household size) that are not observable here.
* **Random split, not out-of-time** — see the churn model card for the same caveat; it applies equally here.
* **Eligibility is a simplification.** "Bought a large bundle in the last 30 days" is a proxy for "does not currently
  need an upsell"; a customer who churns and returns, or who buys two mid-size bundles instead of one large one,
  is not perfectly captured by this rule.
* **Synthetic preference signal.** In the simulation, "data heavy" and "balanced" archetypes are given a higher
  probability of choosing a large bundle (`large_pref` in `10_generate_source_data.py`); the model is expected to
  rediscover data-usage intensity as its top feature, which it does (see `tz_ml.feature_importance`, filtered to
  `model_name LIKE 'upsell:%'`).
