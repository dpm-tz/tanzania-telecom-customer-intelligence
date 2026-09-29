# Model card: Churn prediction

## Overview
| | |
|---|---|
| Task | Binary classification: will the customer have **no billable activity in the next 30 days**? |
| Trained in | `notebooks/50_train_churn_model.py` |
| Scored in | `notebooks/70_score_and_recommend.py` |
| Candidates | Logistic Regression (baseline), Gradient-Boosted Trees (challenger; Random Forest if GBT is unavailable) |
| Champion selection | Highest AUC-PR on a random 20 % hold-out |
| Saved to | Volume `tz_ml.artifacts/models/churn/versions/<version>`, pointer in `champion.json` |
| Tracking | MLflow experiment `tz_telecom_customer_intelligence` (best effort; skipped gracefully if unavailable) |

## Intended use
Rank customers by churn risk so a retention team with limited capacity (able to contact roughly 10-30 % of the base)
knows who to call first, and why. **Not** intended to make automatic decisions (e.g. auto-cancelling a line) or to be
used as a legal/financial risk score.

## Training data
* Population: all customers with a valid Silver record as of the snapshot date (2026-06-29).
* Features: `ML_FEATURES` in `00_config.py` — usage, spend, mobile-money, network and care signals over the
  preceding 90 days, plus 30-day trend ratios. Full list and definitions: `docs/data_dictionary.md`.
* Label: `churn_30d` = 1 if the customer has zero outgoing calls, data sessions, bundle purchases or mobile-money
  transactions in the 30 days *after* the snapshot.
* Split: random 80/20 by customer. **Not** an out-of-time split (see Limitations).

## Metrics (tiny-scale rehearsal, 3,000 customers — Databricks numbers will differ)
| Metric | Value |
|---|---|
| Base rate (actual churn) | ~15 % |
| AUC-ROC | ~0.90-0.95 |
| AUC-PR | ~0.75-0.85 |
| Recall in top 10 % of scores | ~55-60 % |
| Lift in top 10 % of scores | ~5-6x random |

Re-run `notebooks/50_train_churn_model.py` to get the numbers for your own run — the exact values depend on the
random split and on `scale`. The canonical source of truth is `tz_gold.model_performance`.

## How risk bands are set
Rank-based, not threshold-based: top 10 % of scores = **High**, next 20 % = **Medium**, rest = **Low**
(`RISK_HIGH_SHARE`, `RISK_MEDIUM_SHARE` in `00_config.py`). This matches how a campaign team actually operates
(a fixed list size, not "everyone above probability 0.5").

## What the model relies on (global feature importance)
Recency (`days_since_last_activity`) dominates, followed by tenure, short-term activity trends, and care-ticket
signals. See `tz_ml.feature_importance` and chart `06_churn_feature_importance.png` for the full ranked list from
your run.

## Limitations
* **Random split, not out-of-time.** A production deployment should train on one period and validate on a later,
  unseen period. This project uses a single snapshot for simplicity; extending it to several monthly snapshots and
  an out-of-time split is a natural next step (see `docs/data_assumptions.md`).
* **The full scoring table includes training customers.** `tz_ml.retention_actions` scores everyone in the current
  snapshot, so its churn-rate-by-band numbers are optimistic. Trust the **hold-out** metrics
  (`tz_ml.churn_test_scored`, `tz_gold.churn_gains`) for an honest read of quality.
* **Recency-dominated.** The model is good at confirming customers who already went quiet; it is weaker at catching
  churn before the first sign of inactivity. Combine with the driver rules (network, care, mobile-money) to catch
  earlier warning signs.
* **Synthetic data.** Churn is generated from a known formula (see `docs/data_assumptions.md` §2), so this model is
  a demonstration of a sound pipeline, not a measurement of real Tanzanian churn drivers or rates.
* **No calibration step.** Predicted probabilities are usable for ranking (which is how they are used here) but have
  not been formally calibrated to match true probabilities.
* **Drivers are rules, not SHAP values.** `primary_driver` / `top_drivers` are transparent business rules chosen to
  mirror the model's strongest global features; they are not a per-customer explanation of *this* model's decision.

## Fairness and ethical notes
Gender is not used as a feature. Age, region and device type are used, which can act as proxies for
socio-economic status; if this model is ever used for anything beyond a retention offer with clear customer
benefit (a discount or check-in call), we would first check that the model does not disproportionately flag
protected groups without a corresponding, justified difference in behaviour.
