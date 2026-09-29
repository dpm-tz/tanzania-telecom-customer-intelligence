# Data assumptions and limitations

Read this before you quote any number from the project. **All data is synthetic and the company "Simba Telecom" is
fictional.** Nothing here describes a real operator, and no figure should be presented as a real Tanzanian market
statistic.

## 1. What is realistic, and what is invented

| Topic | What the simulation assumes | Status |
|---|---|---|
| Prepaid market | Customers hold no contract. Churn = **no billable activity for 30 days**. | Modelling choice that mirrors how prepaid operators define inactive lines |
| Multi-SIM | 35 % of customers are flagged multi-SIM and churn slightly more. | Invented parameter |
| Mobile money | 85 % of urban and 55 % of rural customers are registered. Only registered customers transact. | Invented parameter |
| Geography | 14 regions, 3 to 5 districts each, real names. The urban/rural label per district and each region's customer share are **illustrative**, not census data. | Invented weights |
| Network | Cells (towers) per district scale with customers. Urban cells are better on average, rural cells fail more and are less likely to have power backup. | Invented parameters |
| Power failures | Cells without power backup produce five times more `power_failure` events. | Invented parameter |
| Bundles | 18 bundles named in Swahili (`SIKU` = day, `WIKI` = week, `MWEZI` = month, `DAKIKA` = minutes). **Prices are illustrative** and do not match any operator's tariff. | Invented prices |
| Devices | Smartphone share 78 % urban and 50 % rural, decreasing slightly with age. | Invented parameter |
| Seasonality | Only one effect: slightly more bundle purchases in the first 5 days of a month (salary timing). No Ramadan, holiday, harvest or school calendar effects. | Deliberately minimal |
| Currency | All money in TZS. | Fact |

## 2. How churn is generated (so nobody mistakes it for a random column)

1. Each customer gets a hidden churn propensity from a logistic formula. The coefficients (documented in notebook
   `10_generate_source_data`) reward realistic causes:

   | Driver | Effect on churn |
   |---|---|
   | Worse home-cell quality | Strong increase |
   | Care tickets (count, and any open or slow-resolved ticket) | Increase |
   | New SIM (under 90 days) | Increase |
   | Multi-SIM | Small increase |
   | Not registered for mobile money | Small increase |
   | Rural | Small increase |
   | Mobile-money power-user profile | Decrease |
   | Low-activity profile | Increase |

2. Customers who churn stop generating events. 60 % stop abruptly within the last 5 days before the snapshot,
   40 % fade first (usage drops to about 45 %) for their last weeks.
3. 18 % of customers who stay go almost silent for 14 days and then come back. These are **false alarms** on purpose,
   otherwise the model would be unrealistically good.
4. The label is then **computed from the events** (no activity in the 30 days after the snapshot). It is never
   stored by the generator.

The hidden truth is written to `landing/_simulation/`. The pipeline never reads it. You can use it to check whether a
model recovers the real drivers.

## 3. Time windows

| Window | Dates | Use |
|---|---|---|
| Observation | 2026-04-01 to 2026-06-29 (90 days) | All features and all business KPIs |
| Prediction | 2026-06-30 to 2026-07-29 (30 days) | Labels only |

Features never look past 2026-06-29. Tickets resolved after that date are treated as still open at the snapshot.
There is a test for this (`test_churn_label_is_derived_from_real_inactivity`, `test_features_do_not_look_into_the_future`).

## 4. Data-quality problems injected on purpose

About 1 to 2 % of rows in each event table are corrupted so Silver has real work to do:

| Problem | Example |
|---|---|
| Exact duplicates | Same `call_id` twice |
| Missing keys | `customer_id` is null |
| Orphans | `customer_id` that does not exist |
| Impossible values | Zero-second call, negative bundle price, age 150, severity 9 |
| Timestamp outside the extract window | Event dated 45 days in the future |
| Logical errors | Ticket resolved before it was created, unknown bundle id |

Rows failing any rule go to `tz_silver.dq_quarantine`. When a **customer** is quarantined, all events of that customer
are quarantined too as orphans (`customer_exists` fails). That is the correct referential-integrity behaviour but it
removes about 0.5 % of customers and their history from Silver.

## 5. Modelling limitations

| Limitation | Why it matters | What a production team would do |
|---|---|---|
| **One snapshot, random 80/20 split** | Rows from the same period appear in train and test. Real deployments predict *future* periods. | Build several monthly snapshots and validate out-of-time |
| **The scoring table includes training customers** | `tz_ml.retention_actions` is optimistic for those customers. | Score only new snapshots; judge quality from hold-out metrics |
| **Recency dominates** | `days_since_last_activity` is the strongest churn feature. That is typical for prepaid, but it means the model detects customers who already went quiet. | Add earlier signals and test proactive interventions |
| **Simulated causality** | Cell quality, tickets and tenure drive churn *by construction*. Recovering them shows the pipeline works, not that these are the real causes in any market. | A/B test retention actions to measure causal uplift |
| **Performance is probably higher than reality** | Real churn is noisier than a simulation with a known formula. | Expect lower AUC on real data |
| **Risk drivers are rules, not explanations** | The per-customer "drivers" are readable business rules, not SHAP values. | Add SHAP or similar for per-prediction explanations |
| **Revenue at risk is a rough estimate** | `ARPU × churn probability × 3 months`. The 3-month horizon is an assumption. | Use customer lifetime value with real margin data |
| **Probabilities** | Models are trained without class weights and are approximately calibrated (see the risk-band chart), but no formal calibration step is applied. | Add calibration and monitor drift |

## 6. Parameters you can change

| Parameter | Where | Default |
|---|---|---|
| Data volume | Widget `scale` (`tiny`, `small`, `standard`) | `small` (20,000 customers) |
| Snapshot date, window lengths | `notebooks/00_config.py` | 2026-06-29, 90 d, 30 d |
| High / Medium risk share | `RISK_HIGH_SHARE`, `RISK_MEDIUM_SHARE` | 10 %, 20 % |
| Upsell "High" share | `UPSELL_HIGH_SHARE` | 15 % |
| Revenue horizon | `REVENUE_HORIZON_MONTHS` | 3 |
| Large-bundle threshold | `LARGE_DATA_MB` | 10,240 MB |
| Regions, districts, bundles | `_GEO`, `BUNDLES` in `00_config.py` | See file |
| Churn drivers and base rate | `INTERCEPT` and the logit formula in `10_generate_source_data.py` | Base rate about 14 to 15 % |
