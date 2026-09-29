-- =====================================================================================
-- Tanzania Telecom Customer Intelligence: dashboard datasets
-- =====================================================================================
-- How to use (AI/BI Dashboards):
--   Dashboards -> Create dashboard -> Data tab -> "Create from SQL" -> paste ONE query per dataset
--   (copy everything between two "-- @query" markers) -> Canvas tab -> add the visual named in the header.
--
-- Catalog: these queries use `workspace`. If your catalog has another name, find & replace `workspace.`.
-- Money is in TZS. All data is synthetic (fictional operator "Simba Telecom").
-- Every query in this file is executed by tests/test_pipeline_outputs.py.
-- =====================================================================================


-- @query: p1_headline_kpis | Page 1 Executive overview | 5 Counter tiles (one per column)
SELECT
  (SELECT COUNT(*) FROM workspace.tz_gold.customer_360)                                              AS customers,
  (SELECT ROUND(AVG(actual_churn_30d) * 100, 1) FROM workspace.tz_ml.retention_actions)              AS churn_rate_pct,
  (SELECT ROUND(SUM(bundle_revenue_tzs) / 1e6, 1) FROM workspace.tz_gold.kpi_region_month)           AS bundle_revenue_90d_million_tzs,
  (SELECT ROUND(AVG(arpu_monthly_tzs), 0) FROM workspace.tz_gold.customer_360)                       AS avg_arpu_monthly_tzs,
  (SELECT ROUND(SUM(revenue_at_risk_tzs) / 1e6, 1)
     FROM workspace.tz_ml.retention_actions WHERE risk_band = 'High')                                AS high_risk_revenue_at_risk_million_tzs


-- @query: p1_daily_trend | Page 1 Executive overview | Line: x=date, y=active_customers (second axis: bundle_revenue_tzs)
SELECT date, active_customers, bundle_revenue_tzs, voice_minutes, data_gb, mm_value_tzs
FROM workspace.tz_gold.kpi_daily
ORDER BY date


-- @query: p1_revenue_by_month | Page 1 Executive overview | Bar: x=month, y=revenue_million_tzs
SELECT month,
       ROUND(SUM(bundle_revenue_tzs) / 1e6, 1) AS revenue_million_tzs,
       SUM(active_customers)                   AS active_customer_months,
       MAX(days_covered)                       AS days_covered
FROM workspace.tz_gold.kpi_region_month
GROUP BY month
ORDER BY month


-- @query: p1_region_overview | Page 1 Executive overview | Table or bar: region vs customers / ARPU
SELECT region, customers, avg_arpu_monthly_tzs, mobile_money_penetration, smartphone_share, urban_share
FROM workspace.tz_gold.region_summary
ORDER BY customers DESC


-- @query: p2_churn_by_region | Page 2 Churn and retention | Bar: x=region, y=actual_churn_pct (colour: high_risk_customers)
SELECT region,
       ROUND(actual_churn_rate * 100, 1) AS actual_churn_pct,
       high_risk_customers,
       ROUND(high_risk_revenue_at_risk_tzs / 1e6, 2) AS high_risk_revenue_at_risk_million_tzs
FROM workspace.tz_gold.region_summary
ORDER BY actual_churn_pct DESC


-- @query: p2_risk_band_holdout | Page 2 Churn and retention | Clustered bar: risk_band, predicted vs actual (HOLD-OUT customers)
SELECT a.risk_band,
       COUNT(*)                                   AS customers,
       ROUND(AVG(a.churn_probability) * 100, 1)   AS predicted_churn_pct,
       ROUND(AVG(a.actual_churn_30d) * 100, 1)    AS actual_churn_pct
FROM workspace.tz_ml.retention_actions a
JOIN workspace.tz_ml.churn_test_scored t USING (customer_id)
GROUP BY a.risk_band
ORDER BY CASE a.risk_band WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END


-- @query: p2_gains | Page 2 Churn and retention | Line: x=cumulative_share_of_customers, y=cumulative_recall
SELECT decile, customers, churners, actual_churn_rate, cumulative_share_of_customers, cumulative_recall, lift
FROM workspace.tz_gold.churn_gains
ORDER BY decile


-- @query: p2_top_drivers_high_risk | Page 2 Churn and retention | Horizontal bar: driver vs customers
SELECT driver, COUNT(*) AS high_risk_customers
FROM workspace.tz_ml.retention_actions
LATERAL VIEW EXPLODE(top_drivers) d AS driver
WHERE risk_band = 'High'
GROUP BY driver
ORDER BY high_risk_customers DESC


-- @query: p2_actions_summary | Page 2 Churn and retention | Table: what each team should do
SELECT campaign_priority, recommended_action, owner_team,
       COUNT(*)                                        AS customers,
       ROUND(SUM(revenue_at_risk_tzs) / 1e6, 2)        AS revenue_at_risk_million_tzs
FROM workspace.tz_ml.retention_actions
GROUP BY campaign_priority, recommended_action, owner_team
ORDER BY campaign_priority, customers DESC


-- @query: p2_target_list | Page 2 Churn and retention | Table (top 200 customers to contact first)
SELECT customer_id, region, district, segment_name,
       ROUND(churn_probability, 3) AS churn_probability,
       risk_band, primary_driver, recommended_action, owner_team,
       arpu_monthly_tzs, revenue_at_risk_tzs
FROM workspace.tz_gold.retention_target_list
ORDER BY revenue_at_risk_tzs DESC
LIMIT 200


-- @query: p2_feature_importance | Page 2 Churn and retention | Horizontal bar: feature vs importance_pct
SELECT feature, ROUND(importance * 100, 1) AS importance_pct
FROM workspace.tz_ml.feature_importance
WHERE model_name NOT LIKE 'upsell:%'
ORDER BY importance DESC
LIMIT 15


-- @query: p3_segment_profile | Page 3 Customers and segments | Table + bar of customers per segment
SELECT segment_name, customers, ROUND(share_of_customers * 100, 1) AS share_pct,
       avg_arpu_monthly_tzs, avg_data_gb_90d, avg_voice_minutes_90d, avg_mm_txns_90d,
       avg_active_days_90d, ROUND(actual_churn_rate * 100, 1) AS churn_pct
FROM workspace.tz_gold.segment_profile
ORDER BY customers DESC


-- @query: p3_urban_rural | Page 3 Customers and segments | Clustered bar: area_type
SELECT c.area_type,
       COUNT(*)                                   AS customers,
       ROUND(AVG(c.arpu_monthly_tzs), 0)          AS avg_arpu_monthly_tzs,
       ROUND(AVG(c.data_gb_90d), 2)               AS avg_data_gb_90d,
       ROUND(AVG(l.churn_30d) * 100, 1)           AS churn_pct
FROM workspace.tz_gold.customer_360 c
JOIN workspace.tz_ml.labels l USING (customer_id)
GROUP BY c.area_type


-- @query: p3_device_and_mobile_money | Page 3 Customers and segments | Grouped bar: churn by device and mobile-money status
SELECT c.device_type,
       CASE WHEN c.mobile_money_registered = 1 THEN 'mobile money' ELSE 'no mobile money' END AS mobile_money_status,
       COUNT(*)                                   AS customers,
       ROUND(AVG(c.arpu_monthly_tzs), 0)          AS avg_arpu_monthly_tzs,
       ROUND(AVG(l.churn_30d) * 100, 1)           AS churn_pct
FROM workspace.tz_gold.customer_360 c
JOIN workspace.tz_ml.labels l USING (customer_id)
GROUP BY c.device_type, c.mobile_money_registered


-- @query: p3_age_band | Page 3 Customers and segments | Bar: age_band vs customers / churn
SELECT CASE WHEN c.age < 25 THEN '18-24' WHEN c.age < 35 THEN '25-34' WHEN c.age < 45 THEN '35-44'
            WHEN c.age < 55 THEN '45-54' ELSE '55+' END AS age_band,
       COUNT(*)                                   AS customers,
       ROUND(AVG(c.arpu_monthly_tzs), 0)          AS avg_arpu_monthly_tzs,
       ROUND(AVG(l.churn_30d) * 100, 1)           AS churn_pct
FROM workspace.tz_gold.customer_360 c
JOIN workspace.tz_ml.labels l USING (customer_id)
GROUP BY 1
ORDER BY 1


-- @query: p4_network_by_region | Page 4 Network and care | Bar: region vs power_failures_per_cell (and events_per_cell)
SELECT region,
       SUM(network_events)                                       AS network_events,
       SUM(power_failures)                                       AS power_failures,
       SUM(cell_down_events)                                     AS cell_down_events,
       SUM(outage_minutes)                                       AS outage_minutes,
       MAX(total_cells)                                          AS total_cells,
       ROUND(SUM(network_events) / MAX(total_cells), 1)          AS events_per_cell,
       ROUND(SUM(power_failures) / MAX(total_cells), 1)          AS power_failures_per_cell
FROM workspace.tz_gold.network_quality_region
GROUP BY region
ORDER BY power_failures_per_cell DESC


-- @query: p4_churn_by_cell_outage | Page 4 Network and care | Bar: outage_quintile vs churn_pct
SELECT quintile AS outage_quintile_1_best_to_5_worst,
       COUNT(*)                          AS customers,
       ROUND(AVG(outage_min), 0)         AS avg_outage_minutes,
       ROUND(AVG(churn_30d) * 100, 1)    AS churn_pct
FROM (
  SELECT c.cell_outage_minutes_90d AS outage_min, l.churn_30d,
         NTILE(5) OVER (ORDER BY c.cell_outage_minutes_90d) AS quintile
  FROM workspace.tz_gold.customer_360 c
  JOIN workspace.tz_ml.labels l USING (customer_id)
) q
GROUP BY quintile
ORDER BY quintile


-- @query: p4_care_by_category | Page 4 Network and care | Table / bar: resolution time by ticket category
SELECT category,
       SUM(tickets)                                            AS tickets,
       SUM(open_at_snapshot)                                   AS open_at_snapshot,
       ROUND(SUM(avg_resolution_hours * resolved) / NULLIF(SUM(resolved), 0), 1) AS avg_resolution_hours,
       ROUND(SUM(share_resolved_within_72h * tickets) / SUM(tickets) * 100, 1)   AS resolved_within_72h_pct
FROM workspace.tz_gold.care_performance
GROUP BY category
ORDER BY tickets DESC


-- @query: p4_churn_by_tickets | Page 4 Network and care | Bar: ticket bucket vs churn_pct
SELECT CASE WHEN c.slow_tickets_90d >= 1 THEN 'slow or open ticket'
            WHEN c.tickets_90d >= 1 THEN 'tickets, all resolved fast'
            ELSE 'no tickets' END AS care_experience,
       COUNT(*)                                   AS customers,
       ROUND(AVG(l.churn_30d) * 100, 1)           AS churn_pct
FROM workspace.tz_gold.customer_360 c
JOIN workspace.tz_ml.labels l USING (customer_id)
GROUP BY 1
ORDER BY churn_pct DESC


-- @query: p5_top_bundles | Page 5 Bundles and mobile money | Horizontal bar: bundle_code vs revenue_million_tzs
SELECT bundle_code, category, price_tzs,
       SUM(purchases)                        AS purchases,
       SUM(unique_buyers)                    AS buyer_months,
       ROUND(SUM(revenue_tzs) / 1e6, 2)      AS revenue_million_tzs
FROM workspace.tz_gold.bundle_performance
GROUP BY bundle_code, category, price_tzs
ORDER BY revenue_million_tzs DESC


-- @query: p5_channel_mix | Page 5 Bundles and mobile money | Stacked bar: month x channel
SELECT month, 'mobile_money' AS channel, SUM(via_mobile_money) AS purchases FROM workspace.tz_gold.bundle_performance GROUP BY month
UNION ALL
SELECT month, 'agent', SUM(via_agent) FROM workspace.tz_gold.bundle_performance GROUP BY month
UNION ALL
SELECT month, 'ussd', SUM(via_ussd) FROM workspace.tz_gold.bundle_performance GROUP BY month
UNION ALL
SELECT month, 'app', SUM(via_app) FROM workspace.tz_gold.bundle_performance GROUP BY month
ORDER BY month, channel


-- @query: p5_mobile_money_by_type | Page 5 Bundles and mobile money | Bar: txn_type vs value_billion_tzs
SELECT txn_type,
       SUM(txns)                              AS txns,
       ROUND(SUM(value_tzs) / 1e9, 3)         AS value_billion_tzs,
       ROUND(SUM(value_tzs) / SUM(txns), 0)   AS avg_txn_tzs
FROM workspace.tz_gold.mobile_money_summary
GROUP BY txn_type
ORDER BY value_billion_tzs DESC


-- @query: p5_upsell_by_segment | Page 5 Bundles and mobile money | Bar: segment vs high-propensity customers
SELECT segment_name,
       COUNT(*)                                  AS high_propensity_customers,
       ROUND(AVG(upsell_probability) * 100, 1)   AS avg_upsell_probability_pct,
       ROUND(AVG(arpu_monthly_tzs), 0)           AS avg_arpu_monthly_tzs
FROM workspace.tz_ml.retention_actions
WHERE upsell_band = 'High' AND risk_band = 'Low'
GROUP BY segment_name
ORDER BY high_propensity_customers DESC


-- @query: p6_model_performance | Page 6 Trust | Table: latest metrics per model
SELECT task, model_name, version, metric, ROUND(value, 4) AS value, is_champion
FROM workspace.tz_gold.model_performance
WHERE metric IN ('auc_roc', 'auc_pr', 'base_rate', 'recall_top10', 'lift_top10', 'recall_top15', 'lift_top15', 'silhouette')
ORDER BY task, model_name, metric


-- @query: p6_data_quality | Page 6 Trust | Bar: pct_rejected per table
SELECT table_name, bronze_rows, silver_rows, quarantined_rows, pct_rejected
FROM workspace.tz_gold.data_quality_summary
ORDER BY pct_rejected DESC
