# Business questions this project answers

Each question links to the table(s) that answer it, the chart in `notebooks/85_visual_analytics.py`, and the
matching dashboard page (`docs/dashboard_guide.md`).

## Churn and retention

**Q1. How many customers are at risk of leaving next month, and where?**
`tz_ml.retention_actions` (risk_band), `tz_gold.region_summary`. Chart `02_churn_by_region`. Dashboard page 2.

**Q2. What drives churn — network, care, price, or falling mobile-money use?**
`tz_ml.feature_importance` (global) + `primary_driver` / `top_drivers` in `tz_ml.retention_actions` (per customer).
Chart `06_churn_feature_importance`. Dashboard page 2.

**Q3. If we contact the top 10 % of highest-risk customers, how much churn do we catch?**
`tz_gold.churn_gains` (hold-out only). Chart `05_churn_gains_curve`. Dashboard page 2.
Answer at tiny scale: roughly 55-60 % of churners are inside the top 10 % of scores — about 6x better than
contacting customers at random. Numbers move slightly with the random split and with scale.

## Revenue and bundles

**Q4. What is ARPU by region and customer group, and how is it moving?**
`tz_gold.region_summary`, `tz_gold.kpi_region_month`. Chart `01_revenue_active_by_month`. Dashboard page 1.

**Q5. Which bundles sell the most, and which earn the most?**
`tz_gold.bundle_performance`. Chart `07_bundle_revenue`. Dashboard page 5.
(Best-seller and top-earner are not always the same bundle — small daily bundles sell more often but a monthly
large bundle can earn more per sale.)

**Q6. Which payment channel brings the most loyal customers?**
`tz_gold.bundle_performance` (channel columns) joined with `tz_ml.retention_actions` (churn). Dashboard page 5, query
`p5_channel_mix`. In this data mobile-money-registered customers churn less (see `p3_device_and_mobile_money`), which
is one input to that question but not the whole answer — cut by channel share per customer for a direct test.

## Network and care

**Q7. How does network quality vary by region, and does it affect churn?**
`tz_gold.network_quality_region` for quality; customer-level link via `tz_gold.customer_360.cell_outage_minutes_90d`
joined to `tz_ml.labels`. Chart `03_churn_vs_cell_outage`. Dashboard page 4.
Answer: customers on the worst-quality quintile of cells churn several times more than those on the best. This is
partly built into the simulation on purpose (see `docs/data_assumptions.md` §2), so treat the *pattern* as the
finding, not the exact multiplier.

**Q8. How much do power failures affect service, and where?**
`tz_gold.network_quality_region` (power_failures, outage_minutes). Chart `08_power_failures_by_region`. Dashboard page 4.

**Q9. Does slow ticket resolution push customers away?**
`tz_gold.customer_360` (slow_tickets_90d, open_tickets) joined to `tz_ml.labels`. Dashboard page 4, query
`p4_churn_by_tickets`.

## Customers and opportunity

**Q10. What customer segments do we have, and how do they differ?**
`tz_gold.segment_profile`, `tz_ml.customer_segments`. Chart `04_segments_arpu_churn`. Dashboard page 3.

**Q11. Which customers are ready for a bigger data bundle?**
`tz_ml.upsell_scores`, `tz_ml.retention_actions` (upsell_band = 'High'). Dashboard page 5, query `p5_upsell_by_segment`.

**Q12. How do urban and rural customers differ in usage and churn?**
`tz_gold.customer_360` (area_type) joined to `tz_ml.labels`. Dashboard page 3, query `p3_urban_rural`.

## How confident should we be in the answers?

See `docs/data_assumptions.md` §5 ("Modelling limitations") before presenting any number outside this project as a
market fact. The short version: this is a **learning / portfolio pipeline on synthetic data**. The patterns and the
engineering are real and transferable; the specific percentages are not measurements of any real operator.
