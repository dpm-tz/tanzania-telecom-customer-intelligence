# Building the AI/BI Dashboard

The SQL for every chart is already written in `sql/dashboard_queries.sql`, one query per dataset, each preceded by
a `-- @query: name | page | suggested chart` header. You only need to create datasets and drop them onto a canvas.

## Steps

1. In Databricks, click **Dashboards** in the left sidebar, then **Create dashboard**.
2. Open the **Data** tab, click **Create from SQL**, paste one query (everything between two `-- @query` markers,
   without the header comment), run it, and give the dataset the name from the header (e.g. `p1_headline_kpis`).
   Repeat for every query you want to use — you do not have to use all of them on your first pass.
3. Switch to the **Canvas** tab and add a visual for each dataset. The header comment names a suggested chart type
   and the columns to put on each axis.
4. Group visuals into pages matching the header's page name. Suggested page order:

   | Page | Purpose | Key visuals |
   |---|---|---|
   | 1 · Executive overview | Leadership snapshot | Headline counters, monthly trend, region table |
   | 2 · Churn & retention | Who is at risk and what to do | Churn by region, hold-out calibration, gains curve, driver bar, action table, target list |
   | 3 · Customers & segments | Who our customers are | Segment table, urban/rural, device × mobile-money, age band |
   | 4 · Network & care | Service quality | Power failures by region, churn vs outage quintile, ticket categories |
   | 5 · Bundles & mobile money | Product and payments | Top bundles, channel mix, mobile-money by type, upsell by segment |
   | 6 · Trust | Is the pipeline healthy | Model performance table, data-quality bar |

5. Add a **region** filter (and optionally a **segment_name** filter) and attach it to every visual that has that
   column, so the whole dashboard can be sliced by region in one click.
6. Click **Publish**.

## Notes

* `p2_risk_band_holdout` and `p2_gains` use `tz_ml.churn_test_scored`, i.e. **hold-out customers only** — this is the
  honest read of model quality. Other tables (e.g. `region_summary`) include every customer and are meant for
  business sizing, not model evaluation.
* If your catalog is not named `workspace`, replace `workspace.` with your catalog name in every query, or run:
  `sed -i 's/workspace\./<your_catalog>./g' sql/dashboard_queries.sql` before copying.
* Every query in this file is executed by `tests/test_pipeline_outputs.py::test_dashboard_query_runs_and_returns_rows`,
  so if a query is broken after you edit the schema, the test suite will catch it.
