# Runbook

## 1. First-time setup (Databricks)

1. Create / open a Databricks workspace (Free Edition is enough for `tiny` or `small` scale).
2. Import the whole `notebooks/` folder: **Workspace → Create → Folder** (e.g. `tz_telecom`), then inside it
   **⋮ → Import**, drag every file from `notebooks/`.
3. Attach to **Serverless** compute (or any cluster with Databricks Runtime ML for MLflow, though MLflow logging is
   best-effort and the pipeline runs fine without it).
4. Run notebooks **in this order** — each one assumes the previous ones have already run:

   | # | Notebook | What it needs | What it produces |
   |---|---|---|---|
   | 1 | `01_setup_environment` | nothing | Schemas + volumes |
   | 2 | `10_generate_source_data` | (1) | Landing files (simulated source systems) |
   | 3 | `20_ingest_bronze` | (2) | Bronze tables |
   | 4 | `30_build_silver_with_dq` | (3) | Silver tables + `dq_quarantine` |
   | 5 | `40_build_customer_360` | (4) | `customer_360`, ML features, labels |
   | 6 | `50_train_churn_model` | (5) | Churn model + metrics |
   | 7 | `55_segment_customers` | (5) | Segments + segment_profile |
   | 8 | `60_train_upsell_model` | (5) | Upsell model + metrics |
   | 9 | `70_score_and_recommend` | (6,7,8) | `retention_actions`, target list |
   | 10 | `80_build_kpi_marts` | (1-9) | All dashboard-ready Gold tables |
   | 11 | `85_visual_analytics` | (10) | 10 PNG charts in the `artifacts/images` volume |

   Notebooks 6, 7 and 8 do not depend on each other and can run in any order (or in parallel jobs) once 5 has run.

5. Set the `scale` widget on each notebook (or as a job parameter): `tiny` for a 2-minute smoke test, `small`
   (default, ~20k customers) for normal use, `standard` (~100k) for a heavier, more "real" run. Every notebook must
   use the **same** `scale` value in one pipeline run.

## 2. Running it as a scheduled Job

1. **Jobs & Pipelines → Create job**.
2. Add one task per notebook, in the order above, wiring each task's "Depends on" to the previous one (tasks 6-8
   can all depend on task 5 directly, and task 9 depends on 6, 7 and 8).
3. Set a job parameter `scale` and reference it as `{{job.parameters.scale}}` in each task's notebook parameters.
4. Optional: a Corn schedule, e.g. daily at 02:00 Africa/Dar_es_Salaam, if you point step 2
   (`10_generate_source_data`) at a real landing feed instead of the simulator — see §4.

`databricks.yml` in the project root describes the same job as code (Databricks Asset Bundle) if you prefer
`databricks bundle deploy` over building the job by hand.

## 3. Rebuilding after a code change

* Changed a **Silver rule**? Re-run from step 4 (`30_build_silver_with_dq`) onward.
* Changed a **feature**? Re-run from step 5 (`40_build_customer_360`) onward, then retrain (6, 7, 8) and rescore (9).
* Changed **only the dashboard SQL**? No notebook re-run needed — edit the dataset in the dashboard's Data tab.
* Changed a **business parameter** (`RISK_HIGH_SHARE`, `LARGE_DATA_MB`, ...) in `00_config.py`? Re-run from step 5
  onward (these parameters affect labels and/or scoring).

## 4. Replacing the simulator with real data

`10_generate_source_data` exists only because there is no real Simba Telecom. To point the pipeline at real source
systems:

1. Land real extracts as Parquet (or any format Spark reads) at the same nine paths under
   `/Volumes/<catalog>/tz_raw/landing/` (or change `LANDING_PATH` in `00_config.py`), with the same column names
   listed in `docs/data_dictionary.md`.
2. Skip `10_generate_source_data` entirely.
3. Run `20_ingest_bronze` onward unchanged — Bronze, Silver, Gold, and the ML notebooks do not know or care that the
   data used to be simulated.
4. Revisit `docs/data_assumptions.md`: every invented parameter there (bundle prices, region weights, churn
   coefficients) becomes irrelevant once real data drives the numbers; only the *structure* of the pipeline carries
   over.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CREATE VOLUME` / `CREATE SCHEMA` fails with "catalog not found" | Your catalog is not named `workspace` | Set the `catalog` widget on `00_config` to your catalog name |
| `30_build_silver_with_dq` fails on an assert | A rule is rejecting more or fewer rows than expected, or a new column was added without an allowed-values list | Check `tz_gold.data_quality_summary` and `dq_rule_failures` for the failing table/rule |
| `40_build_customer_360` assert on nulls | A new feature was added to `ML_FEATURES` without a `fillna` | Add the column to `zero_cols` in `40_build_customer_360.py`, or give it an explicit default |
| Model training is slow or the cluster is out of memory | `scale = standard` on very small compute | Use `scale = small` or `tiny`, or a bigger cluster |
| GBT not available on some serverless environments | Certain Spark builds omit `GBTClassifier` | The notebook already falls back to Random Forest automatically; check the printed message |
| Dashboard chart shows no data | Wrong catalog name in the SQL, or an upstream notebook was not re-run | Re-run `sed -i 's/workspace\./<catalog>./g' sql/dashboard_queries.sql`; re-run `80_build_kpi_marts` |
| `tests/run_local_pipeline.py` runs out of memory | Local laptop rehearsal at `small`/`standard` scale on limited RAM | Use `--scale tiny` for local rehearsal; `small`/`standard` are meant for Databricks compute |

## 6. Local rehearsal (before touching Databricks)

```bash
pip install -r requirements-local.txt
python tests/run_local_pipeline.py --scale tiny --root /tmp/tz_dev   # ~3-5 minutes
pytest -q tests/test_pipeline_outputs.py                              # re-runs the pipeline unless TZ_REUSE_ROOT is set
TZ_REUSE_ROOT=/tmp/tz_dev pytest -q tests/test_pipeline_outputs.py    # reuse the folder above instead of re-running
```

This runs the exact same notebook code locally (Delta → Parquet, Volumes → local folders — see the comment at the
top of `tests/run_local_pipeline.py`), so a change can be checked in minutes before it is run on Databricks.
