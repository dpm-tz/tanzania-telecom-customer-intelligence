# Tanzania Telecom Customer Intelligence

An end-to-end, Databricks-native customer intelligence pipeline for a fictional Tanzanian mobile operator,
**Simba Telecom**: synthetic-but-realistic prepaid telecom data, a governed medallion lakehouse, three production-
style ML models, a business-ready recommendation layer, and an AI/BI dashboard.

> **All data is synthetic.** "Simba Telecom" does not exist. Prices, region shares, churn drivers and every other
> business parameter are documented, invented assumptions — see `docs/data_assumptions.md` before quoting any number.
> The goal of this project is to practice real data-engineering and ML-engineering work end to end, the way a data
> team at a telecom operator in Tanzania would.

## 1. Objective

Build a system that lets a telecom's retention, marketing, network and analytics teams answer, every day:

1. Which customers are about to churn, why, and what should we do about it today?
2. Which customers are ready for a bigger data bundle?
3. What do our customer segments look like, and how do they differ?
4. Where is the network hurting the customer experience, and how much does that cost us?
5. Which bundles and payment channels actually make money?

The full list of 12 business questions, each mapped to the table and chart that answers it, is in
`docs/business_questions.md`.

## 2. Architecture

![Architecture](docs/images/architecture.png)

Medallion lakehouse (Bronze → Silver → Gold) on Unity Catalog, feeding an ML layer (features, labels, scores,
segments) and a serving layer (AI/BI dashboard, a prioritised retention list, and a set of PNG charts). See
`docs/images/data_model.png` for the nine-table source data model and `docs/data_dictionary.md` for every column.

## 3. Datasets (9 source tables, simulated)

| Dataset | Grain | ~Rows at `small` scale (20k customers) |
|---|---|---:|
| `customers` | 1 per customer | 20,000 |
| `regions_cells` | 1 per cell tower | ~750 |
| `bundle_catalog` | 1 per bundle | 18 |
| `voice_usage` | 1 per call | ~1,000,000 |
| `data_sessions` | 1 per session | ~1,600,000 |
| `bundle_purchases` | 1 per purchase | ~300,000 |
| `mobile_money_txns` | 1 per transaction | ~600,000 |
| `network_events` | 1 per network incident | ~165,000 |
| `care_tickets` | 1 per ticket | ~24,000 |

Full column definitions: `docs/data_dictionary.md`. Tanzania-specific modelling choices (prepaid churn definition,
multi-SIM, mobile money, urban/rural, bundle naming in Swahili, power-grid effects on cell reliability) are explained
in `docs/data_assumptions.md`.

## 4. Models

| Model | Question | Algorithm(s) | Notebook |
|---|---|---|---|
| **Churn** | Will this customer go inactive in the next 30 days? | Logistic Regression vs Gradient-Boosted Trees | `50_train_churn_model.py` |
| **Segmentation** | What natural customer groups exist? | K-Means (k chosen by silhouette) | `55_segment_customers.py` |
| **Upsell propensity** | Will this customer buy a large data bundle? | Logistic Regression vs Random Forest | `60_train_upsell_model.py` |

Every model is trained with a champion/challenger comparison, evaluated with metrics appropriate for an imbalanced
business problem (AUC-PR, recall/lift in the top-N% the business can act on — not plain accuracy), saved as a
versioned artifact with a `champion.json` pointer, and documented in its own model card under `docs/model_cards/`.
**Churn is derived from real customer behaviour, not a random column** — see `docs/data_assumptions.md` §2 and the
tests in `tests/test_pipeline_outputs.py` that verify this.

## 5. Business questions this project answers

See `docs/business_questions.md` for the full list of 12 questions across churn/retention, revenue/bundles,
network/care and customer segments/opportunity, each linked to its table, chart and dashboard page.

## 6. Dashboard

A 6-page AI/BI Dashboard built from ready-made SQL in `sql/dashboard_queries.sql`:

| Page | Content |
|---|---|
| 1 · Executive overview | Headline KPIs, revenue/active-customer trend, region table |
| 2 · Churn & retention | Churn by region, hold-out model calibration, gains curve, top drivers, recommended actions, prioritised target list |
| 3 · Customers & segments | Segment profiles, urban vs rural, device & mobile-money cuts, age bands |
| 4 · Network & care | Power failures and outages by region, churn vs network quality, ticket-resolution performance |
| 5 · Bundles & mobile money | Top bundles, payment-channel mix, mobile-money value, upsell candidates by segment |
| 6 · Trust | Latest model metrics, data-quality summary |

Step-by-step build instructions: `docs/dashboard_guide.md`. Every query in the file is executed by the automated
test suite, so it is verified to run and return data.

## 7. Visualisations

`notebooks/85_visual_analytics.py` produces 10 charts (revenue trend, churn by region, churn vs network quality,
segment value/churn, the churn model's gains curve, feature importance, bundle revenue, power failures by region,
mobile-money value, and predicted-vs-actual calibration), shown inline and saved as PNG to
`Catalog → workspace → tz_ml → artifacts → images` for reuse in slides or reports.

## 8. Project layout

```
tanzania-telecom-customer-intelligence/
├── README.md                     <- you are here
├── databricks.yml                <- Databricks Asset Bundle: deploy the whole pipeline as a Job
├── requirements-local.txt        <- only needed to rehearse the pipeline on a laptop
├── notebooks/
│   ├── 00_config.py              <- single source of truth: schemas, dates, geography, bundle catalogue, helpers
│   ├── 00_ml_utils.py            <- shared ML helpers: metrics, model save/load, MLflow logging
│   ├── 01_setup_environment.py   <- create schemas and volumes
│   ├── 10_generate_source_data.py<- simulate the 9 source systems (behaviour-driven churn, injected dirty data)
│   ├── 20_ingest_bronze.py       <- copy raw data into Bronze + audit columns
│   ├── 30_build_silver_with_dq.py<- validate, dedupe, quarantine bad rows
│   ├── 40_build_customer_360.py  <- Customer 360, ML features, behaviour-derived labels
│   ├── 50_train_churn_model.py   <- champion/challenger churn model
│   ├── 55_segment_customers.py   <- K-Means segmentation with business naming
│   ├── 60_train_upsell_model.py  <- champion/challenger upsell model
│   ├── 70_score_and_recommend.py <- scores, risk bands, drivers, recommended actions
│   ├── 80_build_kpi_marts.py     <- Gold marts that feed the dashboard
│   └── 85_visual_analytics.py    <- 10 PNG charts
├── sql/
│   └── dashboard_queries.sql     <- one ready-to-paste SQL query per dashboard visual
├── tests/
│   ├── run_local_pipeline.py     <- runs every notebook locally with local Spark (no Databricks needed)
│   └── test_pipeline_outputs.py  <- 40+ automated checks: data quality, no-leakage, model quality, dashboard SQL
└── docs/
    ├── make_diagrams.py, images/architecture.png, images/data_model.png
    ├── data_dictionary.md        <- every table, every column
    ├── data_assumptions.md       <- what's realistic, what's invented, and every modelling limitation
    ├── business_questions.md     <- 12 questions, each mapped to its answer
    ├── dashboard_guide.md        <- how to build the AI/BI dashboard from sql/dashboard_queries.sql
    ├── runbook.md                <- run order, scheduling, troubleshooting, how to plug in real data
    └── model_cards/              <- one card per model: intended use, metrics, limitations
```

## 9. How to run it

### On Databricks (recommended)

1. Import the `notebooks/` folder into your Workspace (drag-and-drop the 11 files).
2. Run notebooks **01 → 10 → 20 → 30 → 40**, then **50, 55, 60** (any order), then **70 → 80 → 85**.
3. Use the `scale` widget on each notebook: `tiny` (3k customers, smoke test), `small` (20k, default),
   `standard` (100k). Use the **same** scale for every notebook in one run.
4. Build the dashboard from `sql/dashboard_queries.sql` — see `docs/dashboard_guide.md`.

Full details, a dependency table, scheduling as a Job, and troubleshooting: `docs/runbook.md`.

### Locally, before touching Databricks

```bash
pip install -r requirements-local.txt
python tests/run_local_pipeline.py --scale tiny --root /tmp/tz_dev
pytest -q tests/test_pipeline_outputs.py
```

This runs the exact same notebook files with a local Spark session (Delta → Parquet, Volumes → local folders), so
you can validate a change in minutes before running it on Databricks. See the header of
`tests/run_local_pipeline.py` for exactly what differs locally.

## 10. Data quality and testing

* **Silver** enforces declarative rules (not-null keys, ranges, allowed values, referential integrity, duplicates).
  Rejected rows are never silently dropped — they go to `tz_silver.dq_quarantine` with the rule(s) they failed and
  the original row as JSON. `bronze = silver + quarantine` is checked by an assertion in the notebook and by a test.
* **No data leakage**: features only use events on or before the snapshot date; labels only use the 30 days after
  it. Verified by `test_features_do_not_look_into_the_future` and `test_churn_label_is_derived_from_real_inactivity`.
* **40+ automated tests** (`tests/test_pipeline_outputs.py`) cover data quality, no-leakage, model quality bars
  (e.g. "the churn model's AUC-PR must clearly beat the base rate"), segment coverage, risk-band ordering, and that
  every single dashboard SQL query actually runs and returns rows.

## 11. Honest limitations (read before presenting results externally)

This is a **learning / portfolio-grade pipeline on synthetic data**, built to practice the full craft of production
data + ML engineering: medallion architecture with real data-quality gates, leakage-safe feature engineering,
champion/challenger modelling with business-relevant metrics, versioned model artifacts, and a dashboard grounded in
tested SQL. The engineering patterns are production-style and reusable as-is. The specific numbers (churn rate,
ARPU, network impact) are **not** measurements of any real operator and should never be quoted as Tanzanian market
data. Full details, including why a random train/test split is a deliberate simplification and what a production
deployment should do differently, are in `docs/data_assumptions.md` §5 and each model card in `docs/model_cards/`.
