# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Project configuration
# MAGIC
# MAGIC Single source of truth for **Tanzania Telecom Customer Intelligence**.
# MAGIC Every other notebook starts with `%run ./00_config`, so nothing is hard-coded twice.
# MAGIC
# MAGIC | Widget | Meaning |
# MAGIC |---|---|
# MAGIC | `scale` | `tiny` (3k customers, smoke test) · `small` (20k, default) · `standard` (100k) |
# MAGIC | `catalog` | Unity Catalog catalog to write into (Databricks Free Edition: `workspace`) |
# MAGIC
# MAGIC All data is **synthetic** and the company "Simba Telecom" is **fictional**.

# COMMAND ----------

import os
import json
from datetime import date, timedelta
from itertools import chain

from pyspark.sql import functions as F

# ``TZ_LOCAL_TEST=1`` is only set by tests/run_local_pipeline.py (laptop rehearsal).
IS_LOCAL = os.environ.get("TZ_LOCAL_TEST") == "1"


def _param(name, default, choices=None):
    """Read a notebook/job parameter (widget). Locally: read env var TZ_<NAME>."""
    if IS_LOCAL:
        return os.environ.get(f"TZ_{name.upper()}", default)
    try:
        return dbutils.widgets.get(name)  # noqa: F821  (dbutils exists on Databricks)
    except Exception:
        if choices:
            dbutils.widgets.dropdown(name, default, choices, name)  # noqa: F821
        else:
            dbutils.widgets.text(name, default, name)  # noqa: F821
        return dbutils.widgets.get(name)  # noqa: F821


SCALE = _param("scale", "small", ["tiny", "small", "standard"])
CATALOG = "spark_catalog" if IS_LOCAL else _param("catalog", "workspace")

try:
    spark.conf.set("spark.sql.session.timeZone", "Africa/Dar_es_Salaam")  # noqa: F821
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data volumes per scale
# MAGIC Event volumes are *targets*: the generator over-samples and filters, so real counts land within roughly ±15 %.

# COMMAND ----------

# Per-customer multipliers reproduce the 100k "standard" profile (5M calls, 8M data sessions, ...).
_PER_CUSTOMER = dict(voice=50, data_sessions=80, purchases=15, mobile_money=30, tickets=1.2, network_events=8)
_CUSTOMERS = {"tiny": 3_000, "small": 20_000, "standard": 100_000}

N_CUSTOMERS = _CUSTOMERS[SCALE]
VOLUMES = {k: int(v * N_CUSTOMERS) for k, v in _PER_CUSTOMER.items()}
VOLUMES["customers"] = N_CUSTOMERS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time windows
# MAGIC
# MAGIC ```
# MAGIC  2026-04-01                          2026-06-29 | 2026-06-30                2026-07-29
# MAGIC  |<------ observation window (90 d) ------------>|<---- prediction window (30 d) --->|
# MAGIC        features are computed here                       churn / upsell labels here
# MAGIC ```
# MAGIC
# MAGIC Features only ever see events on or before `OBS_END`. Labels only look at the 30 days after it.
# MAGIC This prevents **data leakage** (the model can never see the answer).

# COMMAND ----------

OBS_END = date(2026, 6, 29)
FEATURE_DAYS = 90
LABEL_DAYS = 30

SPAN_START = OBS_END - timedelta(days=FEATURE_DAYS - 1)     # 2026-04-01
LABEL_START = OBS_END + timedelta(days=1)                    # 2026-06-30
LABEL_END = OBS_END + timedelta(days=LABEL_DAYS)             # 2026-07-29
SPAN_END = LABEL_END
SPAN_DAYS = (SPAN_END - SPAN_START).days + 1                 # 120

W30_START = OBS_END - timedelta(days=29)
W90_START = SPAN_START
PRIOR60_END = OBS_END - timedelta(days=30)

GLOBAL_SEED = 2026

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog layout, storage paths and business parameters

# COMMAND ----------

SCHEMAS = {"raw": "tz_raw", "bronze": "tz_bronze", "silver": "tz_silver", "gold": "tz_gold", "ml": "tz_ml"}
TABLE_FORMAT = "parquet" if IS_LOCAL else "delta"

if IS_LOCAL:
    _root = os.environ.get("TZ_LOCAL_ROOT", "/tmp/tz_local")
    LANDING_PATH = f"{_root}/landing"
    ARTIFACTS_PATH = f"{_root}/artifacts"
else:
    LANDING_PATH = f"/Volumes/{CATALOG}/{SCHEMAS['raw']}/landing"
    ARTIFACTS_PATH = f"/Volumes/{CATALOG}/{SCHEMAS['ml']}/artifacts"

MODELS_PATH = f"{ARTIFACTS_PATH}/models"
IMAGES_PATH = f"{ARTIFACTS_PATH}/images"
EXPORTS_PATH = f"{ARTIFACTS_PATH}/exports"

# Business parameters (documented in docs/data_assumptions.md)
RISK_HIGH_SHARE = 0.10          # top 10 % of churn scores  -> High   (retention team capacity)
RISK_MEDIUM_SHARE = 0.20        # next 20 %                 -> Medium
UPSELL_HIGH_SHARE = 0.15        # top 15 % upsell scores    -> High propensity
REVENUE_HORIZON_MONTHS = 3      # assumed remaining lifetime used for "revenue at risk"
CURRENCY = "TZS"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Master data: geography (regions, districts, urban/rural)
# MAGIC Classification into urban/rural is illustrative and only used to drive simulated behaviour.

# COMMAND ----------

_GEO = {
    "Dar es Salaam": (0.20, [("Kinondoni", "urban", .28), ("Ilala", "urban", .24), ("Temeke", "urban", .26),
                             ("Ubungo", "urban", .14), ("Kigamboni", "urban", .08)]),
    "Mwanza": (0.09, [("Nyamagana", "urban", .34), ("Ilemela", "urban", .26), ("Magu", "rural", .16),
                      ("Sengerema", "rural", .14), ("Kwimba", "rural", .10)]),
    "Arusha": (0.08, [("Arusha City", "urban", .42), ("Arumeru", "rural", .24), ("Karatu", "rural", .14),
                      ("Monduli", "rural", .10), ("Longido", "rural", .10)]),
    "Dodoma": (0.07, [("Dodoma City", "urban", .46), ("Bahi", "rural", .16), ("Chamwino", "rural", .20),
                      ("Kondoa", "rural", .18)]),
    "Mbeya": (0.08, [("Mbeya City", "urban", .38), ("Rungwe", "rural", .24), ("Kyela", "rural", .20),
                     ("Mbarali", "rural", .18)]),
    "Morogoro": (0.07, [("Morogoro Urban", "urban", .36), ("Kilombero", "rural", .24), ("Kilosa", "rural", .22),
                        ("Mvomero", "rural", .18)]),
    "Tanga": (0.06, [("Tanga City", "urban", .40), ("Korogwe", "rural", .22), ("Muheza", "rural", .20),
                     ("Lushoto", "rural", .18)]),
    "Kilimanjaro": (0.06, [("Moshi Urban", "urban", .34), ("Moshi Rural", "rural", .24), ("Hai", "rural", .22),
                           ("Rombo", "rural", .20)]),
    "Kagera": (0.06, [("Bukoba Urban", "urban", .30), ("Muleba", "rural", .28), ("Karagwe", "rural", .22),
                      ("Biharamulo", "rural", .20)]),
    "Tabora": (0.05, [("Tabora Urban", "urban", .34), ("Nzega", "rural", .26), ("Igunga", "rural", .22),
                      ("Urambo", "rural", .18)]),
    "Iringa": (0.04, [("Iringa Urban", "urban", .42), ("Mufindi", "rural", .32), ("Kilolo", "rural", .26)]),
    "Zanzibar (Unguja)": (0.05, [("Mjini Magharibi", "urban", .55), ("Kaskazini A", "rural", .25),
                                 ("Kusini", "rural", .20)]),
    "Mtwara": (0.04, [("Mtwara Urban", "urban", .40), ("Masasi", "rural", .32), ("Newala", "rural", .28)]),
    "Kigoma": (0.05, [("Kigoma-Ujiji", "urban", .38), ("Kasulu", "rural", .34), ("Kibondo", "rural", .28)]),
}

REGIONS = list(_GEO.keys())
DISTRICTS = [
    dict(region=r, district=d, area_type=a, weight=rw * dw)
    for r, (rw, ds) in _GEO.items() for d, a, dw in ds
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Master data: bundle catalogue (illustrative prices in TZS, not any operator's real tariff)

# COMMAND ----------

BUNDLES = [
    # id, code, category, price_tzs, validity_days, data_mb, voice_min, sms
    (1, "SIKU_300MB", "data", 500, 1, 300, 0, 0),
    (2, "SIKU_1GB", "data", 1000, 1, 1024, 0, 0),
    (3, "SIKU_3GB", "data", 2000, 1, 3072, 0, 0),
    (4, "WIKI_2GB", "data", 3000, 7, 2048, 0, 0),
    (5, "WIKI_5GB", "data", 5000, 7, 5120, 0, 0),
    (6, "WIKI_10GB", "data", 8000, 7, 10240, 0, 0),
    (7, "MWEZI_5GB", "data", 10000, 30, 5120, 0, 0),
    (8, "MWEZI_15GB", "data", 20000, 30, 15360, 0, 0),
    (9, "MWEZI_30GB", "data", 35000, 30, 30720, 0, 0),
    (10, "MWEZI_60GB", "data", 60000, 30, 61440, 0, 0),
    (11, "SIKU_DAKIKA_30", "voice", 500, 1, 0, 30, 0),
    (12, "WIKI_DAKIKA_120", "voice", 2000, 7, 0, 120, 0),
    (13, "MWEZI_DAKIKA_500", "voice", 8000, 30, 0, 500, 0),
    (14, "MWEZI_DAKIKA_3000", "voice", 15000, 30, 0, 3000, 0),
    (15, "COMBO_SIKU", "combo", 1000, 1, 500, 20, 0),
    (16, "COMBO_WIKI", "combo", 5000, 7, 3072, 80, 0),
    (17, "COMBO_MWEZI", "combo", 15000, 30, 10240, 300, 0),
    (18, "SMS_MWEZI", "sms", 2000, 30, 0, 0, 1000),
]
LARGE_DATA_MB = 10240                       # "large data bundle" = at least 10 GB
LARGE_BUNDLE_IDS = [b[0] for b in BUNDLES if b[5] >= LARGE_DATA_MB]
BUNDLE_PRICE = {b[0]: b[3] for b in BUNDLES}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Allowed values (used by the generator *and* by the Silver data-quality rules)

# COMMAND ----------

GENDERS = ["F", "M"]
AREA_TYPES = ["urban", "rural"]
DEVICE_TYPES = ["smartphone", "feature_phone"]
DIRECTIONS = [("outgoing", .62), ("incoming", .38)]
CALL_TYPES = [("on_net", .55), ("off_net", .40), ("international", .05)]
APP_CATEGORIES = [("whatsapp", .30), ("social_media", .20), ("video", .25), ("browsing", .15), ("other", .10)]
PURCHASE_CHANNELS = [("mobile_money", .52), ("agent", .22), ("ussd", .16), ("app", .10)]
MM_TXN_TYPES = [("send_money", .30), ("receive_money", .28), ("cash_in", .12), ("cash_out", .12),
                ("bill_payment", .08), ("airtime_purchase", .10)]
NETWORK_EVENT_TYPES = ["call_drop_spike", "weak_signal", "congestion", "cell_down", "power_failure"]
TICKET_CATEGORIES = [("network_coverage", .22), ("slow_internet", .18), ("billing", .14),
                     ("mobile_money_issue", .18), ("bundle_dispute", .12), ("sim_registration", .10), ("other", .06)]
TICKET_CHANNELS = [("call_centre", .35), ("shop", .25), ("whatsapp", .25), ("ussd", .15)]
NETWORK_TICKET_CATEGORIES = ["network_coverage", "slow_internet"]
ACQUISITION_CHANNELS = [("agent", .55), ("shop", .25), ("online", .08), ("referral", .12)]
REGISTRATION_STATUSES = [("fully_registered", .97), ("pending_verification", .03)]


def names(options):
    return [o[0] for o in options]

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML feature lists (shared by churn, upsell, scoring)

# COMMAND ----------

ML_FEATURES = [
    "age", "tenure_days", "is_multi_sim", "mobile_money_registered", "area_rural", "device_smartphone",
    "calls_out_90d", "calls_in_90d", "voice_minutes_90d", "avg_call_sec_90d", "offnet_share_90d",
    "intl_calls_90d", "calls_out_30d",
    "sessions_90d", "data_gb_90d", "data_gb_30d", "video_social_share_90d", "active_data_days_90d",
    "purchases_90d", "spend_tzs_90d", "spend_tzs_30d", "avg_purchase_tzs_90d",
    "distinct_bundle_types_90d", "large_bundle_purchases_90d",
    "mm_txns_90d", "mm_value_tzs_90d", "mm_txns_30d", "bill_payments_90d",
    "days_since_last_activity", "active_days_30d", "active_days_90d",
    "cell_events_30d", "cell_power_failures_30d", "cell_outage_minutes_90d",
    "tickets_90d", "network_tickets_90d", "open_tickets", "avg_resolution_hours", "slow_tickets_90d",
    "voice_trend_ratio", "data_trend_ratio", "spend_trend_ratio", "mm_trend_ratio",
]

SEGMENT_FEATURES = [
    "calls_out_90d", "voice_minutes_90d", "data_gb_90d", "spend_tzs_90d",
    "mm_txns_90d", "mm_value_tzs_90d", "active_days_90d", "tenure_days",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def tbl(layer, name):
    """Fully qualified table name, e.g. tbl('silver','customers') -> workspace.tz_silver.customers"""
    return f"{CATALOG}.{SCHEMAS[layer]}.{name}"


def read_table(layer, name):
    return spark.table(tbl(layer, name))  # noqa: F821


def save_table(df, layer, name, mode="overwrite"):
    writer = df.write.format(TABLE_FORMAT).mode(mode)
    if TABLE_FORMAT == "delta" and mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(tbl(layer, name))
    return tbl(layer, name)


def show(df, n=20):
    """display() on Databricks (gives charts/download); plain show() elsewhere."""
    try:
        display(df.limit(n))  # noqa: F821
    except NameError:
        df.show(n, truncate=False)


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def rnd(offset):
    """Deterministic uniform [0,1) column; offset keeps every random column independent."""
    return F.rand(GLOBAL_SEED + offset)


def rndn(offset):
    """Deterministic standard-normal column."""
    return F.randn(GLOBAL_SEED + offset)


def weighted_choice(u, options):
    """Turn a uniform column `u` into a categorical column using (value, weight) pairs.

    IMPORTANT: `u` must be a *materialised* column (F.col("_u_x")), never an inline F.rand(): Spark evaluates
    an inline rand() separately in every CASE WHEN branch and the resulting distribution is wrong.
    Use add_choice() below, which does this correctly.
    """
    total = float(sum(w for _, w in options))
    expr, cum = None, 0.0
    for value, weight in options[:-1]:
        cum += weight / total
        cond = u < F.lit(cum)
        expr = F.when(cond, F.lit(value)) if expr is None else expr.when(cond, F.lit(value))
    return expr.otherwise(F.lit(options[-1][0]))


def add_choice(df, name, offset, options):
    """Add categorical column `name` drawn from (value, weight) options, using one materialised uniform draw."""
    u = f"_u_{name}"
    return df.withColumn(u, rnd(offset)).withColumn(name, weighted_choice(F.col(u), options)).drop(u)


def map_from_dict(d):
    return F.create_map(*list(chain.from_iterable((F.lit(k), F.lit(v)) for k, v in d.items())))


def clip(col, lo, hi):
    return F.least(F.lit(float(hi)), F.greatest(F.lit(float(lo)), col))


def in_window(col, start, end):
    return (F.col(col) >= F.lit(start)) & (F.col(col) <= F.lit(end))


def flag_sum(cond):
    """Count rows where cond is true (inside an aggregation)."""
    return F.sum(F.when(cond, F.lit(1)).otherwise(F.lit(0)))


def sql_list(values):
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


print(f"Config loaded | catalog={CATALOG} | scale={SCALE} ({N_CUSTOMERS:,} customers) | "
      f"observation window {SPAN_START} -> {OBS_END} | label window {LABEL_START} -> {LABEL_END}")
