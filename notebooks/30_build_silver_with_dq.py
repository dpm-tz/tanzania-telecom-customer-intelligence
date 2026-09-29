# Databricks notebook source
# MAGIC %md
# MAGIC # 30 · Silver: validated data with a quarantine
# MAGIC
# MAGIC Every Bronze table goes through **declarative data-quality rules**. Rows that pass move to Silver.
# MAGIC Rows that fail are **not silently deleted**: they go to `tz_silver.dq_quarantine` together with the
# MAGIC names of the rules they broke and the original record (JSON), so data owners can fix the source.
# MAGIC
# MAGIC ```
# MAGIC bronze.X  ──▶  rules (null keys, ranges, allowed values, duplicates, referential integrity)
# MAGIC                 ├── pass ──▶ silver.X
# MAGIC                 └── fail ──▶ silver.dq_quarantine (source_table, record_key, failed_rules, record_json)
# MAGIC ```
# MAGIC
# MAGIC Guarantee checked at the end: **bronze rows = silver rows + quarantined rows**, for every table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import Window

TS_MIN = f"TIMESTAMP'{SPAN_START} 00:00:00'"
TS_MAX = f"TIMESTAMP'{SPAN_END + timedelta(days=1)} 00:00:00'"

spark.sql(f"DROP TABLE IF EXISTS {tbl('silver', 'dq_quarantine')}")  # noqa: F821
summary_rows = []


def build_silver(name, key, rules, refs=(), event_col=None):
    """Validate bronze.<name>, write silver.<name>, append rejects to the quarantine."""
    bronze = read_table("bronze", name)
    df = bronze
    for ref_df, on in refs:
        df = df.join(F.broadcast(ref_df), on, "left")

    df = df.withColumn("_rn", F.row_number().over(Window.partitionBy(key).orderBy("_ingested_at")))
    checks = list(rules) + [("duplicate_key", "_rn = 1")]
    failed = F.filter(
        F.array(*[F.when(~F.coalesce(F.expr(expr), F.lit(False)), F.lit(rule)) for rule, expr in checks]),
        lambda x: x.isNotNull())
    df = df.withColumn("_failed", failed)

    good = df.filter(F.size("_failed") == 0).select(*bronze.columns)
    if event_col:
        good = good.withColumn("event_date", F.to_date(F.col(event_col)))
    good = good.withColumn("_validated_at", F.current_timestamp())
    save_table(good, "silver", name)

    payload_cols = [F.col(c) for c in bronze.columns if not c.startswith("_")]
    bad = (df.filter(F.size("_failed") > 0)
           .select(F.lit(name).alias("source_table"),
                   F.col(key).cast("string").alias("record_key"),
                   F.concat_ws(",", "_failed").alias("failed_rules"),
                   F.to_json(F.struct(*payload_cols)).alias("record_json"),
                   F.current_timestamp().alias("quarantined_at")))
    save_table(bad, "silver", "dq_quarantine", mode="append")

    n_bronze, n_silver = bronze.count(), read_table("silver", name).count()
    summary_rows.append((name, n_bronze, n_silver, n_bronze - n_silver))
    print(f"silver.{name:<20} bronze={n_bronze:>10,}  silver={n_silver:>10,}  quarantined={n_bronze - n_silver:>8,}")


def allowed(col, values):
    return f"{col} IN ({sql_list(values)})"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference tables first (other tables validate against them)

# COMMAND ----------

build_silver("regions_cells", "cell_id", [
    ("cell_id_not_null", "cell_id IS NOT NULL"),
    ("technology_valid", allowed("technology", ["2G", "3G", "4G"])),
    ("area_type_valid", allowed("area_type", AREA_TYPES)),
    ("region_valid", allowed("region", REGIONS)),
])

build_silver("bundle_catalog", "bundle_id", [
    ("bundle_id_not_null", "bundle_id IS NOT NULL"),
    ("price_positive", "price_tzs > 0"),
])

cells_ref = read_table("silver", "regions_cells").select("cell_id", F.lit(True).alias("_ok_cell"))
bundles_ref = read_table("silver", "bundle_catalog").select("bundle_id", F.lit(True).alias("_ok_bundle"))

# COMMAND ----------

build_silver("customers", "customer_id", [
    ("customer_id_not_null", "customer_id IS NOT NULL"),
    ("age_in_range_15_100", "age BETWEEN 15 AND 100"),
    ("region_valid", allowed("region", REGIONS)),
    ("gender_valid", allowed("gender", GENDERS)),
    ("area_type_valid", allowed("area_type", AREA_TYPES)),
    ("device_type_valid", allowed("device_type", DEVICE_TYPES)),
    ("activation_date_valid", f"activation_date IS NOT NULL AND activation_date <= DATE'{OBS_END}'"),
    ("home_cell_exists", "_ok_cell IS NOT NULL"),
], refs=[(cells_ref.select(F.col("cell_id").alias("home_cell_id"), "_ok_cell"), "home_cell_id")])

customers_ref = read_table("silver", "customers").select("customer_id", F.lit(True).alias("_ok_customer"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Event tables

# COMMAND ----------

CUSTOMER_RULES = [
    ("customer_id_not_null", "customer_id IS NOT NULL"),
    ("customer_exists", "_ok_customer IS NOT NULL"),
]
TS_RULE = ("event_ts_in_window", f"event_ts >= {TS_MIN} AND event_ts < {TS_MAX}")

build_silver("voice_usage", "call_id", CUSTOMER_RULES + [
    ("call_id_not_null", "call_id IS NOT NULL"),
    ("duration_1_to_14400s", "duration_sec BETWEEN 1 AND 14400"),
    ("direction_valid", allowed("direction", names(DIRECTIONS))),
    ("call_type_valid", allowed("call_type", names(CALL_TYPES))),
    TS_RULE,
], refs=[(customers_ref, "customer_id")], event_col="event_ts")

build_silver("data_sessions", "session_id", CUSTOMER_RULES + [
    ("session_id_not_null", "session_id IS NOT NULL"),
    ("data_mb_positive_and_sane", "data_mb > 0 AND data_mb <= 50000"),
    ("app_category_valid", allowed("app_category", names(APP_CATEGORIES))),
    TS_RULE,
], refs=[(customers_ref, "customer_id")], event_col="event_ts")

build_silver("bundle_purchases", "purchase_id", CUSTOMER_RULES + [
    ("purchase_id_not_null", "purchase_id IS NOT NULL"),
    ("bundle_exists", "_ok_bundle IS NOT NULL"),
    ("amount_positive", "amount_tzs > 0"),
    ("channel_valid", allowed("channel", names(PURCHASE_CHANNELS))),
    TS_RULE,
], refs=[(customers_ref, "customer_id"), (bundles_ref, "bundle_id")], event_col="event_ts")

build_silver("mobile_money_txns", "txn_id", CUSTOMER_RULES + [
    ("txn_id_not_null", "txn_id IS NOT NULL"),
    ("amount_100_to_5m_tzs", "amount_tzs BETWEEN 100 AND 5000000"),
    ("txn_type_valid", allowed("txn_type", names(MM_TXN_TYPES))),
    ("channel_valid", allowed("channel", ["agent", "ussd", "app"])),
    TS_RULE,
], refs=[(customers_ref, "customer_id")], event_col="event_ts")

build_silver("care_tickets", "ticket_id", CUSTOMER_RULES + [
    ("ticket_id_not_null", "ticket_id IS NOT NULL"),
    ("category_valid", allowed("category", names(TICKET_CATEGORIES))),
    ("status_valid", allowed("status", ["resolved", "open"])),
    ("resolved_not_before_created", "resolved_ts IS NULL OR resolved_ts >= created_ts"),
    ("created_in_window", f"created_ts >= {TS_MIN} AND created_ts < {TS_MAX}"),
], refs=[(customers_ref, "customer_id")], event_col="created_ts")

build_silver("network_events", "network_event_id", [
    ("network_event_id_not_null", "network_event_id IS NOT NULL"),
    ("cell_exists", "_ok_cell IS NOT NULL"),
    ("severity_1_to_3", "severity BETWEEN 1 AND 3"),
    ("event_type_valid", allowed("event_type", NETWORK_EVENT_TYPES)),
    ("duration_non_negative", "duration_minutes >= 0"),
    TS_RULE,
], refs=[(cells_ref, "cell_id")], event_col="event_ts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data-quality report

# COMMAND ----------

summary = (spark.createDataFrame(summary_rows, "table_name string, bronze_rows long, silver_rows long, quarantined_rows long")  # noqa: F821
           .withColumn("pct_rejected", F.round(F.try_divide(F.col("quarantined_rows") * 100.0, F.col("bronze_rows")), 3))
           .withColumn("run_ts", F.current_timestamp()))
save_table(summary, "silver", "dq_summary")

q = read_table("silver", "dq_quarantine")
rule_failures = (q.select("source_table", F.explode(F.split("failed_rules", ",")).alias("rule"))
                 .groupBy("source_table", "rule").count().withColumnRenamed("count", "failed_rows")
                 .withColumn("run_ts", F.current_timestamp()))
save_table(rule_failures, "silver", "dq_rule_failures")

banner("Rows rejected per table")
show(read_table("silver", "dq_summary").orderBy(F.desc("pct_rejected")), 20)
banner("Rows failing each rule")
show(read_table("silver", "dq_rule_failures").orderBy("source_table", F.desc("failed_rows")), 60)

# The conservation law of data quality: nothing disappears without a trace.
bad_tables = [r for r in summary_rows if r[1] != r[2] + r[3]]
assert not bad_tables, f"bronze != silver + quarantine for {bad_tables}"
assert q.count() == sum(r[3] for r in summary_rows), "quarantine row count mismatch"
print("\nOK: bronze = silver + quarantine for every table.")
