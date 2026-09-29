# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Bronze: ingest raw files as-is
# MAGIC
# MAGIC **Rule of Bronze: copy, never fix.** Every row from the landing files is stored exactly as delivered
# MAGIC (duplicates and bad values included) plus two audit columns:
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `_ingested_at` | When the row entered the lakehouse |
# MAGIC | `_source_file` | Which landing file it came from |
# MAGIC
# MAGIC Keeping the untouched original means Silver can always be rebuilt if a cleaning rule changes.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

SOURCES = ["customers", "regions_cells", "bundle_catalog", "voice_usage", "data_sessions",
           "bundle_purchases", "mobile_money_txns", "network_events", "care_tickets"]

log_rows = []
for name in SOURCES:
    raw = spark.read.parquet(f"{LANDING_PATH}/{name}")  # noqa: F821
    bronze = (raw
              .withColumn("_ingested_at", F.current_timestamp())
              .withColumn("_source_file", F.col("_metadata.file_path")))
    save_table(bronze, "bronze", name)
    n = read_table("bronze", name).count()
    log_rows.append((name, n))
    print(f"bronze.{name:<20}{n:>12,} rows")

# COMMAND ----------

log = (spark.createDataFrame(log_rows, "table_name string, row_count long")  # noqa: F821
       .withColumn("ingested_at", F.current_timestamp())
       .withColumn("scale", F.lit(SCALE)))
save_table(log, "bronze", "ingestion_log", mode="append")
show(read_table("bronze", "ingestion_log").orderBy(F.desc("ingested_at")), 12)
