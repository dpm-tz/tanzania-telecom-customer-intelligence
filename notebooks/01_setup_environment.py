# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Set up the environment
# MAGIC Creates the five schemas and two Unity Catalog volumes used by the project, then proves that the
# MAGIC notebook can write to them. Run once (it is safe to re-run).
# MAGIC
# MAGIC | Object | Purpose |
# MAGIC |---|---|
# MAGIC | `tz_raw` + volume `landing` | Raw files delivered by the (simulated) source systems |
# MAGIC | `tz_bronze` | Raw data in Delta, plus ingestion metadata |
# MAGIC | `tz_silver` | Validated data + `dq_quarantine` of rejected rows |
# MAGIC | `tz_gold` | Customer 360 and business marts that feed the dashboard |
# MAGIC | `tz_ml` | Features, labels, scores, segments, recommendations + volume `artifacts` (models, images) |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

if IS_LOCAL:
    print("Local rehearsal: schemas only (no Unity Catalog volumes).")
for layer, schema in SCHEMAS.items():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")  # noqa: F821
    print(f"schema ready : {CATALOG}.{schema}")

if not IS_LOCAL:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMAS['raw']}.landing")  # noqa: F821
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMAS['ml']}.artifacts")  # noqa: F821
    print(f"volume ready : {CATALOG}.{SCHEMAS['raw']}.landing")
    print(f"volume ready : {CATALOG}.{SCHEMAS['ml']}.artifacts")

# COMMAND ----------

# Prove that we can write where the project will write.
for folder in (LANDING_PATH, MODELS_PATH, IMAGES_PATH, EXPORTS_PATH):
    os.makedirs(folder, exist_ok=True)
    probe = os.path.join(folder, "_write_test.txt")
    with open(probe, "w") as fh:
        fh.write("ok")
    os.remove(probe)
    print(f"writable     : {folder}")

print("\nEnvironment is ready. Next: 10_generate_source_data")
