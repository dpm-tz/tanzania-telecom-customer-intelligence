# Databricks notebook source
# MAGIC %md
# MAGIC # 55 · Segment customers by behaviour (K-Means)
# MAGIC
# MAGIC **Question:** what kinds of customers do we have, and how do they differ?
# MAGIC
# MAGIC Method
# MAGIC 1. Take eight behaviour measures (calls, minutes, data GB, spend, mobile-money count and value,
# MAGIC    active days, tenure), apply `log1p` to tame skew, then standardise.
# MAGIC 2. Try k = 4 … 7 clusters (marketing cannot act on fewer than four segments). Score each with the
# MAGIC    **silhouette** (cohesion vs separation).
# MAGIC 3. Choose the k with the best silhouette **whose smallest cluster is at least 3 % of customers**
# MAGIC    (a segment nobody can act on is useless).
# MAGIC 4. Give each cluster a **business name** by comparing its profile with the other clusters.
# MAGIC
# MAGIC Outputs: `tz_ml.customer_segments`, `tz_gold.segment_profile`, `tz_ml.segmentation_eval`, saved models.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %run ./00_ml_utils

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler

c360 = read_table("gold", "customer_360")
seg_in = c360.select("customer_id", *[F.log1p(F.coalesce(F.col(c).cast("double"), F.lit(0.0))).alias(c)
                                      for c in SEGMENT_FEATURES])
seg_in = assemble(seg_in, SEGMENT_FEATURES).withColumnRenamed("features", "raw_features")

scaler_model = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(seg_in)
scaled = scaler_model.transform(seg_in)
n_customers = scaled.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Choose k

# COMMAND ----------

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
eval_rows, models = [], {}
for k in range(4, 8):
    model = KMeans(k=k, seed=42, maxIter=50, featuresCol="features").fit(scaled)
    pred = model.transform(scaled)
    sil = float(evaluator.evaluate(pred))
    sizes = {r["prediction"]: r["count"] for r in pred.groupBy("prediction").count().collect()}
    min_share = min(sizes.values()) / n_customers
    eval_rows.append((k, sil, float(min_share)))
    models[k] = model
    print(f"k={k}  silhouette={sil:.3f}  smallest cluster={min_share:.1%}")

eligible = [r for r in eval_rows if r[2] >= 0.03] or eval_rows
best_k = max(eligible, key=lambda r: r[1])[0]
print(f"\nChosen k = {best_k}")
save_table(spark.createDataFrame(eval_rows, "k int, silhouette double, smallest_cluster_share double")  # noqa: F821
           .withColumn("chosen", F.col("k") == best_k), "ml", "segmentation_eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profile the clusters and name them

# COMMAND ----------

kmeans_model = models[best_k]
assigned = kmeans_model.transform(scaled).select("customer_id", F.col("prediction").alias("cluster_id"))

labels = read_table("ml", "labels").select("customer_id", "churn_30d")
profile = (c360.join(assigned, "customer_id").join(labels, "customer_id")
           .groupBy("cluster_id").agg(
               F.count("*").alias("customers"),
               F.round(F.avg("calls_out_90d"), 1).alias("avg_calls_out_90d"),
               F.round(F.avg("voice_minutes_90d"), 1).alias("avg_voice_minutes_90d"),
               F.round(F.avg("data_gb_90d"), 2).alias("avg_data_gb_90d"),
               F.round(F.avg("spend_tzs_90d"), 0).alias("avg_spend_tzs_90d"),
               F.round(F.avg("arpu_monthly_tzs"), 0).alias("avg_arpu_monthly_tzs"),
               F.round(F.avg("mm_txns_90d"), 1).alias("avg_mm_txns_90d"),
               F.round(F.avg("mm_value_tzs_90d"), 0).alias("avg_mm_value_tzs_90d"),
               F.round(F.avg("active_days_90d"), 1).alias("avg_active_days_90d"),
               F.round(F.avg("tenure_days"), 0).alias("avg_tenure_days"),
               F.round(F.avg("mobile_money_registered"), 3).alias("mobile_money_share"),
               F.round(F.avg("area_rural"), 3).alias("rural_share"),
               F.round(F.avg("churn_30d"), 4).alias("actual_churn_rate"))
           .orderBy("cluster_id"))
prof = profile.toPandas()


def z(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


prof["z_data"] = z(prof["avg_data_gb_90d"])
prof["z_voice"] = z(prof["avg_voice_minutes_90d"])
prof["z_mm"] = z(prof["avg_mm_txns_90d"]) + z(prof["avg_mm_value_tzs_90d"])
prof["z_value"] = z(prof["avg_spend_tzs_90d"]) + 0.5 * z(prof["avg_tenure_days"])
prof["z_active"] = z(prof["avg_active_days_90d"]) + z(prof["avg_spend_tzs_90d"])

# A name is only given when a cluster clearly stands out on that trait (z-score threshold), and the
# business-critical "low activity" cluster is named first. Anything left is a neutral "Everyday" segment.
names_by_cluster = {}


def pick(seg_name, score, threshold=0.5):
    pool = [(score(r), int(r.cluster_id)) for r in prof.itertuples() if int(r.cluster_id) not in names_by_cluster]
    if pool:
        best_score, best_cluster = max(pool)
        if best_score >= threshold:
            names_by_cluster[best_cluster] = seg_name


pick("Low-Activity / Dormant-Prone", lambda r: -r.z_active)
pick("Non-Mobile-Money Users", lambda r: 1.0 - 5.0 * r.mobile_money_share)     # true only if <= ~10 % registered
pick("Heavy Multi-Service Users", lambda r: min(r.z_data, r.z_voice, r.z_value))
pick("Data Heavy Users", lambda r: r.z_data)
pick("Mobile Money Power Users", lambda r: r.z_mm)
pick("Voice-First Users", lambda r: r.z_voice - 0.5 * r.z_data)
pick("High-Value Loyal", lambda r: r.z_value)
rest = [int(c) for c in prof["cluster_id"] if int(c) not in names_by_cluster]
mm_share = dict(zip(prof["cluster_id"].astype(int), prof["mobile_money_share"]))
for c in rest:
    names_by_cluster[c] = "Everyday Users" + (" (with mobile money)" if mm_share[c] >= 0.5 else " (no mobile money)")
names_by_cluster = dict(names_by_cluster)
_dups = [n for n in set(names_by_cluster.values()) if list(names_by_cluster.values()).count(n) > 1]
for n in _dups:                                  # keep names unique if two clusters ended up with the same label
    idx = 0
    for c, v in list(names_by_cluster.items()):
        if v == n:
            idx += 1
            names_by_cluster[c] = f"{v} {idx}"

name_df = spark.createDataFrame(list(names_by_cluster.items()), "cluster_id int, segment_name string")  # noqa: F821

segments = (assigned.join(name_df, "cluster_id")
            .select("customer_id", "cluster_id", "segment_name", F.lit(OBS_END).alias("as_of_date")))
save_table(segments, "ml", "customer_segments")

seg_profile = (profile.join(name_df, "cluster_id")
               .withColumn("share_of_customers", F.round(F.col("customers") / F.lit(float(n_customers)), 4))
               .select("cluster_id", "segment_name", "customers", "share_of_customers",
                       *[c for c in profile.columns if c not in ("cluster_id", "customers")]))
save_table(seg_profile, "gold", "segment_profile")
show(read_table("gold", "segment_profile").orderBy(F.desc("customers")), 10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save models (scaler + K-Means) so new customers can be assigned later

# COMMAND ----------

version = new_version()
base_path = f"{MODELS_PATH}/segmentation/versions/{version}"
os.makedirs(f"{MODELS_PATH}/segmentation", exist_ok=True)
scaler_model.write().overwrite().save(f"{base_path}/scaler")
kmeans_model.write().overwrite().save(f"{base_path}/kmeans")
sil_best = [r[1] for r in eval_rows if r[0] == best_k][0]
with open(f"{MODELS_PATH}/segmentation/champion.json", "w") as fh:
    json.dump({"task": "segmentation", "version": version, "path": base_path, "k": best_k,
               "silhouette": sil_best, "features": SEGMENT_FEATURES, "preprocessing": "log1p then StandardScaler",
               "segment_names": {str(k): v for k, v in names_by_cluster.items()}, "snapshot_date": str(OBS_END),
               "scale": SCALE}, fh, indent=2)
log_metrics_table("segmentation", "kmeans", version, {"silhouette": sil_best, "k": float(best_k)}, True)
log_mlflow(f"segmentation_kmeans_{version}", {"k": best_k, "features": SEGMENT_FEATURES}, {"silhouette": sil_best},
           tags={"task": "segmentation"})
print(f"Saved segmentation models -> {base_path}")
