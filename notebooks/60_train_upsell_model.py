# Databricks notebook source
# MAGIC %md
# MAGIC # 60 · Train the upsell propensity model (large data bundles)
# MAGIC
# MAGIC **Question:** which customers are ready to buy a *large* data bundle (>= 10 GB) in the next 30 days?
# MAGIC
# MAGIC * **Population:** customers who did **not** already buy a large bundle in the last 30 days of the observation
# MAGIC   window (`eligible_for_upsell = 1`). Selling to someone who just bought one is not an upsell.
# MAGIC * **Label:** bought a large bundle during the next 30 days.
# MAGIC * **Models:** Logistic Regression baseline vs Random Forest. Champion = best AUC-PR on a 20 % hold-out.
# MAGIC * **Use:** the top 15 % scores become the "High propensity" list handed to the sales/CRM team (notebook 70).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %run ./00_ml_utils

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

features_df = read_table("ml", "customer_features")
labels_df = (read_table("ml", "labels").filter(F.col("eligible_for_upsell") == 1)
             .select("customer_id", F.col("bought_large_bundle_30d").alias("label")))

data = numeric_frame(features_df.join(labels_df, "customer_id"), ML_FEATURES, label="label")
data = assemble(data, ML_FEATURES)
train, test = data.randomSplit([0.8, 0.2], seed=42)
n_train, n_test = train.count(), test.count()
base_rate = data.agg(F.avg("label")).first()[0]
print(f"eligible customers: {n_train + n_test:,} | train {n_train:,} | test {n_test:,} | positive rate {base_rate:.1%}")

# COMMAND ----------

VERSION = new_version()
candidates = {
    "logistic_regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=100,
                                              regParam=0.01, standardization=True).fit(train),
    "random_forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=150,
                                            maxDepth=8, seed=42).fit(train),
}
params = {"logistic_regression": {"maxIter": 100, "regParam": 0.01},
          "random_forest": {"numTrees": 150, "maxDepth": 8}}

results = {}
for name, model in candidates.items():
    results[name] = binary_metrics(model.transform(test), top_share=0.15)
    print_metrics(name, results[name])

champion_name = max(results, key=lambda k: results[k]["auc_pr"])
champion = candidates[champion_name]
m = results[champion_name]
banner(f"Champion: {champion_name}")
print(f"AUC-PR {m['auc_pr']:.3f} vs base rate {m['base_rate']:.3f} | top-15% lift {m['lift_top15']:.1f}x")

# COMMAND ----------

try:
    importance = champion.featureImportances.toArray()
except Exception:  # noqa: BLE001
    importance = [abs(float(c)) for c in champion.coefficients.toArray()]
total = float(sum(importance)) or 1.0
ranked = sorted(zip(ML_FEATURES, importance), key=lambda x: -x[1])
imp_df = spark.createDataFrame(  # noqa: F821
    [("upsell:" + champion_name, f, float(v) / total, i + 1) for i, (f, v) in enumerate(ranked)],
    "model_name string, feature string, importance double, rank int")
save_table(imp_df.withColumn("version", F.lit(VERSION)), "ml", "feature_importance", mode="append")
show(imp_df.limit(12))

# COMMAND ----------

save_model(champion, "upsell", champion_name, VERSION, results[champion_name], ML_FEATURES,
           extra={"train_rows": n_train, "test_rows": n_test,
                  "label": "bought a >=10GB bundle in the next 30 days (eligible customers only)"})
for name, res in results.items():
    log_metrics_table("upsell", name, VERSION, res, is_champion=(name == champion_name))
    log_mlflow(f"upsell_{name}_{VERSION}", params[name], res, tags={"task": "upsell", "champion": name == champion_name})
print(f"\nupsell model version {VERSION} is ready for scoring (notebook 70).")
