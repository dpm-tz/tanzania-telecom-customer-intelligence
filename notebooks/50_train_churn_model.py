# Databricks notebook source
# MAGIC %md
# MAGIC # 50 · Train the churn model (champion / challenger)
# MAGIC
# MAGIC **Question:** which customers will show *no activity for the next 30 days*?
# MAGIC
# MAGIC | Model | Role |
# MAGIC |---|---|
# MAGIC | Logistic Regression | Baseline: simple, explainable, hard to beat by accident |
# MAGIC | Gradient-Boosted Trees | Challenger (falls back to Random Forest if GBT is not available on your compute) |
# MAGIC
# MAGIC **How we judge them.** Churn is the minority class, so accuracy is misleading. We report:
# MAGIC * **AUC-PR**: quality of the ranking for a rare class (compare it to the base rate!)
# MAGIC * **Recall / precision / lift in the top 10 %**: the retention team can only call 10 % of customers, so
# MAGIC   how many real churners do they reach, and how many times better than random is that?
# MAGIC
# MAGIC The champion is the model with the best AUC-PR on the held-out 20 %. Champion is saved, versioned, to
# MAGIC `artifacts/models/churn/versions/<version>` and `champion.json` points to it.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %run ./00_ml_utils

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier

features_df = read_table("ml", "customer_features")
labels_df = read_table("ml", "labels").select("customer_id", F.col("churn_30d").alias("label"))

data = numeric_frame(features_df.join(labels_df, "customer_id"), ML_FEATURES, label="label")
data = assemble(data, ML_FEATURES)
train, test = data.randomSplit([0.8, 0.2], seed=42)

n_train, n_test = train.count(), test.count()
base_rate = data.agg(F.avg("label")).first()[0]
print(f"train rows: {n_train:,} | test rows: {n_test:,} | churn base rate: {base_rate:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit the candidates

# COMMAND ----------

VERSION = new_version()
candidates = {}
params = {}

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01,
                        elasticNetParam=0.0, standardization=True)
candidates["logistic_regression"] = lr.fit(train)
params["logistic_regression"] = {"maxIter": 100, "regParam": 0.01, "standardization": True}

try:
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=80, maxDepth=4, stepSize=0.1, seed=42)
    candidates["gradient_boosted_trees"] = gbt.fit(train)
    params["gradient_boosted_trees"] = {"maxIter": 80, "maxDepth": 4, "stepSize": 0.1}
except Exception as exc:  # noqa: BLE001
    print(f"GBT unavailable here ({type(exc).__name__}); using Random Forest as challenger.")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=150, maxDepth=8, seed=42)
    candidates["random_forest"] = rf.fit(train)
    params["random_forest"] = {"numTrees": 150, "maxDepth": 8}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate on the held-out 20 %

# COMMAND ----------

results = {}
for name, model in candidates.items():
    results[name] = binary_metrics(model.transform(test), top_share=0.10)
    print_metrics(name, results[name])

champion_name = max(results, key=lambda k: results[k]["auc_pr"])
champion = candidates[champion_name]
m = results[champion_name]
banner(f"Champion: {champion_name}")
print(f"AUC-PR {m['auc_pr']:.3f} vs random baseline {m['base_rate']:.3f}  "
      f"| top-10% recall {m['recall_top10']:.1%} | lift {m['lift_top10']:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global feature importance (what the model relies on)

# COMMAND ----------

importance = None
try:
    importance = champion.featureImportances.toArray()
except Exception:  # noqa: BLE001
    try:
        importance = [abs(float(c)) for c in champion.coefficients.toArray()]
    except Exception:  # noqa: BLE001
        print("Feature importance is not available on this compute; skipping.")

if importance is not None:
    total = float(sum(importance)) or 1.0
    ranked = sorted(zip(ML_FEATURES, importance), key=lambda x: -x[1])
    imp_df = spark.createDataFrame(  # noqa: F821
        [(champion_name, f, float(v) / total, i + 1) for i, (f, v) in enumerate(ranked)],
        "model_name string, feature string, importance double, rank int")
    save_table(imp_df.withColumn("version", F.lit(VERSION)), "ml", "feature_importance")
    show(imp_df.limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the champion, log metrics, keep test predictions for the gains chart

# COMMAND ----------

meta = save_model(champion, "churn", champion_name, VERSION, results[champion_name], ML_FEATURES,
                  extra={"train_rows": n_train, "test_rows": n_test, "label": "no billable activity in 30 days",
                         "split": "random 80/20 by customer (see model card for limitations)"})

for name, res in results.items():
    log_metrics_table("churn", name, VERSION, res, is_champion=(name == champion_name))
    log_mlflow(f"churn_{name}_{VERSION}", params[name], res, tags={"task": "churn", "champion": name == champion_name})

test_scored = (champion.transform(test)
               .select("customer_id", F.col("label").cast("int").alias("churn_30d"),
                       vector_to_array("probability")[1].alias("churn_probability")))
save_table(test_scored, "ml", "churn_test_scored")
print(f"\nchurn model version {VERSION} is ready for scoring (notebook 70).")
