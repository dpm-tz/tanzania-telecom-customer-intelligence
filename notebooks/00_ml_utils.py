# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · ML utilities
# MAGIC Shared helpers for the three models: feature assembly, ranking metrics that matter for churn
# MAGIC (AUC-PR, recall and lift in the top 10 %), versioned model saving and MLflow logging.
# MAGIC Run automatically through `%run ./00_ml_utils`.

# COMMAND ----------

import json
import os
from datetime import datetime

from pyspark.ml.classification import (
    GBTClassificationModel,
    LogisticRegressionModel,
    RandomForestClassificationModel,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

MODEL_CLASSES = {
    "LogisticRegressionModel": LogisticRegressionModel,
    "GBTClassificationModel": GBTClassificationModel,
    "RandomForestClassificationModel": RandomForestClassificationModel,
}

# COMMAND ----------

def numeric_frame(df, features, keep=("customer_id",), label=None):
    """Select id + features (cast to double, nulls -> 0) + optional label."""
    cols = [F.col(k) for k in keep]
    cols += [F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c) for c in features]
    if label:
        cols.append(F.col(label).cast("double").alias("label"))
    return df.select(*cols)


def assemble(df, features):
    return VectorAssembler(inputCols=features, outputCol="features", handleInvalid="error").transform(df)


def binary_metrics(predictions, top_share=0.10):
    """Metrics for an imbalanced binary problem.

    * auc_roc / auc_pr      : ranking quality (AUC-PR is the honest one when positives are rare)
    * recall_top / lift_top : if the business can only act on the top `top_share` of scores,
                              what share of true positives do we catch and how much better than random?
    """
    roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(predictions)
    pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                       metricName="areaUnderPR").evaluate(predictions)
    scored = predictions.select(F.col("label").alias("y"), vector_to_array("probability")[1].alias("p"))
    thr = scored.approxQuantile("p", [1.0 - top_share], 0.001)[0]
    agg = scored.agg(
        F.count("*").alias("n"),
        F.sum("y").alias("positives"),
        F.sum(F.when(F.col("p") >= thr, 1).otherwise(0)).alias("flagged"),
        F.sum(F.when(F.col("p") >= thr, F.col("y")).otherwise(0.0)).alias("tp"),
    ).first()
    n, pos, flagged, tp = float(agg["n"]), float(agg["positives"]), float(agg["flagged"]), float(agg["tp"])
    base_rate = pos / n if n else 0.0
    precision = tp / flagged if flagged else 0.0
    return {
        "auc_roc": float(roc),
        "auc_pr": float(pr),
        "base_rate": base_rate,
        f"recall_top{int(top_share * 100)}": tp / pos if pos else 0.0,
        f"precision_top{int(top_share * 100)}": precision,
        f"lift_top{int(top_share * 100)}": precision / base_rate if base_rate else 0.0,
        "test_rows": n,
    }


def print_metrics(name, m):
    print(f"{name:<28}" + " | ".join(f"{k}={v:,.4f}" if abs(v) < 1000 else f"{k}={v:,.0f}" for k, v in m.items()))

# COMMAND ----------

def new_version():
    return datetime.now().strftime("v%Y%m%d_%H%M%S")


def save_model(model, task, model_name, version, metrics, features, extra=None):
    """Save a Spark ML model under models/<task>/versions/<version> and point champion.json at it."""
    path = f"{MODELS_PATH}/{task}/versions/{version}"
    os.makedirs(f"{MODELS_PATH}/{task}", exist_ok=True)
    model.write().overwrite().save(path)
    meta = {
        "task": task,
        "model_name": model_name,
        "model_class": type(model).__name__,
        "version": version,
        "path": path,
        "features": features,
        "metrics": metrics,
        "snapshot_date": str(OBS_END),
        "scale": SCALE,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta.update(extra or {})
    with open(f"{MODELS_PATH}/{task}/champion.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved {model_name} -> {path}")
    return meta


def load_champion(task):
    """Load the current champion model of `task` together with its metadata."""
    with open(f"{MODELS_PATH}/{task}/champion.json") as fh:
        meta = json.load(fh)
    return MODEL_CLASSES[meta["model_class"]].load(meta["path"]), meta

# COMMAND ----------

def log_metrics_table(task, model_name, version, metrics, is_champion):
    """Append one row per metric to tz_ml.model_metrics (feeds the dashboard's 'model performance' page)."""
    rows = [(task, model_name, version, k, float(v), bool(is_champion)) for k, v in metrics.items()]
    df = (spark.createDataFrame(rows, "task string, model_name string, version string, metric string, "  # noqa: F821
                                      "value double, is_champion boolean")
          .withColumn("run_ts", F.current_timestamp()))
    save_table(df, "ml", "model_metrics", mode="append")


def log_mlflow(run_name, params, metrics, tags=None):
    """Best-effort MLflow tracking. Never breaks the pipeline if MLflow is unavailable."""
    if IS_LOCAL:
        return
    try:
        import mlflow
        user = spark.sql("SELECT current_user()").first()[0]  # noqa: F821
        mlflow.set_experiment(f"/Users/{user}/tz_telecom_customer_intelligence")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.set_tags({k: str(v) for k, v in (tags or {}).items()})
        print(f"MLflow run logged: {run_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"MLflow logging skipped ({type(exc).__name__}: {str(exc)[:120]})")

print("ML utilities loaded.")
