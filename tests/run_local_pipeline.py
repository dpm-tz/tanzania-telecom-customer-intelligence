"""Rehearse the whole Databricks pipeline on a laptop with local Spark.

The notebooks are plain Python files in Databricks "source" format. This runner executes them cell by cell in
one shared namespace, replacing `%run ./x` with the contents of that notebook and dropping `# MAGIC` lines.

Differences from Databricks (all switched by TZ_LOCAL_TEST=1 inside 00_config.py):
  * tables are Parquet (Databricks: Delta) in the catalog `spark_catalog`
  * "volumes" are plain folders under TZ_LOCAL_ROOT
  * MLflow logging is skipped

Usage:
    pip install -r requirements-local.txt
    python tests/run_local_pipeline.py --scale tiny
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
PIPELINE = [
    "01_setup_environment", "10_generate_source_data", "20_ingest_bronze", "30_build_silver_with_dq",
    "40_build_customer_360", "50_train_churn_model", "55_segment_customers", "60_train_upsell_model",
    "70_score_and_recommend", "80_build_kpi_marts", "85_visual_analytics",
]


def load_cells(name):
    text = (NOTEBOOKS / f"{name}.py").read_text()
    return [c for c in re.split(r"\n# COMMAND ----------\n", text)]


def run_cell(cell, ns):
    lines = cell.splitlines()
    code_lines, run_target = [], None
    for line in lines:
        m = re.match(r"# MAGIC %run\s+\./(\S+)", line)
        if m:
            run_target = m.group(1)
        elif line.startswith("# MAGIC"):
            continue
        else:
            code_lines.append(line)
    if run_target:
        for sub in load_cells(run_target):
            run_cell(sub, ns)
        return
    code = "\n".join(code_lines).strip()
    if code:
        exec(compile(code, "<notebook cell>", "exec"), ns)


def make_spark(tmp):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[1]").appName("tz-local")
            .config("spark.sql.warehouse.dir", f"{tmp}/warehouse")
            .config("spark.hadoop.javax.jdo.option.ConnectionURL", f"jdbc:derby:;databaseName={tmp}/metastore_db;create=true")
            .enableHiveSupport()
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.ansi.enabled", "true")          # Databricks serverless runs with ANSI on
            .config("spark.driver.memory", "2g")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "Africa/Dar_es_Salaam")
            .getOrCreate())


def run_pipeline(scale="tiny", steps=None, keep=False, ns=None, spark=None, root=None):
    os.environ["TZ_LOCAL_TEST"] = "1"
    os.environ["TZ_SCALE"] = scale
    os.environ.setdefault("MPLBACKEND", "Agg")
    tmp = root or tempfile.mkdtemp(prefix="tz_local_")
    os.environ["TZ_LOCAL_ROOT"] = f"{tmp}/files"
    spark = spark or make_spark(tmp)
    spark.sparkContext.setLogLevel("ERROR")
    ns = ns if ns is not None else {"spark": spark}
    for step in steps or PIPELINE:
        t0 = time.time()
        print(f"\n######## {step} ########", flush=True)
        for cell in load_cells(step):
            run_cell(cell, ns)
        print(f"-- {step} finished in {time.time() - t0:.0f}s", flush=True)
    if not keep and root is None:
        pass
    return spark, ns, tmp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="tiny", choices=["tiny", "small", "standard"])
    ap.add_argument("--steps", nargs="*", help="subset of notebooks, default = all")
    ap.add_argument("--root", help="reuse a folder (keeps files between runs)")
    a = ap.parse_args()
    run_pipeline(a.scale, a.steps, root=a.root)
