"""End-to-end tests for the local rehearsal of the pipeline.

Run everything (about 5 minutes at tiny scale):
    pytest -q tests/test_pipeline_outputs.py

Re-use an existing rehearsal folder instead of re-running the pipeline:
    python tests/run_local_pipeline.py --scale tiny --root /tmp/tzdev
    TZ_REUSE_ROOT=/tmp/tzdev pytest -q tests/test_pipeline_outputs.py
"""
import json
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import run_local_pipeline as runner  # noqa: E402

OBS_END = date(2026, 6, 29)
LABEL_START, LABEL_END = OBS_END + timedelta(days=1), OBS_END + timedelta(days=30)


def T(layer, name):
    return f"spark_catalog.tz_{layer}.{name}"


@pytest.fixture(scope="session")
def env():
    os.environ["TZ_LOCAL_TEST"] = "1"
    reuse = os.environ.get("TZ_REUSE_ROOT")
    if reuse:
        os.environ["TZ_LOCAL_ROOT"] = f"{reuse}/files"
        spark = runner.make_spark(reuse)
        spark.sparkContext.setLogLevel("ERROR")
        return spark, Path(reuse) / "files"
    import tempfile
    tmp = tempfile.mkdtemp(prefix="tz_test_")
    spark, _, tmp = runner.run_pipeline("tiny", root=tmp)
    return spark, Path(tmp) / "files"


def q(env_, sql):
    return env_[0].sql(sql)


def scalar(env_, sql):
    return env_[0].sql(sql).first()[0]


SILVER_KEYS = {"customers": "customer_id", "voice_usage": "call_id", "data_sessions": "session_id",
               "bundle_purchases": "purchase_id", "mobile_money_txns": "txn_id", "care_tickets": "ticket_id",
               "network_events": "network_event_id", "regions_cells": "cell_id", "bundle_catalog": "bundle_id"}


# ---------------------------------------------------------------- data quality
def test_bronze_equals_silver_plus_quarantine(env):
    for table in SILVER_KEYS:
        b = scalar(env, f"SELECT COUNT(*) FROM {T('bronze', table)}")
        s = scalar(env, f"SELECT COUNT(*) FROM {T('silver', table)}")
        x = scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'dq_quarantine')} WHERE source_table = '{table}'")
        assert b == s + x, f"{table}: bronze {b} != silver {s} + quarantine {x}"


def test_silver_keys_are_unique_and_not_null(env):
    for table, key in SILVER_KEYS.items():
        nulls = scalar(env, f"SELECT COUNT(*) FROM {T('silver', table)} WHERE {key} IS NULL")
        dups = scalar(env, f"SELECT COUNT(*) - COUNT(DISTINCT {key}) FROM {T('silver', table)}")
        assert nulls == 0 and dups == 0, f"{table}.{key}: nulls={nulls} dups={dups}"


def test_quarantine_explains_every_row(env):
    n = scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'dq_quarantine')}")
    assert n > 0, "dirty data was injected; the quarantine must not be empty"
    blank = scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'dq_quarantine')} "
                        "WHERE failed_rules IS NULL OR failed_rules = '' OR record_json IS NULL")
    assert blank == 0


def test_bad_values_never_reach_silver(env):
    assert scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'voice_usage')} WHERE duration_sec <= 0") == 0
    assert scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'bundle_purchases')} WHERE amount_tzs <= 0") == 0
    assert scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'customers')} WHERE age NOT BETWEEN 15 AND 100") == 0
    orphans = scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'voice_usage')} v "
                          f"LEFT JOIN {T('silver', 'customers')} c ON v.customer_id = c.customer_id "
                          "WHERE c.customer_id IS NULL")
    assert orphans == 0


# ---------------------------------------------------------------- gold and labels
def test_customer_360_one_row_per_customer(env):
    n = scalar(env, f"SELECT COUNT(*) FROM {T('gold', 'customer_360')}")
    d = scalar(env, f"SELECT COUNT(DISTINCT customer_id) FROM {T('gold', 'customer_360')}")
    assert n == d and n == scalar(env, f"SELECT COUNT(*) FROM {T('silver', 'customers')}")


def test_ml_features_have_no_nulls(env):
    cols = [c for c in env[0].table(T("ml", "customer_features")).columns]
    expr = " + ".join(f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in cols)
    assert scalar(env, f"SELECT {expr} FROM {T('ml', 'customer_features')}") == 0


def test_churn_rate_is_plausible(env):
    rate = scalar(env, f"SELECT AVG(churn_30d) FROM {T('ml', 'labels')}")
    assert 0.05 <= rate <= 0.30, f"churn rate {rate:.1%} is outside the plausible 5-30% range"


def test_features_do_not_look_into_the_future(env):
    assert scalar(env, f"SELECT MAX(last_activity_date) FROM {T('gold', 'customer_360')}") <= OBS_END
    assert scalar(env, f"SELECT MIN(days_since_last_activity) FROM {T('gold', 'customer_360')}") >= 0


def test_churn_label_is_derived_from_real_inactivity(env):
    """Churned = zero billable events in the label window; retained = at least one."""
    window = f"event_date BETWEEN DATE'{LABEL_START}' AND DATE'{LABEL_END}'"
    events = f"""
      SELECT customer_id FROM {T('silver', 'voice_usage')} WHERE direction = 'outgoing' AND {window}
      UNION ALL SELECT customer_id FROM {T('silver', 'data_sessions')} WHERE {window}
      UNION ALL SELECT customer_id FROM {T('silver', 'bundle_purchases')} WHERE {window}
      UNION ALL SELECT customer_id FROM {T('silver', 'mobile_money_txns')} WHERE {window}"""
    wrong_churn = scalar(env, f"SELECT COUNT(*) FROM {T('ml', 'labels')} l JOIN ({events}) e USING (customer_id) "
                              "WHERE l.churn_30d = 1")
    assert wrong_churn == 0, "customers labelled churned still have activity in the label window"
    wrong_stay = scalar(env, f"SELECT COUNT(*) FROM {T('ml', 'labels')} l WHERE l.churn_30d = 0 AND "
                             f"l.customer_id NOT IN (SELECT customer_id FROM ({events}))")
    assert wrong_stay == 0, "customers labelled retained have no activity in the label window"


# ---------------------------------------------------------------- models
def _metric(env_, task, model, metric):
    return scalar(env_, f"SELECT value FROM {T('gold', 'model_performance')} "
                        f"WHERE task='{task}' AND model_name='{model}' AND metric='{metric}'")


def test_churn_champion_beats_random_by_a_wide_margin(env):
    champ = scalar(env, f"SELECT model_name FROM {T('gold', 'model_performance')} "
                        "WHERE task='churn' AND is_champion LIMIT 1")
    auc_pr, base = _metric(env, "churn", champ, "auc_pr"), _metric(env, "churn", champ, "base_rate")
    assert auc_pr > 2 * base, f"AUC-PR {auc_pr:.3f} is not clearly above base rate {base:.3f}"
    assert _metric(env, "churn", champ, "auc_roc") > 0.75
    assert _metric(env, "churn", champ, "lift_top10") > 2.0


def test_upsell_champion_beats_random(env):
    champ = scalar(env, f"SELECT model_name FROM {T('gold', 'model_performance')} "
                        "WHERE task='upsell' AND is_champion LIMIT 1")
    assert _metric(env, "upsell", champ, "auc_pr") > 1.3 * _metric(env, "upsell", champ, "base_rate")


def test_model_files_and_metadata_exist(env):
    models = env[1] / "artifacts" / "models"
    for task in ("churn", "upsell", "segmentation"):
        meta = json.loads((models / task / "champion.json").read_text())
        assert Path(meta["path"]).exists(), f"{task}: saved model folder is missing"


def test_segments_cover_everyone_with_unique_readable_names(env):
    total = scalar(env, f"SELECT COUNT(*) FROM {T('gold', 'customer_360')}")
    assert scalar(env, f"SELECT COUNT(*) FROM {T('ml', 'customer_segments')}") == total
    k = scalar(env, f"SELECT COUNT(DISTINCT cluster_id) FROM {T('ml', 'customer_segments')}")
    names = scalar(env, f"SELECT COUNT(DISTINCT segment_name) FROM {T('ml', 'customer_segments')}")
    assert k == names and 4 <= k <= 7
    smallest = scalar(env, f"SELECT MIN(share_of_customers) FROM {T('gold', 'segment_profile')}")
    assert smallest >= 0.03


def test_risk_bands_are_ordered_on_holdout_customers(env):
    rows = q(env, f"""SELECT a.risk_band, AVG(a.actual_churn_30d) r FROM {T('ml', 'retention_actions')} a
                      JOIN {T('ml', 'churn_test_scored')} t USING (customer_id) GROUP BY a.risk_band""").collect()
    r = {x["risk_band"]: x["r"] for x in rows}
    assert r["High"] > r["Medium"] > r["Low"], r


def test_risk_band_sizes_follow_capacity_rules(env):
    total = scalar(env, f"SELECT COUNT(*) FROM {T('ml', 'retention_actions')}")
    high = scalar(env, f"SELECT COUNT(*) FROM {T('ml', 'retention_actions')} WHERE risk_band='High'")
    assert abs(high / total - 0.10) < 0.01


def test_every_high_and_medium_customer_has_an_action(env):
    bad = scalar(env, f"SELECT COUNT(*) FROM {T('ml', 'retention_actions')} "
                      "WHERE risk_band IN ('High','Medium') AND (recommended_action IS NULL "
                      "OR recommended_action = 'No action: monitor')")
    assert bad == 0
    assert scalar(env, f"SELECT COUNT(*) FROM {T('gold', 'retention_target_list')}") == scalar(
        env, f"SELECT COUNT(*) FROM {T('ml', 'retention_actions')} WHERE risk_band IN ('High','Medium')")


# ---------------------------------------------------------------- dashboard + visuals
def _dashboard_queries():
    text = (ROOT / "sql" / "dashboard_queries.sql").read_text()
    parts = re.split(r"^-- @query: ", text, flags=re.M)[1:]
    out = []
    for part in parts:
        header, _, body = part.partition("\n")
        out.append((header.split("|")[0].strip(), body.strip()))
    return out


def test_dashboard_sql_covers_all_pages():
    names = [n for n, _ in _dashboard_queries()]
    assert len(names) >= 20
    for page in ("p1_", "p2_", "p3_", "p4_", "p5_", "p6_"):
        assert any(n.startswith(page) for n in names), f"no query for {page}"


@pytest.mark.parametrize("name,sql", _dashboard_queries())
def test_dashboard_query_runs_and_returns_rows(env, name, sql):
    local_sql = sql.replace("workspace.", "spark_catalog.")
    assert env[0].sql(local_sql).limit(5).count() > 0, f"{name} returned no rows"


def test_charts_were_created(env):
    images = sorted(p.name for p in (env[1] / "artifacts" / "images").glob("*.png"))
    assert len(images) == 10, images
