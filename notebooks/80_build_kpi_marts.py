# Databricks notebook source
# MAGIC %md
# MAGIC # 80 · Gold: KPI marts for the dashboard
# MAGIC
# MAGIC Small, pre-aggregated tables so that every dashboard chart is one cheap `SELECT`. Business KPIs use only the
# MAGIC **observation window** (up to 2026-06-29): July data exists only to compute labels.
# MAGIC
# MAGIC | Table | Grain | Dashboard page |
# MAGIC |---|---|---|
# MAGIC | `kpi_region_month` | region × month | 1 · Executive overview |
# MAGIC | `kpi_daily` | day | 1 · Executive overview |
# MAGIC | `region_summary` | region | 1 · Executive, 2 · Churn |
# MAGIC | `churn_scorecard` | region × risk band | 2 · Churn & retention |
# MAGIC | `churn_gains` | score decile (hold-out) | 2 · Churn & retention |
# MAGIC | `retention_target_list` (notebook 70) | customer | 2 · Churn & retention |
# MAGIC | `segment_profile` (notebook 55) | segment | 3 · Customers & segments |
# MAGIC | `network_quality_region` | region × month | 4 · Network & care |
# MAGIC | `care_performance` | region × category × month | 4 · Network & care |
# MAGIC | `bundle_performance` | bundle × month | 5 · Bundles & mobile money |
# MAGIC | `mobile_money_summary` | type × channel × month | 5 · Bundles & mobile money |
# MAGIC | `model_performance`, `data_quality_summary` | model / table | 2 & 6 · Trust |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from functools import reduce

from pyspark.sql import Window

cust = read_table("silver", "customers").select("customer_id", "region", "district", "area_type")
cells = read_table("silver", "regions_cells")
catalog = read_table("silver", "bundle_catalog")


def observed(name):
    return read_table("silver", name).filter(F.col("event_date") <= F.lit(OBS_END))


voice, sessions, purchases, mm = (observed("voice_usage"), observed("data_sessions"),
                                  observed("bundle_purchases"), observed("mobile_money_txns"))
tickets, net = observed("care_tickets"), observed("network_events")

month_ = F.date_trunc("month", F.col("event_date")).cast("date")
days_covered = (F.datediff(F.least(F.last_day("month"), F.lit(OBS_END)), F.greatest(F.col("month"), F.lit(SPAN_START))) + 1)

activity = (voice.filter(F.col("direction") == "outgoing").select("customer_id", "event_date")
            .unionByName(sessions.select("customer_id", "event_date"))
            .unionByName(purchases.select("customer_id", "event_date"))
            .unionByName(mm.select("customer_id", "event_date")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## kpi_region_month and kpi_daily

# COMMAND ----------

def by_region_month(df, aggs):
    return (df.join(cust, "customer_id").withColumn("month", month_).groupBy("region", "month").agg(*aggs))


frames = [
    by_region_month(voice, [F.round(F.sum("duration_sec") / 60.0, 0).alias("voice_minutes"), F.count("*").alias("calls")]),
    by_region_month(sessions, [F.round(F.sum("data_mb") / 1024.0, 1).alias("data_gb"), F.count("*").alias("data_sessions")]),
    by_region_month(purchases, [F.sum("amount_tzs").alias("bundle_revenue_tzs"), F.count("*").alias("bundle_purchases")]),
    by_region_month(mm, [F.count("*").alias("mm_txns"), F.sum("amount_tzs").alias("mm_value_tzs")]),
    by_region_month(tickets, [F.count("*").alias("care_tickets")]),
    by_region_month(activity, [F.count_distinct("customer_id").alias("active_customers")]),
]
kpi = reduce(lambda a, b: a.join(b, ["region", "month"], "full"), frames).na.fill(0)
kpi = (kpi.withColumn("arpu_tzs", F.round(F.try_divide(F.col("bundle_revenue_tzs"), F.col("active_customers")), 0))
       .withColumn("days_covered", days_covered))
save_table(kpi, "gold", "kpi_region_month")

daily_frames = [
    voice.groupBy("event_date").agg(F.round(F.sum("duration_sec") / 60.0, 0).alias("voice_minutes")),
    sessions.groupBy("event_date").agg(F.round(F.sum("data_mb") / 1024.0, 1).alias("data_gb")),
    purchases.groupBy("event_date").agg(F.sum("amount_tzs").alias("bundle_revenue_tzs")),
    mm.groupBy("event_date").agg(F.sum("amount_tzs").alias("mm_value_tzs")),
    activity.groupBy("event_date").agg(F.count_distinct("customer_id").alias("active_customers")),
]
daily = reduce(lambda a, b: a.join(b, "event_date", "full"), daily_frames).na.fill(0).withColumnRenamed("event_date", "date")
save_table(daily, "gold", "kpi_daily")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Network quality and care performance

# COMMAND ----------

cells_per_region = cells.groupBy("region").agg(F.count("*").alias("total_cells"))
customers_per_region = cust.groupBy("region").agg(F.count("*").alias("customers"))

network = (net.join(F.broadcast(cells.select("cell_id", "region")), "cell_id")
           .withColumn("month", month_).groupBy("region", "month")
           .agg(F.count("*").alias("network_events"),
                flag_sum(F.col("event_type") == "power_failure").alias("power_failures"),
                flag_sum(F.col("event_type") == "cell_down").alias("cell_down_events"),
                flag_sum(F.col("event_type") == "call_drop_spike").alias("call_drop_spikes"),
                F.sum(F.when(F.col("event_type").isin("cell_down", "power_failure"), F.col("duration_minutes"))
                      .otherwise(0)).alias("outage_minutes"),
                F.count_distinct("cell_id").alias("affected_cells"))
           .join(cells_per_region, "region").join(customers_per_region, "region")
           .withColumn("events_per_cell", F.round(F.col("network_events") / F.col("total_cells"), 2))
           .withColumn("power_failures_per_cell", F.round(F.col("power_failures") / F.col("total_cells"), 2)))
save_table(network, "gold", "network_quality_region")

obs_end_ts = F.lit(f"{OBS_END + timedelta(days=1)} 00:00:00").cast("timestamp")
resolved_asof = F.col("resolved_ts").isNotNull() & (F.col("resolved_ts") < obs_end_ts)
care = (tickets.join(cust, "customer_id").withColumn("month", month_)
        .groupBy("region", "category", "month")
        .agg(F.count("*").alias("tickets"),
             flag_sum(resolved_asof).alias("resolved"),
             flag_sum(~resolved_asof).alias("open_at_snapshot"),
             F.round(F.avg(F.when(resolved_asof, F.col("resolution_hours"))), 1).alias("avg_resolution_hours"),
             F.round(F.try_divide(flag_sum(resolved_asof & (F.col("resolution_hours") <= 72)), F.count("*")), 4)
             .alias("share_resolved_within_72h")))
save_table(care, "gold", "care_performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bundles and mobile money

# COMMAND ----------

bundles = (purchases.join(F.broadcast(catalog.select("bundle_id", "bundle_code", "category", "price_tzs", "data_mb")),
                          "bundle_id")
           .withColumn("month", month_).groupBy("bundle_code", "category", "price_tzs", "data_mb", "month")
           .agg(F.count("*").alias("purchases"), F.sum("amount_tzs").alias("revenue_tzs"),
                F.count_distinct("customer_id").alias("unique_buyers"),
                flag_sum(F.col("channel") == "mobile_money").alias("via_mobile_money"),
                flag_sum(F.col("channel") == "agent").alias("via_agent"),
                flag_sum(F.col("channel") == "ussd").alias("via_ussd"),
                flag_sum(F.col("channel") == "app").alias("via_app"))
           .withColumn("is_large_data_bundle", F.col("data_mb") >= LARGE_DATA_MB))
save_table(bundles, "gold", "bundle_performance")

mm_sum = (mm.withColumn("month", month_).groupBy("txn_type", "channel", "month")
          .agg(F.count("*").alias("txns"), F.sum("amount_tzs").alias("value_tzs"),
               F.round(F.avg("amount_tzs"), 0).alias("avg_txn_tzs"), F.count_distinct("customer_id").alias("unique_customers")))
save_table(mm_sum, "gold", "mobile_money_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Region summary, churn scorecard and gains table (need notebooks 40, 50 and 70)

# COMMAND ----------

c360 = read_table("gold", "customer_360")
actions = read_table("ml", "retention_actions")

net_total = (read_table("gold", "network_quality_region").groupBy("region")
             .agg(F.sum("network_events").alias("network_events"), F.sum("power_failures").alias("power_failures"),
                  F.max("total_cells").alias("total_cells")))
tk_total = read_table("gold", "care_performance").groupBy("region").agg(F.sum("tickets").alias("tickets"))

region_summary = (
    c360.groupBy("region").agg(
        F.count("*").alias("customers"),
        F.round(F.avg("arpu_monthly_tzs"), 0).alias("avg_arpu_monthly_tzs"),
        F.round(F.avg("mobile_money_registered"), 3).alias("mobile_money_penetration"),
        F.round(F.avg("device_smartphone"), 3).alias("smartphone_share"),
        F.round(F.avg(1 - F.col("area_rural")), 3).alias("urban_share"))
    .join(actions.groupBy("region").agg(
        F.round(F.avg("actual_churn_30d"), 4).alias("actual_churn_rate"),
        flag_sum(F.col("risk_band") == "High").alias("high_risk_customers"),
        F.round(F.sum(F.when(F.col("risk_band") == "High", F.col("revenue_at_risk_tzs")).otherwise(0.0)), 0)
        .alias("high_risk_revenue_at_risk_tzs")), "region")
    .join(net_total, "region").join(tk_total, "region")
    .withColumn("network_events_per_cell", F.round(F.col("network_events") / F.col("total_cells"), 1))
    .withColumn("tickets_per_100_customers", F.round(F.col("tickets") * 100.0 / F.col("customers"), 1))
    .drop("network_events", "power_failures", "tickets"))
save_table(region_summary, "gold", "region_summary")

scorecard = (actions.groupBy("region", "risk_band").agg(
    F.count("*").alias("customers"),
    F.round(F.avg("churn_probability"), 4).alias("avg_churn_probability"),
    F.round(F.avg("actual_churn_30d"), 4).alias("actual_churn_rate"),
    F.round(F.sum("revenue_at_risk_tzs"), 0).alias("revenue_at_risk_tzs"),
    F.round(F.avg("arpu_monthly_tzs"), 0).alias("avg_arpu_monthly_tzs")))
save_table(scorecard, "gold", "churn_scorecard")

# Gains table on HOLD-OUT customers only: if we call the top X % of scores, how many real churners do we reach?
held = read_table("ml", "churn_test_scored")
dec = (held.withColumn("decile", F.ntile(10).over(Window.orderBy(F.desc("churn_probability"))))
       .groupBy("decile").agg(F.count("*").alias("customers"), F.sum("churn_30d").alias("churners"),
                              F.round(F.avg("churn_probability"), 4).alias("avg_predicted_churn")))
tot = dec.agg(F.sum("customers").alias("n"), F.sum("churners").alias("c")).first()
w_cum = Window.orderBy("decile").rowsBetween(Window.unboundedPreceding, Window.currentRow)
gains = (dec.withColumn("actual_churn_rate", F.round(F.col("churners") / F.col("customers"), 4))
         .withColumn("cumulative_recall", F.round(F.sum("churners").over(w_cum) / F.lit(float(tot["c"])), 4))
         .withColumn("cumulative_share_of_customers", F.round(F.sum("customers").over(w_cum) / F.lit(float(tot["n"])), 4))
         .withColumn("lift", F.round(F.col("actual_churn_rate") / F.lit(float(tot["c"]) / float(tot["n"])), 2))
         .orderBy("decile"))
save_table(gains, "gold", "churn_gains")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trust tables: latest model metrics and data quality

# COMMAND ----------

mets = read_table("ml", "model_metrics")
latest = mets.groupBy("task").agg(F.max("version").alias("version"))
model_perf = (mets.join(latest, ["task", "version"]).select("task", "model_name", "version", "metric", "value",
                                                            "is_champion", "run_ts"))
save_table(model_perf, "gold", "model_performance")
save_table(read_table("silver", "dq_summary"), "gold", "data_quality_summary")

banner("Gold marts built")
for t in ["kpi_region_month", "kpi_daily", "region_summary", "churn_scorecard", "churn_gains", "network_quality_region",
          "care_performance", "bundle_performance", "mobile_money_summary", "segment_profile", "retention_target_list",
          "model_performance", "data_quality_summary", "customer_360"]:
    print(f"tz_gold.{t:<26}{read_table('gold', t).count():>10,} rows")

show(read_table("gold", "churn_gains"), 10)
show(read_table("gold", "region_summary").orderBy(F.desc("actual_churn_rate")), 15)
