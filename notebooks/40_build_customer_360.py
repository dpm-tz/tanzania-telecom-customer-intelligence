# Databricks notebook source
# MAGIC %md
# MAGIC # 40 · Gold: Customer 360, ML features and labels
# MAGIC
# MAGIC One row per customer, as of the snapshot date `OBS_END` (2026-06-29).
# MAGIC
# MAGIC | Output table | Content |
# MAGIC |---|---|
# MAGIC | `tz_gold.customer_360` | Profile + usage + money + network + care features (human readable, dashboard-ready) |
# MAGIC | `tz_ml.customer_features` | The same customers as numeric model inputs (`ML_FEATURES` in `00_config`) |
# MAGIC | `tz_ml.labels` | What happened in the **next 30 days**: `churn_30d`, `bought_large_bundle_30d` |
# MAGIC
# MAGIC **Churn definition (prepaid reality):** there is no contract to cancel. A customer has churned when they show
# MAGIC **no billable activity** (outgoing call, data session, bundle purchase or mobile-money transaction) during the
# MAGIC 30 days after the snapshot. Features use only events **on or before** the snapshot, so labels never leak.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

W = dict(
    d90=(W90_START, OBS_END), d30=(W30_START, OBS_END), prior60=(W90_START, PRIOR60_END),
    label=(LABEL_START, LABEL_END),
)

customers = read_table("silver", "customers")
voice = read_table("silver", "voice_usage")
sessions = read_table("silver", "data_sessions")
purchases = read_table("silver", "bundle_purchases")
mm = read_table("silver", "mobile_money_txns")
net = read_table("silver", "network_events")
tickets = read_table("silver", "care_tickets")
catalog = read_table("silver", "bundle_catalog")
cells = read_table("silver", "regions_cells")

purchases = (purchases.join(F.broadcast(catalog.select("bundle_id", "category",
                                                       (F.col("data_mb") >= LARGE_DATA_MB).alias("is_large"))),
                            "bundle_id"))


def d90(df):
    return df.filter(in_window("event_date", *W["d90"]))


def ratio(cur, prior, months_prior=2.0):
    """Trend: last 30 days vs the average 30 days of the 60 days before. 1.0 = stable, <1 = declining."""
    r = F.try_divide(F.col(cur), F.col(prior) / F.lit(months_prior))
    return F.least(F.lit(3.0), F.coalesce(r, F.lit(1.0)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Behavioural aggregates (one small table per source)

# COMMAND ----------

f_voice = (d90(voice).groupBy("customer_id").agg(
    flag_sum(F.col("direction") == "outgoing").alias("calls_out_90d"),
    flag_sum(F.col("direction") == "incoming").alias("calls_in_90d"),
    F.round(F.sum("duration_sec") / 60.0, 1).alias("voice_minutes_90d"),
    F.round(F.avg("duration_sec"), 1).alias("avg_call_sec_90d"),
    F.round(F.try_divide(flag_sum(F.col("call_type") == "off_net"), F.count("*")), 4).alias("offnet_share_90d"),
    flag_sum(F.col("call_type") == "international").alias("intl_calls_90d"),
    flag_sum((F.col("direction") == "outgoing") & in_window("event_date", *W["d30"])).alias("calls_out_30d"),
    flag_sum((F.col("direction") == "outgoing") & in_window("event_date", *W["prior60"])).alias("calls_out_prior60"),
))

f_data = (d90(sessions).groupBy("customer_id").agg(
    F.count("*").alias("sessions_90d"),
    F.round(F.sum("data_mb") / 1024.0, 3).alias("data_gb_90d"),
    F.round(F.sum(F.when(in_window("event_date", *W["d30"]), F.col("data_mb")).otherwise(0.0)) / 1024.0, 3)
    .alias("data_gb_30d"),
    F.round(F.sum(F.when(in_window("event_date", *W["prior60"]), F.col("data_mb")).otherwise(0.0)) / 1024.0, 3)
    .alias("data_gb_prior60"),
    F.round(F.try_divide(F.sum(F.when(F.col("app_category").isin("video", "social_media"), F.col("data_mb"))
                                .otherwise(0.0)), F.sum("data_mb")), 4).alias("video_social_share_90d"),
    F.count_distinct("event_date").alias("active_data_days_90d"),
))

f_purch = (d90(purchases).groupBy("customer_id").agg(
    F.count("*").alias("purchases_90d"),
    F.sum("amount_tzs").alias("spend_tzs_90d"),
    F.sum(F.when(in_window("event_date", *W["d30"]), F.col("amount_tzs")).otherwise(0)).alias("spend_tzs_30d"),
    F.sum(F.when(in_window("event_date", *W["prior60"]), F.col("amount_tzs")).otherwise(0)).alias("spend_prior60"),
    F.round(F.avg("amount_tzs"), 1).alias("avg_purchase_tzs_90d"),
    F.count_distinct("category").alias("distinct_bundle_types_90d"),
    flag_sum(F.col("is_large")).alias("large_bundle_purchases_90d"),
    flag_sum(F.col("is_large") & in_window("event_date", *W["d30"])).alias("large_bundles_30d"),
    F.round(F.try_divide(flag_sum(F.col("channel") == "mobile_money"), F.count("*")), 4).alias("share_purchases_via_mm"),
    F.max("event_date").alias("last_purchase_date"),
))

f_mm = (d90(mm).groupBy("customer_id").agg(
    F.count("*").alias("mm_txns_90d"),
    F.sum("amount_tzs").alias("mm_value_tzs_90d"),
    flag_sum(in_window("event_date", *W["d30"])).alias("mm_txns_30d"),
    flag_sum(in_window("event_date", *W["prior60"])).alias("mm_txns_prior60"),
    flag_sum(F.col("txn_type") == "bill_payment").alias("bill_payments_90d"),
    F.round(F.try_divide(flag_sum(F.col("txn_type") == "cash_out"), F.count("*")), 4).alias("cash_out_share_90d"),
))

# Any billable activity, by day: used for recency, active days and the churn label.
def activity(start, end):
    parts = [
        voice.filter(F.col("direction") == "outgoing").select("customer_id", "event_date"),
        sessions.select("customer_id", "event_date"),
        purchases.select("customer_id", "event_date"),
        mm.select("customer_id", "event_date"),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out.filter(in_window("event_date", start, end))


act90 = activity(*W["d90"]).distinct()
f_act = act90.groupBy("customer_id").agg(
    F.max("event_date").alias("last_activity_date"),
    F.count_distinct("event_date").alias("active_days_90d"),
    flag_sum(in_window("event_date", *W["d30"])).alias("active_days_30d"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Network experience of the customer's home cell, and care history

# COMMAND ----------

net_obs = net.filter(F.col("event_date") <= F.lit(OBS_END))
f_net = (net_obs.groupBy(F.col("cell_id").alias("home_cell_id")).agg(
    flag_sum(in_window("event_date", *W["d30"])).alias("cell_events_30d"),
    flag_sum((F.col("event_type") == "power_failure") & in_window("event_date", *W["d30"]))
    .alias("cell_power_failures_30d"),
    F.sum(F.when(in_window("event_date", *W["d90"]) & F.col("event_type").isin("cell_down", "power_failure"),
                 F.col("duration_minutes")).otherwise(0)).alias("cell_outage_minutes_90d"),
))

# A ticket resolved after the snapshot was still OPEN at the snapshot (avoids leakage).
obs_end_ts = F.lit(f"{OBS_END + timedelta(days=1)} 00:00:00").cast("timestamp")
tk_obs = tickets.filter(F.to_date("created_ts") <= F.lit(OBS_END))
open_asof = F.col("resolved_ts").isNull() | (F.col("resolved_ts") >= obs_end_ts)
f_care = (tk_obs.filter(F.to_date("created_ts") >= F.lit(W90_START)).groupBy("customer_id").agg(
    F.count("*").alias("tickets_90d"),
    flag_sum(F.col("category").isin(*NETWORK_TICKET_CATEGORIES)).alias("network_tickets_90d"),
    flag_sum(open_asof).alias("open_tickets"),
    F.round(F.avg(F.when(~open_asof, F.col("resolution_hours"))), 1).alias("avg_resolution_hours"),
    flag_sum(open_asof | (F.col("resolution_hours") > 72)).alias("slow_tickets_90d"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble Customer 360

# COMMAND ----------

profile = (customers.join(F.broadcast(cells.select(F.col("cell_id").alias("home_cell_id"),
                                                   F.col("technology").alias("cell_technology"),
                                                   F.col("power_backup").alias("cell_has_power_backup"))),
                          "home_cell_id", "left")
           .select("customer_id", "customer_code", "region", "district", "area_type", "gender", "age",
                   "device_type", "acquisition_channel", "registration_status", "is_multi_sim",
                   "mobile_money_registered", "activation_date", "home_cell_id", "cell_technology",
                   "cell_has_power_backup")
           .withColumn("tenure_days", F.datediff(F.lit(OBS_END), F.col("activation_date"))))

c360 = profile
for part, key in [(f_voice, "customer_id"), (f_data, "customer_id"), (f_purch, "customer_id"),
                  (f_mm, "customer_id"), (f_act, "customer_id"), (f_care, "customer_id"),
                  (f_net, "home_cell_id")]:
    c360 = c360.join(part, key, "left")

zero_cols = [c for c in ML_FEATURES if c in c360.columns and c not in
             ("days_since_last_activity", "voice_trend_ratio", "data_trend_ratio", "spend_trend_ratio",
              "mm_trend_ratio", "age", "tenure_days")]
zero_cols += ["calls_out_prior60", "data_gb_prior60", "spend_prior60", "mm_txns_prior60", "large_bundles_30d",
              "share_purchases_via_mm", "cash_out_share_90d"]
c360 = c360.fillna(0, subset=[c for c in dict.fromkeys(zero_cols) if c in c360.columns])

c360 = (c360
        .withColumn("days_since_last_activity",
                    F.coalesce(F.datediff(F.lit(OBS_END), F.col("last_activity_date")), F.lit(FEATURE_DAYS)))
        .withColumn("voice_trend_ratio", ratio("calls_out_30d", "calls_out_prior60"))
        .withColumn("data_trend_ratio", ratio("data_gb_30d", "data_gb_prior60"))
        .withColumn("spend_trend_ratio", ratio("spend_tzs_30d", "spend_prior60"))
        .withColumn("mm_trend_ratio", ratio("mm_txns_30d", "mm_txns_prior60"))
        .withColumn("arpu_monthly_tzs", F.round(F.col("spend_tzs_90d") / 3.0, 0))
        .withColumn("bought_large_bundle_last30d", (F.col("large_bundles_30d") > 0).cast("int"))
        .withColumn("area_rural", (F.col("area_type") == "rural").cast("int"))
        .withColumn("device_smartphone", (F.col("device_type") == "smartphone").cast("int"))
        .withColumn("is_multi_sim", F.col("is_multi_sim").cast("int"))
        .withColumn("mobile_money_registered", F.col("mobile_money_registered").cast("int"))
        .withColumn("cell_has_power_backup", F.col("cell_has_power_backup").cast("int"))
        .withColumn("as_of_date", F.lit(OBS_END))
        .drop("calls_out_prior60", "data_gb_prior60", "spend_prior60", "mm_txns_prior60"))

save_table(c360, "gold", "customer_360")
c360 = read_table("gold", "customer_360")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML feature table

# COMMAND ----------

feature_table = c360.select("customer_id", "as_of_date", *ML_FEATURES,
                            "arpu_monthly_tzs", "bought_large_bundle_last30d")
save_table(feature_table, "ml", "customer_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labels from the next 30 days

# COMMAND ----------

active_next30 = activity(*W["label"]).select("customer_id").distinct().withColumn("_active", F.lit(1))
large_next30 = (purchases.filter(in_window("event_date", *W["label"]) & F.col("is_large"))
                .select("customer_id").distinct().withColumn("_large", F.lit(1)))

labels = (c360.select("customer_id", "bought_large_bundle_last30d")
          .join(active_next30, "customer_id", "left")
          .join(large_next30, "customer_id", "left")
          .select("customer_id",
                  F.when(F.col("_active").isNull(), 1).otherwise(0).alias("churn_30d"),
                  F.coalesce(F.col("_large"), F.lit(0)).alias("bought_large_bundle_30d"),
                  (1 - F.col("bought_large_bundle_last30d")).alias("eligible_for_upsell"),
                  F.lit(OBS_END).alias("as_of_date"),
                  F.lit(LABEL_START).alias("label_window_start"),
                  F.lit(LABEL_END).alias("label_window_end")))
save_table(labels, "ml", "labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity checks

# COMMAND ----------

c360 = read_table("gold", "customer_360")
labels = read_table("ml", "labels")
n = c360.count()
assert n == c360.select("customer_id").distinct().count(), "customer_360 must have one row per customer"

stats = labels.agg(F.avg("churn_30d").alias("churn_rate"),
                   F.avg("bought_large_bundle_30d").alias("large_bundle_rate"),
                   F.avg("eligible_for_upsell").alias("eligible_share")).first()
banner("Snapshot summary")
print(f"customers in customer_360        : {n:,}")
print(f"observed churn rate (next 30 d)  : {stats['churn_rate']:.1%}")
print(f"bought a >=10GB bundle (next 30d): {stats['large_bundle_rate']:.1%}")
print(f"eligible for upsell              : {stats['eligible_share']:.1%}")

null_counts = c360.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in ML_FEATURES]).first().asDict()
nulls = {k: v for k, v in null_counts.items() if v}
assert not nulls, f"ML features contain nulls: {nulls}"
print("OK: no nulls in ML features.")

show(c360.groupBy("region").agg(F.count("*").alias("customers"), F.round(F.avg("arpu_monthly_tzs"), 0).alias("avg_arpu_tzs"),
                               F.round(F.avg("days_since_last_activity"), 1).alias("avg_days_inactive"))
     .orderBy(F.desc("customers")), 20)
