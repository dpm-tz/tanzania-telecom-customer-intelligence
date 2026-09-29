# Databricks notebook source
# MAGIC %md
# MAGIC # 70 · Score customers and recommend an action
# MAGIC
# MAGIC Turns model outputs into something a retention team can act on this week.
# MAGIC
# MAGIC 1. **Churn score** for every customer (champion churn model, loaded from `champion.json`).
# MAGIC 2. **Risk band by team capacity**: top 10 % of scores = *High*, next 20 % = *Medium*, rest = *Low*.
# MAGIC    (Bands are rank-based on purpose: a campaign team calls a fixed number of customers, not "everyone above 0.5".)
# MAGIC 3. **Upsell score** for customers eligible for a large-bundle offer; top 15 % = *High propensity*.
# MAGIC 4. **Risk drivers** per customer (rule-based, readable by non-analysts) such as
# MAGIC    *"Repeated power failures at home cell"* or *"Open or slow care tickets"*.
# MAGIC 5. **Recommended action + owning team**, and **revenue at risk** in TZS.
# MAGIC
# MAGIC > Drivers are transparent business rules that mirror the strongest model features. They are **not** per-customer
# MAGIC > SHAP explanations. Notebook 50 shows the model's global feature importance for comparison.
# MAGIC
# MAGIC > This notebook scores *all* customers of the snapshot, including customers the model was trained on. Judge model
# MAGIC > quality with the hold-out metrics from notebooks 50 and 60, not with this table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %run ./00_ml_utils

# COMMAND ----------

from pyspark.sql import Window

c360 = read_table("gold", "customer_360")
features = read_table("ml", "customer_features")
labels = read_table("ml", "labels")
segments = read_table("ml", "customer_segments").select("customer_id", "segment_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Churn scores and risk bands

# COMMAND ----------

churn_model, churn_meta = load_champion("churn")
print(f"churn model: {churn_meta['model_name']} {churn_meta['version']} "
      f"(AUC-PR {churn_meta['metrics']['auc_pr']:.3f} on hold-out)")

x = assemble(numeric_frame(features, churn_meta["features"]), churn_meta["features"])
churn = (churn_model.transform(x)
         .select("customer_id", vector_to_array("probability")[1].alias("churn_probability")))

rank_w = Window.orderBy(F.desc("churn_probability"))
churn = (churn.withColumn("_pr", F.percent_rank().over(rank_w))
         .withColumn("risk_band", F.when(F.col("_pr") < RISK_HIGH_SHARE, "High")
                     .when(F.col("_pr") < RISK_HIGH_SHARE + RISK_MEDIUM_SHARE, "Medium").otherwise("Low"))
         .withColumn("risk_rank_pct", F.round(F.col("_pr") * 100, 2))
         .withColumn("model_version", F.lit(churn_meta["version"]))
         .withColumn("as_of_date", F.lit(OBS_END))
         .drop("_pr"))
save_table(churn, "ml", "churn_scores")
churn = read_table("ml", "churn_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Upsell scores (eligible customers only)

# COMMAND ----------

up_model, up_meta = load_champion("upsell")
eligible = features.join(labels.filter(F.col("eligible_for_upsell") == 1).select("customer_id"), "customer_id")
xu = assemble(numeric_frame(eligible, up_meta["features"]), up_meta["features"])
upsell = (up_model.transform(xu)
          .select("customer_id", vector_to_array("probability")[1].alias("upsell_probability")))
upsell = (upsell.withColumn("_pr", F.percent_rank().over(Window.orderBy(F.desc("upsell_probability"))))
          .withColumn("upsell_band", F.when(F.col("_pr") < UPSELL_HIGH_SHARE, "High").otherwise("Standard"))
          .withColumn("model_version", F.lit(up_meta["version"]))
          .withColumn("as_of_date", F.lit(OBS_END))
          .drop("_pr"))
save_table(upsell, "ml", "upsell_scores")
upsell = read_table("ml", "upsell_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Risk drivers (transparent rules, ordered by business priority)

# COMMAND ----------

# Network thresholds are relative to the population (top quartile), so a driver only fires when a customer's home
# cell is genuinely worse than most.
outage_q75 = max(1.0, float(c360.approxQuantile("cell_outage_minutes_90d", [0.75], 0.01)[0]))
power_q75 = max(2.0, float(c360.approxQuantile("cell_power_failures_30d", [0.75], 0.01)[0]))

D = {
    "power": "Frequent power failures at home cell",
    "outage": "Long network outages at home cell",
    "ticket": "Open or slow-resolved care tickets",
    "inactive": "No activity for 5+ days",
    "declining": "Usage declining vs previous 60 days",
    "mm_stopped": "Stopped using mobile money",
    "new": "New customer (under 90 days)",
    "multisim": "Multi-SIM user (may shift usage to another SIM)",
}
driver_rules = [
    (D["power"], F.col("cell_power_failures_30d") >= F.lit(power_q75)),
    (D["outage"], F.col("cell_outage_minutes_90d") >= F.lit(outage_q75)),
    (D["ticket"], (F.col("open_tickets") >= 1) | (F.col("slow_tickets_90d") >= 1)),
    (D["inactive"], F.col("days_since_last_activity") >= 5),
    (D["declining"], F.least(F.col("voice_trend_ratio"), F.col("data_trend_ratio"), F.col("spend_trend_ratio")) < 0.6),
    (D["mm_stopped"], (F.col("mm_txns_90d") > 0) & (F.col("mm_txns_30d") == 0)),
    (D["new"], F.col("tenure_days") < 90),
    (D["multisim"], F.col("is_multi_sim") == 1),
]
drivers = F.filter(F.array(*[F.when(cond, F.lit(label)) for label, cond in driver_rules]), lambda v: v.isNotNull())

base = (c360.select("customer_id", "region", "district", "arpu_monthly_tzs",
                    drivers.alias("_drivers"))
        .withColumn("top_drivers", F.slice("_drivers", 1, 3))
        .withColumn("primary_driver", F.coalesce(F.try_element_at("_drivers", F.lit(1)), F.lit("No dominant driver")))
        .drop("_drivers"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Recommended action, owner and revenue at risk

# COMMAND ----------

pd_ = F.col("primary_driver")
seg = F.col("segment_name")
high = F.col("risk_band") == "High"
medium = F.col("risk_band") == "Medium"

action = (
    F.when(high & pd_.isin(D["power"], D["outage"]),
           "Network care: apology SMS + 1 GB free data; raise the home cell with Network Ops")
    .when(high & (pd_ == D["ticket"]), "Priority care callback within 24 h + goodwill credit")
    .when(high & pd_.isin(D["inactive"], D["declining"]),
          "Win-back offer: 20% off a weekly bundle via SMS / USSD push")
    .when(high & (pd_ == D["mm_stopped"]), "Mobile money cashback on the next 3 transactions")
    .when(high & (pd_ == D["new"]), "Onboarding call + first-bundle bonus")
    .when(high, "Loyalty offer: bonus data on the next bundle purchase")
    .when(medium & (seg == "Data Heavy Users"), "Targeted SMS: monthly 15 GB bundle offer")
    .when(medium & (seg == "Mobile Money Power Users"), "Targeted SMS: mobile money cashback campaign")
    .when(medium & (seg == "Voice-First Users"), "Targeted SMS: monthly voice bundle with bonus minutes")
    .when(medium, "Targeted SMS: bundle recommendation matched to usage")
    .when(F.col("upsell_band") == "High", "Upsell: offer a monthly 15 GB+ data bundle")
    .otherwise("No action: monitor")
)
owner = (
    F.when(high & pd_.isin(D["power"], D["outage"]), "Network Ops + Retention")
    .when(high & (pd_ == D["ticket"]), "Customer Care")
    .when(high & (pd_ == D["mm_stopped"]), "Mobile Money Team")
    .when(high, "Retention Marketing")
    .when(medium, "CRM")
    .when(F.col("upsell_band") == "High", "Sales / CRM")
    .otherwise("None")
)
priority = (F.when(high, 1).when(medium, 2).when(F.col("upsell_band") == "High", 3).otherwise(4))

actions = (base.join(churn.select("customer_id", "churn_probability", "risk_band", "risk_rank_pct"), "customer_id")
           .join(upsell.select("customer_id", "upsell_probability", "upsell_band"), "customer_id", "left")
           .join(segments, "customer_id", "left")
           .join(labels.select("customer_id", F.col("churn_30d").alias("actual_churn_30d")), "customer_id")
           .fillna({"upsell_band": "Not eligible", "segment_name": "Unassigned"})
           .withColumn("revenue_at_risk_tzs",
                       F.round(F.col("arpu_monthly_tzs") * F.col("churn_probability") * REVENUE_HORIZON_MONTHS, 0))
           .withColumn("recommended_action", action)
           .withColumn("owner_team", owner)
           .withColumn("campaign_priority", priority)
           .withColumn("as_of_date", F.lit(OBS_END))
           .select("customer_id", "region", "district", "segment_name", "churn_probability", "risk_band",
                   "risk_rank_pct", "top_drivers", "primary_driver", "upsell_probability", "upsell_band",
                   "arpu_monthly_tzs", "revenue_at_risk_tzs", "recommended_action", "owner_team",
                   "campaign_priority", "actual_churn_30d", "as_of_date"))
save_table(actions, "ml", "retention_actions")
actions = read_table("ml", "retention_actions")

target_list = (actions.filter(F.col("risk_band").isin("High", "Medium"))
               .orderBy(F.desc("revenue_at_risk_tzs")))
save_table(target_list, "gold", "retention_target_list")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · What the business gets

# COMMAND ----------

test_ids = read_table("ml", "churn_test_scored").select("customer_id")
holdout = actions.join(test_ids, "customer_id")
banner("Risk bands vs what actually happened: HOLD-OUT customers only (not used for training)")
show(holdout.groupBy("risk_band")
     .agg(F.count("*").alias("customers"), F.round(F.avg("churn_probability"), 3).alias("avg_predicted_churn"),
          F.round(F.avg("actual_churn_30d"), 3).alias("actual_churn_rate"))
     .orderBy(F.expr("CASE risk_band WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END")))

banner("Risk bands, all customers (includes training customers: optimistic, for business sizing only)")
band_stats = (actions.groupBy("risk_band")
              .agg(F.count("*").alias("customers"),
                   F.round(F.avg("churn_probability"), 3).alias("avg_predicted_churn"),
                   F.round(F.avg("actual_churn_30d"), 3).alias("actual_churn_rate"),
                   F.round(F.sum("revenue_at_risk_tzs") / 1e6, 2).alias("revenue_at_risk_million_tzs")))
show(band_stats.orderBy(F.expr("CASE risk_band WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END")))

banner("Recommended actions")
show(actions.groupBy("campaign_priority", "recommended_action", "owner_team").count().orderBy("campaign_priority", F.desc("count")), 30)

high_rate = holdout.filter(high).agg(F.avg("actual_churn_30d")).first()[0]
low_rate = holdout.filter(F.col("risk_band") == "Low").agg(F.avg("actual_churn_30d")).first()[0]
assert high_rate > low_rate, "High-risk customers must churn more than Low-risk customers"
assert actions.filter(high & (F.col("recommended_action") == "No action: monitor")).count() == 0
print(f"\nOK (hold-out): High-risk churn {high_rate:.1%} vs Low-risk churn {low_rate:.1%}.")
