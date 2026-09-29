# Databricks notebook source
# MAGIC %md
# MAGIC # 85 · Visual analytics (charts saved as PNG)
# MAGIC
# MAGIC Ten charts that answer the business questions in `docs/business_questions.md`.
# MAGIC Each chart is shown in the notebook **and** saved to the `artifacts/images` volume folder:
# MAGIC
# MAGIC `Catalog → tz_ml → artifacts → images`  (download from the Catalog UI, or copy with the CLI)
# MAGIC
# MAGIC The interactive, shareable version of the same content is the AI/BI dashboard (`docs/dashboard_guide.md`).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if IS_LOCAL:
    matplotlib.use("Agg")
os.makedirs(IMAGES_PATH, exist_ok=True)

PALETTE = {"main": "#1f6f8b", "accent": "#e07a1f", "bad": "#c0392b", "good": "#2e8b57", "grey": "#8a8f98"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})


def finish(name, title, subtitle=None):
    plt.title(title, loc="left")
    if subtitle:
        plt.figtext(0.01, -0.02, subtitle, fontsize=8, color=PALETTE["grey"], ha="left")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_PATH}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved {IMAGES_PATH}/{name}.png")


def pdf(df):
    return df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Revenue and active customers by month
# MAGIC *Q4 · How is ARPU / revenue evolving?*

# COMMAND ----------

m = pdf(read_table("gold", "kpi_region_month").groupBy("month")
        .agg(F.sum("bundle_revenue_tzs").alias("rev"), F.sum("active_customers").alias("active"),
             F.max("days_covered").alias("days")).orderBy("month"))
m["label"] = m["month"].astype(str).str[:7]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(m["label"], m["rev"] / 1e6, color=PALETTE["main"])
ax.set_ylabel("Bundle revenue (million TZS)")
ax2 = ax.twinx()
ax2.plot(m["label"], m["active"], color=PALETTE["accent"], marker="o")
ax2.set_ylabel("Active customers", color=PALETTE["accent"])
ax2.spines["right"].set_visible(True)
finish("01_revenue_active_by_month", "Bundle revenue and active customers by month",
       "Synthetic data. June covers 29 of 30 days (snapshot date 2026-06-29).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Churn rate by region
# MAGIC *Q1 · Which regions lose the most customers?*

# COMMAND ----------

r = pdf(read_table("gold", "region_summary")).sort_values("actual_churn_rate")
overall = float(np.average(r["actual_churn_rate"], weights=r["customers"]))
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(r["region"], r["actual_churn_rate"] * 100, color=PALETTE["main"])
ax.axvline(overall * 100, color=PALETTE["accent"], linestyle="--", label=f"average {overall:.1%}")
ax.set_xlabel("Customers with no activity in the next 30 days (%)")
ax.legend(frameon=False)
finish("02_churn_by_region", "30-day churn rate by region")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Does a poor home cell push customers away?
# MAGIC *Q7 / Q8 · Network quality, power failures and churn*

# COMMAND ----------

from pyspark.sql import Window

lab = read_table("ml", "labels").select("customer_id", "churn_30d")
q = (read_table("gold", "customer_360").join(lab, "customer_id")
     .withColumn("quintile", F.ntile(5).over(Window.orderBy("cell_outage_minutes_90d")))
     .groupBy("quintile").agg(F.round(F.avg("churn_30d") * 100, 2).alias("churn"),
                              F.round(F.avg("cell_outage_minutes_90d"), 0).alias("outage_min"),
                              F.count("*").alias("n")).orderBy("quintile"))
q = pdf(q)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([f"Q{i}\n({int(o):,} min)" for i, o in zip(q["quintile"], q["outage_min"])], q["churn"],
              color=[PALETTE["good"], "#7fb77e", PALETTE["grey"], "#e39a5b", PALETTE["bad"]])
ax.bar_label(bars, fmt="%.1f%%", padding=2)
ax.set_ylabel("30-day churn rate (%)")
ax.set_xlabel("Home-cell outage minutes in the last 90 days, quintiles (Q1 = best cells)")
finish("03_churn_vs_cell_outage", "Customers on the worst cells churn more",
       "Association, not proof of causation: in this synthetic data cell quality is a built-in churn driver.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Customer segments
# MAGIC *Q10 · What kinds of customers do we have?*

# COMMAND ----------

s = pdf(read_table("gold", "segment_profile")).sort_values("customers", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(s))
ax.bar(x, s["avg_arpu_monthly_tzs"] / 1000, color=PALETTE["main"])
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n{c:,} cust." for n, c in zip(s["segment_name"], s["customers"])], fontsize=8)
ax.set_ylabel("Average monthly spend (thousand TZS)")
ax2 = ax.twinx()
ax2.plot(x, s["actual_churn_rate"] * 100, color=PALETTE["bad"], marker="o")
ax2.set_ylabel("30-day churn (%)", color=PALETTE["bad"])
ax2.spines["right"].set_visible(True)
finish("04_segments_arpu_churn", "Segments: value versus churn")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · How good is the churn model? (hold-out gains curve)
# MAGIC *Q3 · If we contact the top 10 %, how many churners do we reach?*

# COMMAND ----------

g = pdf(read_table("gold", "churn_gains")).sort_values("decile")
x = np.r_[0, g["cumulative_share_of_customers"].to_numpy() * 100]
y = np.r_[0, g["cumulative_recall"].to_numpy() * 100]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, y, marker="o", color=PALETTE["main"], label="churn model")
ax.plot([0, 100], [0, 100], linestyle="--", color=PALETTE["grey"], label="random targeting")
top10 = float(g.loc[g["decile"] == 1, "cumulative_recall"].iloc[0]) * 100
ax.annotate(f"top 10% of customers\nreaches {top10:.0f}% of churners", xy=(10, top10), xytext=(25, top10 - 25),
            arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]), color=PALETTE["accent"])
ax.set_xlabel("Customers contacted, highest risk first (%)")
ax.set_ylabel("Churners reached (%)")
ax.legend(frameon=False, loc="lower right")
finish("05_churn_gains_curve", "Cumulative gains on hold-out customers",
       "Hold-out customers only (never used for training).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · What drives churn? (global feature importance)
# MAGIC *Q2 · Which factors matter most?*

# COMMAND ----------

fi = read_table("ml", "feature_importance").filter(~F.col("model_name").startswith("upsell:")).orderBy("rank").limit(12)
fi = pdf(fi).sort_values("importance")
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(fi["feature"], fi["importance"] * 100, color=PALETTE["main"])
ax.set_xlabel("Share of model importance (%)")
finish("06_churn_feature_importance", "What the churn model relies on")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Bundles that earn the money
# MAGIC *Q5 · Which bundles sell and which earn?*

# COMMAND ----------

b = pdf(read_table("gold", "bundle_performance").groupBy("bundle_code", "category")
        .agg(F.sum("revenue_tzs").alias("rev"), F.sum("purchases").alias("n")).orderBy(F.desc("rev")).limit(12))
b = b.sort_values("rev")
colors = {"data": PALETTE["main"], "voice": PALETTE["accent"], "combo": PALETTE["good"], "sms": PALETTE["grey"]}
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(b["bundle_code"], b["rev"] / 1e6, color=[colors[c] for c in b["category"]])
ax.set_xlabel("Revenue (million TZS), 90 days")
handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
ax.legend(handles, colors.keys(), frameon=False, loc="lower right", title="category")
finish("07_bundle_revenue", "Top bundles by revenue")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Power failures per cell by region
# MAGIC *Q8 · How much does the power grid hurt the network?*

# COMMAND ----------

n = pdf(read_table("gold", "network_quality_region").groupBy("region")
        .agg(F.sum("power_failures").alias("pf"), F.max("total_cells").alias("cells")))
n["pf_per_cell"] = n["pf"] / n["cells"]
n = n.sort_values("pf_per_cell")
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(n["region"], n["pf_per_cell"], color=PALETTE["accent"])
ax.set_xlabel("Power-failure events per cell (90 days)")
finish("08_power_failures_by_region", "Power failures per cell by region")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Mobile money: where the value moves
# MAGIC *Q6 · Which channels and transaction types matter?*

# COMMAND ----------

mmv = pdf(read_table("gold", "mobile_money_summary").groupBy("txn_type")
          .agg(F.sum("value_tzs").alias("v"), F.sum("txns").alias("n")).orderBy("v"))
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(mmv["txn_type"], mmv["v"] / 1e9, color=PALETTE["good"])
ax.set_xlabel("Transaction value (billion TZS), 90 days")
finish("09_mobile_money_value", "Mobile-money value by transaction type")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Can we trust the risk bands? (predicted vs actual, hold-out)

# COMMAND ----------

held = (read_table("ml", "retention_actions").join(read_table("ml", "churn_test_scored").select("customer_id"), "customer_id")
        .groupBy("risk_band").agg(F.avg("churn_probability").alias("pred"), F.avg("actual_churn_30d").alias("act"),
                                  F.count("*").alias("n")))
h = pdf(held).set_index("risk_band").reindex(["High", "Medium", "Low"]).reset_index()
x = np.arange(len(h))
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x - 0.18, h["pred"] * 100, 0.36, label="predicted", color=PALETTE["grey"])
ax.bar(x + 0.18, h["act"] * 100, 0.36, label="actual", color=PALETTE["main"])
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n(n={int(n_)})" for b, n_ in zip(h["risk_band"], h["n"])])
ax.set_ylabel("30-day churn (%)")
ax.legend(frameon=False)
finish("10_risk_band_calibration", "Predicted versus actual churn by risk band (hold-out)")

print(f"\nAll charts are in: {IMAGES_PATH}")
