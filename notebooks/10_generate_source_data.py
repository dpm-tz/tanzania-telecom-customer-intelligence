# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Generate the source-system data (simulation)
# MAGIC
# MAGIC Simulates nine source systems of the fictional operator **Simba Telecom** and lands them as Parquet
# MAGIC in the `landing` volume. In a real company this notebook would not exist: the files would be delivered
# MAGIC by billing, network, mobile-money and care systems.
# MAGIC
# MAGIC **How churn is simulated (important):** churn is *not* a random 0/1 column.
# MAGIC 1. Every customer gets a hidden *churn propensity* driven by realistic causes: poor home-cell quality,
# MAGIC    unresolved/slow care tickets, new SIM, multi-SIM behaviour, no mobile money, rural coverage, ...
# MAGIC 2. Customers who churn simply **stop generating events** (abruptly, or after a period of fading usage).
# MAGIC    18 % of the customers who stay go almost silent for two weeks and then return: realistic false alarms.
# MAGIC 3. The churn label is later **derived from the data** (no activity in the 30 days after the snapshot).
# MAGIC
# MAGIC The hidden truth is stored in `landing/_simulation/` for validation only, and is never read by the pipeline.
# MAGIC Small amounts of dirty data (duplicates, nulls, orphans, impossible values) are injected on purpose so
# MAGIC that the Silver data-quality layer has real work to do.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

N = VOLUMES["customers"]
SIM = f"{LANDING_PATH}/_simulation"
os.makedirs(LANDING_PATH, exist_ok=True)


def write_landing(df, name):
    df.write.mode("overwrite").parquet(f"{LANDING_PATH}/{name}")


def write_sim(df, name):
    """Materialise a simulation step and read it back so every later step sees identical values."""
    df.write.mode("overwrite").parquet(f"{SIM}/{name}")
    return spark.read.parquet(f"{SIM}/{name}")  # noqa: F821


def corrupt(df, col, prob, bad_value, seed):
    return df.withColumn(col, F.when(rnd(seed) < prob, bad_value).otherwise(F.col(col)))


def add_duplicates(df, fraction, seed):
    return df.unionByName(df.sample(False, fraction, GLOBAL_SEED + seed))


NULL_ID = F.lit(None).cast("long")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Reference data: bundle catalogue and network cells (towers)

# COMMAND ----------

bundle_catalog = spark.createDataFrame(  # noqa: F821
    BUNDLES, "bundle_id int, bundle_code string, category string, price_tzs int, validity_days int, "
             "data_mb int, voice_minutes int, sms_count int")
write_landing(bundle_catalog.coalesce(1), "bundle_catalog")

total_cells = max(60, N // 70)
alloc, base = [], 0
for d in DISTRICTS:
    k = max(2, round(d["weight"] * total_cells))
    alloc.append((d["region"], d["district"], d["area_type"], base, k))
    base += k
N_CELLS = base

dist_df = spark.createDataFrame(  # noqa: F821
    alloc, "region string, district string, area_type string, cell_base int, cell_count int")

_urban = F.col("area_type") == "urban"
cells_all = (
    spark.range(1, N_CELLS + 1).withColumnRenamed("id", "cell_id")  # noqa: F821
    .join(F.broadcast(dist_df),
          (F.col("cell_id") > F.col("cell_base")) & (F.col("cell_id") <= F.col("cell_base") + F.col("cell_count")))
    .withColumn("_u_tech", rnd(11))
    .withColumn("technology", F.when(_urban, weighted_choice(F.col("_u_tech"), [("4G", .70), ("3G", .25), ("2G", .05)]))
                .otherwise(weighted_choice(F.col("_u_tech"), [("4G", .28), ("3G", .45), ("2G", .27)])))
    .drop("_u_tech")
    .withColumn("power_backup", rnd(12) < F.when(_urban, F.lit(0.85)).otherwise(F.lit(0.35)))
    .withColumn("cell_quality", F.round(clip(F.when(_urban, F.lit(0.78)).otherwise(F.lit(0.55))
                                             + (rnd(13) - 0.5) * 0.45, 0.05, 0.99), 3))
)
cells_sim = write_sim(cells_all.select("cell_id", "region", "district", "area_type", "technology",
                                       "power_backup", "cell_quality"), "cells")
write_landing(cells_sim.drop("cell_quality").coalesce(1), "regions_cells")
print(f"cells: {N_CELLS:,} | bundles: {len(BUNDLES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Customers and their hidden behavioural traits
# MAGIC Five behavioural archetypes (data-heavy, voice-first, mobile-money power users, balanced, low-activity)
# MAGIC drive how much each customer uses each service. The archetype itself is hidden: the K-Means model must
# MAGIC *rediscover* the structure from behaviour.

# COMMAND ----------

ARCHETYPES = [("data_heavy", .18), ("voice_first", .22), ("mm_power", .18), ("balanced", .27), ("low_activity", .15)]
# multipliers: voice, data, purchases, mobile money; large_pref = propensity to pick a >=10GB bundle
_MULT = {
    "data_heavy":   (0.8, 2.6, 1.6, 0.9, 0.50),
    "voice_first":  (2.4, 0.7, 1.0, 0.8, 0.05),
    "mm_power":     (1.0, 1.0, 1.0, 2.8, 0.12),
    "balanced":     (1.2, 1.2, 1.2, 1.2, 0.14),
    "low_activity": (0.35, 0.30, 0.45, 0.30, 0.03),
}
m_voice = map_from_dict({k: v[0] for k, v in _MULT.items()})
m_data = map_from_dict({k: v[1] for k, v in _MULT.items()})
m_purch = map_from_dict({k: v[2] for k, v in _MULT.items()})
m_mm = map_from_dict({k: v[3] for k, v in _MULT.items()})
m_large = map_from_dict({k: v[4] for k, v in _MULT.items()})

customers0 = add_choice(
    spark.range(1, N + 1).withColumnRenamed("id", "customer_id"),  # noqa: F821
    "district", 21, [(d["district"], d["weight"]) for d in DISTRICTS])
customers0 = (
    customers0
    .join(F.broadcast(dist_df), "district")
    .withColumn("home_cell_id", (F.col("cell_base") + 1 + F.floor(rnd(22) * F.col("cell_count"))).cast("long"))
    .join(F.broadcast(cells_sim.select(F.col("cell_id").alias("home_cell_id"), "cell_quality")), "home_cell_id")
    .drop("cell_base", "cell_count")
)
urban = F.col("area_type") == "urban"
customers0 = add_choice(customers0, "acquisition_channel", 30, ACQUISITION_CHANNELS)
customers0 = add_choice(customers0, "archetype", 31, ARCHETYPES)
customers0 = (
    customers0
    .withColumn("age", (18 + F.floor(F.pow(rnd(23), 1.4) * 52)).cast("int"))
    .withColumn("gender", F.when(rnd(24) < 0.5, "F").otherwise("M"))
    .withColumn("_p_smart", clip(F.when(urban, F.lit(0.78)).otherwise(F.lit(0.50)) - (F.col("age") - 35) * 0.004,
                                 0.15, 0.95))
    .withColumn("device_type", F.when(rnd(25) < F.col("_p_smart"), "smartphone").otherwise("feature_phone"))
    .withColumn("tenure_days", (10 + F.floor(F.pow(rnd(26), 1.6) * 2500)).cast("int"))
    .withColumn("activation_date", F.expr(f"date_sub(DATE'{OBS_END}', tenure_days)"))
    .withColumn("mobile_money_registered", rnd(27) < F.when(urban, F.lit(0.85)).otherwise(F.lit(0.55)))
    .withColumn("is_multi_sim", rnd(28) < 0.35)
    .withColumn("registration_status", F.when(rnd(29) < 0.97, "fully_registered").otherwise("pending_verification"))
    .withColumn("_eng", F.exp(rndn(32) * 0.35))
    .withColumn("_smart", (F.col("device_type") == "smartphone"))
    .withColumn("w_voice", m_voice[F.col("archetype")] * F.col("_eng"))
    .withColumn("w_data", m_data[F.col("archetype")] * F.col("_eng")
                * F.when(F.col("_smart"), F.lit(1.0)).otherwise(F.lit(0.35))
                * F.when(urban, F.lit(1.15)).otherwise(F.lit(1.0)))
    .withColumn("w_purchase", m_purch[F.col("archetype")] * F.col("_eng"))
    .withColumn("w_mm", F.when(F.col("mobile_money_registered"), m_mm[F.col("archetype")] * F.col("_eng"))
                .otherwise(F.lit(0.0)))
    .withColumn("large_pref", m_large[F.col("archetype")] * F.when(F.col("_smart"), F.lit(1.0)).otherwise(F.lit(0.2)))
    .withColumn("w_ticket", 0.35 + 2.2 * (1.0 - F.col("cell_quality")))
    .drop("_p_smart", "_eng", "_smart")
)
base_traits = write_sim(customers0, "base_traits")
print("customers:", f"{base_traits.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Event generator helper
# MAGIC Draw more candidate events than needed, then keep each one with probability proportional to the
# MAGIC customer's activity weight. Customers only produce events while they are active (after activation, and
# MAGIC before their hidden `last_active_date` if they churn).

# COMMAND ----------

def propose(n_target, traits, wcol, seed, cols, behavioural=True, retention=0.90):
    st = traits.agg(F.avg(wcol).alias("m"), F.max(wcol).alias("x")).first()
    wmax = float(st["x"])
    n_prop = int(n_target * wmax / float(st["m"]) / (retention if behavioural else 1.0) * 1.02)
    behaviour_cols = ["activation_date", "last_active_date", "fade", "lull", "lull_start", "lull_end"] if behavioural else []
    ev = (spark.range(n_prop).withColumnRenamed("id", "event_id")  # noqa: F821
          .withColumn("customer_id", (F.floor(rnd(seed) * N) + 1).cast("long"))
          .withColumn("_day", F.floor(rnd(seed + 1) * SPAN_DAYS).cast("int"))
          .join(F.broadcast(traits.select("customer_id", wcol, *behaviour_cols, *cols)), "customer_id")
          .withColumn("event_date", F.expr(f"date_add(DATE'{SPAN_START}', _day)"))
          .withColumn("_keep", rnd(seed + 2))
          .filter(F.col("_keep") < F.col(wcol) / F.lit(wmax)))
    if behavioural:
        ev = (ev.filter(F.col("event_date") >= F.col("activation_date"))
              .filter(F.col("event_date") <= F.col("last_active_date"))
              .withColumn("_ufade", rnd(seed + 4))
              .filter(~F.col("fade") | (F.col("event_date") < F.expr("date_sub(last_active_date, 21)"))
                      | (F.col("_ufade") < 0.45))
              .withColumn("_ulull", rnd(seed + 5))
              .filter(~F.col("lull") | (F.col("event_date") < F.col("lull_start"))
                      | (F.col("event_date") > F.col("lull_end")) | (F.col("_ulull") < 0.15)))
    return ev.withColumn(
        "event_ts",
        F.timestamp_seconds(F.unix_timestamp(F.col("event_date").cast("timestamp")) + F.floor(rnd(seed + 3) * 86400)))


def make_id(prefix, width=9):
    return F.concat(F.lit(prefix), F.lpad(F.col("event_id").cast("string"), width, "0"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Care tickets (clean version first: they influence churn propensity)

# COMMAND ----------

tk = propose(VOLUMES["tickets"], base_traits, "w_ticket", 500, ["area_type"], behavioural=False)
tk = (tk
      .transform(add_choice, "category", 510, TICKET_CATEGORIES)
      .transform(add_choice, "channel", 511, TICKET_CHANNELS)
      .transform(add_choice, "priority", 512, [("low", .50), ("medium", .35), ("high", .15)])
      .withColumn("_hours", F.least(F.lit(720), F.ceil(-F.log(1 - rnd(513)) * 30
                                                          * F.when(F.col("area_type") == "rural", F.lit(1.5))
                                                          .otherwise(F.lit(1.0)))).cast("int"))
      .withColumn("_resolved_ts", F.timestamp_seconds(F.unix_timestamp("event_ts") + F.col("_hours") * 3600))
      .withColumn("_is_resolved", (rnd(514) < 0.88)
                  & (F.col("_resolved_ts") <= F.lit(f"{SPAN_END + timedelta(days=1)} 00:00:00").cast("timestamp")))
      .withColumn("status", F.when(F.col("_is_resolved"), "resolved").otherwise("open"))
      .withColumn("resolved_ts", F.when(F.col("_is_resolved"), F.col("_resolved_ts")))
      .withColumn("resolution_hours", F.when(F.col("_is_resolved"), F.col("_hours")).cast("int"))
      .withColumn("ticket_id", make_id("TK"))
      .withColumnRenamed("event_ts", "created_ts")
      .select("ticket_id", "customer_id", "created_ts", "category", "channel", "priority", "status",
              "resolved_ts", "resolution_hours"))
tickets_clean = write_sim(tk, "tickets_clean")
print("tickets (clean):", f"{tickets_clean.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Hidden churn propensity and each customer's last active day
# MAGIC The coefficients below are the *simulation's assumptions* about what drives prepaid churn in Tanzania.
# MAGIC They are documented in `docs/data_assumptions.md`; the ML model has to discover them from behaviour.

# COMMAND ----------

tk_agg = (tickets_clean.filter(F.to_date("created_ts") <= F.lit(OBS_END))
          .groupBy("customer_id")
          .agg(F.count("*").alias("tk_cnt"),
               F.max(F.when(F.col("status") != "resolved", 1).when(F.col("resolution_hours") > 72, 1)
                     .otherwise(0)).alias("tk_bad")))

INTERCEPT = -3.55
traits = (
    base_traits.join(tk_agg, "customer_id", "left")
    .fillna({"tk_cnt": 0, "tk_bad": 0})
    .withColumn("_logit",
                F.lit(INTERCEPT)
                + 0.9 * (F.col("tenure_days") < 90).cast("double")
                + 0.35 * F.col("is_multi_sim").cast("double")
                + 0.30 * (~F.col("mobile_money_registered")).cast("double")
                + 1.4 * (1.0 - F.col("cell_quality"))
                + 0.45 * F.least(F.col("tk_cnt"), F.lit(4)).cast("double")
                + 0.80 * F.col("tk_bad").cast("double")
                + 0.30 * (F.col("area_type") == "rural").cast("double")
                - 0.60 * (F.col("archetype") == "mm_power").cast("double")
                + 0.40 * (F.col("archetype") == "low_activity").cast("double")
                + rndn(600) * 0.5)
    .withColumn("churn_propensity", F.round(1.0 / (1.0 + F.exp(-F.col("_logit"))), 4))
    .withColumn("churn_latent", (rnd(601) < F.col("churn_propensity")).cast("int"))
    .withColumn("_abrupt", rnd(602) < 0.6)
    .withColumn("_back", F.when(F.col("_abrupt"), F.floor(rnd(603) * 5)).otherwise(3 + F.floor(rnd(603) * 12)).cast("int"))
    .withColumn("last_active_date",
                F.when(F.col("churn_latent") == 1, F.expr(f"date_sub(DATE'{OBS_END}', _back)"))
                .otherwise(F.lit(SPAN_END)))
    .withColumn("fade", (F.col("churn_latent") == 1) & (~F.col("_abrupt")))
    # 18 % of NON-churners go quiet for two weeks (travel, illness, exams, holiday) and then come back:
    # these are the false alarms every real churn model has to live with.
    .withColumn("lull", (F.col("churn_latent") == 0) & (rnd(604) < 0.18))
    .withColumn("_lull_back", F.floor(rnd(605) * 9).cast("int"))
    .withColumn("lull_end", F.expr(f"date_sub(DATE'{OBS_END}', _lull_back)"))
    .withColumn("lull_start", F.expr("date_sub(lull_end, 13)"))
    .drop("_logit", "_abrupt", "_back", "_lull_back")
)
traits = write_sim(traits, "traits")
latent_rate = traits.agg(F.avg("churn_latent")).first()[0]
print(f"hidden churn rate used by the simulation: {latent_rate:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Customers (published)

# COMMAND ----------

customers = (traits
             .withColumn("customer_code", F.concat(F.lit("SIM-"), F.lpad(F.col("customer_id").cast("string"), 7, "0")))
             .select("customer_id", "customer_code", "region", "district", "area_type", "home_cell_id", "age",
                     "gender", "device_type", "is_multi_sim", "mobile_money_registered", "registration_status",
                     "acquisition_channel", "activation_date"))
customers = corrupt(customers, "age", 0.003, F.lit(0) + (rnd(701) * 0 + 150).cast("int"), 701)   # impossible age
customers = corrupt(customers, "region", 0.0015, F.lit(None).cast("string"), 702)                 # missing region
customers = add_duplicates(customers, 0.002, 703)                                                 # duplicate rows
write_landing(customers, "customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Voice usage

# COMMAND ----------

ev = propose(VOLUMES["voice"], traits, "w_voice", 100,
             ["home_cell_id"])
voice = (ev
         .transform(add_choice, "direction", 110, DIRECTIONS)
         .transform(add_choice, "call_type", 111, CALL_TYPES)
         .withColumn("duration_sec", F.least(F.lit(3600), F.ceil(-F.log(1 - rnd(112)) * 95)).cast("int"))
         .withColumn("cell_id", F.when(rnd(113) < 0.85, F.col("home_cell_id"))
                     .otherwise((F.floor(rnd(114) * N_CELLS) + 1).cast("long")))
         .withColumn("call_id", make_id("CL"))
         .select("call_id", "customer_id", "event_ts", "direction", "call_type", "duration_sec", "cell_id"))
voice = corrupt(voice, "duration_sec", 0.003, F.lit(0), 121)                                       # zero-length call
voice = corrupt(voice, "customer_id", 0.002, NULL_ID, 122)                                         # missing customer
voice = corrupt(voice, "customer_id", 0.0015, (F.lit(N + 1000) + F.floor(rnd(123) * 500)).cast("long"), 124)  # orphan
voice = corrupt(voice, "event_ts", 0.001, F.lit(f"{SPAN_END + timedelta(days=45)} 10:00:00").cast("timestamp"), 125)
voice = add_duplicates(voice, 0.004, 126)
write_landing(voice, "voice_usage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Data sessions

# COMMAND ----------

ev = propose(VOLUMES["data_sessions"], traits, "w_data", 200,
             ["home_cell_id"])
sessions = (ev
            .transform(add_choice, "app_category", 210, APP_CATEGORIES)
            .withColumn("data_mb", F.round(clip(F.exp(rndn(211) * 0.9 + 3.1), 0.5, 20000.0), 1))
            .withColumn("session_minutes", F.least(F.lit(240), F.ceil(-F.log(1 - rnd(212)) * 8)).cast("int"))
            .withColumn("cell_id", F.when(rnd(213) < 0.85, F.col("home_cell_id"))
                        .otherwise((F.floor(rnd(214) * N_CELLS) + 1).cast("long")))
            .withColumn("session_id", make_id("DS", 10))
            .select("session_id", "customer_id", "event_ts", "app_category", "data_mb", "session_minutes", "cell_id"))
sessions = corrupt(sessions, "data_mb", 0.002, F.lit(-1.0), 221)
sessions = corrupt(sessions, "customer_id", 0.002, NULL_ID, 222)
sessions = corrupt(sessions, "customer_id", 0.0015, (F.lit(N + 1000) + F.floor(rnd(223) * 500)).cast("long"), 224)
sessions = add_duplicates(sessions, 0.004, 226)
write_landing(sessions, "data_sessions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Bundle purchases
# MAGIC Data-heavy smartphone customers prefer large (>= 10 GB) bundles: this is the signal the upsell model learns.
# MAGIC Purchases are slightly more frequent in the first five days of the month (salary effect).

# COMMAND ----------

price_map = map_from_dict(BUNDLE_PRICE)
LARGE_OPTIONS = [(6, .25), (8, .30), (9, .20), (10, .05), (17, .20)]
SMALL_OPTIONS = [(1, .14), (2, .14), (3, .08), (4, .10), (5, .08), (7, .10), (11, .05), (12, .06),
                 (13, .06), (14, .02), (15, .06), (16, .05), (18, .06)]

ev = propose(VOLUMES["purchases"], traits, "w_purchase", 300,
             ["large_pref", "mobile_money_registered"],
             retention=0.72)
ev = ev.filter((F.dayofmonth("event_date") <= 5) | (rnd(305) < 0.78))       # month-start salary bump
purchases = (ev
             .withColumn("_u_large", rnd(311)).withColumn("_u_small", rnd(312))
             .withColumn("bundle_id", F.when(rnd(310) < F.col("large_pref"),
                                             weighted_choice(F.col("_u_large"), LARGE_OPTIONS))
                         .otherwise(weighted_choice(F.col("_u_small"), SMALL_OPTIONS)).cast("int"))
             .withColumn("amount_tzs", price_map[F.col("bundle_id")].cast("int"))
             .transform(add_choice, "_ch", 313, PURCHASE_CHANNELS)
             .withColumn("channel", F.when((F.col("_ch") == "mobile_money") & (~F.col("mobile_money_registered")),
                                           F.lit("agent")).otherwise(F.col("_ch")))
             .withColumn("purchase_id", make_id("BP"))
             .select("purchase_id", "customer_id", "event_ts", "bundle_id", "amount_tzs", "channel"))
purchases = corrupt(purchases, "amount_tzs", 0.002, F.lit(-500), 321)
purchases = corrupt(purchases, "bundle_id", 0.0015, F.lit(99), 322)                               # unknown bundle
purchases = corrupt(purchases, "customer_id", 0.002, NULL_ID, 323)
purchases = corrupt(purchases, "customer_id", 0.0015, (F.lit(N + 1000) + F.floor(rnd(324) * 500)).cast("long"), 325)
purchases = add_duplicates(purchases, 0.004, 326)
write_landing(purchases, "bundle_purchases")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Mobile-money transactions (only customers registered for mobile money)

# COMMAND ----------

ev = propose(VOLUMES["mobile_money"], traits, "w_mm", 400, [], retention=0.90)
mm = (ev
      .transform(add_choice, "txn_type", 410, MM_TXN_TYPES)
      .withColumn("amount_tzs", F.round(clip(F.exp(rndn(411) * 1.05 + 9.6), 200.0, 4000000.0), 0).cast("long"))
      .withColumn("_u_ch", rnd(412))
      .withColumn("channel", F.when(F.col("txn_type").isin("cash_in", "cash_out"), F.lit("agent"))
                  .when(F.col("_u_ch") < 0.55, F.lit("ussd")).otherwise(F.lit("app")))
      .withColumn("txn_id", make_id("MM", 10))
      .select("txn_id", "customer_id", "event_ts", "txn_type", "amount_tzs", "channel"))
mm = corrupt(mm, "amount_tzs", 0.002, F.lit(-1).cast("long"), 421)
mm = corrupt(mm, "customer_id", 0.002, NULL_ID, 422)
mm = corrupt(mm, "customer_id", 0.0015, (F.lit(N + 1000) + F.floor(rnd(423) * 500)).cast("long"), 424)
mm = add_duplicates(mm, 0.004, 426)
write_landing(mm, "mobile_money_txns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 · Network events (per cell; poor cells and cells without power backup fail more)

# COMMAND ----------

keep_prob = float(cells_sim.agg(F.avg(1.05 - F.col("cell_quality"))).first()[0])
n_prop = int(VOLUMES["network_events"] / keep_prob * 1.02)
ne = (spark.range(n_prop).withColumnRenamed("id", "event_id")  # noqa: F821
      .withColumn("cell_id", (F.floor(rnd(800) * N_CELLS) + 1).cast("long"))
      .join(F.broadcast(cells_sim.select("cell_id", "cell_quality", "power_backup")), "cell_id")
      .filter(rnd(801) < (1.05 - F.col("cell_quality")))
      .withColumn("_day", F.floor(rnd(802) * SPAN_DAYS).cast("int"))
      .withColumn("event_ts", F.timestamp_seconds(
          F.unix_timestamp(F.expr(f"date_add(DATE'{SPAN_START}', _day)").cast("timestamp")) + F.floor(rnd(803) * 86400)))
      .withColumn("_u_pwr", rnd(804)).withColumn("_u_ev", rnd(805))
      .withColumn("event_type", F.when(F.col("_u_pwr") < F.when(F.col("power_backup"), F.lit(0.04)).otherwise(F.lit(0.20)),
                                       F.lit("power_failure"))
                  .otherwise(weighted_choice(F.col("_u_ev"), [("call_drop_spike", .30), ("weak_signal", .30),
                                                              ("congestion", .25), ("cell_down", .15)])))
      .transform(add_choice, "severity", 806, [(1, .55), (2, .30), (3, .15)])
      .withColumn("severity", F.col("severity").cast("int"))
      .withColumn("_mean", F.when(F.col("event_type") == "power_failure", F.lit(180.0))
                  .when(F.col("event_type") == "cell_down", F.lit(90.0)).otherwise(F.lit(45.0)))
      .withColumn("duration_minutes", F.least(F.lit(2880), F.ceil(-F.log(1 - rnd(807)) * F.col("_mean"))).cast("int"))
      .withColumn("network_event_id", make_id("NE"))
      .select("network_event_id", "cell_id", "event_ts", "event_type", "severity", "duration_minutes"))
ne = corrupt(ne, "severity", 0.002, F.lit(9), 811)
ne = corrupt(ne, "cell_id", 0.0015, F.lit(N_CELLS + 500).cast("long"), 812)
ne = add_duplicates(ne, 0.004, 813)
write_landing(ne, "network_events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12 · Care tickets (published with dirty rows)

# COMMAND ----------

tickets = tickets_clean
tickets = corrupt(tickets, "customer_id", 0.002, NULL_ID, 901)
tickets = corrupt(tickets, "customer_id", 0.0015, (F.lit(N + 1000) + F.floor(rnd(902) * 500)).cast("long"), 903)
tickets = corrupt(tickets, "resolved_ts", 0.002,
                  F.col("created_ts") - F.expr("INTERVAL 2 DAYS"), 904)                            # resolved before created
tickets = tickets.withColumn("resolved_ts", F.when(F.col("status") == "resolved", F.col("resolved_ts")))
tickets = add_duplicates(tickets, 0.004, 905)
write_landing(tickets, "care_tickets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13 · Manifest and summary

# COMMAND ----------

counts = {}
for name in ["customers", "regions_cells", "bundle_catalog", "voice_usage", "data_sessions", "bundle_purchases",
             "mobile_money_txns", "network_events", "care_tickets"]:
    counts[name] = spark.read.parquet(f"{LANDING_PATH}/{name}").count()  # noqa: F821

manifest = {
    "company": "Simba Telecom (fictional)", "data": "synthetic", "scale": SCALE, "seed": GLOBAL_SEED,
    "observation_window": [str(SPAN_START), str(OBS_END)], "label_window": [str(LABEL_START), str(LABEL_END)],
    "hidden_churn_rate": round(float(latent_rate), 4), "row_counts": counts,
}
with open(f"{LANDING_PATH}/_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

banner("Landing zone summary")
for k, v in counts.items():
    print(f"{k:<20}{v:>12,}")
print(f"{'TOTAL':<20}{sum(counts.values()):>12,}")
