# Data dictionary

All monetary columns are in **TZS**. All dates/timestamps are naive (no timezone offset stored), business timezone
is Africa/Dar_es_Salaam. `as_of_date` / snapshot date throughout is **2026-06-29**.

## 1. Bronze / Silver (source-aligned tables)

### `customers`
| Column | Type | Notes |
|---|---|---|
| customer_id | long | PK |
| customer_code | string | External-looking code, `SIM-0000123` |
| region, district | string | See `docs/data_assumptions.md` for the region list |
| area_type | string | `urban` / `rural` |
| home_cell_id | long | FK → `regions_cells.cell_id` |
| age | int | 15-100 after Silver |
| gender | string | `F` / `M` |
| device_type | string | `smartphone` / `feature_phone` |
| is_multi_sim | boolean | |
| mobile_money_registered | boolean | |
| registration_status | string | `fully_registered` / `pending_verification` |
| acquisition_channel | string | `agent` / `shop` / `online` / `referral` |
| activation_date | date | |

### `regions_cells`
| Column | Type | Notes |
|---|---|---|
| cell_id | long | PK |
| region, district, area_type | string | |
| technology | string | `2G` / `3G` / `4G` |

### `bundle_catalog`
| Column | Type | Notes |
|---|---|---|
| bundle_id | int | PK |
| bundle_code | string | e.g. `MWEZI_15GB` |
| category | string | `data` / `voice` / `combo` / `sms` |
| price_tzs | int | |
| validity_days | int | |
| data_mb, voice_minutes, sms_count | int | Bundle contents |

### `voice_usage`
| Column | Type | Notes |
|---|---|---|
| call_id | string | PK |
| customer_id | long | FK |
| event_ts | timestamp | |
| direction | string | `outgoing` / `incoming` |
| call_type | string | `on_net` / `off_net` / `international` |
| duration_sec | int | 1-14400 after Silver |
| cell_id | long | FK |

### `data_sessions`
| Column | Type | Notes |
|---|---|---|
| session_id | string | PK |
| customer_id | long | FK |
| event_ts | timestamp | |
| app_category | string | `whatsapp` / `social_media` / `video` / `browsing` / `other` |
| data_mb | double | > 0, ≤ 50,000 after Silver |
| session_minutes | int | |
| cell_id | long | FK |

### `bundle_purchases`
| Column | Type | Notes |
|---|---|---|
| purchase_id | string | PK |
| customer_id | long | FK |
| event_ts | timestamp | |
| bundle_id | int | FK |
| amount_tzs | int | > 0 after Silver |
| channel | string | `mobile_money` / `agent` / `ussd` / `app` |

### `mobile_money_txns`
| Column | Type | Notes |
|---|---|---|
| txn_id | string | PK |
| customer_id | long | FK |
| event_ts | timestamp | |
| txn_type | string | `send_money` / `receive_money` / `cash_in` / `cash_out` / `bill_payment` / `airtime_purchase` |
| amount_tzs | long | 100-5,000,000 after Silver |
| channel | string | `agent` / `ussd` / `app` |

### `network_events`
| Column | Type | Notes |
|---|---|---|
| network_event_id | string | PK |
| cell_id | long | FK |
| event_ts | timestamp | |
| event_type | string | `call_drop_spike` / `weak_signal` / `congestion` / `cell_down` / `power_failure` |
| severity | int | 1-3 |
| duration_minutes | int | |

### `care_tickets`
| Column | Type | Notes |
|---|---|---|
| ticket_id | string | PK |
| customer_id | long | FK |
| created_ts | timestamp | |
| category | string | `network_coverage` / `slow_internet` / `billing` / `mobile_money_issue` / `bundle_dispute` / `sim_registration` / `other` |
| channel | string | `call_centre` / `shop` / `whatsapp` / `ussd` |
| priority | string | `low` / `medium` / `high` |
| status | string | `resolved` / `open` |
| resolved_ts | timestamp, nullable | Null if still open |
| resolution_hours | int, nullable | |

### `dq_quarantine`
| Column | Type | Notes |
|---|---|---|
| source_table | string | Which Bronze table the row came from |
| record_key | string | The row's primary key, as text |
| failed_rules | string | Comma-separated rule names |
| record_json | string | The original row, as JSON |
| quarantined_at | timestamp | |

## 2. Gold: `customer_360` (one row per customer, as of 2026-06-29)

| Group | Columns |
|---|---|
| Profile | customer_id, customer_code, region, district, area_type, gender, age, device_type, acquisition_channel, registration_status, is_multi_sim, mobile_money_registered, activation_date, tenure_days, home_cell_id, cell_technology, cell_has_power_backup |
| Voice (90d) | calls_out_90d, calls_in_90d, voice_minutes_90d, avg_call_sec_90d, offnet_share_90d, intl_calls_90d, calls_out_30d, voice_trend_ratio |
| Data (90d) | sessions_90d, data_gb_90d, data_gb_30d, video_social_share_90d, active_data_days_90d, data_trend_ratio |
| Spend | purchases_90d, spend_tzs_90d, spend_tzs_30d, avg_purchase_tzs_90d, distinct_bundle_types_90d, large_bundle_purchases_90d, bought_large_bundle_last30d, arpu_monthly_tzs, spend_trend_ratio |
| Mobile money | mm_txns_90d, mm_value_tzs_90d, mm_txns_30d, bill_payments_90d, mm_trend_ratio |
| Activity / recency | last_activity_date, active_days_90d, active_days_30d, days_since_last_activity |
| Network (home cell) | cell_events_30d, cell_power_failures_30d, cell_outage_minutes_90d |
| Care | tickets_90d, network_tickets_90d, open_tickets, avg_resolution_hours, slow_tickets_90d |

`voice_trend_ratio`, `data_trend_ratio`, `spend_trend_ratio`, `mm_trend_ratio`: last-30-days activity divided by the
monthly average of the **prior** 60 days (capped at 3.0). 1.0 = stable, below 1.0 = declining, `1.0` default when
there is no prior activity to compare against.

## 3. ML tables (`tz_ml`)

| Table | Grain | Key columns |
|---|---|---|
| `customer_features` | customer | `ML_FEATURES` list (see `00_config.py`) + arpu, bought_large_bundle_last30d |
| `labels` | customer | `churn_30d`, `bought_large_bundle_30d`, `eligible_for_upsell`, label window dates |
| `churn_scores` | customer | `churn_probability`, `risk_band` (High/Medium/Low), `risk_rank_pct`, `model_version` |
| `churn_test_scored` | customer (hold-out only) | `churn_probability`, `churn_30d` — used for all "hold-out" charts and metrics |
| `upsell_scores` | customer (eligible only) | `upsell_probability`, `upsell_band` (High/Standard) |
| `customer_segments` | customer | `cluster_id`, `segment_name` |
| `retention_actions` | customer | scores + drivers + recommendation, see below |
| `feature_importance` | model × feature | global importance, all models |
| `model_metrics` | model × metric | every metric from every training run (not just the champion) |
| `segmentation_eval` | k | silhouette per candidate k |
| `dq_quarantine` (Silver, referenced from ML docs) | row | see above |

### `retention_actions` (the table the business actually uses)
| Column | Meaning |
|---|---|
| churn_probability, risk_band, risk_rank_pct | From the churn model |
| top_drivers, primary_driver | Up to 3 readable reasons (rule-based, see notebook 70) |
| upsell_probability, upsell_band | From the upsell model (`Not eligible` if they already bought a large bundle) |
| arpu_monthly_tzs, revenue_at_risk_tzs | `arpu × churn_probability × 3 months` |
| recommended_action, owner_team, campaign_priority | 1 = urgent retention, 2 = targeted marketing, 3 = upsell, 4 = monitor |
| actual_churn_30d | What really happened — **for evaluation, never as a model input** |

## 4. Gold marts for the dashboard

| Table | Grain |
|---|---|
| `kpi_region_month`, `kpi_daily` | Region×month / day: revenue, minutes, GB, mobile-money value, active customers |
| `region_summary` | Region: customers, ARPU, churn, high-risk counts and revenue at risk |
| `churn_scorecard` | Region × risk band |
| `churn_gains` | Decile (hold-out): cumulative recall and lift |
| `retention_target_list` | Customer (High/Medium risk only), sorted by revenue at risk |
| `segment_profile` | Segment: size, averages, churn rate |
| `network_quality_region` | Region × month: events, power failures, outage minutes |
| `care_performance` | Region × category × month: resolution stats |
| `bundle_performance` | Bundle × month: purchases, revenue, channel mix |
| `mobile_money_summary` | Transaction type × channel × month |
| `model_performance` | Latest metrics per model (feeds the "Trust" page) |
| `data_quality_summary` | Rows rejected per source table |
