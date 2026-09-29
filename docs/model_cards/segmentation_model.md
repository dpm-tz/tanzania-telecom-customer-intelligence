# Model card: Customer segmentation

## Overview
| | |
|---|---|
| Task | Unsupervised clustering of customers by behaviour |
| Trained in | `notebooks/55_segment_customers.py` |
| Algorithm | K-Means (Spark ML), `k` chosen automatically |
| Preprocessing | `log1p` on all eight inputs (to tame skew), then `StandardScaler` (zero mean, unit variance) |
| Saved to | Volume `tz_ml.artifacts/models/segmentation/versions/<version>/{scaler,kmeans}`, pointer in `champion.json` |

## Intended use
Give marketing and product teams a small number of readable customer groups to design campaigns and offers around
(e.g. "Data Heavy Users" vs "Mobile Money Power Users"). Not intended as a precise causal explanation of why any
individual customer behaves the way they do.

## Inputs (`SEGMENT_FEATURES` in `00_config.py`)
`calls_out_90d`, `voice_minutes_90d`, `data_gb_90d`, `spend_tzs_90d`, `mm_txns_90d`, `mm_value_tzs_90d`,
`active_days_90d`, `tenure_days`.

## How `k` is chosen
1. Fit K-Means for k = 4 … 7 (fewer than 4 segments is not useful for a marketing team).
2. Score each with the **silhouette coefficient** (higher = better-separated, more cohesive clusters).
3. Reject any k whose smallest cluster is under 3 % of customers (too small to run a campaign against).
4. Pick the best silhouette among what remains.

## How segments are named
Each cluster's profile (data usage, voice usage, mobile-money activity, ARPU, tenure, activity days, mobile-money
penetration) is compared with the *other* clusters using z-scores. Names are assigned in a fixed priority order —
low activity first (the most business-critical group), then non-mobile-money, data-heavy, mobile-money power users,
voice-first, high-value — and only when a cluster clearly stands out on that trait. Anything left over is labelled
"Everyday Users", split by whether they use mobile money. Naming is therefore descriptive and rule-based, not a
separate model.

## What to expect (tiny-scale rehearsal — your run will differ)
Typically 4-5 segments emerge, silhouette around 0.4-0.6. A "Low-Activity / Dormant-Prone" segment reliably appears
with a churn rate well above the rest — this is a useful early-warning list even without the churn model.

## Limitations
* **K-Means assumes roughly spherical, similarly-sized clusters** in the scaled feature space. Segments with very
  different shapes or densities (e.g. a few very high-value outliers) may be absorbed into a larger cluster rather
  than standing out.
* **Sensitive to feature choice.** Adding or removing an input column (e.g. adding network/care features) will change
  the segments. The current eight features focus on commercial value and engagement, not network experience.
* **Not guaranteed stable over time.** Re-running on a later snapshot can shuffle cluster numbers and slightly change
  boundaries; always join on `segment_name`, never assume `cluster_id` 0 means the same thing across runs (the
  `champion.json` for a given run records the `cluster_id → segment_name` mapping actually used).
* **Descriptive, not causal.** "Mobile Money Power Users churn less" is an association in this segment's profile, not
  proof that promoting mobile money *causes* retention.
