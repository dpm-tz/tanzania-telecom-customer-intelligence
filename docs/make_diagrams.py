"""Draw docs/images/architecture.png and docs/images/data_model.png (matplotlib only, no extra tools).

    python docs/make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)

C = {"src": "#f3e5d0", "bronze": "#e6c9a8", "silver": "#d5dbe0", "gold": "#f6e3a1", "ml": "#cfe6d8",
     "out": "#cfe0f2", "ink": "#22303c", "line": "#5b6b78"}


def box(ax, x, y, w, h, title, lines=(), color="#ffffff", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=C["line"], lw=1.1))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=fs + 1, fontweight="bold", color=C["ink"])
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.5 - i * 0.28, line, ha="center", va="top", fontsize=fs - 1, color=C["ink"])


def arrow(ax, x1, y1, x2, y2, style="-|>", color=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=1.4,
                                 color=color or C["line"], connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(0.2, 6.95, "Tanzania Telecom Customer Intelligence: architecture", fontsize=15, fontweight="bold", color=C["ink"])
    ax.text(0.2, 6.6, "Databricks Lakehouse, medallion layers, three ML models, one dashboard. All data is synthetic.",
            fontsize=10, color=C["line"])

    box(ax, 0.2, 1.4, 2.2, 4.6, "Source systems", ["(simulated)", "", "customers", "voice usage", "data sessions",
                                                    "bundle purchases", "mobile money", "network events", "care tickets",
                                                    "cells · bundle catalogue"], C["src"])
    box(ax, 2.9, 1.4, 1.9, 4.6, "Landing", ["Volume", "tz_raw.landing", "", "Parquet files", "exactly as", "delivered"], C["src"])
    box(ax, 5.3, 1.4, 2.1, 4.6, "Bronze", ["tz_bronze", "", "raw copy", "+ _ingested_at", "+ _source_file", "", "never fix,", "only copy"], C["bronze"])
    box(ax, 7.9, 1.4, 2.3, 4.6, "Silver", ["tz_silver", "", "data-quality rules", "dedupe · ranges ·", "allowed values ·", "referential checks", "", "dq_quarantine"], C["silver"])
    box(ax, 10.7, 1.4, 2.2, 4.6, "Gold", ["tz_gold", "", "customer_360", "KPI marts:", "region · month ·", "network · care ·", "bundles · money"], C["gold"])
    box(ax, 13.4, 3.9, 2.4, 2.1, "ML layer", ["tz_ml", "features · labels", "scores · segments"], C["ml"])
    box(ax, 13.4, 1.4, 2.4, 2.1, "Serving", ["AI/BI dashboard", "retention target list", "PNG charts"], C["out"])

    for x1, x2 in [(2.4, 2.9), (4.8, 5.3), (7.4, 7.9), (10.2, 10.7)]:
        arrow(ax, x1, 3.7, x2, 3.7)
    arrow(ax, 12.9, 4.4, 13.4, 4.8)
    arrow(ax, 13.4, 4.2, 12.9, 3.9, rad=-0.2)
    arrow(ax, 12.9, 3.2, 13.4, 2.5)
    arrow(ax, 14.6, 3.9, 14.6, 3.5)

    ax.add_patch(FancyBboxPatch((0.6, 0.15), 14.8, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fbfbfb",
                                ec=C["line"], lw=1, ls="--"))
    ax.text(8.0, 0.93, "Models (saved in Volume tz_ml.artifacts/models, versioned, champion.json)", ha="center",
            fontsize=9.5, fontweight="bold", color=C["ink"])
    ax.text(8.0, 0.5, "Churn: logistic regression vs gradient-boosted trees   |   Segments: K-Means (k by silhouette)   |   "
                      "Upsell: logistic regression vs random forest", ha="center", fontsize=8.6, color=C["ink"])
    fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def data_model():
    fig, ax = plt.subplots(figsize=(19.6, 8.6))
    ax.set_xlim(0, 19.6)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.text(0.2, 8.3, "Source data model (nine tables)", fontsize=15, fontweight="bold", color=C["ink"])
    ax.text(0.2, 7.95, "PK = primary key, FK = foreign key. Arrows show the join keys the pipeline validates in Silver "
                       "(voice_usage and data_sessions also carry FK cell_id to regions_cells).", fontsize=10, color=C["line"])

    def entity(x, y, name, cols, color, w=3.7):
        h = 0.5 + 0.27 * len(cols)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec=C["line"], lw=1.1))
        ax.text(x + w / 2, y + h - 0.1, name, ha="center", va="top", fontsize=10.5, fontweight="bold", color=C["ink"])
        for i, c in enumerate(cols):
            ax.text(x + 0.15, y + h - 0.5 - i * 0.27, c, fontsize=8.3, va="top", color=C["ink"], family="monospace")
        return (x, y, w, h)

    voice = entity(0.3, 5.5, "voice_usage", ["PK call_id", "FK customer_id", "FK cell_id", "   event_ts, direction", "   call_type, duration_sec"], C["bronze"])
    data = entity(0.3, 3.0, "data_sessions", ["PK session_id", "FK customer_id", "FK cell_id", "   event_ts, app_category", "   data_mb, session_minutes"], C["bronze"])
    tick = entity(0.3, 0.4, "care_tickets", ["PK ticket_id", "FK customer_id", "   created_ts, category, channel", "   priority, status", "   resolved_ts, resolution_hours"], C["bronze"])
    cust = entity(6.3, 3.3, "customers", ["PK customer_id", "   region, district, area_type", "FK home_cell_id", "   age, gender, device_type",
                                          "   is_multi_sim", "   mobile_money_registered", "   registration_status",
                                          "   acquisition_channel", "   activation_date"], C["gold"], w=4.0)
    cells = entity(6.3, 0.4, "regions_cells", ["PK cell_id", "   region, district, area_type", "   technology (2G/3G/4G)", "   power_backup"], C["silver"], w=4.0)
    purch = entity(11.7, 5.5, "bundle_purchases", ["PK purchase_id", "FK customer_id", "FK bundle_id", "   event_ts, amount_tzs", "   channel"], C["bronze"])
    mm = entity(11.7, 3.0, "mobile_money_txns", ["PK txn_id", "FK customer_id", "   event_ts, txn_type", "   amount_tzs, channel"], C["bronze"])
    net = entity(11.7, 0.4, "network_events", ["PK network_event_id", "FK cell_id", "   event_ts, event_type", "   severity, duration_minutes"], C["bronze"])
    cat = entity(15.8, 5.5, "bundle_catalog", ["PK bundle_id", "   bundle_code, category", "   price_tzs, validity_days", "   data_mb, voice_minutes, sms"], C["silver"])

    def mid(e, side):
        x, y, w, h = e
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]

    links = [(voice, "r", cust, "l"), (data, "r", cust, "l"), (tick, "r", cust, "l"), (purch, "l", cust, "r"),
             (mm, "l", cust, "r"), (cust, "b", cells, "t"), (net, "l", cells, "r"), (purch, "r", cat, "l")]
    for a, sa, b, sb in links:
        (x1, y1), (x2, y2) = mid(a, sa), mid(b, sb)
        arrow(ax, x1, y1, x2, y2, style="-|>", rad=0.0)
    fig.savefig(OUT / "data_model.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    data_model()
    print("wrote", OUT / "architecture.png", "and", OUT / "data_model.png")
