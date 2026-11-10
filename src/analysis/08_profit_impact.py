"""
PromoGuard AI — Phase 8: Gross-Profit Impact and Promotion ROI
==============================================================

Location in repo:  PromoGuard-AI/src/analysis/08_profit_impact.py
Run from repo root:
    python src/analysis/08_profit_impact.py

Inputs
    data/processed/tna_counterfactual_rows.parquet   (Phase 6: actual and counterfactual units per row)
    data/processed/tna_event_cannibalization.csv     (Phase 7: events + cannibalized units)
Outputs
    data/processed/tna_event_profit.csv              one row per event: the final PromoGuard table
    reports/phase8/phase8_profit_report.txt
    reports/phase8/figures/*.png

Run from repo root:
    python src/analysis/08_profit_impact.py              # canned tuna (default, --cat tna)
    python src/analysis/08_profit_impact.py --cat sdr    # carbonated soft drinks
    (soft drinks: files start with sdr_, report reports/phase8/phase8_sdr_profit_report.txt,
     figures in reports/phase8/sdr/figures/)

What "profit" means here
    The data has PROFIT = gross margin % on each row (manual): cents of margin per dollar of
    sales, computed with the AVERAGE ACQUISITION COST (AAC) of the inventory. It excludes
    labour, rent, advertising, slotting fees, etc. All results are therefore
    ESTIMATED GROSS-PROFIT IMPACT, never net profit.

    unit cost   = unit price x (1 - margin% / 100)
    gross profit = units x (unit price - unit cost)

Two cost views (because of the AAC caveat in the manual)
    RECORDED cost   : the cost in the data during the promotion. When manufacturers fund a
                      promotion with a trade deal, the chain buys cheaply in advance and the AAC
                      falls — so part of the discount is paid by the manufacturer.
    PRE-PROMO cost  : the product's regular cost just before the event (median over the previous
                      13 regular weeks in that store, using only plausible margins of 0-95%). Profit as if the retailer paid the whole
                      discount itself. Conservative view.
    The true retailer outcome lies between the two (the data does not show trade payments directly).

Event-level decomposition (gross profit, $)
    CF_GP            counterfactual units x (regular price - cost)          what the weeks would have earned
    PROMO_GP         actual units x (promoted price - cost)                  what they did earn
    GROSS_GP_LIFT    PROMO_GP - CF_GP
    POST_GP_EFFECT   (actual - counterfactual units after the event) x regular unit margin
    CANNIB_GP_LOSS   cannibalized units (Phase 7) x average regular unit margin of other tuna that week
    NET_GP_IMPACT    GROSS_GP_LIFT + POST_GP_EFFECT - CANNIB_GP_LOSS

ROI proxy (promotion cost is not in the data — not fabricated)
    DISCOUNT_COST    actual units x (regular price - promoted price)   = revenue given away
    SUBSIDISED_COST  counterfactual units x (regular price - promoted price)
                     = discount paid on units that would have sold anyway
    GP_ROI           NET_GP_IMPACT / DISCOUNT_COST   (gross-profit return per $ of discount)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports" / "phase8"
FIG_DIR = REPORT_DIR / "figures"

CATEGORIES = {"tna": {"name": "canned tuna", "unit": "tuna"},
              "sdr": {"name": "carbonated soft drinks", "unit": "soft drinks"}}
CAT, CAT_NAME, UNIT = "tna", "canned tuna", "tuna"     # set in main()

COST_HISTORY_WEEKS = 13
SEED = 42
N_BOOT = 1000

SURFACE, INK, INK_2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, RED, AQUA = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def section(title: str) -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {(FIG_DIR / name).relative_to(REPO_ROOT).as_posix()}")


def money(v: float) -> str:
    sign = "-" if v < 0 else ""
    a = abs(v)
    return f"{sign}${a:,.0f}" if a < 1e6 else f"{sign}${a / 1e6:,.2f}M"


# ----------------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------------
def add_costs(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values(["PRODUCT_ID", "STORE", "WEEK"]).reset_index(drop=True)
    r["COST"] = r["UNIT_PRICE"] * (1 - r["MARGIN_PCT"] / 100)
    # normal cost history: regular-price weeks with a plausible margin (0-95%); margins >= 95%
    # mean a near-zero accounting cost (AAC artefact flagged in Phase 2) and would distort costs
    plausible = r["MARGIN_PCT"].between(0, 95)
    reg_cost = r["COST"].where(r["PROMO_TYPE"].eq("regular") & plausible)
    r["COST_REG_HIST"] = reg_cost.groupby([r["PRODUCT_ID"], r["STORE"]]).transform(
        lambda s: s.shift(1).rolling(COST_HISTORY_WEEKS, min_periods=1).median())
    # cost and regular price frozen at the first event week of each (event, store)
    promo = r[r["PHASE"].eq("promo")].sort_values("WEEK")
    first = promo.groupby(["EVENT_ID", "STORE"])[["COST_REG_HIST", "REG_PRICE"]].first() \
        .rename(columns={"COST_REG_HIST": "COST_PRE", "REG_PRICE": "REG_PRICE_PRE"})
    r = r.join(first, on=["EVENT_ID", "STORE"])
    fallback = reg_cost.groupby([r["PRODUCT_ID"], r["STORE"]]).transform("median")
    r["COST_PRE"] = r["COST_PRE"].fillna(r["COST_REG_HIST"]).fillna(fallback)
    r["REG_PRICE_PRE"] = r["REG_PRICE_PRE"].fillna(r["REG_PRICE"])
    return r


def regular_unit_margin_by_week(r: pd.DataFrame) -> pd.Series:
    """Average gross profit per unit of non-promoted regular-price tuna, by week (for cannibalization)."""
    x = r[r["PROMO_TYPE"].eq("regular") & ~r["PROMO"] & r["COVERAGE_OK"].astype(bool)]
    g = x.groupby("WEEK").agg(gp=("GROSS_PROFIT", "sum"), u=("MOVE", "sum"))
    return g["gp"] / g["u"]


# ----------------------------------------------------------------------------
# Event table
# ----------------------------------------------------------------------------
def event_profit(r: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    p = r[r["PHASE"].eq("promo")].copy()
    p["REG_PRICE_E"] = p["REG_PRICE"].fillna(p["REG_PRICE_PRE"])
    p["CF_GP_REC"] = p["PRED_CF"] * (p["REG_PRICE_E"] - p["COST"])         # recorded cost of that week
    p["CF_GP_PRE"] = p["PRED_CF"] * (p["REG_PRICE_E"] - p["COST_PRE"])
    p["PROMO_GP_REC"] = p["GROSS_PROFIT"]
    p["PROMO_GP_PRE"] = p["MOVE"] * (p["UNIT_PRICE"] - p["COST_PRE"])
    disc = (p["REG_PRICE_E"] - p["UNIT_PRICE"]).clip(lower=0)
    p["DISCOUNT_COST"] = p["MOVE"] * disc
    p["SUBSIDISED_COST"] = p["PRED_CF"] * disc
    p["CF_REVENUE"] = p["PRED_CF"] * p["REG_PRICE_E"]
    agg = p.groupby("EVENT_ID")[["CF_GP_REC", "CF_GP_PRE", "PROMO_GP_REC", "PROMO_GP_PRE", "DISCOUNT_COST",
                                 "SUBSIDISED_COST", "CF_REVENUE", "DOLLARS"]].sum() \
        .rename(columns={"DOLLARS": "PROMO_REVENUE"})

    post = r[r["PHASE"].eq("post")].copy()
    post["UNIT_GP"] = post["REG_PRICE"] - post["COST"]
    post["POST_GP_EFFECT"] = (post["MOVE"] - post["PRED_CF"]) * post["UNIT_GP"]
    agg = agg.join(post.groupby("EVENT_ID")["POST_GP_EFFECT"].sum())

    out = ev.merge(agg, left_on="EVENT_ID", right_index=True, how="left")
    out["POST_GP_EFFECT"] = out["POST_GP_EFFECT"].fillna(0.0)

    # cannibalization: lost units x average regular unit margin of other tuna in the event weeks
    um = regular_unit_margin_by_week(r)
    out["OTHER_UNIT_GP"] = [um.reindex(range(s, e + 1)).mean() for s, e in zip(out["START_WEEK"], out["END_WEEK"])]
    out["CANNIB_GP_LOSS"] = out["CANNIBALIZED_UNITS"].fillna(0) * out["OTHER_UNIT_GP"].fillna(um.mean())

    for v in ("REC", "PRE"):
        out[f"GROSS_GP_LIFT_{v}"] = out[f"PROMO_GP_{v}"] - out[f"CF_GP_{v}"]
        out[f"NET_GP_IMPACT_{v}"] = out[f"GROSS_GP_LIFT_{v}"] + out["POST_GP_EFFECT"] - out["CANNIB_GP_LOSS"]
        out[f"GP_ROI_{v}"] = out[f"NET_GP_IMPACT_{v}"] / out["DISCOUNT_COST"].replace(0, np.nan)
        out[f"PROFITABLE_{v}"] = out[f"NET_GP_IMPACT_{v}"] > 0
    out["TRADE_FUNDING_EST"] = out["NET_GP_IMPACT_REC"] - out["NET_GP_IMPACT_PRE"]
    out["SUBSIDISED_SHARE"] = out["SUBSIDISED_COST"] / out["DISCOUNT_COST"].replace(0, np.nan)

    # uncertainty: counterfactual units at the event's 90% bounds (from Phase 6 placebo errors)
    scale_hi = out["PROMO_ACTUAL"] / (1 + out["LIFT_PCT_LO90"] / 100) / out["PROMO_CF"]
    scale_lo = out["PROMO_ACTUAL"] / (1 + out["LIFT_PCT_HI90"] / 100) / out["PROMO_CF"]
    for v in ("REC", "PRE"):
        base = out[f"NET_GP_IMPACT_{v}"] + out[f"CF_GP_{v}"]
        out[f"NET_GP_IMPACT_{v}_LO90"] = base - out[f"CF_GP_{v}"] * scale_hi
        out[f"NET_GP_IMPACT_{v}_HI90"] = base - out[f"CF_GP_{v}"] * scale_lo
    out["DEPTH_BUCKET"] = pd.cut(out["MEAN_DEPTH"], [-np.inf, 0.05, 0.10, 0.20, np.inf],
                                 labels=["<5%", "5-10%", "10-20%", ">=20%"], right=False)
    return out


def pooled_table(ev: pd.DataFrame, by: str | None) -> pd.DataFrame:
    groups = [("all", ev)] if by is None else list(ev.groupby(by, observed=True))
    rows = []
    for key, g in groups:
        rows.append({
            "group": key, "events": len(g),
            "cf_gp": g["CF_GP_PRE"].sum(),
            "net_gp_recorded": g["NET_GP_IMPACT_REC"].sum(),
            "net_gp_pre_cost": g["NET_GP_IMPACT_PRE"].sum(),
            "gp_change_%_pre": g["NET_GP_IMPACT_PRE"].sum() / g["CF_GP_PRE"].sum() * 100,
            "discount_cost": g["DISCOUNT_COST"].sum(),
            "roi_recorded": g["NET_GP_IMPACT_REC"].sum() / g["DISCOUNT_COST"].sum(),
            "roi_pre_cost": g["NET_GP_IMPACT_PRE"].sum() / g["DISCOUNT_COST"].sum(),
            "%_profitable_rec": g["PROFITABLE_REC"].mean() * 100,
            "%_profitable_pre": g["PROFITABLE_PRE"].mean() * 100,
        })
    return pd.DataFrame(rows).set_index("group")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_waterfall(ev: pd.DataFrame) -> None:
    gross = ev["GROSS_GP_LIFT_PRE"].sum()
    post, can = ev["POST_GP_EFFECT"].sum(), -ev["CANNIB_GP_LOSS"].sum()
    trade = ev["TRADE_FUNDING_EST"].sum()
    steps = [("Gross-profit lift\n(promoted products)", gross), ("After-event\nweeks", post),
             (f"Cannibalized\nother {UNIT}", can), ("Net impact,\npre-promo cost", None),
             ("Lower recorded\ncost (trade deals)", trade), ("Net impact,\nrecorded cost", None)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    style_axis(ax)
    level = 0.0
    for i, (lab, v) in enumerate(steps):
        if v is None:
            ax.bar(i, level, color=BLUE, width=0.6)
            ax.text(i, level + abs(gross) * 0.02, money(level), ha="center", fontsize=8.5, color=INK)
        else:
            bottom = level if v >= 0 else level + v
            ax.bar(i, abs(v), bottom=bottom, color=ORANGE if v >= 0 else RED, width=0.6)
            ax.text(i, max(level, level + v) + abs(gross) * 0.02, ("+" if v >= 0 else "") + money(v),
                    ha="center", fontsize=8.5, color=INK)
            level += v
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
    ax.set_ylabel("Gross profit, all in-scope events", color=INK_2, fontsize=9)
    ax.set_title("From sales lift to gross-profit impact", color=INK, fontsize=11, loc="left")
    save(fig, "gp_waterfall.png")


def plot_by_depth(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4), facecolor=SURFACE)
    style_axis(ax)
    x, w = np.arange(len(tbl)), 0.38
    ax.bar(x - w / 2, tbl["roi_pre_cost"], width=w, color=BLUE, label="Pre-promotion cost (retailer pays discount)")
    ax.bar(x + w / 2, tbl["roi_recorded"], width=w, color=AQUA, label="Recorded cost (incl. trade-deal savings)")
    ax.axhline(0, color=INK_2, lw=1)
    for xi, a, b in zip(x, tbl["roi_pre_cost"], tbl["roi_recorded"]):
        ax.text(xi - w / 2, a + (0.03 if a >= 0 else -0.08), f"{a:+.2f}", ha="center", fontsize=8, color=INK)
        ax.text(xi + w / 2, b + (0.03 if b >= 0 else -0.08), f"{b:+.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{i}\n({n} events)" for i, n in zip(tbl.index.astype(str), tbl["events"])])
    ax.set_xlabel("Average discount depth of the event", color=INK_2, fontsize=9)
    ax.set_ylabel("Net gross profit per $1 of discount", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, labelcolor=INK_2)
    fig.suptitle("Gross-profit return on discount, by depth", color=INK, fontsize=11, x=0.01, ha="left", y=1.08)
    save(fig, "gp_roi_by_depth.png")


def plot_scatter(ev: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    pos = ev["NET_GP_IMPACT_PRE"] >= 0
    ax.scatter(ev.loc[pos, "LIFT_PCT"].clip(-80, 2000), ev.loc[pos, "NET_GP_IMPACT_PRE"], s=14, color=BLUE,
               alpha=0.55, edgecolor="none", label="Gross profit increased")
    ax.scatter(ev.loc[~pos, "LIFT_PCT"].clip(-80, 2000), ev.loc[~pos, "NET_GP_IMPACT_PRE"], s=14, color=RED,
               alpha=0.65, edgecolor="none", label="Gross profit decreased")
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_yscale("symlog", linthresh=500)
    ax.set_xlabel("Unit sales lift % (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("Net gross-profit impact, pre-promo cost ($, log scale)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    ax.set_title("More units does not always mean more profit", color=INK, fontsize=11, loc="left")
    save(fig, "lift_vs_gp_scatter.png")


# ----------------------------------------------------------------------------
def main() -> None:
    global CAT, CAT_NAME, UNIT, FIG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", choices=sorted(CATEGORIES))
    CAT = parser.parse_args().cat
    CAT_NAME, UNIT = CATEGORIES[CAT]["name"], CATEGORIES[CAT]["unit"]
    if CAT != "tna":
        FIG_DIR = REPORT_DIR / CAT / "figures"
    report = "phase8_profit_report.txt" if CAT == "tna" else f"phase8_{CAT}_profit_report.txt"
    tee = Tee(REPORT_DIR / report)
    sys.stdout = tee
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        section(f"PROMOGUARD AI — PHASE 8 GROSS-PROFIT IMPACT — {CAT_NAME}")
        print("All figures are ESTIMATED GROSS-PROFIT impacts (margin on average acquisition cost),")
        print("not net profit. Model-based counterfactuals; see Phases 5-7 for assumptions.")
        r = pd.read_parquet(PROC / f"{CAT}_counterfactual_rows.parquet")
        ev_all = pd.read_csv(PROC / f"{CAT}_event_cannibalization.csv")
        r = add_costs(r)
        prom = r[r["PHASE"].eq("promo")]
        ratio = (prom["COST"] / prom["COST_PRE"]).replace([np.inf, -np.inf], np.nan)
        print(f"\nPromotion rows with a pre-promotion cost: {prom['COST_PRE'].notna().mean() * 100:.1f}%")
        print("Recorded cost during promotions / pre-promotion cost:")
        print(ratio.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
        print("-> Median ratio < 1 means the recorded cost usually drops during promotions (trade-deal buying);")
        print("   ~1 means mostly unchanged.")

        ev = event_profit(r, ev_all)
        scope = ev[ev["IN_SCOPE"].astype(bool)].copy()

        section("1. POOLED GROSS-PROFIT IMPACT (all in-scope events)")
        t = pooled_table(scope, None)
        g = scope
        print(f"Events: {len(g):,}")
        print(f"  Counterfactual gross profit (no promotion)      : {money(g['CF_GP_PRE'].sum())}")
        print(f"  Gross-profit lift, promoted products (pre cost) : {money(g['GROSS_GP_LIFT_PRE'].sum())}")
        print(f"  After-event weeks                               : {money(g['POST_GP_EFFECT'].sum())}")
        print(f"  Cannibalized other {UNIT:<29}: {money(-g['CANNIB_GP_LOSS'].sum())}")
        print(f"  NET impact, pre-promotion cost                  : {money(g['NET_GP_IMPACT_PRE'].sum())}"
              f"  ({t.loc['all', 'gp_change_%_pre']:+.1f}% vs. counterfactual)")
        print(f"  NET impact, recorded cost                       : {money(g['NET_GP_IMPACT_REC'].sum())}")
        print(f"     difference = estimated trade-deal funding    : {money(g['TRADE_FUNDING_EST'].sum())}")
        print(f"\n  Revenue given away as discounts                 : {money(g['DISCOUNT_COST'].sum())}")
        print(f"     of which on units that would have sold anyway: {money(g['SUBSIDISED_COST'].sum())} "
              f"({g['SUBSIDISED_COST'].sum() / g['DISCOUNT_COST'].sum() * 100:.1f}%)")
        print(f"  Gross profit per $1 of discount: pre cost {t.loc['all', 'roi_pre_cost']:+.2f} | "
              f"recorded cost {t.loc['all', 'roi_recorded']:+.2f}")
        rng = np.random.default_rng(SEED)
        n = len(g)
        boots = [g["NET_GP_IMPACT_PRE"].to_numpy()[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)]
        print(f"  90% bootstrap interval (events), net pre-cost: {money(np.quantile(boots, .05))} to "
              f"{money(np.quantile(boots, .95))}")
        print(f"\nEvents that INCREASED gross profit: pre cost {g['PROFITABLE_PRE'].mean() * 100:.1f}% | "
              f"recorded cost {g['PROFITABLE_REC'].mean() * 100:.1f}%")
        sure_loss = (g["NET_GP_IMPACT_PRE_HI90"] < 0).mean() * 100
        sure_gain = (g["NET_GP_IMPACT_PRE_LO90"] > 0).mean() * 100
        print(f"  with 90% confidence (pre cost): gain {sure_gain:.1f}% | loss {sure_loss:.1f}% | "
              f"uncertain {100 - sure_gain - sure_loss:.1f}%")
        plot_waterfall(g)

        section("2. BY DISCOUNT DEPTH")
        tbl = pooled_table(g, "DEPTH_BUCKET")
        print(tbl.round(2).to_string())
        plot_by_depth(tbl)
        section("3. BY PROMOTION KIND, BRAND, LENT, DURATION")
        g["LENT"] = g["LENT"].map({1.0: "Lent", 0.0: "not Lent", True: "Lent", False: "not Lent"})
        for by in ["PROMO_KIND", "BG", "LENT"]:
            print(f"\nBy {by}:")
            print(pooled_table(g, by).round(2).to_string())
        g["DUR"] = g["DURATION_WEEKS"].clip(upper=5).map(lambda v: str(v) if v < 5 else "5+")
        print("\nBy DURATION (weeks):")
        print(pooled_table(g, "DUR").round(2).to_string())
        plot_scatter(g)

        section("4. BEST AND WORST EVENTS (net gross-profit impact, pre-promotion cost)")
        cols = ["EVENT_ID", "DESCRIP", "START_DATE", "DURATION_WEEKS", "MEAN_DEPTH", "PROMO_KIND", "LIFT_PCT",
                "NET_GP_IMPACT_PRE", "NET_GP_IMPACT_REC", "GP_ROI_PRE"]
        print("Best 8:")
        print(g.nlargest(8, "NET_GP_IMPACT_PRE")[cols].round(2).to_string(index=False))
        print("\nWorst 8:")
        print(g.nsmallest(8, "NET_GP_IMPACT_PRE")[cols].round(2).to_string(index=False))

        section("5. KEY TAKEAWAYS (generated from the numbers above)")
        best = tbl["roi_pre_cost"].idxmax()
        worst = tbl["roi_pre_cost"].idxmin()
        print(f"  * Highest return per $ of discount (pre cost): depth {best} ({tbl.loc[best, 'roi_pre_cost']:+.2f}); "
              f"lowest: {worst} ({tbl.loc[worst, 'roi_pre_cost']:+.2f}).")
        print(f"  * {100 - g['PROFITABLE_PRE'].mean() * 100:.0f}% of events reduced gross profit if the retailer "
              "paid the whole discount.")
        print(f"  * {g['SUBSIDISED_COST'].sum() / g['DISCOUNT_COST'].sum() * 100:.0f}% of discount dollars went to "
              "units that would have sold anyway.")

        section("OUTPUTS")
        ev.to_csv(PROC / f"{CAT}_event_profit.csv", index=False)
        print(f"data/processed/{CAT}_event_profit.csv  ({len(ev):,} events, {ev.shape[1]} columns)")
        section("DONE")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
