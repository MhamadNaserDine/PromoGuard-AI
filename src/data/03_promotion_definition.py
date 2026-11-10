"""
PromoGuard AI — Phase 3: Promotion Definition
=============================================

Location in repo:  PromoGuard-AI/src/data/03_promotion_definition.py
Run from repo root:
    python src/data/03_promotion_definition.py             # canned tuna (default, --cat tna)
    python src/data/03_promotion_definition.py --cat sdr   # carbonated soft drinks

Inputs
    data/processed/<cat>_movement_clean.parquet   (Phase 2)
    data/processed/<cat>_products.csv             (Phase 2)

Outputs
    data/processed/<cat>_<main>_panel.parquet   PRODUCT_ID x STORE x WEEK panel with promotion labels
                                                (tna_tuna_panel.parquet / sdr_csd_panel.parquet)
    data/processed/<cat>_promo_events.csv       one row per chain-level promotion event
    reports/phase3/phase3_promotion_report.txt  (tuna) / phase3_<cat>_promotion_report.txt
    reports/phase3/figures/ (tuna) / reports/phase3/<cat>/figures/   diagnostic figures

Scope: the main sub-category of the category (tuna = 78% of tuna-file units; carbonated
soft drinks = 95.5% of in-scope soft-drink units). The definition itself is in
src/features/promotions.py and is identical for every category.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import promotions as pr  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports" / "phase3"
FIG_DIR = REPORT_DIR / "figures"                       # tuna; other categories use reports/phase3/<cat>/figures
CATEGORIES = {"tna": {"main": "tuna", "name": "canned tuna"},
              "sdr": {"main": "csd", "name": "carbonated soft drinks"}}
CAT_NAME = "canned tuna"

# Chart styling (light surface, recessive axes; series colours from a CVD-validated palette)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"

pd.set_option("display.width", 200)
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


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
def validate(p: pd.DataFrame) -> None:
    known = p[p["PROMO_TYPE"].ne("no_history")].copy()

    section("1. PROMOTION TYPES")
    print(f"Panel rows (PRODUCT_ID x STORE x WEEK): {len(p):,}")
    vc = p["PROMO_TYPE"].value_counts()
    print(pd.DataFrame({"rows": vc, "pct": (vc / len(p) * 100).round(1)}).to_string())
    print(f"\nPROMO (price_promo or deal_only): {known['PROMO'].mean() * 100:.1f}% of rows with history")

    section("2. DEAL CODE vs. OBSERVED PRICE CUT (>= 5% below regular)")
    ct = pd.crosstab(known["DEAL_CODE"], known["PRICE_CUT"], margins=True)
    print(ct.to_string())
    print("\nRow %:")
    print((pd.crosstab(known["DEAL_CODE"], known["PRICE_CUT"], normalize="index") * 100).round(1).to_string())
    uncoded_cuts = known["PRICE_CUT"] & known["DEAL_CODE"].eq("NONE")
    print(f"\nPrice cuts WITHOUT any deal code: {int(uncoded_cuts.sum()):,} "
          f"({uncoded_cuts.sum() / known['PRICE_CUT'].sum() * 100:.1f}% of all price cuts)")
    print(f"Price cuts that are temporary (back to regular within {pr.RETURN_WINDOW} weeks): "
          f"{known.loc[known['PRICE_CUT'], 'TEMPORARY'].mean() * 100:.1f}%")

    section("3. DESCRIPTIVE SALES RATIO BY TYPE (not causal)")
    print("SALES_RATIO = units / median units of the same product-store in 'regular' weeks.")
    print("This is a sanity check of the labels, NOT an estimate of promotion effect\n"
          "(no control for seasonality, trend, or other products).")
    base = known[known["PROMO_TYPE"].eq("regular")].groupby(["PRODUCT_ID", "STORE"])["MOVE"].median()
    known["SALES_RATIO"] = known["MOVE"] / pd.Series(
        list(zip(known["PRODUCT_ID"], known["STORE"]))).map(base).to_numpy()
    detail = np.select(
        [known["PROMO_TYPE"].eq("price_promo") & known["DEAL_CODE"].ne("NONE"),
         known["PROMO_TYPE"].eq("price_promo"),
         known["PROMO_TYPE"].eq("deal_only"),
         known["PROMO_TYPE"].eq("permanent_cut")],
        ["price_promo + deal code", "price_promo, no code", "deal_only (code, no cut)",
         "permanent_cut (price change)"], default="regular")
    known["DETAIL"] = detail
    print(known.groupby("DETAIL")["SALES_RATIO"].describe(percentiles=[.25, .5, .75]).round(2).to_string())

    section("4. DISCOUNT DEPTH vs. SALES RATIO (price_promo rows + regular rows)")
    sub = known[known["PROMO_TYPE"].isin(["price_promo", "regular"])].copy()
    bins = [-np.inf, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf]
    labels = ["<5% (regular)", "5-10%", "10-15%", "15-20%", "20-30%", ">=30%"]
    sub["DEPTH_BUCKET"] = pd.cut(sub["DEPTH"].where(sub["PROMO_TYPE"].eq("price_promo"), 0), bins=bins,
                                 labels=labels, right=False)
    depth_tbl = sub.groupby("DEPTH_BUCKET", observed=True).agg(
        rows=("SALES_RATIO", "size"), median_ratio=("SALES_RATIO", "median"),
        mean_margin_pct=("MARGIN_PCT", "mean")).round(2)
    print(depth_tbl.to_string())
    plot_depth_vs_ratio(depth_tbl)

    section("5. SENSITIVITY TO THE CUT THRESHOLD")
    for th in (0.05, 0.10, 0.15):
        cut = known["DEPTH"].ge(th)
        print(f"threshold {th:.0%}: {cut.mean() * 100:5.1f}% of rows are price cuts; "
              f"{(known['DEAL_CODE'].ne('NONE') & cut).sum() / max(cut.sum(), 1) * 100:5.1f}% of them carry a deal code")

    section("6. ARE PROMOTIONS CHAINWIDE? (share of stores on PROMO per product-week)")
    pw = known.groupby(["PRODUCT_ID", "WEEK"]).agg(
        promo_share=("PROMO", "mean"), code_share=("DEAL_CODE", lambda s: s.ne("NONE").mean()))
    bins_s = [-0.001, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(pd.DataFrame({
        "promo_share": pd.cut(pw["promo_share"], bins_s).value_counts(sort=False),
        "deal_code_share": pd.cut(pw["code_share"], bins_s).value_counts(sort=False),
    }).to_string())
    print("Interpretation: deal codes are almost all-or-nothing across stores (chain decision);")
    print("price cuts are less uniform because regular prices differ across the 16 pricing zones.")


def describe_events(events: pd.DataFrame, p: pd.DataFrame, products: pd.DataFrame) -> None:
    section("7. CHAIN-LEVEL PROMOTION EVENTS (>= 50% of stores on PROMO, consecutive weeks merged)")
    print(f"Events: {len(events):,} across {events['PRODUCT_ID'].nunique()} products")
    print("\nDuration (weeks):")
    print(events["DURATION_WEEKS"].value_counts().sort_index().head(12).to_string())
    print("\nEvent summary:")
    print(events[["DURATION_WEEKS", "STORES", "MEAN_DEPTH", "SHARE_PRICE_PROMO", "SHARE_DEAL_CODED"]]
          .describe().round(3).to_string())
    print("\nMain deal code of events:")
    print(events["MAIN_DEAL_CODE"].value_counts().to_string())
    in_event = p["EVENT_ID"].ge(0)
    print(f"\nPROMO rows inside a chain event: {in_event.sum() / p['PROMO'].sum() * 100:.1f}% "
          f"(the rest are store-level/local promotions)")

    top = (events.merge(products[["UPC", "DESCRIP", "BRAND"]], left_on="PRODUCT_ID", right_on="UPC", how="left")
           .groupby(["PRODUCT_ID", "DESCRIP", "BRAND"]).size().sort_values(ascending=False).head(10))
    print("\nProducts with the most events:")
    print(top.rename("events").to_string())


# ----------------------------------------------------------------------------
def plot_depth_vs_ratio(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    style_axis(ax)
    x = np.arange(len(tbl))
    colors = [GREY] + [BLUE] * (len(tbl) - 1)
    ax.bar(x, tbl["median_ratio"], width=0.6, color=colors)
    for xi, v in zip(x, tbl["median_ratio"]):
        ax.text(xi, v + 0.06, f"{v:.2f}x", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, tbl.index.astype(str))
    ax.set_ylabel("Median units vs. regular weeks", color=INK_2, fontsize=9)
    ax.set_xlabel("Discount depth (price below regular price)", color=INK_2, fontsize=9)
    ax.set_title(f"Deeper discounts, larger sales spikes (descriptive, {CAT_NAME})",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_depth_vs_sales_ratio.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_example(p: pd.DataFrame, products: pd.DataFrame) -> None:
    """Price, regular price and promotions for the best-selling product in one store."""
    top_pid = p.groupby("PRODUCT_ID")["MOVE"].sum().idxmax()
    store = p[p["PRODUCT_ID"].eq(top_pid)].groupby("STORE")["WEEK"].nunique().idxmax()
    s = p[p["PRODUCT_ID"].eq(top_pid) & p["STORE"].eq(store)].sort_values("WEEK")
    s = s[s["WEEK"].between(100, 200)]
    name = products.loc[products["UPC"].eq(top_pid), "DESCRIP"].iloc[0].strip()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1, 1]})
    for ax in (a1, a2):
        style_axis(ax)
    a1.step(s["WEEK_START"], s["REG_PRICE"], where="mid", color=GREY, lw=2, ls="--", label="Regular price (modal)")
    a1.plot(s["WEEK_START"], s["UNIT_PRICE"], color=BLUE, lw=2, label="Actual price")
    promo = s[s["PROMO_TYPE"].eq("price_promo")]
    deal = s[s["PROMO_TYPE"].eq("deal_only")]
    perm = s[s["PROMO_TYPE"].eq("permanent_cut")]
    a1.scatter(promo["WEEK_START"], promo["UNIT_PRICE"], s=36, color=ORANGE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Price promotion")
    if len(perm):
        a1.scatter(perm["WEEK_START"], perm["UNIT_PRICE"], s=36, facecolor=SURFACE, edgecolor=GREY,
                   linewidth=1.5, zorder=3, label="Permanent price cut (not a promotion)")
    a1.set_ylabel("Unit price ($)", color=INK_2, fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, labelcolor=INK_2)
    fig.suptitle(f"{name} — store {store}: price, regular price and promotions",
                 color=INK, fontsize=11, x=0.01, ha="left")

    a2.bar(s["WEEK_START"], s["MOVE"], width=5, color=GREY)
    a2.bar(promo["WEEK_START"], promo["MOVE"], width=5, color=ORANGE, label="Price promotion week")
    a2.bar(deal["WEEK_START"], deal["MOVE"], width=5, color=BLUE, label="Deal code, no price cut")
    a2.set_ylabel("Units sold", color=INK_2, fontsize=9)
    a2.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_price_promo_example.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> None:
    global FIG_DIR, CAT_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", choices=sorted(CATEGORIES))
    cat = parser.parse_args().cat
    main_sub, CAT_NAME = CATEGORIES[cat]["main"], CATEGORIES[cat]["name"]
    if cat != "tna":
        FIG_DIR = REPORT_DIR / cat / "figures"
    report = "phase3_promotion_report.txt" if cat == "tna" else f"phase3_{cat}_promotion_report.txt"
    tee = Tee(REPORT_DIR / report)
    sys.stdout = tee
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        section(f"PROMOGUARD AI — PHASE 3 PROMOTION DEFINITION — {CAT_NAME}")
        print(f"Regular price : modal unit price of previous {pr.REG_WINDOW} observed weeks "
              f"(min {pr.REG_MIN_OBS}), same product & store")
        print(f"Price cut     : unit price >= {pr.CUT_THRESHOLD:.0%} below regular price")
        print(f"Temporary     : price back to regular within {pr.RETURN_WINDOW} observed weeks "
              "(labelling only — never a model feature)")

        mv = pd.read_parquet(PROC / f"{cat}_movement_clean.parquet")
        products = pd.read_csv(PROC / f"{cat}_products.csv")
        sub = mv[mv["SUBCATEGORY"].eq(main_sub)]
        del mv
        print(f"\n{main_sub} rows (UPC x store x week): {len(sub):,}")

        panel = pr.aggregate_to_product(sub)
        del sub
        print(f"After merging barcode versions (PRODUCT_ID x store x week): {len(panel):,} "
              f"({int(panel['N_UPC'].gt(1).sum()):,} rows combine 2+ UPCs)")

        panel = pr.label_promotions(panel)
        validate(panel)
        events, panel = pr.build_events(panel)
        describe_events(events, panel, products)
        plot_example(panel, products)

        section("OUTPUTS")
        out_panel = PROC / f"{cat}_{main_sub}_panel.parquet"
        out_events = PROC / f"{cat}_promo_events.csv"
        panel.to_parquet(out_panel, index=False)
        events.to_csv(out_events, index=False)
        print(f"{out_panel.relative_to(REPO_ROOT)}  ({len(panel):,} rows x {panel.shape[1]} cols)")
        print(f"{out_events.relative_to(REPO_ROOT)}  ({len(events):,} events)")
        for f in sorted(FIG_DIR.glob("*.png")):
            print(f.relative_to(REPO_ROOT))
        print("\nPanel columns:", list(panel.columns))
        section("DONE")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
