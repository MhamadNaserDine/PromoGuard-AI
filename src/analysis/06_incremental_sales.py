"""
PromoGuard AI — Phase 6: Incremental Sales of Promotions
========================================================

Location in repo:  PromoGuard-AI/src/analysis/06_incremental_sales.py
Run from repo root:
    python src/analysis/06_incremental_sales.py

Inputs
Run from repo root:
    python src/analysis/06_incremental_sales.py              # canned tuna (default, --cat tna)
    python src/analysis/06_incremental_sales.py --cat sdr    # carbonated soft drinks
    (soft drinks: every file below starts with sdr_ instead of tna_, the model is
     models/sdr_baseline_model.joblib, the report is reports/phase6/phase6_sdr_incrementality_report.txt)

    data/processed/tna_tuna_panel.parquet, tna_products.csv, tna_promo_events.csv,
    tna_week_features.csv, data/reference/dominicks_stores.csv   (Phases 2-4)
    models/baseline_model.joblib                                  (Phase 5: model specification)

Outputs
    data/processed/tna_event_incrementality.csv     one row per promotion event
    data/processed/tna_counterfactual_rows.parquet  every product-store-week with actual and
                                                    counterfactual (no-promotion) units
    reports/phase6/phase6_incrementality_report.txt
    reports/phase6/figures/*.png

Definitions (per chain-level promotion EVENT e of product p, summed over all stores selling p)
    PROMO window : the event weeks                      [START, END], all stores selling p
                   (excluded: rows of products with < 4 weeks of sales history, whose
                    counterfactual cannot be estimated — e.g. launch promotions)
    POST window  : the 4 weeks after the event          [END+1, END+4]
    PRE window   : the 4 weeks before the event         [START-4, START-1]  (diagnostic only)
                   POST and PRE use CLEAN rows only (regular price, product not promoted in
                   >= 20% of other stores, enough history) — the same rows the baseline model is
                   built for. Rows where the price stays low after the event or part of the chain
                   still promotes are partial promotions, not clean post-promotion weeks; they are
                   counted and reported but not used to measure the dip.

    CF (counterfactual) = out-of-sample baseline prediction of units WITHOUT the event:
        price at the regular price, product not promoted anywhere in the chain, and the
        product's own promotion history frozen at the event start ("the event never happened").

    Gross lift          = Actual(PROMO) - CF(PROMO)
    Post-promotion dip  = Actual(POST)  - CF(POST)     (usually negative: borrowed future sales)
    Net incremental     = Gross lift + Post-promotion dip
    Lift %              = Actual / CF - 1

    These are ESTIMATED effects under the assumption that the baseline model captures what
    would have happened without the promotion (no unobserved demand shock that coincides
    with the event). Phase 7 adds cannibalization of OTHER products; Phase 8 adds profit.

Uncertainty
    Placebo windows (1-4 weeks, no promotion of the product nearby) give the distribution of
    the counterfactual error e = Actual / CF - 1 when the true effect is zero.
      * per event : 90% interval from the placebo quantiles for the same duration
      * pooled    : bootstrap over events (sampling error) + placebo mean (systematic bias)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import baseline_features as bf  # noqa: E402
from src.models import counterfactual as cfm  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
REF = REPO_ROOT / "data" / "reference"
MODEL_PATH = REPO_ROOT / "models" / "baseline_model.joblib"
REPORT_DIR = REPO_ROOT / "reports" / "phase6"
FIG_DIR = REPORT_DIR / "figures"

PRE_WEEKS, POST_WEEKS = 4, 4
MIN_EVENT_STORES = 20          # events promoted in fewer stores are too small to measure
N_PLACEBO_PER_DURATION = 300
N_BOOT = 1000
SEED = 42
MAJOR_BRANDS = ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's"]

CATEGORIES = {
    "tna": {"main": "tuna", "name": "canned tuna",
            "majors": ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's"]},
    "sdr": {"main": "csd", "name": "carbonated soft drinks",       # brand groups as in Phase 4
            "majors": ["Coca-Cola Co.", "PepsiCo", "7-Up", "Dominick's"]},
}
CAT, MAIN, CAT_NAME = "tna", "tuna", "canned tuna"     # set in main()

SURFACE, INK, INK_2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

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


# ----------------------------------------------------------------------------
# 1. Assign every row to an event phase
# ----------------------------------------------------------------------------
def assign_phases(d: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    PHASE: 'promo' (row's week is an event week of its product), 'post' (1-4 weeks after
    the product's most recent event and the product is not promoted in this store),
    'pre' (1-4 weeks before an event, not promo/post), or 'none'.
    EVENT_ID links promo/post/pre rows to their event; REL_WEEK = WEEK - event START.
    """
    promo_map, post_map, pre_map = {}, {}, {}
    for e in events.itertuples(index=False):
        for w in range(e.START_WEEK, e.END_WEEK + 1):
            promo_map[(e.PRODUCT_ID, w)] = e.EVENT_ID
        for k in range(1, POST_WEEKS + 1):
            post_map.setdefault((e.PRODUCT_ID, e.END_WEEK + k), e.EVENT_ID)
            post_map[(e.PRODUCT_ID, e.END_WEEK + k)] = e.EVENT_ID     # most recent event wins
        for k in range(1, PRE_WEEKS + 1):
            pre_map.setdefault((e.PRODUCT_ID, e.START_WEEK - k), e.EVENT_ID)

    keys = list(zip(d["PRODUCT_ID"], d["WEEK"]))
    ev_promo = np.array([promo_map.get(k, -1) for k in keys])
    ev_post = np.array([post_map.get(k, -1) for k in keys])
    ev_pre = np.array([pre_map.get(k, -1) for k in keys])

    phase = np.full(len(d), "none", dtype=object)
    event = np.full(len(d), -1)
    is_promo = ev_promo >= 0
    phase[is_promo], event[is_promo] = "promo", ev_promo[is_promo]
    is_post = ~is_promo & (ev_post >= 0)
    phase[is_post], event[is_post] = "post", ev_post[is_post]
    # a post row where this store promotes the product (local promotion) is not a clean post week
    phase[is_post & d["PROMO"].to_numpy()] = "post_excluded"
    is_pre = (phase == "none") & (ev_pre >= 0)
    phase[is_pre], event[is_pre] = "pre", ev_pre[is_pre]

    # PRE/POST must be clean regular weeks; PROMO rows need sales history for a counterfactual
    clean = cfm.clean_training_rows(d).to_numpy()
    phase[(phase == "post") & ~clean] = "post_excluded"
    phase[(phase == "pre") & ~clean] = "pre_excluded"
    no_hist = (d["N_REG_HIST"].lt(cfm.MIN_HISTORY) | d["BASE_LOG_13"].isna()).to_numpy()
    phase[(phase == "promo") & no_hist] = "promo_excluded"

    d = d.assign(PHASE=phase, EVENT_ID=event)
    start = events.set_index("EVENT_ID")["START_WEEK"]
    d["REL_WEEK"] = np.where(d["EVENT_ID"] >= 0, d["WEEK"] - d["EVENT_ID"].map(start).fillna(0), np.nan)
    return d


def counterfactual_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    Feature view 'the event never happened':
      promo rows : regular price, no chain promotion, own promotion history frozen at event start
      post rows  : own promotion history frozen at event start (the event did not create a
                   recent promotion), price/competition as observed
      other rows : unchanged
    """
    keep = list(dict.fromkeys(bf.CATEGORICAL + bf.NUMERIC + ["PHASE", "EVENT_ID", "STORE", "WEEK"]))
    cf = d[keep].copy()          # only the model inputs: same predictions, half the memory
    in_ev = cf["PHASE"].isin(["promo", "promo_excluded", "post", "post_excluded"])
    promo = cf["PHASE"].eq("promo")
    cf.loc[promo, "PRICE_INDEX"] = 1.0
    cf.loc[promo, "CHAIN_PROMO_SHARE"] = 0.0

    # promotion-history features at the first event week of each (event, store)
    first = (cf[promo].sort_values("WEEK").groupby(["EVENT_ID", "STORE"])
             [["WEEK", "WEEKS_SINCE_PROMO", "PROMOS_LAST_13"]].first()
             .rename(columns={"WEEK": "_W0", "WEEKS_SINCE_PROMO": "_WSP0", "PROMOS_LAST_13": "_PL0"}))
    cf = cf.join(first, on=["EVENT_ID", "STORE"])
    has0 = in_ev & cf["_W0"].notna()
    cf.loc[has0, "WEEKS_SINCE_PROMO"] = (cf.loc[has0, "_WSP0"] + cf.loc[has0, "WEEK"] - cf.loc[has0, "_W0"]) \
        .clip(upper=bf.WEEKS_SINCE_CAP)
    cf.loc[has0, "PROMOS_LAST_13"] = cf.loc[has0, "_PL0"]
    return cf.drop(columns=["_W0", "_WSP0", "_PL0"])


# ----------------------------------------------------------------------------
# 2. Placebo windows -> counterfactual error distribution
# ----------------------------------------------------------------------------
def placebo_errors(d: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    blocked = set()
    for e in events.itertuples(index=False):
        for w in range(e.START_WEEK - 6, e.END_WEEK + 7):
            blocked.add((e.PRODUCT_ID, w))
    base = d[cfm.clean_training_rows(d)]          # same row definition as the post window
    pw = base.groupby(["PRODUCT_ID", "WEEK"]).agg(A=("MOVE", "sum"), P=("PRED_ACTUAL", "sum"),
                                                  stores=("STORE", "nunique")).reset_index()
    pw = pw[pw["stores"].ge(min(30, max(1, d["STORE"].nunique() // 2)))]    # 30 on the full data
    ok = np.array([(p, w) not in blocked for p, w in zip(pw["PRODUCT_ID"], pw["WEEK"])])
    pw = pw[ok].set_index(["PRODUCT_ID", "WEEK"]).sort_index()
    starts = pw.index.to_list()
    available = set(starts)
    rows = []
    for dur in (1, 2, 3, 4):
        cand = [(p, w) for p, w in starts if all((p, w + k) in available for k in range(dur))]
        if not cand:
            continue
        pick = rng.choice(len(cand), min(N_PLACEBO_PER_DURATION, len(cand)), replace=False)
        for i in pick:
            p, w = cand[i]
            sub = pw.loc[[(p, w + k) for k in range(dur)]]
            rows.append({"DURATION": dur, "PRODUCT_ID": p, "START": w,
                         "A": sub["A"].sum(), "P": sub["P"].sum()})
    pl = pd.DataFrame(rows)
    pl["ERR"] = pl["A"] / pl["P"] - 1
    return pl


# ----------------------------------------------------------------------------
# 3. Event-level incrementality
# ----------------------------------------------------------------------------
def event_table(d: pd.DataFrame, events: pd.DataFrame, products: pd.DataFrame,
                placebo: pd.DataFrame) -> pd.DataFrame:
    def agg(phase: str, prefix: str) -> pd.DataFrame:
        s = d[d["PHASE"].eq(phase)].groupby("EVENT_ID").agg(
            **{f"{prefix}_ACTUAL": ("MOVE", "sum"), f"{prefix}_CF": ("PRED_CF", "sum"),
               f"{prefix}_ROWS": ("MOVE", "size")})
        return s

    ev = events.set_index("EVENT_ID").join(agg("promo", "PROMO")).join(agg("post", "POST")).join(agg("pre", "PRE"))
    ev = ev.join(d[d["PHASE"].eq("promo")].groupby("EVENT_ID").agg(
        PROMO_DOLLARS=("DOLLARS", "sum"), STORES_SELLING=("STORE", "nunique"),
        LENT=("LENT_WEEK", "max"), ALL_WEEKS_COVERED=("COVERAGE_OK", "min")))
    ev = ev.merge(products[["UPC", "DESCRIP", "BRAND", "SIZE"]], left_on="PRODUCT_ID", right_on="UPC", how="left") \
        .drop(columns="UPC").set_index(events.set_index("EVENT_ID").index)
    for c in ("POST_ACTUAL", "POST_CF", "PRE_ACTUAL", "PRE_CF"):
        ev[c] = ev[c].fillna(0.0)

    ev["GROSS_LIFT_UNITS"] = ev["PROMO_ACTUAL"] - ev["PROMO_CF"]
    ev["LIFT_PCT"] = (ev["PROMO_ACTUAL"] / ev["PROMO_CF"] - 1) * 100
    ev["POST_DIP_UNITS"] = ev["POST_ACTUAL"] - ev["POST_CF"]
    ev["NET_INCREMENTAL_UNITS"] = ev["GROSS_LIFT_UNITS"] + ev["POST_DIP_UNITS"]
    ev["NET_LIFT_PCT"] = ev["NET_INCREMENTAL_UNITS"] / ev["PROMO_CF"] * 100
    ev["PRE_ERROR_PCT"] = np.where(ev["PRE_CF"] > 0, (ev["PRE_ACTUAL"] / ev["PRE_CF"] - 1) * 100, np.nan)

    # 90% interval for each event from placebo errors of the same duration
    dur = ev["DURATION_WEEKS"].clip(upper=4)
    q = placebo.groupby("DURATION")["ERR"].quantile([0.05, 0.95]).unstack()
    e_lo, e_hi = dur.map(q[0.05]), dur.map(q[0.95])
    ev["LIFT_PCT_LO90"] = (ev["PROMO_ACTUAL"] / (ev["PROMO_CF"] * (1 + e_hi)) - 1) * 100
    ev["LIFT_PCT_HI90"] = (ev["PROMO_ACTUAL"] / (ev["PROMO_CF"] * (1 + e_lo)) - 1) * 100
    ev["LIFT_SIGNIFICANT"] = ev["LIFT_PCT_LO90"] > 0

    ev["PROMO_KIND"] = np.where(ev["SHARE_PRICE_PROMO"] >= 0.5, "price promotion", "deal code, no price cut")
    ev["DEPTH_BUCKET"] = pd.cut(ev["MEAN_DEPTH"], [-np.inf, 0.05, 0.10, 0.20, np.inf],
                                labels=["<5%", "5-10%", "10-20%", ">=20%"], right=False)
    ev["BRAND_GROUP"] = brand_group(ev["PRODUCT_ID"], products)
    min_stores = min(MIN_EVENT_STORES, max(1, d["STORE"].nunique() // 2))     # 20 on the full data
    ev["IN_SCOPE"] = ev["STORES"].ge(min_stores) & ev["ALL_WEEKS_COVERED"].astype(bool) & ev["PROMO_CF"].gt(0)
    return ev.reset_index()


def brand_group(product_ids: pd.Series, products: pd.DataFrame) -> np.ndarray:
    """Tuna: brand (4 majors + Other). Soft drinks: Coca-Cola Co. / PepsiCo (all their brands), 7-Up,
    Dominick's, Other — the same groups as Phase 4."""
    p = products.drop_duplicates("UPC").set_index("UPC")
    brand = product_ids.map(p["BRAND"])
    if CAT == "sdr":
        maker = product_ids.map(p["MAKER"])
        brand = pd.Series(np.select([maker.isin(["Coca-Cola Co.", "PepsiCo"]).to_numpy()], [maker.to_numpy()],
                                    default=brand.to_numpy()), index=product_ids.index)
    return np.where(brand.isin(MAJOR_BRANDS), brand, "Other")


def pooled(ev: pd.DataFrame, rng: np.random.Generator) -> dict:
    a, c = ev["PROMO_ACTUAL"].to_numpy(), ev["PROMO_CF"].to_numpy()
    pa, pc = ev["POST_ACTUAL"].to_numpy(), ev["POST_CF"].to_numpy()
    lift = a.sum() / c.sum() - 1
    net = (a.sum() - c.sum() + pa.sum() - pc.sum()) / c.sum()
    boot_l, boot_n = [], []
    n = len(ev)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        boot_l.append(a[i].sum() / c[i].sum() - 1)
        boot_n.append((a[i].sum() - c[i].sum() + pa[i].sum() - pc[i].sum()) / c[i].sum())
    return {"events": n, "actual": a.sum(), "cf": c.sum(), "post_dip": pa.sum() - pc.sum(),
            "lift": lift, "lift_lo": np.quantile(boot_l, 0.05), "lift_hi": np.quantile(boot_l, 0.95),
            "net": net, "net_lo": np.quantile(boot_n, 0.05), "net_hi": np.quantile(boot_n, 0.95)}


def concentration(ev: pd.DataFrame) -> None:
    """How concentrated are incremental units across events?"""
    net = ev["NET_INCREMENTAL_UNITS"].clip(lower=0).sort_values(ascending=False)
    total = net.sum()
    for share in (0.05, 0.10, 0.25):
        k = max(1, int(round(len(net) * share)))
        print(f"  Top {share:.0%} of events ({k}) produce {net.head(k).sum() / total * 100:.1f}% of all net incremental units")
    print(f"  Median event: lift {ev['LIFT_PCT'].median():+.1f}%  |  pooled (units-weighted) lift is dominated by a few")
    if CAT == "tna":
        print("  deep, Lent-time price cuts on the best-selling cans (see top events below).")
    else:
        print("  deep price cuts on the best-selling products (see top events below).")


def breakdown(ev: pd.DataFrame, by: str, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for key, g in ev.groupby(by, observed=True):
        if len(g) < 5:
            continue
        r = pooled(g, rng)
        rows.append({by: key, "events": r["events"],
                     "gross_lift_%": r["lift"] * 100, "lift_90ci": f"[{r['lift_lo'] * 100:.1f}, {r['lift_hi'] * 100:.1f}]",
                     "post_dip_%_of_cf": r["post_dip"] / r["cf"] * 100,
                     "net_lift_%": r["net"] * 100, "net_90ci": f"[{r['net_lo'] * 100:.1f}, {r['net_hi'] * 100:.1f}]",
                     "share_significant_%": g["LIFT_SIGNIFICANT"].mean() * 100})
    return pd.DataFrame(rows).set_index(by)


def tier_breakdown(d: pd.DataFrame, ev_ids: set, stores: pd.DataFrame) -> pd.DataFrame:
    x = d[d["EVENT_ID"].isin(ev_ids) & d["PHASE"].isin(["promo", "post"])] \
        .merge(stores[["STORE", "PRICE_TIER"]].rename(columns={"PRICE_TIER": "TIER"}), on="STORE", how="left")
    x["TIER"] = x["TIER"].fillna("unknown")
    g = x.groupby(["TIER", "PHASE"]).agg(A=("MOVE", "sum"), C=("PRED_CF", "sum")).unstack()
    out = pd.DataFrame({
        "gross_lift_%": (g[("A", "promo")] / g[("C", "promo")] - 1) * 100,
        "post_dip_%_of_cf": (g[("A", "post")] - g[("C", "post")]) / g[("C", "promo")] * 100,
    })
    out["net_lift_%"] = out["gross_lift_%"] + out["post_dip_%_of_cf"]
    depth = x[x["PHASE"].eq("promo")].groupby("TIER")["DEPTH"].mean() * 100
    out["mean_depth_%"] = depth
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_event_example(d: pd.DataFrame, ev: pd.DataFrame) -> None:
    cand = ev[ev["IN_SCOPE"] & ev["DURATION_WEEKS"].eq(2) & ev["BRAND_GROUP"].isin(MAJOR_BRANDS)
              & ev["POST_ROWS"].fillna(0).gt(0)]
    e = cand.loc[(cand["STORES"]).idxmax()] if len(cand) else ev[ev["IN_SCOPE"]].iloc[0]
    rows = d[d["PRODUCT_ID"].eq(e["PRODUCT_ID"]) & d["WEEK"].between(e["START_WEEK"] - PRE_WEEKS, e["END_WEEK"] + POST_WEEKS)]
    s = rows.groupby("WEEK").agg(actual=("MOVE", "sum"), cf=("PRED_CF", "sum"))
    rel = s.index - e["START_WEEK"]
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    style_axis(ax)
    in_ev = (rel >= 0) & (rel < e["DURATION_WEEKS"])
    ax.bar(rel, s["actual"], color=np.where(in_ev, ORANGE, GREY), width=0.7, label="Actual units")
    ax.plot(rel, s["cf"], color=BLUE, lw=2.2, marker="o", ms=4, label="Counterfactual (no promotion)")
    ax.set_xticks(rel)
    ax.set_xlabel("Weeks relative to promotion start (orange = promotion weeks)", color=INK_2, fontsize=9)
    ax.set_ylabel("Units, all stores", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK_2)
    fig.suptitle(f"{str(e['DESCRIP']).strip()} — event {int(e['EVENT_ID'])} ({e['START_DATE']}): "
                 f"lift {e['LIFT_PCT']:+.0f}%, net {e['NET_LIFT_PCT']:+.0f}%",
                 color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, "event_example_actual_vs_counterfactual.png")


def plot_lift_distribution(ev: pd.DataFrame, placebo: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.9), facecolor=SURFACE)
    style_axis(ax)
    bins = np.linspace(-60, 400, 47)
    ax.hist(placebo["ERR"].clip(-60, 400) * 100, bins=bins, color=GREY, alpha=0.8, density=True,
            label="Placebo windows (true lift = 0)")
    ax.hist(ev["LIFT_PCT"].clip(-60, 400), bins=bins, color=ORANGE, alpha=0.75, density=True,
            label="Real promotion events")
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_xlabel("Lift % = actual / counterfactual − 1 (clipped at 400%)", color=INK_2, fontsize=9)
    ax.set_ylabel("Density", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_2)
    ax.set_title("Real promotions vs. placebo: the effect is far outside the noise", color=INK, fontsize=11, loc="left")
    save(fig, "lift_distribution_vs_placebo.png")


def plot_by_depth(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.9), facecolor=SURFACE)
    style_axis(ax)
    x = np.arange(len(tbl))
    w = 0.38
    ax.bar(x - w / 2, tbl["gross_lift_%"], width=w, color=ORANGE, label="Gross lift (promotion weeks)")
    ax.bar(x + w / 2, tbl["net_lift_%"], width=w, color=BLUE, label="Net (after post-promotion dip)")
    for xi, g, n in zip(x, tbl["gross_lift_%"], tbl["net_lift_%"]):
        ax.text(xi - w / 2, g + 3, f"{g:.0f}%", ha="center", fontsize=8, color=INK)
        ax.text(xi + w / 2, n + 3, f"{n:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{i}\n({n} events)" for i, n in zip(tbl.index.astype(str), tbl["events"])])
    ax.set_xlabel("Average discount depth of the event", color=INK_2, fontsize=9)
    ax.set_ylabel("% of counterfactual units", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK_2)
    fig.suptitle("Incremental units by discount depth", color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, "lift_by_depth.png")


# ----------------------------------------------------------------------------
def main() -> None:
    global CAT, MAIN, CAT_NAME, MAJOR_BRANDS, REPORT_DIR, FIG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", choices=sorted(CATEGORIES))
    CAT = parser.parse_args().cat
    cfg = CATEGORIES[CAT]
    MAIN, CAT_NAME, MAJOR_BRANDS = cfg["main"], cfg["name"], cfg["majors"]
    bf.set_category(CAT)
    if CAT != "tna":
        FIG_DIR = REPORT_DIR / CAT / "figures"
    report = "phase6_incrementality_report.txt" if CAT == "tna" else f"phase6_{CAT}_incrementality_report.txt"
    model_path = MODEL_PATH if CAT == "tna" else REPO_ROOT / "models" / f"{CAT}_baseline_model.joblib"

    tee = Tee(REPORT_DIR / report)
    sys.stdout = tee
    rng = np.random.default_rng(SEED)
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        section(f"PROMOGUARD AI — PHASE 6 INCREMENTAL SALES — {CAT_NAME}")
        panel = pd.read_parquet(PROC / f"{CAT}_{MAIN}_panel.parquet")
        products = pd.read_csv(PROC / f"{CAT}_products.csv")
        events = pd.read_csv(PROC / f"{CAT}_promo_events.csv")
        stores = pd.read_csv(REF / "dominicks_stores.csv")
        weeks = pd.read_csv(PROC / f"{CAT}_week_features.csv")
        bundle = joblib.load(model_path)
        print(f"Baseline model specification: {bundle['model_name']} (Phase 5)")

        d = bf.build_features(panel, products, stores, weeks, bundle["train_end_week"])
        d = assign_phases(d, events)
        print("\nRows by phase:")
        ph = d.groupby("PHASE").agg(rows=("MOVE", "size"), units=("MOVE", "sum"))
        ph["units_%"] = (ph["units"] / ph["units"].sum() * 100).round(1)
        print(ph.to_string())
        excl = d["PHASE"].eq("promo_excluded")
        print(f"\nPromotion-week rows excluded (no sales history to build a counterfactual): "
              f"{int(excl.sum()):,} rows, {d.loc[excl, 'MOVE'].sum() / d.loc[d['PHASE'].str.startswith('promo'), 'MOVE'].sum() * 100:.1f}% of promotion-week units")
        pe = d["PHASE"].eq("post_excluded")
        print(f"Post-window rows that are partial promotions (price still low / chain still promoting): "
              f"{int(pe.sum()):,} — not used to measure the dip")
        cf_view = counterfactual_features(d)

        section("1. CROSS-FITTED COUNTERFACTUAL PREDICTIONS (one model per year, predicts the held-out year)")
        preds, diag = cfm.crossfit(d, bundle["model"], {"actual": d, "counterfactual": cf_view})
        d["PRED_ACTUAL"], d["PRED_CF"] = preds["actual"], preds["counterfactual"]
        print("\nOut-of-sample check on clean regular weeks of each held-out fold:")
        print(diag.round(4).to_string())

        section("2. PLACEBO WINDOWS (no promotion nearby; true lift = 0)")
        placebo = placebo_errors(d, events)
        summ = placebo.groupby("DURATION")["ERR"].describe(percentiles=[.05, .5, .95]).mul(100).round(1)
        summ["count"] = summ["count"] / 100
        print("Counterfactual error in % by window length (weeks):")
        print(summ.to_string())
        bias = placebo["A"].sum() / placebo["P"].sum() - 1
        print(f"\nPooled placebo bias: {bias * 100:+.2f}%  (systematic error of the counterfactual)")

        section("3. EVENT-LEVEL INCREMENTALITY")
        ev = event_table(d, events, products, placebo)
        scope = ev[ev["IN_SCOPE"]]
        print(f"Events: {len(ev):,} total | in scope: {len(scope):,} "
              f"(>= {MIN_EVENT_STORES} stores promoting, all weeks with good data coverage)")
        pre_err = (scope["PRE_ACTUAL"].sum() / scope["PRE_CF"].sum() - 1) * 100
        print(f"Pre-period check (4 weeks before events, should be ~0): {pre_err:+.1f}%")

        r = pooled(scope, rng)
        adj = (1 + r["lift"]) / (1 + bias) - 1
        print(f"\nPOOLED RESULTS ({r['events']} events)")
        print(f"  Actual units in promotion weeks : {r['actual']:,.0f}")
        print(f"  Counterfactual units             : {r['cf']:,.0f}")
        print(f"  Gross lift                       : {r['lift'] * 100:+.1f}%  (90% CI {r['lift_lo'] * 100:+.1f}% to {r['lift_hi'] * 100:+.1f}%)")
        print(f"     bias-adjusted (placebo)       : {adj * 100:+.1f}%")
        print(f"  Post-promotion dip (4 weeks)     : {r['post_dip']:,.0f} units = {r['post_dip'] / r['cf'] * 100:+.1f}% of counterfactual")
        print(f"     = {-r['post_dip'] / (r['actual'] - r['cf']) * 100:.1f}% of the gross lift is borrowed from later weeks")
        print(f"  Net incremental                  : {r['net'] * 100:+.1f}%  (90% CI {r['net_lo'] * 100:+.1f}% to {r['net_hi'] * 100:+.1f}%)")
        print("\n  Post-promotion dip: measured against the counterfactual on clean weeks only (the model's")
        print("  expected sales without the event). Phase 4 compared post weeks with PRE-event weeks, which")
        print("  also mixes in season, holidays, competitor promotions and price changes; this estimate")
        print("  removes those. Partial-promotion weeks after events (price still low) are excluded, so a")
        print("  small extra dip in those weeks cannot be ruled out.")
        print(f"\nEvents with a significant lift (90% interval above 0): {scope['LIFT_SIGNIFICANT'].mean() * 100:.1f}%")
        print("\nConcentration of the effect:")
        concentration(scope)
        print("Event-level lift % distribution:")
        print(scope["LIFT_PCT"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())

        section("4. BREAKDOWNS (pooled within group; 90% CI by bootstrap over events)")
        for by in ["PROMO_KIND", "DEPTH_BUCKET", "BRAND_GROUP", "DURATION_WEEKS", "LENT"]:
            sub = scope.copy()
            if by == "DURATION_WEEKS":
                sub[by] = sub[by].clip(upper=5).map(lambda v: f"{v}" if v < 5 else "5+")
            if by == "LENT":
                sub[by] = sub[by].map({1.0: "Lent", 0.0: "not Lent", True: "Lent", False: "not Lent"})
            tbl = breakdown(sub, by, rng)
            print(f"\nBy {by}:")
            print(tbl.round(1).to_string())
            if by == "DEPTH_BUCKET":
                plot_by_depth(tbl)
        print("\nBy store PRICE_TIER (store-level sums over in-scope events):")
        print(tier_breakdown(d, set(scope["EVENT_ID"]), stores).round(1).to_string())
        print("\nNote: season (Lent), promotion depth and brand are correlated with each other;")
        print("these breakdowns are descriptive. Phase 7 (causal) separates the effects.")

        section("5. TOP AND BOTTOM EVENTS (in scope, by net incremental units)")
        cols = ["EVENT_ID", "DESCRIP", "START_DATE", "DURATION_WEEKS", "STORES", "MEAN_DEPTH",
                "PROMO_KIND", "LIFT_PCT", "LIFT_PCT_LO90", "LIFT_PCT_HI90", "NET_INCREMENTAL_UNITS"]
        print("Top 8:")
        print(scope.nlargest(8, "NET_INCREMENTAL_UNITS")[cols].round(2).to_string(index=False))
        print("\nBottom 8:")
        print(scope.nsmallest(8, "NET_INCREMENTAL_UNITS")[cols].round(2).to_string(index=False))

        plot_event_example(d, ev)
        plot_lift_distribution(scope, placebo)

        section("OUTPUTS")
        ev.to_csv(PROC / f"{CAT}_event_incrementality.csv", index=False)
        keep = ["PRODUCT_ID", "STORE", "WEEK", "WEEK_START", "MOVE", "DOLLARS", "GROSS_PROFIT", "UNIT_PRICE",
                "REG_PRICE", "MARGIN_PCT", "DEPTH", "DEAL_CODE", "PROMO_TYPE", "PROMO", "PHASE", "EVENT_ID",
                "REL_WEEK", "LENT_WEEK", "COVERAGE_OK", "PRED_ACTUAL", "PRED_CF"]
        d[keep].to_parquet(PROC / f"{CAT}_counterfactual_rows.parquet", index=False)
        print(f"data/processed/{CAT}_event_incrementality.csv   ({len(ev):,} events)")
        print(f"data/processed/{CAT}_counterfactual_rows.parquet ({len(d):,} rows)")
        section("DONE")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
