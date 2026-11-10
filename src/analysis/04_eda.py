"""
PromoGuard AI — Phase 4: Exploratory Data Analysis (canned tuna, soft drinks)
=============================================================================

Location in repo:  PromoGuard-AI/src/analysis/04_eda.py
Run from repo root:
    python src/analysis/04_eda.py              # canned tuna (default, --cat tna)
    python src/analysis/04_eda.py --cat sdr    # carbonated soft drinks

Inputs (shown for tuna; soft drinks use sdr_csd_panel.parquet, sdr_products.csv, sdr_promo_events.csv)
    data/processed/tna_tuna_panel.parquet     (Phase 3)
    data/processed/tna_products.csv           (Phase 2)
    data/processed/tna_promo_events.csv       (Phase 3)
    data/reference/dominicks_weeks.csv        (week calendar, from the manual)
    data/reference/dominicks_stores.csv       (store -> zone / price tier, from the manual)

Outputs
    reports/phase4/phase4_eda_report.txt          (soft drinks: phase4_sdr_eda_report.txt)
    reports/phase4/figures/*.png                  (soft drinks: reports/phase4/sdr/figures/)
    data/processed/<cat>_week_features.csv        week-level calendar features (Lent etc.) for modelling

All results in this phase are DESCRIPTIVE. Nothing here is a causal estimate.
Questions answered
    Q1  How did category demand evolve over time?
    Q2  Is there a Lent effect on tuna demand, and how large?
        (soft drinks: which holidays and seasons move demand?)
    Q3  How did brand shares evolve?
    Q4  Do competing brands promote in the same weeks?
    Q5  Do sales dip after a promotion (sign of stockpiling / pull-forward)?
    Q6  How do stores differ by price tier?
    Q7  First look at substitution between the major brands
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dateutil.easter import easter  # noqa: E402  (installed with pandas)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC = REPO_ROOT / "data" / "processed"
REF = REPO_ROOT / "data" / "reference"
REPORT_DIR = REPO_ROOT / "reports" / "phase4"
FIG_DIR = REPORT_DIR / "figures"

MAJOR_BRANDS = ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's"]
MIN_STORE_COVERAGE = 0.5   # a week is usable if at least 50% of the usual number of stores report sales

# Category settings. GROUP_COL is the brand grouping used in Q3/Q4/Q7.
# Soft drinks: brands are grouped by maker for Coke and Pepsi (all Coca-Cola Co. / PepsiCo trademarks),
# plus the biggest other national brand (7-Up) and the store brand.
CATEGORIES = {
    "tna": {"main": "tuna", "name": "canned tuna", "unit": "tuna",
            "majors": ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's"]},
    "sdr": {"main": "csd", "name": "carbonated soft drinks", "unit": "soft-drink",
            "majors": ["Coca-Cola Co.", "PepsiCo", "7-Up", "Dominick's"]},
}
CAT, MAIN, CAT_NAME, UNIT = "tna", "tuna", "canned tuna", "tuna"     # set in main()

# Chart styling — light surface, recessive axes, CVD-validated categorical order
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
GREY = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BRAND_COLOR = dict(zip(MAJOR_BRANDS + ["Other"], SERIES[:4] + [GREY]))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", NEUTRAL, "#2a78d6"])

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ----------------------------------------------------------------------------
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
    print(f"  [figure] {(FIG_DIR / name).relative_to(REPO_ROOT)}")


def ols(y: np.ndarray, X: pd.DataFrame, hac_lags: int = 4) -> pd.DataFrame:
    """OLS with Newey-West (HAC) standard errors — weekly data are autocorrelated."""
    Xv = X.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    resid = y - Xv @ beta
    n, k = Xv.shape
    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    u = Xv * resid[:, None]
    S = u.T @ u
    for lag in range(1, hac_lags + 1):
        w = 1 - lag / (hac_lags + 1)
        g = u[lag:].T @ u[:-lag]
        S += w * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ S @ xtx_inv) * n / (n - k))
    r2 = 1 - resid.var() / y.var()
    out = pd.DataFrame({"coef": beta, "se_hac": se, "t": beta / se}, index=X.columns)
    out.attrs["r2"], out.attrs["n"] = r2, n
    return out


# ----------------------------------------------------------------------------
# Calendar features
# ----------------------------------------------------------------------------
def build_week_features() -> pd.DataFrame:
    """
    Week-level calendar features. Dominick's weeks run Thursday -> Wednesday,
    so every week contains exactly one Friday (WEEK_START + 1 day).
    LENT_FRIDAY = that Friday falls between Ash Wednesday and Holy Saturday.
    LENT_WEEK   = the week overlaps Lent at all (Ash Wednesday .. Holy Saturday). This also
                  catches the Thu-Wed week that ENDS on Ash Wednesday, when shoppers stock up.
    LENT_DAYS   = number of Lent days in the week (0-7).
    Easter dates are computed (dateutil), not taken from the data.
    """
    w = pd.read_csv(REF / "dominicks_weeks.csv", parse_dates=["START", "END"])
    friday = w["START"] + pd.Timedelta(days=1)
    lent = np.zeros(len(w), dtype=bool)
    lent_days = np.zeros(len(w), dtype=int)
    weeks_to_easter = np.full(len(w), np.nan)
    for i, (start, f) in enumerate(zip(w["START"], friday)):
        e = pd.Timestamp(easter(f.year))
        ash_wed = e - pd.Timedelta(days=46)
        holy_sat = e - pd.Timedelta(days=1)
        lent[i] = ash_wed <= f <= holy_sat
        days = pd.date_range(start, start + pd.Timedelta(days=6))
        lent_days[i] = int(((days >= ash_wed) & (days <= holy_sat)).sum())
        weeks_to_easter[i] = (e - f).days // 7
    w["FRIDAY"] = friday
    w["LENT_FRIDAY"] = lent
    w["LENT_DAYS"] = lent_days
    w["LENT_WEEK"] = lent_days > 0
    w["WEEKS_TO_EASTER"] = weeks_to_easter
    w["YEAR"] = w["START"].dt.year
    w["MONTH"] = w["START"].dt.month
    w["WEEK_OF_YEAR"] = w["START"].dt.isocalendar().week.astype(int)
    return w.rename(columns={"START": "WEEK_START", "END": "WEEK_END", "EVENT": "HOLIDAY"})


# ----------------------------------------------------------------------------
# Q1 + Q2: category demand over time and Lent
# ----------------------------------------------------------------------------
def coverage_check(p: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Flag weeks where too few stores report tuna sales (data gap), and add COVERAGE_OK."""
    section("DATA COVERAGE BY WEEK")
    n_stores = p.groupby("WEEK")["STORE"].nunique()
    usual = n_stores.median()
    weeks = weeks.copy()
    col = f"{MAIN.upper()}_STORES"
    weeks[col] = weeks["WEEK"].map(n_stores).fillna(0).astype(int)
    weeks["COVERAGE_OK"] = weeks[col] >= MIN_STORE_COVERAGE * usual
    bad = weeks.loc[~weeks["COVERAGE_OK"], "WEEK"].tolist()
    print(f"Usual number of stores reporting {UNIT} sales: {usual:.0f}")
    print(f"Weeks below {MIN_STORE_COVERAGE:.0%} coverage (no data or a handful of stores): {len(bad)}")
    print(f"  {bad}")
    bad_s = pd.Series(bad)
    runs = bad_s.groupby(bad_s.diff().ne(1).cumsum()).agg(["min", "max", "size"])
    longest = runs.loc[runs["size"].idxmax()]
    w = weeks.set_index("WEEK")
    print(f"Longest gap: weeks {longest['min']}-{longest['max']} ({longest['size']} weeks, "
          f"{w.loc[longest['min'], 'WEEK_START'].date()} to {w.loc[longest['max'], 'WEEK_END'].date()})")
    print("-> These weeks are excluded from weekly analyses here and must be excluded from model training.")
    return weeks


def category_weekly(p: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    p = p[p["WEEK"].isin(weeks.loc[weeks["COVERAGE_OK"], "WEEK"])]
    wk = p.groupby("WEEK").agg(UNITS=("MOVE", "sum"), DOLLARS=("DOLLARS", "sum"),
                               STORES=("STORE", "nunique"),
                               PROMO_SHARE=("PROMO", "mean")).reset_index()
    wk["UNITS_PER_STORE"] = wk["UNITS"] / wk["STORES"]
    return wk.merge(weeks, on="WEEK", how="left")


def q1_trend(wk: pd.DataFrame) -> None:
    section(f"Q1. CATEGORY DEMAND OVER TIME ({UNIT} units per active store per week)")
    yr = wk.groupby("YEAR").agg(weeks=("WEEK", "size"), units_per_store=("UNITS_PER_STORE", "mean"),
                                promo_share=("PROMO_SHARE", "mean"), stores=("STORES", "mean")).round(2)
    print(yr.to_string())
    print("\nNote: 1989 and 1997 are partial years; low-coverage weeks are excluded (see DATA COVERAGE).")

    fig, ax = plt.subplots(figsize=(11, 4), facecolor=SURFACE)
    style_axis(ax)
    for _, r in wk[wk["LENT_WEEK"]].iterrows():
        ax.axvspan(r["WEEK_START"], r["WEEK_START"] + pd.Timedelta(days=7), color="#cde2fb", lw=0)
    line = wk.set_index("WEEK")[["WEEK_START", "UNITS_PER_STORE"]].reindex(
        range(wk["WEEK"].min(), wk["WEEK"].max() + 1))          # NaN rows break the line at data gaps
    ax.plot(line["WEEK_START"].ffill(), line["UNITS_PER_STORE"], color=SERIES[0], lw=1.5)
    ax.set_ylabel(f"{UNIT.capitalize()} units per store per week", color=INK_2, fontsize=9)
    ax.set_title(f"{CAT_NAME.capitalize()} demand, 1989–1997 (shaded: Lent; gaps = missing data)",
                 color=INK, fontsize=11, loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "q1_category_demand_lent.png")


def _lent_regression(wk: pd.DataFrame, y: np.ndarray, with_promo: bool) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0, "LENT_WEEK": wk["LENT_WEEK"].astype(float)}, index=wk.index)
    if with_promo:
        X["PROMO_SHARE"] = wk["PROMO_SHARE"]
    for h in sorted(wk["HOLIDAY"].dropna().unique()):
        if h != "Easter":                       # Easter week is the last Lent week
            X[f"HOL_{h}"] = wk["HOLIDAY"].eq(h).astype(float)
    X = X.join(pd.get_dummies(wk["MONTH"], prefix="M", drop_first=True, dtype=float))
    X = X.join(pd.get_dummies(wk["YEAR"], prefix="Y", drop_first=True, dtype=float))
    return ols(y, X)


def _fmt_effect(res: pd.DataFrame, name: str = "LENT_WEEK") -> str:
    b, se = res.loc[name, "coef"], res.loc[name, "se_hac"]
    pct = lambda v: (np.exp(v) - 1) * 100
    return (f"{pct(b):+.1f}%  (95% CI {pct(b - 1.96 * se):+.1f}% to {pct(b + 1.96 * se):+.1f}%; "
            f"t = {b / se:.2f}; n = {res.attrs['n']} weeks; R2 = {res.attrs['r2']:.2f})")


def q2_lent(wk: pd.DataFrame, p: pd.DataFrame) -> None:
    section("Q2. LENT EFFECT ON TUNA DEMAND")
    feb_apr = wk[wk["MONTH"].between(2, 4)]
    comp = feb_apr.groupby("LENT_WEEK")[["UNITS_PER_STORE", "PROMO_SHARE"]].mean().round(3)
    print("February–April weeks only (raw means):")
    print(comp.to_string())
    print("-> The chain promotes tuna much more often during Lent. Lent and promotion are entangled.")

    print("\nWeekly regressions, month FE + year FE + holiday dummies, Newey-West SE (4 lags).")
    print("Identification comes from Easter moving between late March and late April across years.")
    y_cat = np.log(wk["UNITS_PER_STORE"].to_numpy())
    print("\n(a) Total tuna units per store")
    print("    without promotion control :", _fmt_effect(_lent_regression(wk, y_cat, with_promo=False)))
    print("    with promotion share      :", _fmt_effect(_lent_regression(wk, y_cat, with_promo=True)))

    r = p[p["PROMO_TYPE"].eq("regular") & p["WEEK"].isin(wk["WEEK"])].copy()
    r["LOG_UNITS"] = np.log(r["MOVE"])
    r["LOG_DEV"] = r["LOG_UNITS"] - r.groupby(["PRODUCT_ID", "STORE"])["LOG_UNITS"].transform("mean")
    reg = wk.merge(r.groupby("WEEK")["LOG_DEV"].mean().rename("Y_REG").reset_index(), on="WEEK")
    print("\n(b) Products sold at their REGULAR price only (log units vs. own product-store average)")
    print("    with promotion share      :", _fmt_effect(_lent_regression(reg, reg["Y_REG"].to_numpy(), True)))
    print("\nReading:")
    print(" - Total tuna demand is clearly higher in weeks overlapping Lent, even after holding the")
    print("   share of promoted products fixed (the interval is wide: only ~8 Lent seasons in the data).")
    print(" - Products sold at their REGULAR price do NOT rise in Lent (if anything they fall): the extra")
    print("   Lent demand flows to promoted items, which the chain concentrates in Lent.")
    print(" - Consequences: (1) the baseline/counterfactual model must include Lent features;")
    print("   (2) Lent is a CONFOUNDER for promotion effects - comparing promoted vs. non-promoted")
    print("   weeks without adjusting for Lent would overstate promotion lift.")

    prof = wk[wk["WEEKS_TO_EASTER"].between(-4, 12)].groupby("WEEKS_TO_EASTER")[["UNITS_PER_STORE", "PROMO_SHARE"]].mean()
    base = wk[~wk["LENT_WEEK"] & wk["MONTH"].between(1, 5)]["UNITS_PER_STORE"].mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, facecolor=SURFACE)
    for ax in (a1, a2):
        style_axis(ax)
    x = -prof.index.to_numpy()
    in_lent = (prof.index >= 0) & (prof.index <= 7)
    a1.bar(x, prof["UNITS_PER_STORE"].to_numpy() / base, color=np.where(in_lent, SERIES[0], GREY), width=0.7)
    a1.axhline(1, color=INK_2, lw=1, ls="--")
    a1.set_ylabel("Units/store vs. Jan–May\nnon-Lent average", color=INK_2, fontsize=9)
    a2.bar(x, prof["PROMO_SHARE"].to_numpy() * 100, color=np.where(in_lent, SERIES[1], GREY), width=0.7)
    a2.set_ylabel("% of product-store\nweeks on promotion", color=INK_2, fontsize=9)
    a2.set_xlabel("Weeks relative to Easter (0 = Easter week; coloured = weeks overlapping Lent; "
                  "-7 = week ending on Ash Wednesday)", color=INK_2, fontsize=9)
    a2.set_xticks(x)
    fig.suptitle("Around Easter: tuna demand and promotion intensity", color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, "q2_lent_profile.png")


# ----------------------------------------------------------------------------
# Q3: brand shares
# ----------------------------------------------------------------------------
def q3_brands(p: pd.DataFrame, weeks: pd.DataFrame) -> None:
    section(f"Q3. BRAND SHARES BY YEAR (% of {UNIT} units)")
    d = p.merge(weeks[["WEEK", "YEAR"]], on="WEEK")
    d["BRAND_GROUP"] = np.where(d["BRAND"].isin(MAJOR_BRANDS), d["BRAND"], "Other")
    sh = d.pivot_table(index="YEAR", columns="BRAND_GROUP", values="MOVE", aggfunc="sum")
    sh = sh.div(sh.sum(axis=1), axis=0).mul(100)[MAJOR_BRANDS + ["Other"]]
    print(sh.round(1).to_string())

    promo = d.groupby("BRAND_GROUP").agg(promo_week_share=("PROMO", "mean"),
                                         units_on_promo=("MOVE", lambda s: 0))
    promo["units_on_promo"] = d[d["PROMO"]].groupby("BRAND_GROUP")["MOVE"].sum() / d.groupby("BRAND_GROUP")["MOVE"].sum()
    print("\nShare of weeks on promotion and share of units sold on promotion, by brand:")
    print(promo.mul(100).round(1).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    style_axis(ax)
    for b in MAJOR_BRANDS + ["Other"]:
        ax.plot(sh.index, sh[b], color=BRAND_COLOR[b], lw=2, marker="o", ms=5)
        ax.text(sh.index[-1] + 0.15, sh[b].iloc[-1], b, color=INK_2, fontsize=8, va="center")
    ax.set_ylabel(f"Share of {UNIT} units (%)", color=INK_2, fontsize=9)
    ax.set_xlim(sh.index[0] - 0.3, sh.index[-1] + 2.2)
    ax.set_title(f"{CAT_NAME.capitalize()} brand shares by year", color=INK, fontsize=11, loc="left")
    save(fig, "q3_brand_shares.png")


# ----------------------------------------------------------------------------
# Q4: promotion calendar and overlap
# ----------------------------------------------------------------------------
def q4_calendar(events: pd.DataFrame, products: pd.DataFrame, weeks: pd.DataFrame) -> None:
    section("Q4. PROMOTION CALENDAR — DO COMPETING BRANDS PROMOTE IN THE SAME WEEKS?")
    ev = events.merge(products[["UPC", "BRAND"]], left_on="PRODUCT_ID", right_on="UPC", how="left")
    ev = ev[ev["BRAND"].isin(MAJOR_BRANDS)]
    rows = [(b, w) for b, s, e in ev[["BRAND", "START_WEEK", "END_WEEK"]].itertuples(index=False)
            for w in range(s, e + 1)]
    bw = pd.DataFrame(rows, columns=["BRAND", "WEEK"]).drop_duplicates()
    all_weeks = weeks["WEEK"].to_numpy()
    on = pd.DataFrame(0, index=all_weeks, columns=MAJOR_BRANDS)
    for b, w in bw.itertuples(index=False):
        on.loc[w, b] = 1
    observed_weeks = sorted(set(all_weeks) & set(range(events["START_WEEK"].min(), events["END_WEEK"].max() + 1)))
    on = on.loc[observed_weeks]

    rate = on.mean()
    print("Share of weeks each brand has at least one chain promotion event running:")
    print(rate.mul(100).round(1).to_string())

    n_on = on.sum(axis=1)
    sim = np.random.default_rng(0).random((20_000, len(MAJOR_BRANDS))) < rate.to_numpy()
    exp_dist = pd.Series(sim.sum(axis=1)).value_counts(normalize=True).sort_index()
    obs_dist = n_on.value_counts(normalize=True).sort_index()
    print("\nNumber of major brands promoting in the same week — observed vs. expected if independent (%):")
    print(pd.DataFrame({"observed": obs_dist, "expected_if_independent": exp_dist}).fillna(0).mul(100).round(1).to_string())

    print("\nPairwise co-promotion: observed share of weeks both promote / expected if independent:")
    pair = pd.DataFrame(index=MAJOR_BRANDS, columns=MAJOR_BRANDS, dtype=float)
    for a in MAJOR_BRANDS:
        for b in MAJOR_BRANDS:
            if a != b:
                pair.loc[a, b] = (on[a] & on[b]).mean() / (rate[a] * rate[b])
    print(pair.round(2).to_string())
    print("(> 1: promote together more often than chance; < 1: they alternate)")

    wk_dates = weeks.set_index("WEEK")["WEEK_START"]
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=SURFACE)
    style_axis(ax, grid_axis="")
    for i, b in enumerate(MAJOR_BRANDS):
        wks = on.index[on[b].eq(1)]
        ax.vlines(wk_dates.loc[wks], i - 0.38, i + 0.38, color=BRAND_COLOR[b], lw=1.6)
    ax.set_yticks(range(len(MAJOR_BRANDS)), MAJOR_BRANDS)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Chain promotion weeks by brand", color=INK, fontsize=11, loc="left")
    save(fig, "q4_promotion_calendar.png")


# ----------------------------------------------------------------------------
# Q5: post-promotion dip
# ----------------------------------------------------------------------------
def q5_post_promo(p: pd.DataFrame, events: pd.DataFrame) -> None:
    section("Q5. WHAT HAPPENS AFTER A PROMOTION? (event study, descriptive)")
    ev = events[events["DURATION_WEEKS"].le(2)][["EVENT_ID", "PRODUCT_ID", "START_WEEK", "END_WEEK"]]
    print(f"Events used: {len(ev):,} chain events lasting 1–2 weeks (the most common, cleanest windows)")
    # Build the (event, week) windows first, then join the panel: same rows as joining every panel row
    # to every event of its product and keeping weeks -8..+8, but far less memory for large categories.
    win = ev.loc[ev.index.repeat(17)].copy()
    win["REL"] = np.tile(np.arange(-8, 9), len(ev))
    win["WEEK"] = win["START_WEEK"] + win["REL"]
    d = p[["PRODUCT_ID", "STORE", "WEEK", "MOVE", "PROMO_TYPE"]].merge(win, on=["PRODUCT_ID", "WEEK"])

    pre = d[d["REL"].between(-8, -3) & d["PROMO_TYPE"].eq("regular")]
    base = pre.groupby(["EVENT_ID", "STORE"])["MOVE"].median().rename("BASE")
    d = d.join(base, on=["EVENT_ID", "STORE"]).dropna(subset=["BASE"])
    d = d[d["BASE"] > 0]
    d["RATIO"] = d["MOVE"] / d["BASE"]
    keep = (d["WEEK"].between(d["START_WEEK"], d["END_WEEK"])) | d["PROMO_TYPE"].eq("regular")
    d = d[keep]
    d["LOG_RATIO"] = np.log(d["RATIO"])
    prof = d.groupby("REL").agg(geo_mean=("LOG_RATIO", lambda v: np.exp(v.mean())),
                                median=("RATIO", "median"), obs=("RATIO", "size"))
    print("RATIO = units / median units in regular weeks -8..-3 of the same store (1.0 = normal).")
    print("geo_mean = geometric mean of RATIO (robust to the huge spikes, not stuck at integer medians).")
    print("Post-event weeks that were themselves promotions are excluded.")
    print(prof.round(3).to_string())
    pre = prof.loc[-8:-3, "geo_mean"].mean()
    post = prof.loc[2:8, "geo_mean"].mean()
    print(f"\nGeo-mean ratio, pre weeks -8..-3: {pre:.3f} | post weeks +2..+8: {post:.3f} "
          f"-> post-promotion level {(post / pre - 1) * 100:+.1f}% vs. pre-promotion")
    print("A dip after promotions is consistent with stockpiling / purchase acceleration: part of the")
    print("promotion spike is sales borrowed from future weeks. The incremental-sales phase must count it.")

    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style_axis(ax)
    colors = [SERIES[1] if 0 <= r <= 1 else SERIES[0] for r in prof.index]
    ax.bar(prof.index, prof["geo_mean"], color=colors, width=0.7)
    ax.axhline(1, color=INK_2, lw=1, ls="--")
    ax.set_xticks(prof.index)
    ax.set_xlabel("Weeks relative to promotion start (orange = promotion weeks 0–1)", color=INK_2, fontsize=9)
    ax.set_ylabel("Units vs. pre-promotion normal (geo. mean)", color=INK_2, fontsize=9)
    ax.set_title("Sales before, during and after 1–2 week promotions", color=INK, fontsize=11, loc="left")
    save(fig, "q5_post_promotion.png")


# ----------------------------------------------------------------------------
# Q6: stores by price tier
# ----------------------------------------------------------------------------
def q6_stores(p: pd.DataFrame, stores: pd.DataFrame) -> None:
    section("Q6. STORES BY PRICE TIER")
    d = p.merge(stores[["STORE", "PRICE_TIER", "ZONE"]], on="STORE", how="left")
    d["PRICE_TIER"] = d["PRICE_TIER"].fillna("unknown (opened after 1992)")
    top = p.groupby("PRODUCT_ID")["MOVE"].sum().idxmax()
    out = d.groupby("PRICE_TIER").agg(
        stores=("STORE", "nunique"),
        units_per_store_week=("MOVE", lambda s: s.sum()),
        promo_share=("PROMO", "mean"),
        mean_margin_pct=("MARGIN_PCT", "mean"))
    sw = d.groupby("PRICE_TIER").apply(lambda g: g.groupby(["STORE", "WEEK"]).ngroups, include_groups=False)
    out["units_per_store_week"] = out["units_per_store_week"] / sw
    reg = d[d["PRODUCT_ID"].eq(top) & d["PROMO_TYPE"].eq("regular")].groupby("PRICE_TIER")["UNIT_PRICE"].median()
    out[f"regular_price_top_product"] = reg
    print(f"(top product = PRODUCT_ID {top})")
    print(out.round(3).to_string())
    print("\nReading: the tier sets the REGULAR price; promoted prices are set chainwide (manual),")
    print("so the same promotion is a deeper discount in high-tier stores. Tier is a key store feature.")


# ----------------------------------------------------------------------------
# Q7: substitution, first look
# ----------------------------------------------------------------------------
def q7_substitution(p: pd.DataFrame, products: pd.DataFrame) -> None:
    section("Q7. FIRST LOOK AT SUBSTITUTION BETWEEN MAJOR BRANDS (descriptive, NOT causal)")
    d = p[p["BRAND"].isin(MAJOR_BRANDS)]
    top = (d.groupby(["BRAND", "PRODUCT_ID"])["MOVE"].sum().reset_index()
           .sort_values("MOVE", ascending=False).drop_duplicates("BRAND").set_index("BRAND")["PRODUCT_ID"])
    top = top.reindex(MAJOR_BRANDS)
    names = products.set_index("UPC")["DESCRIP"].str.strip()
    print("Lead product per brand:")
    for b, pid in top.items():
        print(f"  {b:<20s} {pid}  {names.get(pid, '')}")

    s = d[d["PRODUCT_ID"].isin(top.values)].copy()
    s = s.sort_values(["PRODUCT_ID", "STORE", "WEEK"])
    reg_units = s["MOVE"].where(s["PROMO_TYPE"].eq("regular"))
    s["NORMAL"] = (reg_units.groupby([s["PRODUCT_ID"], s["STORE"]])
                   .transform(lambda x: x.rolling(13, center=True, min_periods=4).median()))
    s["RATIO"] = s["MOVE"] / s["NORMAL"]
    wide_ratio = s.pivot_table(index=["STORE", "WEEK"], columns="PRODUCT_ID", values="RATIO")
    wide_promo = s.pivot_table(index=["STORE", "WEEK"], columns="PRODUCT_ID", values="PROMO", aggfunc="max")

    mat = pd.DataFrame(index=MAJOR_BRANDS, columns=MAJOR_BRANDS, dtype=float)
    for a in MAJOR_BRANDS:
        pa = top[a]
        a_on = wide_promo[pa].eq(1)
        others_off = wide_promo.drop(columns=[pa]).fillna(0).eq(0).all(axis=1)
        for b in MAJOR_BRANDS:
            pb = top[b]
            if a == b:
                mask = a_on
            else:
                mask = a_on & others_off
            mat.loc[a, b] = np.exp(np.log(wide_ratio.loc[mask, pb]).mean())
    none_on = wide_promo.fillna(0).eq(0).all(axis=1)
    ref = pd.Series({b: np.exp(np.log(wide_ratio.loc[none_on, top[b]]).mean()) for b in MAJOR_BRANDS})
    mat = mat.div(ref, axis=1)          # 1.0 = same as weeks when none of the four is promoted
    print("\nGeometric-mean sales of the COLUMN brand's lead product, relative to weeks when none of the")
    print("four lead products is promoted, in store-weeks where the ROW brand's lead product is on")
    print("promotion (off-diagonal: and no other lead product is):")
    print(mat.round(3).to_string())
    print("Diagonal = own promotion; off-diagonal < 1 is consistent with sales being drawn away (substitution).")
    off = mat.where(~np.eye(len(MAJOR_BRANDS), dtype=bool)).stack()
    print(f"Off-diagonal range: {off.min():.3f} to {off.max():.3f} (mean {off.mean():.3f})")
    print("Caveat: NORMAL uses a centred 13-week window (descriptive only); not a counterfactual model.")

    fig, ax = plt.subplots(figsize=(6.4, 5), facecolor=SURFACE)
    vals = mat.to_numpy(dtype=float)
    shown = np.where(np.eye(len(MAJOR_BRANDS), dtype=bool), np.nan, vals)
    span = max(0.15, np.nanmax(np.abs(shown - 1)))
    norm = TwoSlopeNorm(vmin=1 - span, vcenter=1.0, vmax=1 + span)
    ax.imshow(shown, cmap=DIVERGING, norm=norm)
    for i in range(len(MAJOR_BRANDS)):
        for j in range(len(MAJOR_BRANDS)):
            txt = f"{vals[i, j]:.2f}" + (" (own)" if i == j else "")
            ax.text(j, i, txt, ha="center", va="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(MAJOR_BRANDS)), MAJOR_BRANDS, rotation=20, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(MAJOR_BRANDS)), MAJOR_BRANDS, fontsize=9, color=INK_2)
    ax.set_xlabel("Sales of this brand's lead product ...", color=INK_2, fontsize=9)
    ax.set_ylabel("... when this brand is on promotion", color=INK_2, fontsize=9)
    ax.set_title("Who loses sales when a competitor promotes?\n(1.00 = normal week; red = below normal)",
                 color=INK, fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "q7_substitution_matrix.png")


# ----------------------------------------------------------------------------
def check_negative_depth_events(events: pd.DataFrame) -> None:
    section("CHECK: EVENTS WITH PRICE ABOVE THE REGULAR PRICE (negative mean depth)")
    neg = events[events["MEAN_DEPTH"] < -0.05]
    print(f"Events with mean depth below -5%: {len(neg)} of {len(events)}")
    if len(neg):
        print(neg[["EVENT_ID", "PRODUCT_ID", "START_WEEK", "DURATION_WEEKS", "STORES", "MEAN_DEPTH",
                   "SHARE_PRICE_PROMO", "MAIN_DEAL_CODE"]].round(3).head(15).to_string(index=False))
        print("These are deal-coded weeks where the shelf price was ABOVE the recent regular price")
        print("(e.g. a regular price increase in the same week). Kept, labelled deal_only.")


# ----------------------------------------------------------------------------
def q2_season(wk: pd.DataFrame, p: pd.DataFrame) -> None:
    """Holiday and season effects (used for categories without a Lent story, e.g. soft drinks)."""
    section(f"Q2. HOLIDAYS AND SEASONS — {CAT_NAME.upper()}")
    print("Weekly regression of log(units per store) on holiday-week dummies + Lent + month FE + year FE,")
    print("Newey-West SE (4 lags). Effects are % differences vs. a normal week of the same month and year.")
    y = np.log(wk["UNITS_PER_STORE"].to_numpy())
    rows = []
    for with_promo in (False, True):
        res = _lent_regression(wk, y, with_promo)
        terms = ["LENT_WEEK"] + [c for c in res.index if c.startswith("HOL_")] + (["PROMO_SHARE"] if with_promo else [])
        for t in terms:
            b, se = res.loc[t, "coef"], res.loc[t, "se_hac"]
            rows.append({"model": "with promo share" if with_promo else "no promo control",
                         "term": t.replace("HOL_", ""), "effect_%": (np.exp(b) - 1) * 100,
                         "ci_low_%": (np.exp(b - 1.96 * se) - 1) * 100, "ci_high_%": (np.exp(b + 1.96 * se) - 1) * 100,
                         "t": b / se})
    tab = pd.DataFrame(rows)
    print(tab.round(1).to_string(index=False))
    print("(PROMO_SHARE row: effect of going from 0% to 100% of product-store weeks on promotion.)")

    month = wk.groupby("MONTH").agg(units_per_store=("UNITS_PER_STORE", "mean"), promo_share=("PROMO_SHARE", "mean"))
    month["index_vs_avg"] = month["units_per_store"] / month["units_per_store"].mean()
    print("\nSeasonality by month (raw means over all years):")
    print(month.round(3).to_string())

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, facecolor=SURFACE)
    for ax in (a1, a2):
        style_axis(ax)
    a1.bar(month.index, month["index_vs_avg"], color=SERIES[0], width=0.7)
    a1.axhline(1, color=INK_2, lw=1, ls="--")
    a1.set_ylabel("Units/store vs.\nyearly average", color=INK_2, fontsize=9)
    a2.bar(month.index, month["promo_share"] * 100, color=SERIES[1], width=0.7)
    a2.set_ylabel("% of product-store\nweeks on promotion", color=INK_2, fontsize=9)
    a2.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    fig.suptitle(f"{CAT_NAME.capitalize()}: demand and promotion intensity by month", color=INK, fontsize=11,
                 x=0.01, ha="left")
    save(fig, "q2_seasonality.png")


def add_brand_group(p: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """BRAND used by Q3/Q4/Q7: tuna = brand; soft drinks = Coke family / Pepsi family / 7-Up / Dominick's / Other."""
    cols = ["UPC", "BRAND"] + (["MAKER"] if "MAKER" in products.columns else [])
    p = p.merge(products[cols].rename(columns={"UPC": "PRODUCT_ID"}), on="PRODUCT_ID", how="left")
    if CAT == "sdr":
        p["BRAND"] = np.select([p["MAKER"].isin(["Coca-Cola Co.", "PepsiCo"]), p["BRAND"].isin(["7-Up", "Dominick's"])],
                               [p["MAKER"], p["BRAND"]], default="Other")
        p = p.drop(columns="MAKER")
    return p


def main() -> None:
    global CAT, MAIN, CAT_NAME, UNIT, MAJOR_BRANDS, BRAND_COLOR, FIG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", choices=sorted(CATEGORIES))
    CAT = parser.parse_args().cat
    cfg = CATEGORIES[CAT]
    MAIN, CAT_NAME, UNIT, MAJOR_BRANDS = cfg["main"], cfg["name"], cfg["unit"], cfg["majors"]
    BRAND_COLOR = dict(zip(MAJOR_BRANDS + ["Other"], SERIES[:4] + [GREY]))
    if CAT != "tna":
        FIG_DIR = REPORT_DIR / CAT / "figures"
    report = "phase4_eda_report.txt" if CAT == "tna" else f"phase4_{CAT}_eda_report.txt"
    tee = Tee(REPORT_DIR / report)
    sys.stdout = tee
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        section(f"PROMOGUARD AI — PHASE 4 EXPLORATORY DATA ANALYSIS — {CAT_NAME}")
        print("All results are descriptive. Causal estimates come in later phases.")

        p = pd.read_parquet(PROC / f"{CAT}_{MAIN}_panel.parquet")
        products = pd.read_csv(PROC / f"{CAT}_products.csv")
        events = pd.read_csv(PROC / f"{CAT}_promo_events.csv")
        stores = pd.read_csv(REF / "dominicks_stores.csv")
        weeks = build_week_features()
        p = add_brand_group(p, products)
        products = products.assign(BRAND=add_brand_group(
            products[["UPC"]].rename(columns={"UPC": "PRODUCT_ID"}), products)["BRAND"].to_numpy())
        print(f"Panel rows: {len(p):,} | products: {p['PRODUCT_ID'].nunique()} | stores: {p['STORE'].nunique()}")

        weeks = coverage_check(p, weeks)
        p = p[p["WEEK"].isin(weeks.loc[weeks["COVERAGE_OK"], "WEEK"])]
        print(f"Panel rows kept for analysis: {len(p):,}")
        wk = category_weekly(p, weeks)
        q1_trend(wk)
        if CAT == "tna":
            q2_lent(wk, p)
        else:
            q2_season(wk, p)
        q3_brands(p, weeks)
        q4_calendar(events, products, weeks)
        q5_post_promo(p, events)
        q6_stores(p, stores)
        q7_substitution(p, products)
        check_negative_depth_events(events)

        out = PROC / f"{CAT}_week_features.csv"
        weeks.to_csv(out, index=False, date_format="%Y-%m-%d")
        section("OUTPUTS")
        print(f"{out.relative_to(REPO_ROOT)}  (calendar features: LENT_WEEK, LENT_DAYS, WEEKS_TO_EASTER, COVERAGE_OK, ...)")
        print(f"Figures in {FIG_DIR.relative_to(REPO_ROOT)}/")
        section("DONE")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
