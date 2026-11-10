"""
PromoGuard AI — Phase 7: Cannibalization (Causal Estimate)
==========================================================

Location in repo:  PromoGuard-AI/src/analysis/07_cannibalization.py
Run from repo root:
    python src/analysis/07_cannibalization.py

Question
    When a tuna product is promoted, how many of its extra units are taken from the
    OTHER tuna products (substitution / cannibalization), and how many are new
    category sales?

Inputs
    data/processed/tna_tuna_panel.parquet, tna_products.csv, tna_week_features.csv,
    tna_event_incrementality.csv, tna_counterfactual_rows.parquet (Phase 6),
    data/reference/dominicks_stores.csv, models/baseline_model.joblib (Phase 5)

Outputs
    data/processed/tna_event_cannibalization.csv   events with cannibalized and net CATEGORY units
    reports/phase7/phase7_cannibalization_report.txt
    reports/phase7/figures/*.png

Run from repo root:
    python src/analysis/07_cannibalization.py              # canned tuna (default, --cat tna)
    python src/analysis/07_cannibalization.py --cat sdr    # carbonated soft drinks
    (soft drinks: files start with sdr_, model models/sdr_baseline_model.joblib, outputs in reports/phase7/sdr/;
     brand groups = Coca-Cola Co. / PepsiCo / 7-Up / Dominick's / Other; sub-type = regular vs. diet)

Method (store x week panel, canned tuna, weeks with good coverage)
------------------------------------------------------------------
  Treatment  L[s,t] = incremental units of all PROMOTED tuna products in store s, week t
                      = sum(actual - counterfactual) over promotion-event rows (Phase 6).
  Outcome    Y[s,t] = deviation of BYSTANDER tuna products from their expected sales
                      = sum(actual - expected) over clean regular rows of products that have no
                        promotion event of their own in the surrounding weeks (not in the 4 weeks
                        before or after one of their own events). "Expected" comes from a
                        cross-fitted baseline WITHOUT competitor-promotion features, so the effect
                        of competitors' promotions is left in the outcome.
  Model      Y[s,t] = beta * L[s,t] + gamma * LOCAL_PROMOS[s,t] + store FE + week FE + error

  beta = DIVERSION RATIO: change in the other products' units per extra unit of the promoted
         products. beta = -0.30 means 30% of the promotion's extra units were taken from other
         tuna products; the remaining 70% are new category sales.

Identification
  * Week fixed effects remove everything common to all stores in a week: chainwide promotion
    calendars, Lent, holidays, weather, category-wide demand shocks.
  * Store fixed effects remove permanent store differences (size, location, clientele).
  * The remaining variation: the SAME chainwide promotion produces different lifts in different
    stores, mainly because promoted prices are chainwide while REGULAR prices differ by pricing
    zone (manual) -> the discount depth differs by store.
  * L is estimated with noise (it contains the promoted product's own forecast error), which
    would bias OLS toward zero. We therefore instrument L with the EXPECTED discount volume
    Z1 = sum(counterfactual units x discount depth) and Z2 = sum(counterfactual units of promoted
    products): both are fixed before the week's sales happen. Two-stage least squares, standard
    errors clustered by week.
  * Assumption: within a week, how deep a store's discount is does not otherwise affect demand
    for the non-promoted products (exclusion restriction). Checked with a LEAD test: next weeks'
    promotions must not "affect" this week's outcome.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import baseline_features as bf  # noqa: E402
from src.models import counterfactual as cfm  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
REF = REPO_ROOT / "data" / "reference"
MODEL_PATH = REPO_ROOT / "models" / "baseline_model.joblib"
REPORT_DIR = REPO_ROOT / "reports" / "phase7"
FIG_DIR = REPORT_DIR / "figures"

COMPETITOR_FEATURES = ["N_OTHER_PROMO", "OTHER_BRAND_PROMO_W", "SAME_BRAND_PROMO_W", "OTHER_MAX_DEPTH"]
NOCOMP_NUMERIC = [c for c in bf.NUMERIC if c not in COMPETITOR_FEATURES]
GROUPS = ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's", "Other"]
SUBTYPE_COL, SUBTYPES = "TUNA_MEAT", ["light", "white"]

CATEGORIES = {
    "tna": {"main": "tuna", "name": "canned tuna", "unit": "tuna",
            "groups": ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's", "Other"],
            "subtype_col": "TUNA_MEAT", "subtypes": ["light", "white"], "subtype_name": "TUNA TYPE (light vs. white)"},
    "sdr": {"main": "csd", "name": "carbonated soft drinks", "unit": "soft-drink",
            "groups": ["Coca-Cola Co.", "PepsiCo", "7-Up", "Dominick's", "Other"],
            "subtype_col": "SD_DIET", "subtypes": ["regular", "diet"], "subtype_name": "DRINK TYPE (regular vs. diet)"},
}
CAT, MAIN, CAT_NAME, UNIT, SUBTYPE_NAME = "tna", "tuna", "canned tuna", "tuna", "TUNA TYPE (light vs. white)"
SEED = 42

SURFACE, INK, INK_2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, RED, AQUA = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])

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
# Econometrics helpers
# ----------------------------------------------------------------------------
def demean_two_way(df: pd.DataFrame, cols: list[str], a: str = "STORE", b: str = "WEEK",
                   tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    """Remove store and week fixed effects (alternating projections; works for unbalanced panels)."""
    out = df[cols].astype(float).copy()
    ga, gb = df[a].to_numpy(), df[b].to_numpy()
    for _ in range(max_iter):
        before = out.to_numpy().copy()
        out = out - out.groupby(ga).transform("mean")
        out = out - out.groupby(gb).transform("mean")
        if np.max(np.abs(out.to_numpy() - before)) < tol:
            break
    return out


def iv_2sls(y: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray | None,
            clusters: np.ndarray, names: list[str]) -> pd.DataFrame:
    """
    2SLS with cluster-robust standard errors. y, X (endogenous), Z (excluded instruments),
    W (exogenous controls) must already be demeaned for the fixed effects.
    If Z is None the model is plain OLS.
    """
    Wm = W if W is not None else np.empty((len(y), 0))
    Xfull = np.column_stack([X, Wm])
    if Z is None:
        Xhat = Xfull
    else:
        Zfull = np.column_stack([Z, Wm])
        coef_fs, *_ = np.linalg.lstsq(Zfull, Xfull, rcond=None)
        Xhat = Zfull @ coef_fs
    beta, *_ = np.linalg.lstsq(Xhat, y, rcond=None)
    u = y - Xfull @ beta
    bread = np.linalg.inv(Xhat.T @ Xhat)
    meat = np.zeros((Xhat.shape[1], Xhat.shape[1]))
    order = np.argsort(clusters, kind="stable")
    cs, xs, us = clusters[order], Xhat[order], u[order]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    for i in range(len(bounds) - 1):
        s = xs[bounds[i]:bounds[i + 1]].T @ us[bounds[i]:bounds[i + 1]]
        meat += np.outer(s, s)
    g = len(bounds) - 1
    V = bread @ meat @ bread * g / (g - 1)
    se = np.sqrt(np.diag(V))
    k = X.shape[1]
    res = pd.DataFrame({"coef": beta[:k], "se": se[:k]}, index=names)
    res["t"] = res["coef"] / res["se"]
    res["ci90_lo"] = res["coef"] - 1.645 * res["se"]
    res["ci90_hi"] = res["coef"] + 1.645 * res["se"]
    res.attrs["V"] = V[:k, :k]
    res.attrs["n"], res.attrs["clusters"] = len(y), g
    if Z is not None:   # first-stage F for each endogenous regressor (excluded instruments only)
        fs = []
        for j in range(k):
            full = np.column_stack([Z, Wm])
            b_full, *_ = np.linalg.lstsq(full, X[:, j], rcond=None)
            rss_full = np.sum((X[:, j] - full @ b_full) ** 2)
            if Wm.shape[1]:
                b_r, *_ = np.linalg.lstsq(Wm, X[:, j], rcond=None)
                rss_r = np.sum((X[:, j] - Wm @ b_r) ** 2)
            else:
                rss_r = np.sum(X[:, j] ** 2)
            q = Z.shape[1]
            fs.append(((rss_r - rss_full) / q) / (rss_full / (len(y) - full.shape[1])))
        res["first_stage_F"] = fs
    return res


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def nocomp_model(bundle: dict) -> Pipeline:
    """Same Random Forest as Phase 5, without competitor-promotion features."""
    rf = bundle["model"].named_steps["model"]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), bf.CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NOCOMP_NUMERIC)])
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(**rf.get_params()))])


def brand_group(b: pd.Series, maker: pd.Series | None = None) -> pd.Series:
    """Tuna: brand. Soft drinks: Coca-Cola Co. / PepsiCo for all their brands (needs MAKER)."""
    if maker is not None and CAT == "sdr":
        b = b.where(~maker.isin(["Coca-Cola Co.", "PepsiCo"]), maker)
    return b.where(b.isin(GROUPS[:-1]), "Other")


def build_store_week(d: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to store x week: outcome Y (non-promoted products), treatment L, instruments Z."""
    need = ["STORE", "WEEK", "MOVE", "PRED_NOCOMP", "PRED_CF", "DEPTH", "PROMO", "PHASE", "BRAND", SUBTYPE_COL,
            "PROMO_TYPE", "COVERAGE_OK", "CHAIN_PROMO_SHARE", "N_REG_HIST", "BASE_LOG_13", "PRICE_TIER"]
    need += ["MAKER"] if "MAKER" in d.columns else []
    d = d.loc[d["COVERAGE_OK"].astype(bool), list(dict.fromkeys(need))].copy()   # only the columns used
    d["BG"] = brand_group(d["BRAND"], d["MAKER"] if "MAKER" in d.columns else None)
    d["MEAT"] = d[SUBTYPE_COL].where(d[SUBTYPE_COL].isin(SUBTYPES), "unknown")

    # bystanders: clean regular rows of products with no own event nearby (PHASE 'none')
    out_rows = d[cfm.clean_training_rows(d) & d["PHASE"].eq("none")].copy()
    out_rows["DEV"] = out_rows["MOVE"] - out_rows["PRED_NOCOMP"]
    tr = d[d["PHASE"].eq("promo")].copy()
    tr["LIFT"] = tr["MOVE"] - tr["PRED_CF"]
    tr["Z1"] = tr["PRED_CF"] * tr["DEPTH"].clip(lower=0).fillna(0)
    tr["Z2"] = tr["PRED_CF"]

    keys = ["STORE", "WEEK"]
    sw = d.groupby(keys).size().rename("ROWS").to_frame()
    sw["Y"] = out_rows.groupby(keys)["DEV"].sum()
    sw["Y_ACTUAL"] = out_rows.groupby(keys)["MOVE"].sum()
    # all clean non-promoted tuna sales (bystanders + products near their own events): used to
    # scale the bystander estimate up to the whole non-promoted assortment
    all_np = d[cfm.clean_training_rows(d) & ~d["PHASE"].eq("promo")]
    sw["Y_ALL_ACTUAL"] = all_np.groupby(keys)["MOVE"].sum()
    sw["L"] = tr.groupby(keys)["LIFT"].sum()
    sw["Z1"] = tr.groupby(keys)["Z1"].sum()
    sw["Z2"] = tr.groupby(keys)["Z2"].sum()
    local = d["PROMO"] & ~d["PHASE"].isin(["promo"])
    sw["LOCAL_PROMOS"] = local.groupby([d["STORE"], d["WEEK"]]).sum()
    for g in GROUPS:
        sw[f"Y_{g}"] = out_rows[out_rows["BG"].eq(g)].groupby(keys)["DEV"].sum()
        sw[f"L_{g}"] = tr[tr["BG"].eq(g)].groupby(keys)["LIFT"].sum()
        sw[f"Z1_{g}"] = tr[tr["BG"].eq(g)].groupby(keys)["Z1"].sum()
        sw[f"Z2_{g}"] = tr[tr["BG"].eq(g)].groupby(keys)["Z2"].sum()
    for m in SUBTYPES:
        sw[f"Y_{m}"] = out_rows[out_rows["MEAT"].eq(m)].groupby(keys)["DEV"].sum()
        sw[f"L_{m}"] = tr[tr["MEAT"].eq(m)].groupby(keys)["LIFT"].sum()
        sw[f"Z1_{m}"] = tr[tr["MEAT"].eq(m)].groupby(keys)["Z1"].sum()
        sw[f"Z2_{m}"] = tr[tr["MEAT"].eq(m)].groupby(keys)["Z2"].sum()
    sw = sw.fillna(0.0).reset_index()

    # leads: the same store's promotion volume 2 weeks LATER (placebo: cannot cause this week)
    lead = sw[["STORE", "WEEK", "Z1", "Z2", "L"]].copy()
    lead["WEEK"] = lead["WEEK"] - 2
    sw = sw.merge(lead.rename(columns={"Z1": "Z1_LEAD2", "Z2": "Z2_LEAD2", "L": "L_LEAD2"}),
                  on=["STORE", "WEEK"], how="left")
    return sw


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_binscatter(sw: pd.DataFrame, dm: pd.DataFrame, beta_ols: float) -> None:
    x, y = dm["L"].to_numpy(), dm["Y"].to_numpy()
    q = pd.qcut(x, 20, duplicates="drop")
    b = pd.DataFrame({"x": x, "y": y, "q": q}).groupby("q", observed=True).mean()
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.scatter(b["x"], b["y"], s=40, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    xs = np.linspace(b["x"].min(), b["x"].max(), 50)
    ax.plot(xs, beta_ols * xs, color=ORANGE, lw=2)
    ax.axhline(0, color=INK_2, lw=0.8, ls="--")
    ax.set_xlabel(f"Extra units of PROMOTED {UNIT} in the store-week\n(after removing store and week effects)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel(f"Units of NON-promoted {UNIT} vs. expected", color=INK_2, fontsize=9)
    ax.set_title(f"Stores where promotions gained more, other {UNIT} lost more", color=INK, fontsize=11, loc="left")
    save(fig, "binscatter_substitution.png")


def plot_decomposition(gross: float, cannib: float, label_rate: str) -> None:
    net = gross - cannib
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    style_axis(ax)
    vals = [gross, -cannib, net]
    bottoms = [0, gross - cannib, 0]
    heights = [gross, cannib, net]
    colors = [ORANGE, RED, BLUE]
    labels = ["Extra units of\npromoted products", f"Taken from other\n{UNIT} products", f"Net new {UNIT}\n(category) units"]
    ax.bar(range(3), heights, bottom=bottoms, color=colors, width=0.6)
    for i, (v, btm, h) in enumerate(zip(vals, bottoms, heights)):
        ax.text(i, btm + h + gross * 0.02, f"{v / 1e6:+.2f}M", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), labels)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
    ax.set_ylabel("Units (all in-scope events)", color=INK_2, fontsize=9)
    ax.set_title(f"Where do promotion sales come from? ({label_rate} cannibalized)", color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, gross * 1.15)
    save(fig, "decomposition_gross_cannibalized_net.png")


def plot_matrix(mat: pd.DataFrame, se: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), facecolor=SURFACE)
    vals = mat.to_numpy(dtype=float)
    span = max(0.05, np.nanmax(np.abs(vals)))
    ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            sig = abs(vals[i, j]) > 1.645 * se.to_numpy()[i, j]
            ax.text(j, i, f"{vals[i, j]:+.3f}" + ("*" if sig else ""), ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=20, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=9, color=INK_2)
    ax.set_xlabel("Brand LOSING / gaining units (non-promoted products)", color=INK_2, fontsize=9)
    ax.set_ylabel("Brand being PROMOTED", color=INK_2, fontsize=9)
    ax.set_title("Diversion: units gained (+) or lost (−) per extra promoted unit\n(* = 90% CI excludes 0)",
                 color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "diversion_matrix_brands.png")


# ----------------------------------------------------------------------------
def main() -> None:
    global CAT, MAIN, CAT_NAME, UNIT, SUBTYPE_NAME, GROUPS, SUBTYPE_COL, SUBTYPES, FIG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", choices=sorted(CATEGORIES))
    CAT = parser.parse_args().cat
    cfg = CATEGORIES[CAT]
    MAIN, CAT_NAME, UNIT, GROUPS = cfg["main"], cfg["name"], cfg["unit"], cfg["groups"]
    SUBTYPE_COL, SUBTYPES, SUBTYPE_NAME = cfg["subtype_col"], cfg["subtypes"], cfg["subtype_name"]
    bf.set_category(CAT)
    out_dir = REPORT_DIR if CAT == "tna" else REPORT_DIR / CAT
    FIG_DIR = out_dir / "figures"
    report = "phase7_cannibalization_report.txt" if CAT == "tna" else f"phase7_{CAT}_cannibalization_report.txt"
    model_path = MODEL_PATH if CAT == "tna" else REPO_ROOT / "models" / f"{CAT}_baseline_model.joblib"

    tee = Tee(REPORT_DIR / report)
    sys.stdout = tee
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        section(f"PROMOGUARD AI — PHASE 7 CANNIBALIZATION — {CAT_NAME}")
        panel = pd.read_parquet(PROC / f"{CAT}_{MAIN}_panel.parquet")
        products = pd.read_csv(PROC / f"{CAT}_products.csv")
        stores = pd.read_csv(REF / "dominicks_stores.csv")
        weeks = pd.read_csv(PROC / f"{CAT}_week_features.csv")
        events = pd.read_csv(PROC / f"{CAT}_event_incrementality.csv")
        cf_rows = pd.read_parquet(PROC / f"{CAT}_counterfactual_rows.parquet",
                                  columns=["PRODUCT_ID", "STORE", "WEEK", "PHASE", "EVENT_ID", "PRED_CF"])
        bundle = joblib.load(model_path)

        d = bf.build_features(panel, products, stores, weeks, bundle["train_end_week"])
        d = d.merge(cf_rows[["PRODUCT_ID", "STORE", "WEEK", "PHASE", "EVENT_ID", "PRED_CF"]],
                    on=["PRODUCT_ID", "STORE", "WEEK"], how="left", validate="one_to_one")
        del cf_rows
        if "MAKER" in products.columns:
            d["MAKER"] = d["PRODUCT_ID"].map(products.drop_duplicates("UPC").set_index("UPC")["MAKER"])

        section("1. EXPECTED SALES WITHOUT COMPETITOR-PROMOTION FEATURES (cross-fitted by year)")
        print("Features removed:", COMPETITOR_FEATURES)
        cols = bf.CATEGORICAL + NOCOMP_NUMERIC
        preds, diag = cfm.crossfit(d, nocomp_model(bundle), {"actual": d}, feature_cols=cols)
        d["PRED_NOCOMP"] = preds["actual"]
        print(diag[["clean_rows", "WAPE", "BIAS"]].round(4).to_string())

        section("2. STORE x WEEK PANEL")
        sw = build_store_week(d)
        print(f"Store-weeks: {len(sw):,} | stores: {sw['STORE'].nunique()} | weeks: {sw['WEEK'].nunique()}")
        print(f"Store-weeks with promotion-event lift: {(sw['L'] != 0).mean() * 100:.1f}%")
        print(sw[["L", "Y", "Y_ACTUAL", "Z1", "Z2", "LOCAL_PROMOS"]].describe().round(1).to_string())
        dm_cols = ["Y", "L", "Z1", "Z2", "LOCAL_PROMOS", "Z1_LEAD2", "Z2_LEAD2", "L_LEAD2"]
        dm_cols += [f"{p}_{g}" for g in GROUPS + SUBTYPES for p in ("Y", "L", "Z1", "Z2")]
        sw_l = sw.dropna(subset=["Z1_LEAD2"])
        dm = demean_two_way(sw, [c for c in dm_cols if not c.endswith("LEAD2")])
        dm_lead = demean_two_way(sw_l, ["Y", "L", "Z1", "Z2", "LOCAL_PROMOS", "Z1_LEAD2", "Z2_LEAD2"])
        within = dm["L"].var() / sw["L"].var()
        print(f"\nShare of the variation in L that remains after store and week effects: {within * 100:.1f}%")
        print("(this within-week, across-store variation identifies the effect)")
        cl = sw["WEEK"].to_numpy()

        section(f"3. OVERALL DIVERSION RATIO (all {UNIT})")
        W = dm[["LOCAL_PROMOS"]].to_numpy()
        ols = iv_2sls(dm["Y"].to_numpy(), dm[["L"]].to_numpy(), None, W, cl, ["L"])
        iv = iv_2sls(dm["Y"].to_numpy(), dm[["L"]].to_numpy(), dm[["Z1", "Z2"]].to_numpy(), W, cl, ["L"])
        print("OLS (two-way FE):")
        print(ols.round(4).to_string())
        print("\n2SLS (two-way FE, instruments: expected discount volume Z1, Z2):")
        print(iv.round(4).to_string())
        beta = iv.loc["L", "coef"]
        print(f"\n-> For every extra unit sold by promoted {UNIT} products, the non-promoted {UNIT} products sell")
        print(f"   {beta:+.3f} units (90% CI {iv.loc['L', 'ci90_lo']:+.3f} to {iv.loc['L', 'ci90_hi']:+.3f}).")
        print(f"   Cannibalization rate = {-beta * 100:.1f}% of the gross lift; "
              f"net new category units = {(1 + beta) * 100:.1f}%.")
        print("   OLS is closer to zero than 2SLS, as expected from measurement error in L (attenuation).")
        scale = sw["Y_ALL_ACTUAL"].sum() / sw["Y_ACTUAL"].sum()
        print(f"\nScope of the outcome: bystander products = {1 / scale * 100:.0f}% of all clean non-promoted {UNIT}")
        print("sales. Products close to their own promotion events are left out (they would break the")
        print("lead test), so the estimate above is a LOWER BOUND for the whole assortment.")
        print(f"If those products lose sales at the same rate (per unit sold), the diversion scales by {scale:.2f}:")
        print(f"   scaled diversion = {beta * scale:+.3f}  ->  cannibalization rate ~ {-beta * scale * 100:.1f}% "
              f"(90% CI {-iv.loc['L', 'ci90_hi'] * scale * 100:.1f}% to {-iv.loc['L', 'ci90_lo'] * scale * 100:.1f}%)")
        print(f"Reported range: {-beta * 100:.1f}% (bystanders only) to {-beta * scale * 100:.1f}% (scaled). "
              "The scaled value is used for event attribution (conservative for profit).")
        plot_binscatter(sw, dm, ols.loc["L", "coef"])

        section("4. VALIDITY CHECKS")
        print(f"Interpretation of 'net new category units': extra {UNIT} bought at Dominick's that did not")
        print(f"come from other {UNIT} products in the same store-week. It can be higher consumption, earlier")
        print("purchases stored at home (no post-promotion dip was measured in Phase 6), or shoppers who")
        print(f"would otherwise have bought {UNIT} at another retailer. Store-level data cannot separate these.\n")
        lead = iv_2sls(dm_lead["Y"].to_numpy(), dm_lead[["Z1", "Z2", "Z1_LEAD2", "Z2_LEAD2"]].to_numpy(), None,
                       dm_lead[["LOCAL_PROMOS"]].to_numpy(), sw_l["WEEK"].to_numpy(),
                       ["Z1 (this week)", "Z2 (this week)", "Z1 (2 weeks later)", "Z2 (2 weeks later)"])
        print("(a) Lead test, reduced form: outcome on this week's AND 2-weeks-later promotion volume.")
        print("    Future promotions cannot cause this week's sales -> their coefficients should be ~0.")
        print(lead[["coef", "se", "t"]].round(5).to_string())
        print("\n(b) Instrument strength (first-stage F > 10 is the usual rule of thumb):")
        print(f"    F = {iv.loc['L', 'first_stage_F']:.1f}")
        tr_rows = d[d["PHASE"].eq("promo")]
        print("\n(c) Where does the instrument's variation come from? Mean discount depth of promotion rows by tier:")
        print((tr_rows.groupby(tr_rows["PRICE_TIER"].fillna("unknown"))["DEPTH"].mean() * 100).round(1).to_string())

        section("5. DIVERSION BY PROMOTED BRAND AND WHO LOSES (2SLS, same fixed effects)")
        Lc = [f"L_{g}" for g in GROUPS]
        Zc = [f"Z1_{g}" for g in GROUPS] + [f"Z2_{g}" for g in GROUPS]
        by_brand = iv_2sls(dm["Y"].to_numpy(), dm[Lc].to_numpy(), dm[Zc].to_numpy(), W, cl, GROUPS)
        print(f"Units lost by ALL other {UNIT} per extra unit of the promoted brand:")
        print(by_brand.round(4).to_string())

        mat = pd.DataFrame(index=GROUPS, columns=GROUPS, dtype=float)
        se = mat.copy()
        for g_out in GROUPS:
            r = iv_2sls(dm[f"Y_{g_out}"].to_numpy(), dm[Lc].to_numpy(), dm[Zc].to_numpy(), W, cl, GROUPS)
            mat[g_out], se[g_out] = r["coef"], r["se"]
        print("\nDiversion matrix: row = promoted brand, column = brand whose NON-promoted products gain/lose")
        print(mat.round(4).to_string())
        print("\nStandard errors:")
        print(se.round(4).to_string())
        plot_matrix(mat, se)

        section(f"6. BY {SUBTYPE_NAME}")
        Lm = [f"L_{m}" for m in SUBTYPES]
        Zm = [f"Z1_{m}" for m in SUBTYPES] + [f"Z2_{m}" for m in SUBTYPES]
        mt = pd.DataFrame(index=SUBTYPES, columns=SUBTYPES, dtype=float)
        mse = mt.copy()
        for m_out in SUBTYPES:
            r = iv_2sls(dm[f"Y_{m_out}"].to_numpy(), dm[Lm].to_numpy(), dm[Zm].to_numpy(), W, cl, SUBTYPES)
            mt[m_out], mse[m_out] = r["coef"], r["se"]
        print("Row = promoted type, column = type losing units:")
        print(mt.round(4).to_string())
        print("Standard errors:")
        print(mse.round(4).to_string())
        print("Closer substitutes (same type) are expected to lose more.")

        section("7. EVENT-LEVEL ATTRIBUTION")
        ev = events[events["IN_SCOPE"]].copy()
        ev["BG"] = brand_group(ev["BRAND"]) if CAT == "tna" else ev["BRAND_GROUP"]
        div = by_brand["coef"]
        ev["DIVERSION"] = ev["BG"].map(div) * scale        # scaled to the whole non-promoted assortment
        ev["CANNIBALIZED_UNITS"] = -ev["DIVERSION"] * ev["GROSS_LIFT_UNITS"].clip(lower=0)
        ev["NET_CATEGORY_UNITS"] = ev["NET_INCREMENTAL_UNITS"] - ev["CANNIBALIZED_UNITS"]
        g, c = ev["GROSS_LIFT_UNITS"].clip(lower=0).sum(), ev["CANNIBALIZED_UNITS"].sum()
        print(f"In-scope events: {len(ev):,}")
        print(f"  Gross extra units of promoted products : {g:,.0f}")
        print(f"  Taken from other {UNIT} products          : {c:,.0f}  ({c / g * 100:.1f}%)")
        print(f"  Net new category units                  : {ev['NET_CATEGORY_UNITS'].sum():,.0f}")
        print("\nBy promoted brand:")
        t = ev.groupby("BG").agg(events=("EVENT_ID", "size"), gross=("GROSS_LIFT_UNITS", "sum"),
                                 cannibalized=("CANNIBALIZED_UNITS", "sum"), net_category=("NET_CATEGORY_UNITS", "sum"))
        t["cannibalized_%"] = t["cannibalized"] / t["gross"] * 100
        print(t.round(1).to_string())
        print(f"\nAttribution uses the brand-level diversion ratios (section 5) scaled by {scale:.2f};")
        print("event-specific diversion is not separately identified. Phase 8 converts these units into")
        print("gross profit.")
        plot_decomposition(g, c, f"{c / g * 100:.0f}%")

        section("OUTPUTS")
        keep = list(events.columns) + ["BG", "DIVERSION", "CANNIBALIZED_UNITS", "NET_CATEGORY_UNITS"]
        out = events.merge(ev[["EVENT_ID", "BG", "DIVERSION", "CANNIBALIZED_UNITS", "NET_CATEGORY_UNITS"]],
                           on="EVENT_ID", how="left")[keep]
        out.to_csv(PROC / f"{CAT}_event_cannibalization.csv", index=False)
        div_out = by_brand.copy()
        div_out.attrs = {}
        div_out.to_csv(out_dir / "diversion_by_promoted_brand.csv")
        mat.to_csv(out_dir / "diversion_matrix_brands.csv")
        print(f"data/processed/{CAT}_event_cannibalization.csv")
        print(f"{out_dir.relative_to(REPO_ROOT).as_posix()}/diversion_by_promoted_brand.csv, diversion_matrix_brands.csv")
        section("DONE")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
