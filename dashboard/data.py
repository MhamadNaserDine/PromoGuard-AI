"""
PromoGuard AI — dashboard data layer.

Location: dashboard/data.py

All file loading and table logic for the dashboard lives here (no Streamlit code),
so it can be tested and reused (e.g. by the AI agent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROC = REPO_ROOT / "data" / "processed"
REF = REPO_ROOT / "data" / "reference"
REPORTS = REPO_ROOT / "reports"
MODELS = REPO_ROOT / "models"

# ----------------------------------------------------------------------------
# Product categories. The dashboard shows one category at a time (sidebar switch);
# set_category() changes the files every loader reads. Loaders also accept cat=... directly.
# ----------------------------------------------------------------------------
CATEGORIES = {
    "tna": {"name": "Canned tuna", "unit": "tuna", "items": "cans", "short": "tuna",
            "gap": "1996: data gap Jan–Oct", "gap_text": "Jan–Oct 1996: tuna data missing for almost all stores.",
            "season": "Lent"},
    "sdr": {"name": "Soft drinks", "unit": "soft drinks", "items": "units", "short": "soft drinks",
            "gap": "1994: data gap Jul–Sep", "gap_text": "Jul–Sep 1994: soft-drink data missing for almost all "
            "stores (8 weeks).", "season": "Lent"},
}
CAT = "tna"


def set_category(cat: str) -> None:
    global CAT
    CAT = cat if cat in CATEGORIES else "tna"


def info(cat: str | None = None) -> dict:
    return CATEGORIES[cat or CAT]


def _proc(name: str, cat: str | None = None) -> Path:
    return PROC / f"{cat or CAT}_{name}"


def _report_dir(phase: int, cat: str | None = None) -> Path:
    c = cat or CAT
    return REPORTS / f"phase{phase}" if c == "tna" else REPORTS / f"phase{phase}" / c


def _model_path(cat: str | None = None) -> Path:
    c = cat or CAT
    return MODELS / ("promo_response_model.joblib" if c == "tna" else f"{c}_promo_response_model.joblib")


def available_categories() -> list[str]:
    """Categories whose pipeline outputs exist (tuna is always listed)."""
    return [c for c in CATEGORIES if c == "tna" or not missing_files(c)]


ROW_COLS = ["PRODUCT_ID", "STORE", "WEEK", "WEEK_START", "MOVE", "DOLLARS", "GROSS_PROFIT", "UNIT_PRICE",
            "REG_PRICE", "DEPTH", "PROMO_TYPE", "PHASE", "EVENT_ID", "PRED_CF", "PRED_ACTUAL"]

def required_files(cat: str | None = None) -> dict:
    return {
        "Phase 6 rows": _proc("counterfactual_rows.parquet", cat),
        "Phase 8 events": _proc("event_profit.csv", cat),
        "Phase 9 SHAP": _proc("event_shap.csv", cat),
        "Phase 9 simulator model": _model_path(cat),
        "Phase 7 diversion": _report_dir(7, cat) / "diversion_by_promoted_brand.csv",
    }


REQUIRED_FILES = required_files("tna")


def clean_name(s: pd.Series) -> pd.Series:
    """Remove the leading '$' / '~' marks some Dominick's descriptions carry (e.g. '$GEISHA', '~STAR K')."""
    return s.astype(str).str.strip().str.lstrip("$~#* ").str.strip()


def readable_name(descrip: str, brand: str, size: str, cat: str | None = None) -> str:
    """'BUM BEE/WATER CHK LI' + Bumble Bee + '6.12 O' -> 'Bumble Bee chunk light in water 6.12 oz'.
    Soft drinks: 'PEPSI COLA CANS' + '24/12O' -> 'Pepsi Cola Cans · 24 × 12 oz'."""
    import re
    if (cat or CAT) != "tna":
        z = str(size).upper().strip()
        m = re.match(r"^(\d+)\s*/\s*(\d*\.?\d+)", z)
        z = f"{m.group(1)} × {m.group(2)} oz" if m else z.replace("LT", " L").replace("O", " oz").replace("  ", " ")
        return f"{str(descrip).strip().title()} · {z.strip()}"
    d = f" {str(descrip).upper()} "
    parts = [str(brand)]
    for pat, word in [(r"SOLID|SLD", "solid"), (r"CHUNK|CHNK|CHK|\bCH\b", "chunk")]:
        if re.search(pat, d):
            parts.append(word)
            break
    if re.search(r"WHITE|WHIT|WHT|\bWH\b|ALBACORE", d):
        parts.append("white")
    elif re.search(r"LIGHT|LITE|LGHT|LGT|\bLT\b|\bLI\b|LIGH", d):
        parts.append("light")
    for pat, word in [(r"L/S|LS/", "low-sodium"), (r"DIET", "diet"), (r"REDUCED", "reduced-fat"),
                      (r"LUN|\bLU\b", "lunch kit"), (r"SALA", "salad kit"), (r"PRE-MIX", "pre-mix")]:
        if re.search(pat, d):
            parts.append(word)
    if re.search(r"\bOIL\b|\bOI\b|/O\s", d):
        parts.append("in oil")
    elif re.search(r"WATER|WATE|/WA|\bWA\b|\bW\s*$", d):
        parts.append("in water")
    sz = str(size).upper().replace("OZ", "").replace(" O", "").strip().rstrip("O").strip()
    sz = sz.replace("3/3", "3×3")
    return " ".join(parts) + (f" {sz} oz" if sz else "")


def add_readable_names(df: pd.DataFrame, id_col: str = "PRODUCT_ID", cat: str | None = None) -> pd.DataFrame:
    """Adds NAME; identical names get the last 3 digits of the product id so they stay distinct."""
    df = df.copy()
    df["NAME"] = [readable_name(d, b, z, cat) for d, b, z in zip(df["DESCRIP"], df["BRAND"], df["SIZE"])]
    names = df.drop_duplicates(id_col)[[id_col, "NAME"]]
    dup = names["NAME"].duplicated(keep=False)
    fix = {pid: f"{n} (#{str(pid)[-3:]})" for pid, n in zip(names.loc[dup, id_col], names.loc[dup, "NAME"])}
    df["NAME"] = [fix.get(pid, n) for pid, n in zip(df[id_col], df["NAME"])]
    return df


def missing_files(cat: str | None = None) -> list[str]:
    return [f"{k}: {v.relative_to(REPO_ROOT)}" for k, v in required_files(cat).items() if not v.exists()]


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_events(cat: str | None = None) -> pd.DataFrame:
    ev = pd.read_csv(_proc("event_profit.csv", cat), parse_dates=["START_DATE"])
    ev["DESCRIP"] = clean_name(ev["DESCRIP"])
    ev["IN_SCOPE"] = ev["IN_SCOPE"].astype(bool)
    ev["LENT_LABEL"] = np.where(ev["LENT"].astype(float) > 0, "Lent", "not Lent")
    ev["DEPTH_BUCKET"] = ev["DEPTH_BUCKET"].astype(str)
    ev["LABEL"] = (ev["DESCRIP"] + " — " + ev["START_DATE"].dt.strftime("%Y-%m-%d") + " ("
                   + ev["DURATION_WEEKS"].astype(str) + " wk, " + (ev["MEAN_DEPTH"] * 100).round(0).astype(int).astype(str)
                   + "% off) #" + ev["EVENT_ID"].astype(str))
    return ev


def load_rows(cat: str | None = None) -> pd.DataFrame:
    return pd.read_parquet(_proc("counterfactual_rows.parquet", cat), columns=ROW_COLS)


def load_stores() -> pd.DataFrame:
    return pd.read_csv(REF / "dominicks_stores.csv")


def load_shap(cat: str | None = None) -> pd.DataFrame:
    return pd.read_csv(_proc("event_shap.csv", cat))


def load_sim_bundle(cat: str | None = None) -> dict:
    return joblib.load(_model_path(cat))


def load_diversion(cat: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    d = _report_dir(7, cat)
    by_brand = pd.read_csv(d / "diversion_by_promoted_brand.csv", index_col=0)
    p = d / "diversion_matrix_brands.csv"
    mat = pd.read_csv(p, index_col=0) if p.exists() else None
    return by_brand, mat


def load_model_comparison(cat: str | None = None) -> pd.DataFrame | None:
    p = _report_dir(5, cat) / "model_comparison_validation.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


def figure_path(phase: int, name: str, cat: str | None = None) -> Path | None:
    p = _report_dir(phase, cat) / "figures" / name
    return p if p.exists() else None


def validity_numbers(ev: pd.DataFrame, cat: str | None = None) -> dict:
    """Headline validity checks from the pipeline outputs (shown on the Model performance page)."""
    s = ev[ev["IN_SCOPE"]]
    pre = (s["PRE_ACTUAL"].sum() / s["PRE_CF"].sum() - 1) * 100 if s["PRE_CF"].sum() else np.nan
    comp = load_model_comparison(cat)
    wape = np.nan
    if comp is not None and {"WAPE_units", "WAPE_product_week"} <= set(comp.columns):
        ml = comp.drop(index=[i for i in ("naive_last", "moving_avg_13") if i in comp.index])
        wape = float(ml.loc[ml["WAPE_units"].idxmin(), "WAPE_product_week"]) * 100
    return {"pre_period_error_pct": pre, "baseline_wape_pct": wape, "events": len(s)}


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def kpis(ev: pd.DataFrame) -> dict:
    s = ev[ev["IN_SCOPE"]]
    gross = s["GROSS_LIFT_UNITS"].clip(lower=0).sum()
    return {
        "events": len(s),
        "products": s["PRODUCT_ID"].nunique(),
        "median_lift_pct": s["LIFT_PCT"].median(),
        "pooled_lift_pct": (s["PROMO_ACTUAL"].sum() / s["PROMO_CF"].sum() - 1) * 100,
        "incremental_units": s["NET_INCREMENTAL_UNITS"].sum(),
        "cannibalization_pct": s["CANNIBALIZED_UNITS"].sum() / gross * 100 if gross else np.nan,
        "net_category_units": s["NET_CATEGORY_UNITS"].sum(),
        "net_gp_pre": s["NET_GP_IMPACT_PRE"].sum(),
        "net_gp_rec": s["NET_GP_IMPACT_REC"].sum(),
        "trade_funding": s["TRADE_FUNDING_EST"].sum(),
        "discount_cost": s["DISCOUNT_COST"].sum(),
        "share_profitable_pre": s["PROFITABLE_PRE"].mean() * 100,
        "share_profitable_rec": s["PROFITABLE_REC"].mean() * 100,
        "share_significant": s["LIFT_SIGNIFICANT"].mean() * 100,
    }


def pooled_by(ev: pd.DataFrame, by: str) -> pd.DataFrame:
    s = ev[ev["IN_SCOPE"]]
    g = s.groupby(by, observed=True)
    out = pd.DataFrame({
        "events": g.size(),
        "lift_pct": (g["PROMO_ACTUAL"].sum() / g["PROMO_CF"].sum() - 1) * 100,
        "median_event_lift_pct": g["LIFT_PCT"].median(),
        "net_gp_pre": g["NET_GP_IMPACT_PRE"].sum(),
        "net_gp_recorded": g["NET_GP_IMPACT_REC"].sum(),
        "gp_per_discount_dollar_pre": g["NET_GP_IMPACT_PRE"].sum() / g["DISCOUNT_COST"].sum(),
        "gp_per_discount_dollar_rec": g["NET_GP_IMPACT_REC"].sum() / g["DISCOUNT_COST"].sum(),
        "share_profitable_pre_%": g["PROFITABLE_PRE"].mean() * 100,
    })
    return out


def event_weekly(rows: pd.DataFrame, ev_row: pd.Series, pad: int = 4) -> pd.DataFrame:
    """Weekly actual vs counterfactual units of the event's product, summed over stores."""
    w0, w1 = int(ev_row["START_WEEK"]) - pad, int(ev_row["END_WEEK"]) + pad
    x = rows[rows["PRODUCT_ID"].eq(ev_row["PRODUCT_ID"]) & rows["WEEK"].between(w0, w1)]
    g = x.groupby("WEEK").agg(WEEK_START=("WEEK_START", "first"), actual=("MOVE", "sum"),
                              counterfactual=("PRED_CF", "sum"), stores=("STORE", "nunique"),
                              price=("UNIT_PRICE", "median"), regular_price=("REG_PRICE", "median"))
    g["REL_WEEK"] = g.index - int(ev_row["START_WEEK"])
    g["in_event"] = g.index.to_series().between(int(ev_row["START_WEEK"]), int(ev_row["END_WEEK"]))
    return g.reset_index()


def event_by_tier(rows: pd.DataFrame, stores: pd.DataFrame, ev_row: pd.Series) -> pd.DataFrame:
    x = rows[rows["EVENT_ID"].eq(ev_row["EVENT_ID"]) & rows["PHASE"].eq("promo")]
    x = x.merge(stores[["STORE", "PRICE_TIER"]], on="STORE", how="left")
    x["PRICE_TIER"] = x["PRICE_TIER"].fillna("unknown")
    g = x.groupby("PRICE_TIER").agg(stores=("STORE", "nunique"), actual=("MOVE", "sum"),
                                    counterfactual=("PRED_CF", "sum"), mean_depth=("DEPTH", "mean"))
    g["lift_pct"] = (g["actual"] / g["counterfactual"] - 1) * 100
    g["mean_depth_%"] = g.pop("mean_depth") * 100
    return g


def shap_long(shap: pd.DataFrame, event_id: int) -> pd.DataFrame:
    row = shap[shap["EVENT_ID"].eq(event_id)]
    if row.empty:
        return pd.DataFrame(columns=["feature", "shap"])
    cols = [c for c in shap.columns if c.startswith("SHAP_") and c != "SHAP_BASE"]
    s = row[cols].iloc[0].rename(lambda c: c.replace("SHAP_", ""))
    out = s.rename("shap").reset_index().rename(columns={"index": "feature"})
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index)


def shap_global(shap: pd.DataFrame) -> pd.Series:
    cols = [c for c in shap.columns if c.startswith("SHAP_") and c != "SHAP_BASE"]
    return shap[cols].abs().mean().rename(lambda c: c.replace("SHAP_", "")).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Phase 10 — customer groups (optional: the page shows a setup message if missing)
# ----------------------------------------------------------------------------
GROUP_FILES = {"stores": PROC / "tna_store_groups.csv", "profiles": REPORTS / "phase10" / "group_profiles.csv",
               "response": REPORTS / "phase10" / "group_response.csv",
               "lift": REPORTS / "phase10" / "group_lift_by_depth.csv",
               "plan": REPORTS / "phase10" / "personalized_discounts.csv"}


def load_groups(cat: str | None = None) -> dict | None:
    """Customer groups were computed for canned tuna only (Phase 10)."""
    if (cat or CAT) != "tna" or not all(p.exists() for p in GROUP_FILES.values()):
        return None
    out = {k: pd.read_csv(p) for k, p in GROUP_FILES.items()}
    out["profiles"] = out["profiles"].set_index("GROUP")
    out["plan"]["DESCRIP"] = clean_name(out["plan"]["DESCRIP"])
    out["plan"] = add_readable_names(out["plan"], cat="tna")
    return out


# ----------------------------------------------------------------------------
# Post-event report card
# ----------------------------------------------------------------------------
CLASSES = ["Winner", "Funded win", "Volume trap", "Loser"]
VERDICTS = {"Winner": "Keep — repeat this design", "Funded win": "Keep only with manufacturer funding",
            "Volume trap": "Redesign — shallower discount", "Loser": "Stop or replace"}


def classify_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Sort in-scope events into four groups and give each a verdict.

    Winner       net GP > 0 even when the retailer pays the whole discount
    Funded win   net GP <= 0 if the retailer pays, > 0 with the recorded (trade-funded) cost
    Volume trap  not profitable in either view, but a significant positive sales lift
    Loser        not profitable and no clear lift
    EVIDENCE = 'clear' when the 90% range of the retailer-pays net GP has one sign, else 'uncertain'.
    """
    s = ev[ev["IN_SCOPE"]].copy()
    pre, rec = s["NET_GP_IMPACT_PRE"], s["NET_GP_IMPACT_REC"]
    s["CLASS"] = np.select([pre > 0, rec > 0, s["LIFT_SIGNIFICANT"].astype(bool) & (s["LIFT_PCT"] > 0)],
                           CLASSES[:3], CLASSES[3])
    s["VERDICT"] = s["CLASS"].map(VERDICTS)
    disc = s["DISCOUNT_COST"].where(s["DISCOUNT_COST"] >= 1)
    s["FUNDED_SHARE"] = s["TRADE_FUNDING_EST"] / disc
    s["BREAK_EVEN_SHARE"] = (-pre / disc).clip(lower=0)
    lo, hi = s["NET_GP_IMPACT_PRE_LO90"], s["NET_GP_IMPACT_PRE_HI90"]
    s["EVIDENCE"] = np.where((lo > 0) | (hi < 0), "clear", "uncertain")

    def reason(r) -> str:
        d = f"{r.MEAN_DEPTH * 100:.0f}% off"
        if r.CLASS == "Winner":
            return f"Profitable even if Dominick's pays the discount ({money(r.NET_GP_IMPACT_PRE)}); lift {r.LIFT_PCT:+.0f}% at {d}."
        if r.CLASS == "Funded win":
            need = "—" if pd.isna(r.BREAK_EVEN_SHARE) else f"{r.BREAK_EVEN_SHARE:.0%}"
            got = "—" if pd.isna(r.FUNDED_SHARE) else f"{r.FUNDED_SHARE:.0%}"
            return f"Profitable only with manufacturer funding: needs ≥ {need} of the discount funded (estimated {got})."
        if r.CLASS == "Volume trap":
            return (f"Sales up {r.LIFT_PCT:+.0f}% but gross profit {money(r.NET_GP_IMPACT_PRE)} if Dominick's pays; "
                    f"{d} gave away {money(r.DISCOUNT_COST)}.")
        return f"No clear lift ({r.LIFT_PCT:+.0f}%) and gross profit {money(r.NET_GP_IMPACT_PRE)} at {d}."

    s["REASON"] = [reason(r) for r in s.itertuples(index=False)]
    return s


def money(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.2f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:,.1f}k"
    return f"{sign}${a:,.0f}"
