"""
PromoGuard AI — tools for the AI Promotion Analyst.

Location: src/agent/tools.py

Every tool reads the project's computed results (Phases 6-9) and returns plain
JSON-able dicts. The language model never invents numbers: it can only quote what
these functions return. Each result carries a "provenance" field
(observed / causal estimate / model estimate / simulation) so the answer can say
what kind of number it is.

The same functions are used by the evaluation script (src/agent/evaluate_agent.py)
to compute the ground truth for the test questions.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "dashboard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data as D  # noqa: E402  (dashboard/data.py)
from src.features.event_features import LABELS  # noqa: E402
from src.simulation.whatif import PromoSimulator  # noqa: E402

DEPTH_ORDER = ["<5%", "5-10%", "10-20%", ">=20%"]
GROUPS = {"brand": "BRAND_GROUP", "depth": "DEPTH_BUCKET", "season": "LENT_LABEL", "year": "YEAR",
          "promo_type": "PROMO_KIND"}


# ----------------------------------------------------------------------------
# Cached data
# ----------------------------------------------------------------------------
# Two product categories. Tools take category="tuna" (default) or "soft_drinks".
CATEGORY_CODES = {"tuna": "tna", "soft_drinks": "sdr"}
CATEGORY_TEXT = {
    "tna": {"scope": "canned tuna, Dominick's Finer Foods (Chicago), 1989-1997; chain events in >= 20 stores",
            "unit": "tuna", "max_discount": 40, "scale": 2.17,
            "names": "Descriptions are Dominick's abbreviations, e.g. 'STAR K CH LT TUNA WA' = StarKist chunk light "
                     "tuna in water; 'C O S' = Chicken of the Sea; 'BUM BEE' = Bumble Bee; 'HH' = Dominick's brand."},
    "sdr": {"scope": "carbonated soft drinks, Dominick's Finer Foods (Chicago), 1989-1997; chain events in >= 20 "
                     "stores; brand groups: Coca-Cola Co. (Coke, Diet Coke, Sprite...), PepsiCo (Pepsi, Mountain "
                     "Dew...), 7-Up, Dominick's, Other",
            "unit": "soft drinks", "max_discount": 50, "scale": 5.68,
            "names": "Descriptions are Dominick's abbreviations, e.g. 'PEPSI COLA N/R' = Pepsi 2-litre bottle, "
                     "'COKE CLASSIC CANS' = Coca-Cola 12-pack/24-pack cans; SIZE '24/12O' = 24 cans x 12 oz, "
                     "'2 LT' = 2 litres; 'DOM' = Dominick's store brand."},
}
AGENT_CAT = "tna"      # default category when a question does not name one


def _code(category: str | None) -> str:
    c = (category or "tuna").strip().lower().replace(" ", "_").replace("-", "_")
    c = {"soda": "soft_drinks", "soft_drink": "soft_drinks", "csd": "soft_drinks", "sdr": "soft_drinks",
         "tna": "tuna", "canned_tuna": "tuna"}.get(c, c)
    if c not in CATEGORY_CODES:
        raise ValueError(f"category must be one of {list(CATEGORY_CODES)}")
    return CATEGORY_CODES[c]


@lru_cache(maxsize=4)
def _events(cat: str = AGENT_CAT) -> pd.DataFrame:
    ev = D.load_events(cat=cat)
    ev["YEAR"] = ev["START_DATE"].dt.year
    return ev


@lru_cache(maxsize=4)
def _scope(cat: str = AGENT_CAT) -> pd.DataFrame:
    return _events(cat)[_events(cat)["IN_SCOPE"]]


@lru_cache(maxsize=4)
def _card(cat: str = AGENT_CAT) -> pd.DataFrame:
    return D.classify_events(_events(cat))


@lru_cache(maxsize=4)
def _sim(cat: str = AGENT_CAT) -> PromoSimulator:
    return PromoSimulator(D.load_sim_bundle(cat=cat))


@lru_cache(maxsize=4)
def _shap(cat: str = AGENT_CAT) -> pd.DataFrame:
    return D.load_shap(cat=cat)


def _r(x, nd: int = 2):
    """Round numbers for compact JSON; keep None for missing."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), nd)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _records(df: pd.DataFrame, nd: int = 2) -> list[dict]:
    return [{k: _r(v, nd) for k, v in row.items()} for row in df.to_dict("records")]


def _event_summary(e: pd.Series) -> dict:
    return {
        "event_id": int(e["EVENT_ID"]), "product_id": int(e["PRODUCT_ID"]), "product": e["DESCRIP"],
        "brand": e["BRAND"], "size": e["SIZE"], "start_date": e["START_DATE"].strftime("%Y-%m-%d"),
        "weeks": int(e["DURATION_WEEKS"]), "stores": int(e["STORES"]), "mean_discount_pct": _r(e["MEAN_DEPTH"] * 100, 1),
        "promo_type": e["PROMO_KIND"], "lent": e["LENT_LABEL"] == "Lent",
        "lift_pct": _r(e["LIFT_PCT"], 1), "lift_90pct_range": [_r(e["LIFT_PCT_LO90"], 1), _r(e["LIFT_PCT_HI90"], 1)],
        "lift_significant": bool(e["LIFT_SIGNIFICANT"]),
        "actual_units": _r(e["PROMO_ACTUAL"], 0), "expected_units_without_promo": _r(e["PROMO_CF"], 0),
        "extra_units": _r(e["GROSS_LIFT_UNITS"], 0), "cannibalized_units": _r(e["CANNIBALIZED_UNITS"], 0),
        "discount_cost_usd": _r(e["DISCOUNT_COST"], 0),
        "net_gross_profit_retailer_pays_usd": _r(e["NET_GP_IMPACT_PRE"], 0),
        "net_gross_profit_recorded_cost_usd": _r(e["NET_GP_IMPACT_REC"], 0),
        "estimated_supplier_funding_usd": _r(e["TRADE_FUNDING_EST"], 0),
    }


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
def get_overview(category: str = "tuna") -> dict:
    """Headline results over all in-scope promotion events."""
    c = _code(category)
    k = D.kpis(_scope(c))
    return {
        "provenance": "causal estimate (counterfactual sales model + cannibalization IV), under stated assumptions",
        "category": category, "scope": CATEGORY_TEXT[c]["scope"],
        "events": k["events"], "products": k["products"],
        "median_event_lift_pct": _r(k["median_lift_pct"], 1), "pooled_lift_pct": _r(k["pooled_lift_pct"], 1),
        "extra_units_net_of_post_weeks": _r(k["incremental_units"], 0),
        "cannibalization_pct_of_extra_units": _r(k["cannibalization_pct"], 1),
        "net_gross_profit_retailer_pays_usd": _r(k["net_gp_pre"], 0),
        "net_gross_profit_recorded_cost_usd": _r(k["net_gp_rec"], 0),
        "estimated_supplier_funding_usd": _r(k["trade_funding"], 0),
        "discount_cost_usd": _r(k["discount_cost"], 0),
        "share_events_profitable_retailer_pays_pct": _r(k["share_profitable_pre"], 1),
        "share_events_significant_lift_pct": _r(k["share_significant"], 1),
        "note": "Gross profit = margin on average acquisition cost, not net profit.",
    }


def compare_groups(group_by: str, category: str = "tuna") -> dict:
    """Pooled lift and gross profit by brand, discount depth, season (Lent), year or promotion type."""
    if group_by not in GROUPS:
        return {"error": f"group_by must be one of {list(GROUPS)}"}
    t = D.pooled_by(_scope(_code(category)), GROUPS[group_by])
    if group_by == "depth":
        t = t.reindex(DEPTH_ORDER).dropna(subset=["events"])
    t = t.reset_index().rename(columns={GROUPS[group_by]: group_by})
    return {"provenance": "causal estimate under stated assumptions", "category": category, "group_by": group_by,
            "columns_explained": {
                "lift_pct": "pooled extra sales vs. no promotion",
                "net_gp_pre": "net gross-profit impact if the retailer pays the whole discount (USD)",
                "net_gp_recorded": "net gross-profit impact with the recorded (supplier-funded) cost (USD)",
                "gp_per_discount_dollar_pre": "gross profit returned per $1 of discount, retailer pays"},
            "rows": _records(t)}


def list_products(search: str = "", brand: str = "", category: str = "tuna") -> dict:
    """Products that can be simulated (active at the end of the data) with their ids and prices."""
    c = _code(category)
    p = _sim(c).product_list()
    if brand:
        p = p[p["BRAND"].str.contains(brand, case=False, na=False)]
    if search:
        toks = [t for t in search.lower().replace("-", " ").split() if t]
        txt = (p["DESCRIP"] + " " + p["BRAND"] + " " + p["SIZE"].astype(str)).str.lower()
        score = sum(txt.str.contains(t, regex=False).astype(int) for t in toks)
        p = p.assign(_s=score)
        p = p[p["_s"] > 0].sort_values("_s", ascending=False).drop(columns="_s")
    p = p.rename(columns={"PRODUCT_ID": "product_id", "DESCRIP": "description", "BRAND": "brand", "SIZE": "size",
                          "REG_PRICE": "regular_price_usd", "UNIT_COST": "unit_cost_usd"})
    return {"category": category, "products": _records(p.head(25), 2), "count": int(len(p)),
            "note": CATEGORY_TEXT[c]["names"]}


def find_events(brand: str = "", product_search: str = "", year: int | None = None, sort_by: str = "profit",
                ascending: bool = False, limit: int = 5, category: str = "tuna") -> dict:
    """Search past promotion events; sort by 'profit' (retailer pays), 'lift', 'discount' or 'date'."""
    s = _scope(_code(category))
    if brand:
        s = s[s["BRAND"].str.contains(brand, case=False, na=False) |
              s["BRAND_GROUP"].str.contains(brand, case=False, na=False)]
    if product_search:
        for t in product_search.lower().split():
            s = s[s["DESCRIP"].str.lower().str.contains(t, regex=False)]
    if year:
        s = s[s["YEAR"].eq(int(year))]
    key = {"profit": "NET_GP_IMPACT_PRE", "lift": "LIFT_PCT", "discount": "MEAN_DEPTH", "date": "START_DATE"}.get(
        sort_by, "NET_GP_IMPACT_PRE")
    s = s.sort_values(key, ascending=ascending).head(max(1, min(int(limit), 15)))
    return {"provenance": "causal estimate under stated assumptions", "category": category, "matches": int(len(s)),
            "events": [_event_summary(e) for _, e in s.iterrows()]}


def get_event(event_id: int, category: str = "tuna") -> dict:
    """Full detail of one past event: lift, cannibalization, profit, report-card verdict and top drivers."""
    cat = _code(category)
    c = _card(cat)
    row = c[c["EVENT_ID"].eq(int(event_id))]
    if row.empty:
        return {"error": f"event {event_id} not found among in-scope events"}
    e = row.iloc[0]
    out = _event_summary(e)
    out.update({"provenance": "causal estimate under stated assumptions; drivers are model explanations (SHAP)",
                "report_card_group": e["CLASS"], "verdict": e["VERDICT"], "reason": e["REASON"],
                "evidence": e["EVIDENCE"]})
    sh = D.shap_long(_shap(cat), int(event_id)).head(5)
    out["top_drivers_of_lift"] = [{"factor": LABELS.get(f, f), "multiplier": _r(np.exp(v), 2)}
                                  for f, v in zip(sh["feature"], sh["shap"])]
    return out


def report_card_summary(group_by: str = "", category: str = "tuna") -> dict:
    """Counts of Winner / Funded win / Volume trap / Loser, overall or by brand or depth."""
    c = _card(_code(category))
    if group_by in ("brand", "depth"):
        col = GROUPS[group_by]
        m = (pd.crosstab(c[col], c["CLASS"], normalize="index") * 100).reindex(columns=D.CLASSES, fill_value=0)
        if group_by == "depth":
            m = m.reindex([d for d in DEPTH_ORDER if d in m.index])
        return {"provenance": "causal estimate", "unit": "% of events in each row",
                "rows": _records(m.reset_index().rename(columns={col: group_by}), 1)}
    g = c.groupby("CLASS").agg(events=("EVENT_ID", "size"), net_gp_retailer_pays_usd=("NET_GP_IMPACT_PRE", "sum"),
                               net_gp_recorded_usd=("NET_GP_IMPACT_REC", "sum")).reindex(D.CLASSES)
    g["share_pct"] = g["events"] / g["events"].sum() * 100
    return {"provenance": "causal estimate", "definitions": {
        "Winner": "profitable even when the retailer pays the whole discount",
        "Funded win": "profitable only thanks to supplier funding",
        "Volume trap": "significant extra sales but not profitable",
        "Loser": "no clear lift and not profitable"}, "verdicts": D.VERDICTS,
        "rows": _records(g.reset_index().rename(columns={"CLASS": "group"}), 0)}


def cannibalization(category: str = "tuna") -> dict:
    """How many extra units were taken from other products of the category, overall and by promoted brand."""
    c = _code(category)
    s = _scope(c)
    gross, can = s["GROSS_LIFT_UNITS"].clip(lower=0).sum(), s["CANNIBALIZED_UNITS"].sum()
    by_brand, mat = D.load_diversion(cat=c)
    b = by_brand.copy()
    b["lost_per_100_extra_units"] = -b["coef"] * 100
    b["ci90_low"], b["ci90_high"] = -b["ci90_hi"] * 100, -b["ci90_lo"] * 100
    return {"provenance": "causal estimate (store x week fixed effects + instrumental variables)",
            "category": category, "extra_units": _r(gross, 0), "cannibalized_units": _r(can, 0),
            "cannibalization_pct": _r(can / gross * 100, 1), "net_new_units": _r(gross - can, 0),
            "by_promoted_brand_note": f"units of NON-promoted {CATEGORY_TEXT[c]['unit']} lost per 100 extra promoted "
                                      f"units (lower bound; the full-assortment figure is about "
                                      f"{CATEGORY_TEXT[c]['scale']}x larger); a CI that includes 0 = not significant",
            "by_promoted_brand": _records(b[["lost_per_100_extra_units", "ci90_low", "ci90_high"]]
                                          .reset_index().rename(columns={"index": "brand"}), 2)}


def _product(product_id: int, cat: str) -> tuple[int | None, str | None]:
    sim = _sim(cat)
    if int(product_id) in sim.products.index:
        return int(product_id), None
    return None, (f"product_id {product_id} is not an active product of this category; call list_products "
                  "(with the same category) to find valid ids")


def _clip_disc(discount_pct: float, cat: str) -> float:
    return max(0.0, min(float(discount_pct), float(CATEGORY_TEXT[cat]["max_discount"]))) / 100


def simulate_promotion(product_id: int, discount_pct: float, weeks: int = 2, lent: bool = False,
                       supplier_share_pct: float = 0.0, category: str = "tuna") -> dict:
    """Simulate one promotion design for one product (model estimate, not a guarantee)."""
    c = _code(category)
    pid, err = _product(product_id, c)
    if err:
        return {"error": err}
    d = _clip_disc(discount_pct, c)
    r = _sim(c).simulate(pid, d, duration=int(max(1, min(weeks, 4))), lent=bool(lent),
                         trade_funding=max(0.0, min(float(supplier_share_pct), 100.0)) / 100)
    return {"provenance": f"simulation from the promotion-response model (trained on {len(_scope(c)):,} past events)",
            "category": category, "product": r["description"], "discount_pct": _r(d * 100, 0), "weeks": r["duration"], "lent": r["lent"],
            "supplier_share_pct": _r(r["trade_funding"] * 100, 0),
            "regular_price_usd": _r(r["regular_price"]), "promo_price_usd": _r(r["promo_price"]),
            "expected_units": _r(r["units"], 0), "units_without_promo": _r(r["cf_units"], 0),
            "lift_pct": _r(r["lift_pct"], 1), "cannibalized_units": _r(r["cannibalized_units"], 0),
            "net_gross_profit_usd": _r(r["net_gp_impact"], 0),
            "net_gross_profit_80pct_range_usd": [_r(r["net_gp_impact_low"], 0), _r(r["net_gp_impact_high"], 0)],
            "gp_per_discount_dollar": _r(r["gp_per_discount_dollar"]),
            "extrapolation_warning": r["extrapolation"]}


def best_discount(product_id: int, weeks: int = 2, lent: bool = False, supplier_share_pct: float = 0.0,
                  category: str = "tuna") -> dict:
    """Compare no promotion and 0-40% discounts for a product; return the most profitable option."""
    c = _code(category)
    pid, err = _product(product_id, c)
    if err:
        return {"error": err}
    t = _sim(c).scenario_table(pid, duration=int(max(1, min(weeks, 4))), lent=bool(lent),
                              trade_funding=max(0.0, min(float(supplier_share_pct), 100.0)) / 100)
    best = t.iloc[1:].loc[t.iloc[1:]["net_gp_impact"].idxmax()]
    cols = ["scenario", "promo_price", "units", "lift_pct", "cannibalized_units", "net_gp_impact",
            "gp_per_discount_dollar", "extrapolation"]
    return {"provenance": "simulation from the promotion-response model", "category": category,
            "product": _sim(c).products.loc[pid, "DESCRIP"],
            "best_option": best["scenario"] if best["net_gp_impact"] > 0 else "no promotion (no option is profitable)",
            "best_net_gross_profit_usd": _r(best["net_gp_impact"], 0),
            "scenarios": _records(t[cols], 2)}


def break_even_funding(product_id: int, discount_pct: float, weeks: int = 2, lent: bool = False,
                       category: str = "tuna") -> dict:
    """Minimum share of the discount the supplier must fund so the retailer breaks even."""
    c = _code(category)
    pid, err = _product(product_id, c)
    if err:
        return {"error": err}
    d = _clip_disc(discount_pct, c)
    if d == 0:
        return {"error": "a promotion without a price cut has no discount to fund"}
    b = _sim(c).break_even(pid, d, duration=int(max(1, min(weeks, 4))), lent=bool(lent))
    return {"provenance": "simulation", "category": category, "product": b["description"], "discount_pct": _r(d * 100, 0),
            "net_gross_profit_retailer_pays_usd": _r(b["net_gp_retailer_pays"], 0),
            "discount_cost_usd": _r(b["discount_cost"], 0),
            "supplier_share_needed_pct": _r(b["break_even_share"] * 100, 0),
            "can_break_even": b["reachable"], "extrapolation_warning": b["extrapolation"]}


def lift_drivers(category: str = "tuna") -> dict:
    """Global importance of the factors that drive promotion lift (SHAP)."""
    g = D.shap_global(_shap(_code(category)))
    return {"provenance": "model explanation (SHAP on the promotion-response model) — associations, not causes",
            "category": category,
            "importance": [{"factor": LABELS.get(k, k), "mean_abs_shap": _r(v, 3)} for k, v in g.head(10).items()]}


def customer_groups() -> dict:
    """Store customer groups (from census demographics) and how strongly each reacts to discounts."""
    gd = D.load_groups(cat=AGENT_CAT)
    if gd is None:
        return {"error": "customer groups not computed: run src/analysis/10_customer_groups.py"}
    prof = gd["profiles"].reset_index()
    resp = gd["response"]
    r = resp[resp["DEPTH"].round(2).isin([0.10, 0.15, 0.25])].copy()
    r["extra_sales_vs_average_pct"] = (r["MULTIPLIER"] - 1) * 100
    r["range90_pct"] = [[_r((lo - 1) * 100, 1), _r((hi - 1) * 100, 1)] for lo, hi in zip(r["MULT_LO90"], r["MULT_HI90"])]
    plan = gd["plan"]
    per_store = plan[plan["BEST_GP"] > 0].assign(v=lambda t: t["BEST_GP"] / t["STORES"]).groupby("GROUP")["v"].max()
    return {"provenance": "causal estimate of promotion response (within-promotion comparison) + census "
                          "demographics; store groups, not individual customers; ethnicity not used",
            "profiles": _records(prof.rename(columns={"income": "median_income_usd", "educ": "share_college",
                                                      "age60": "share_age60plus", "hsizeavg": "household_size",
                                                      "poverty": "share_income_under_15k", "workwom": "share_working_women",
                                                      "shpavid": "share_avid_shoppers"}), 3),
            "response_vs_average_store": _records(r[["GROUP", "DEPTH", "extra_sales_vs_average_pct", "range90_pct",
                                                     "SIGNIFICANT"]].rename(columns={"GROUP": "group",
                                                                                     "DEPTH": "discount"}), 3),
            "best_promotion_gross_profit_per_store_usd": {k: _r(v, 0) for k, v in per_store.items()},
            "key_finding": "best discount is about 15% in every group; personalization matters most for WHERE "
                           "(profit per store) and WHAT (which products), not HOW MUCH"}


def personalized_plan(group: str, product_id: int | None = None, top: int = 5) -> dict:
    """Recommended products and discount for one customer group (2-week promotion, retailer pays)."""
    gd = D.load_groups(cat=AGENT_CAT)
    if gd is None:
        return {"error": "customer groups not computed: run src/analysis/10_customer_groups.py"}
    plan = gd["plan"]
    names = sorted(plan["GROUP"].unique())
    match = [g for g in names if group.lower() in g.lower()] or [g for g in names if
                                                                 any(w in g.lower() for w in group.lower().split())]
    if not match:
        return {"error": f"unknown group; choose one of {names}"}
    p = plan[plan["GROUP"].eq(match[0])].copy()
    if product_id:
        p = p[p["PRODUCT_ID"].eq(int(product_id))]
    p = p.sort_values("BEST_GP", ascending=False).head(max(1, min(int(top), 15)))
    p["best_discount_pct"] = p["BEST_DEPTH"] * 100
    p["chain_wide_best_pct"] = p["UNIFORM_DEPTH"] * 100
    p["gp_per_store_usd"] = p["BEST_GP"] / p["STORES"]
    return {"provenance": "simulation (Phase 9 product response x group sensitivity, group price and volume)",
            "group": match[0], "products": _records(p[["PRODUCT_ID", "DESCRIP", "BRAND", "SIZE", "STORES",
                                                       "REG_PRICE", "best_discount_pct", "BEST_GP",
                                                       "gp_per_store_usd", "chain_wide_best_pct", "EXTRAPOLATION"]]
                                                    .rename(columns={"BEST_GP": "gross_profit_2wk_usd"}), 2),
            "note": "0% best discount = better not to promote this product in this group"}


def compare_categories() -> dict:
    """Side-by-side headline results for canned tuna and soft drinks (same pipeline, same tests)."""
    out = {"provenance": "causal estimates under stated assumptions (same method for both categories)"}
    for name, c in CATEGORY_CODES.items():
        try:
            s = _scope(c)
        except FileNotFoundError:
            out[name] = {"error": "results not computed for this category"}
            continue
        k = D.kpis(s)
        bd = D.pooled_by(s, "DEPTH_BUCKET").reindex(DEPTH_ORDER).dropna(subset=["events"])
        out[name] = {
            "events": k["events"], "products": k["products"],
            "median_event_lift_pct": _r(k["median_lift_pct"], 1), "pooled_lift_pct": _r(k["pooled_lift_pct"], 1),
            "cannibalization_pct_of_extra_units": _r(k["cannibalization_pct"], 1),
            "net_gross_profit_retailer_pays_usd": _r(k["net_gp_pre"], 0),
            "net_gross_profit_recorded_cost_usd": _r(k["net_gp_rec"], 0),
            "gp_per_discount_dollar_retailer_pays": _r(k["net_gp_pre"] / k["discount_cost"]),
            "share_events_profitable_retailer_pays_pct": _r(k["share_profitable_pre"], 1),
            "by_depth": _records(bd[["events", "lift_pct", "gp_per_discount_dollar_pre"]].reset_index()
                                 .rename(columns={"DEPTH_BUCKET": "depth"}), 2)}
    out["note"] = ("Same tool, different advice: compare how strongly sales react and whether promotions pay. "
                   "Gross profit, not net profit.")
    return out


def method_and_limitations(category: str = "tuna") -> dict:
    """How the numbers were produced, the key assumptions and the limitations."""
    if _code(category) == "sdr":
        return {
            "category": category,
            "pipeline": "same 9 phases as tuna (clean data, define promotions, baseline Random Forest, cross-fitted "
                        "counterfactual, lift, cannibalization IV, gross profit, response model + SHAP, simulator)",
            "scope": "carbonated soft drinks only (waters, teas, juices and mixers excluded)",
            "validation": {"baseline_error_unseen_weeks_pct": 21, "pre_period_error_pct": 0.3,
                           "lead_test": "future promotions have no effect on today's sales"},
            "assumptions": ["no hidden demand shock during promotions",
                            "other soft drinks react only through the promoted product",
                            "cost = recorded average acquisition cost; supplier funding is inferred"],
            "limitations": ["Jul-Sep 1994 soft-drink data missing", "gross profit, not net profit",
                            "soft drinks may bring shoppers who buy other things: that basket effect is not measured",
                            "simulations beyond 45% discount are extrapolations",
                            "store-level data: no individual customers"]}
    return {
        "pipeline": ["clean scanner data", "define promotions (price >= 5% below the 12-week modal price, or a deal code)",
                     "baseline model of normal sales (Random Forest, validated on unseen years)",
                     "cross-fitted counterfactual sales", "incremental sales (lift)",
                     "cannibalization (store and week fixed effects + instrumental variables)",
                     "gross-profit impact in two cost views", "promotion-response model + SHAP", "what-if simulator"],
        "category": "tuna",
        "validation": {"baseline_error_unseen_year_pct": 13.3, "pre_period_error_pct": 4,
                       "lead_test": "future promotions have no effect on today's sales"},
        "assumptions": ["no hidden demand shock during promotions", "other tuna react only through the promoted product",
                        "cost = recorded average acquisition cost; supplier funding is inferred"],
        "limitations": ["stock-outs cannot be seen", "hidden chain promotions excluded from training",
                        "Jan-Oct 1996 tuna data missing", "gross profit, not net profit",
                        "simulations beyond 30% discount are extrapolations",
                        "store-level data: no individual customers"]}


# ----------------------------------------------------------------------------
# Registry with JSON schemas (Anthropic tool-use format)
# ----------------------------------------------------------------------------
def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


PID = {"type": "integer", "description": "product_id from list_products"}
WEEKS = {"type": "integer", "minimum": 1, "maximum": 4, "description": "promotion length in weeks (default 2)"}
LENT = {"type": "boolean", "description": "promotion during Lent (default false)"}
SHARE = {"type": "number", "minimum": 0, "maximum": 100, "description": "% of the discount paid by the supplier (default 0)"}
DISC = {"type": "number", "minimum": 0, "maximum": 50,
        "description": "discount in percent, e.g. 15 (max 40 for tuna, 50 for soft drinks)"}
CAT = {"type": "string", "enum": ["tuna", "soft_drinks"],
       "description": "product category: 'tuna' (canned tuna, default) or 'soft_drinks' (carbonated soft drinks)"}


def _with_cat(schema: dict) -> dict:
    schema["properties"]["category"] = CAT
    return schema

TOOLS = {
    "get_overview": (get_overview, "Headline results of all promotion events of one category: lift, "
                     "cannibalization, profit in both cost views, supplier funding.", _with_cat(_schema())),
    "compare_groups": (compare_groups, "Compare pooled lift and gross profit by brand, depth (discount size), "
                       "season (Lent vs not), year or promo_type.",
                       _with_cat(_schema({"group_by": {"type": "string", "enum": list(GROUPS)}}, ["group_by"]))),
    "list_products": (list_products, "Find active products (ids, names, regular price, unit cost). Use before any "
                      "simulation (same category).",
                      _with_cat(_schema({"search": {"type": "string"}, "brand": {"type": "string"}}))),
    "find_events": (find_events, "Search past promotion events and sort them by profit, lift, discount or date.",
                    _with_cat(_schema({"brand": {"type": "string"}, "product_search": {"type": "string"},
                                       "year": {"type": "integer"},
                                       "sort_by": {"type": "string", "enum": ["profit", "lift", "discount", "date"]},
                                       "ascending": {"type": "boolean", "description": "true = worst/smallest first"},
                                       "limit": {"type": "integer", "minimum": 1, "maximum": 15}}))),
    "get_event": (get_event, "Full detail of one past event, its report-card verdict and its top lift drivers.",
                  _with_cat(_schema({"event_id": {"type": "integer"}}, ["event_id"]))),
    "report_card_summary": (report_card_summary, "Winner / Funded win / Volume trap / Loser counts and profit, "
                            "overall or by brand or depth.",
                            _with_cat(_schema({"group_by": {"type": "string", "enum": ["", "brand", "depth"]}}))),
    "cannibalization": (cannibalization, "How many extra units were taken from other products of the same "
                        "category, overall and by brand.", _with_cat(_schema())),
    "simulate_promotion": (simulate_promotion, "Simulate one future promotion for one product.",
                           _with_cat(_schema({"product_id": PID, "discount_pct": DISC, "weeks": WEEKS, "lent": LENT,
                                              "supplier_share_pct": SHARE}, ["product_id", "discount_pct"]))),
    "best_discount": (best_discount, "Test no promotion and 0-40% discounts for one product and return the most "
                      "profitable option.", _with_cat(_schema({"product_id": PID, "weeks": WEEKS, "lent": LENT,
                                                               "supplier_share_pct": SHARE}, ["product_id"]))),
    "break_even_funding": (break_even_funding, "Minimum % of the discount the supplier must pay so the retailer "
                           "does not lose gross profit.",
                           _with_cat(_schema({"product_id": PID, "discount_pct": DISC, "weeks": WEEKS, "lent": LENT},
                                             ["product_id", "discount_pct"]))),
    "customer_groups": (customer_groups, "CANNED TUNA ONLY. The 4 store customer groups (income, age, family "
                        "size…), how strongly each reacts to discounts, and where promotions pay most per store.",
                        _schema()),
    "personalized_plan": (personalized_plan, "CANNED TUNA ONLY. Recommended products and discount for one customer "
                          "group, e.g. "
                          "'Young families', 'Affluent professionals', 'Older, budget-conscious', 'Urban low-income'.",
                          _schema({"group": {"type": "string"}, "product_id": {"type": "integer"},
                                   "top": {"type": "integer", "minimum": 1, "maximum": 15}}, ["group"])),
    "lift_drivers": (lift_drivers, "Which factors drive promotion lift most (SHAP importance).", _with_cat(_schema())),
    "compare_categories": (compare_categories, "Side-by-side results of canned tuna vs soft drinks (lift, "
                           "cannibalization, profit, by discount depth). Use for any cross-category question.",
                           _schema()),
    "method_and_limitations": (method_and_limitations, "Method, validation, assumptions and limitations.",
                               _with_cat(_schema())),
}


def anthropic_tools() -> list[dict]:
    return [{"name": n, "description": d, "input_schema": s} for n, (_, d, s) in TOOLS.items()]


def run_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOLS[name][0](**(args or {}))
    except Exception as exc:  # noqa: BLE001 — errors go back to the model, not to the user
        return {"error": f"{type(exc).__name__}: {exc}"}
