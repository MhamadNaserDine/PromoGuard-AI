"""
PromoGuard AI — Decision dashboard (Streamlit)
==============================================

Location: PromoGuard-AI/dashboard/app.py
Run from the repo root (with the virtual environment active):
    streamlit run dashboard/app.py

Needs the outputs of Phases 6-9 (see dashboard/data.py: REQUIRED_FILES).
Look & feel lives in dashboard/theme.py; data logic in dashboard/data.py.
Every number is labelled as OBSERVED data, a MODEL estimate, a CAUSAL estimate
(under stated assumptions) or a SIMULATION.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data as D  # noqa: E402
import theme as T  # noqa: E402
from src.features.event_features import LABELS  # noqa: E402
from src.simulation.whatif import PromoSimulator  # noqa: E402

st.set_page_config(page_title="PromoGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

BLUE, ORANGE, AQUA, YELLOW, RED, GREY, INK_2 = T.BLUE, T.ORANGE, T.AQUA, T.YELLOW, T.RED, T.GREY, T.INK_2
DEPTH_ORDER = ["<5%", "5-10%", "10-20%", ">=20%"]
CC = st.column_config


# ----------------------------------------------------------------------------
# Data (cached)
# ----------------------------------------------------------------------------
# Every loader is cached per product category (the sidebar switch sets D.CAT).
@st.cache_data(show_spinner="Loading promotion events…")
def _events(cat: str) -> pd.DataFrame:
    ev = D.load_events(cat)
    ev["YEAR"] = ev["START_DATE"].dt.year
    return ev


def events() -> pd.DataFrame:
    return _events(D.CAT)


@st.cache_data(show_spinner="Loading weekly sales and counterfactuals…")
def _rows(cat: str) -> pd.DataFrame:
    return D.load_rows(cat)


def rows() -> pd.DataFrame:
    return _rows(D.CAT)


@st.cache_data
def stores() -> pd.DataFrame:
    return D.load_stores()


@st.cache_data
def _shap(cat: str) -> pd.DataFrame:
    return D.load_shap(cat)


def shap() -> pd.DataFrame:
    return _shap(D.CAT)


@st.cache_resource(show_spinner="Loading the promotion-response model…")
def _simulator(cat: str) -> PromoSimulator:
    return PromoSimulator(D.load_sim_bundle(cat))


def simulator() -> PromoSimulator:
    return _simulator(D.CAT)


@st.cache_data
def simulator_bundle_cv() -> pd.DataFrame:
    return D.load_sim_bundle(D.CAT)["cv_results"].round(3)


@st.cache_data
def _diversion(cat: str):
    return D.load_diversion(cat)


def diversion():
    return _diversion(D.CAT)


def U() -> str:
    """Unit word of the selected category: 'tuna' or 'soft drinks'."""
    return D.info()["unit"]


def CATNAME() -> str:
    return D.info()["name"]


def category_switch() -> None:
    """Sidebar switch between product categories; the brand/year filters reset on change."""
    cats = D.available_categories()
    names = {c: D.CATEGORIES[c]["name"] for c in cats}
    if st.session_state.get("cat") not in cats:
        st.session_state["cat"] = "tna"

    def reset_filters():
        for k in ("f_brands", "f_years"):
            st.session_state.pop(k, None)

    T.side_label("Product category")
    st.sidebar.radio("Product category", cats, format_func=names.get, key="cat", horizontal=True,
                     label_visibility="collapsed", on_change=reset_filters)
    if len(cats) == 1:
        st.sidebar.caption("Run the pipeline with --cat sdr to add soft drinks.")
    D.set_category(st.session_state["cat"])


# ----------------------------------------------------------------------------
# Global filters (sidebar) — apply to every event-based page
# ----------------------------------------------------------------------------
def global_filters() -> None:
    ev = events()
    s = ev[ev["IN_SCOPE"]]
    brands = sorted(s["BRAND_GROUP"].dropna().unique().tolist())
    y0, y1 = int(s["YEAR"].min()), int(s["YEAR"].max())
    T.side_label("Filters · all pages")
    if hasattr(st.sidebar, "pills"):
        st.sidebar.pills("Brand", brands, selection_mode="multi", default=brands, key="f_brands")
    else:
        st.sidebar.multiselect("Brand", brands, default=brands, key="f_brands")
    st.sidebar.slider("Promotion start year", y0, y1, (y0, y1), key="f_years")


def scoped() -> pd.DataFrame:
    """In-scope events after the sidebar filters."""
    ev = events()
    s = ev[ev["IN_SCOPE"]]
    brands = st.session_state.get("f_brands")
    years = st.session_state.get("f_years")
    if brands is not None:
        s = s[s["BRAND_GROUP"].isin(brands)]
    if years is not None:
        s = s[s["YEAR"].between(years[0], years[1])]
    return s


def filter_chips(s: pd.DataFrame) -> list[str]:
    all_s = events()[events()["IN_SCOPE"]]
    chips = [f"{len(s):,} of {len(all_s):,} events", f"{s['PRODUCT_ID'].nunique()} products"]
    if len(s):
        chips.append(f"{int(s['YEAR'].min())}–{int(s['YEAR'].max())}")
    n_b = s["BRAND_GROUP"].nunique()
    chips.append("all brands" if n_b == all_s["BRAND_GROUP"].nunique() else f"{n_b} brand group(s)")
    return chips


def no_events() -> None:
    T.empty_state("No promotion events match the filters",
                  "Widen the brand or year selection in the sidebar.")


def cols(spec):
    """st.columns with widgets aligned on their bottom edge (Streamlit >= 1.36)."""
    try:
        return st.columns(spec, vertical_alignment="bottom")
    except TypeError:
        return st.columns(spec)


def pct(v: float, signed: bool = True) -> str:
    return "—" if pd.isna(v) else (f"{v:+,.0f}%" if signed else f"{v:,.0f}%")


def tone(v: float) -> str:
    return "muted" if pd.isna(v) else ("good" if v > 0 else "bad")


# ----------------------------------------------------------------------------
# Page 1 — Executive overview
# ----------------------------------------------------------------------------
def recommendations(s: pd.DataFrame, k: dict) -> list[dict]:
    recs = []
    by_depth = D.pooled_by(s, "DEPTH_BUCKET").reindex(DEPTH_ORDER).dropna(subset=["events"])
    if len(by_depth):
        best = by_depth["gp_per_discount_dollar_pre"].idxmax()
        recs.append(dict(
            title="Favour shallow, well-placed discounts", tone="good", tag="Causal estimate",
            body=f"<b>{best}</b> discounts return the most gross profit per $1 of discount "
                 f"(<b>{by_depth.loc[best, 'gp_per_discount_dollar_pre']:+.2f}</b> when the retailer pays). "
                 f"Lift grows with depth, profit does not."))
    if ">=20%" in by_depth.index:
        d = by_depth.loc[">=20%"]
        recs.append(dict(
            title="Run deep discounts only when the manufacturer pays", tone="bad", tag="Causal estimate",
            body=f"20%+ discounts change gross profit by <b>{D.money(d['net_gp_pre'])}</b> if Dominick's pays the "
                 f"discount, but <b>{D.money(d['net_gp_recorded'])}</b> with the recorded trade-deal cost."))
    recs.append(dict(
        title="Treat trade funding as the profit lever", tone="warn", tag="Estimate · funding inferred",
        body=f"Estimated manufacturer funding of <b>{D.money(k['trade_funding'])}</b> explains most of the gap "
             f"between {D.money(k['net_gp_pre'])} (retailer pays) and {D.money(k['net_gp_rec'])} (recorded cost). "
             f"Negotiate funding before committing to depth."))
    by_lent = D.pooled_by(s, "LENT_LABEL")
    if D.CAT == "tna" and {"Lent", "not Lent"} <= set(by_lent.index):
        L, N = by_lent.loc["Lent"], by_lent.loc["not Lent"]
        recs.append(dict(
            title="Lent sells itself — don't over-discount it", tone="neutral", tag="Causal estimate",
            body=f"Lent promotions lift sales more (<b>{L['lift_pct']:+.0f}%</b> vs {N['lift_pct']:+.0f}%), yet return "
                 f"<b>{L['gp_per_discount_dollar_pre']:+.2f}</b> per $1 of discount vs "
                 f"{N['gp_per_discount_dollar_pre']:+.2f} outside Lent (retailer pays)."))
    if D.CAT != "tna":
        kind = D.pooled_by(s, "PROMO_KIND")
        if {"price promotion", "deal code, no price cut"} <= set(kind.index):
            dl, pr = kind.loc["deal code, no price cut"], kind.loc["price promotion"]
            recs.append(dict(
                title="Deals without a price cut lose least", tone="neutral", tag="Causal estimate",
                body=f"Deal-coded weeks with no price cut return <b>{dl['gp_per_discount_dollar_pre']:+.2f}</b> per $1 "
                     f"of discount vs <b>{pr['gp_per_discount_dollar_pre']:+.2f}</b> for price cuts (retailer pays), "
                     f"while still lifting sales {dl['lift_pct']:+.0f}%."))
    if pd.notna(k["cannibalization_pct"]):
        recs.append(dict(
            title="Cannibalization is a minor cost", tone="good", tag="Causal estimate (IV)",
            body=f"Only <b>{k['cannibalization_pct']:.1f}%</b> of the extra units come from other {U()} in the same "
                 f"store and week; promotions mostly add category volume."))
    return recs


def summary_markdown(s: pd.DataFrame, k: dict, recs: list[dict]) -> str:
    import re
    strip = lambda t: re.sub(r"<[^>]+>", "", t)  # noqa: E731
    lines = [f"# PromoGuard AI — Executive summary ({date.today():%Y-%m-%d})", "",
             f"{CATNAME()} · Dominick's Finer Foods, Chicago · " + " · ".join(filter_chips(s)), "",
             "## Key figures (causal estimates under stated assumptions)",
             f"- Typical event sales lift (median): {k['median_lift_pct']:+.0f}%",
             f"- Share of extra units taken from other {U()}: {k['cannibalization_pct']:.1f}%",
             f"- Net gross-profit impact, retailer pays the discount: {D.money(k['net_gp_pre'])}",
             f"- Net gross-profit impact, recorded cost: {D.money(k['net_gp_rec'])}",
             f"- Estimated manufacturer (trade) funding: {D.money(k['trade_funding'])}",
             f"- Revenue given away as discounts: {D.money(k['discount_cost'])}", "",
             "## Recommendations"]
    lines += [f"{i}. **{r['title']}** — {strip(r['body'])}" for i, r in enumerate(recs, 1)]
    lines += ["", "_Gross profit = margin on average acquisition cost, not net profit. Trade funding is inferred._"]
    return "\n".join(lines)


def page_overview() -> None:
    s = scoped()
    T.hero("Did our promotions create real value?",
           f"Sales lift, cannibalization and gross-profit impact of chain promotions on {CATNAME().lower()} — "
           "Dominick's Finer Foods, Chicago, 1989–1997.", "Executive overview", "causal", filter_chips(s))
    if s.empty:
        return no_events()
    k = D.kpis(s)
    T.kpi_row([
        T.kpi("Typical sales lift", pct(k["median_lift_pct"]), f"pooled <b>{pct(k['pooled_lift_pct'])}</b> across events",
              "good", "Median of actual vs. counterfactual units during the promotion weeks."),
        T.kpi(f"Taken from other {U()}", f"{k['cannibalization_pct']:.1f}%", "of the extra units (cannibalization)",
              "neutral", "Phase 7: two-way fixed effects + instrumental variables."),
        T.kpi("Net GP · retailer pays", D.money(k["net_gp_pre"]), "cost frozen before the promotion",
              tone(k["net_gp_pre"]), "All events, discount fully paid by Dominick's."),
        T.kpi("Net GP · recorded cost", D.money(k["net_gp_rec"]), "includes manufacturer deal funding",
              tone(k["net_gp_rec"]), "Uses the cost recorded in the data, which falls when manufacturers fund deals."),
    ])
    T.kpi_row([
        T.kpi("Extra units", f"{k['incremental_units'] / 1e6:,.2f}M", "net of after-event weeks", "neutral"),
        T.kpi("Discounts given", D.money(k["discount_cost"]), "revenue given away at the shelf", "warn"),
        T.kpi("Trade funding (est.)", D.money(k["trade_funding"]), "paid by manufacturers · inferred", "neutral"),
        T.kpi("Profitable events", f"{k['share_profitable_pre']:.0f}%",
              f"<b>{k['share_significant']:.0f}%</b> have a significant lift", "neutral", "Retailer-pays view."),
    ])

    recs = recommendations(s, k)
    T.section("Recommendations", "What the evidence suggests for the next promotion calendar.")
    T.rec_cards(recs, cols=3 if len(recs) != 4 else 2)

    T.section("Depth decides profit", "Pooled lift and gross profit per $1 of discount, by discount depth.", "causal")
    by_depth = D.pooled_by(s, "DEPTH_BUCKET").reindex(DEPTH_ORDER).dropna(subset=["events"])
    left, right = st.columns(2)
    fig = go.Figure(go.Bar(x=by_depth.index, y=by_depth["lift_pct"], marker_color=ORANGE,
                           text=[pct(v) for v in by_depth["lift_pct"]], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Pooled sales lift (%)")
    T.chart(fig, left, 360, "Sales lift by discount depth")
    fig = go.Figure()
    fig.add_bar(x=by_depth.index, y=by_depth["gp_per_discount_dollar_pre"], name="Retailer pays discount",
                marker_color=BLUE)
    fig.add_bar(x=by_depth.index, y=by_depth["gp_per_discount_dollar_rec"], name="Recorded cost (trade deals)",
                marker_color=AQUA)
    fig.add_hline(y=0, line_color=INK_2)
    fig.update_yaxes(title="$ gross profit per $1 discount")
    T.chart(fig, right, 360, "Gross profit per $1 of discount")

    T.section("Over time", "Net gross-profit impact of the year's promotions under both cost views.", "causal")
    by_year = D.pooled_by(s, "YEAR")
    fig = go.Figure()
    fig.add_bar(x=by_year.index, y=by_year["net_gp_pre"], name="Retailer pays", marker_color=BLUE)
    fig.add_bar(x=by_year.index, y=by_year["net_gp_recorded"], name="Recorded cost", marker_color=AQUA)
    fig.add_scatter(x=by_year.index, y=by_year["events"], name="Events", yaxis="y2", mode="lines+markers",
                    line=dict(color=GREY, width=2, dash="dot"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Events"))
    fig.update_layout(yaxis=dict(title="Net gross-profit impact"))
    T.money_axis(fig)
    fig.update_xaxes(dtick=1)
    T.chart(fig, height=360, title=f"Profit impact by year ({D.info()['gap']})")

    c1, c2 = st.columns([1, 3])
    c1.download_button("⬇ Executive summary (.md)", summary_markdown(s, k, recs).encode("utf-8"),
                       f"promoguard_executive_summary_{D.CAT}.md", "text/markdown")
    c2.caption("Scope: chain events promoted in ≥ 20 stores with good data coverage. Gross profit = margin on "
               "average acquisition cost (not net profit).")


# ----------------------------------------------------------------------------
# Page 2 — Promotion analysis (one event)
# ----------------------------------------------------------------------------
def page_promotion() -> None:
    s = scoped()
    T.hero("Promotion deep-dive", "Pick one chain promotion and compare actual sales with what the model expects "
           "without the promotion.", "Promotion analysis", "causal", filter_chips(s))
    if s.empty:
        return no_events()
    with st.container(border=True):
        c = st.columns([1, 2, 3])
        brand = c[0].selectbox("Brand", ["All"] + sorted(s["BRAND"].dropna().unique().tolist()))
        sub = s if brand == "All" else s[s["BRAND"].eq(brand)]
        prod = c[1].selectbox("Product", sorted(sub["DESCRIP"].unique().tolist()))
        sub = sub[sub["DESCRIP"].eq(prod)].sort_values("START_DATE")
        label = c[2].selectbox("Promotion event", sub["LABEL"].tolist(), index=len(sub) - 1)
    e = sub[sub["LABEL"].eq(label)].iloc[0]

    can = e["CANNIBALIZED_UNITS"]
    T.kpi_row([
        T.kpi("Sales lift", pct(e["LIFT_PCT"]), f"90% range {pct(e['LIFT_PCT_LO90'])} to {pct(e['LIFT_PCT_HI90'])}",
              "good" if e["LIFT_SIGNIFICANT"] else "muted", "We are 90% sure the real lift is inside this range."),
        T.kpi("Extra units", f"{e['GROSS_LIFT_UNITS']:,.0f}", f"{int(e['STORES'])} stores · "
              f"{int(e['DURATION_WEEKS'])} week(s)", "neutral"),
        T.kpi(f"Taken from other {U()}", "—" if pd.isna(can) else f"{can:,.0f}", "cannibalized units", "neutral"),
        T.kpi("Net GP · retailer pays", D.money(e["NET_GP_IMPACT_PRE"]), "", tone(e["NET_GP_IMPACT_PRE"])),
        T.kpi("Net GP · recorded cost", D.money(e["NET_GP_IMPACT_REC"]), "", tone(e["NET_GP_IMPACT_REC"])),
    ])
    sig = "significant lift" if e["LIFT_SIGNIFICANT"] else "lift not distinguishable from noise"
    T.callout(f"<b>{T.esc(e['PROMO_KIND'])}</b> · average depth <b>{e['MEAN_DEPTH'] * 100:.0f}%</b> · "
              f"{'during Lent' if e['LENT_LABEL'] == 'Lent' else 'outside Lent'} · {sig}",
              "good" if e["LIFT_SIGNIFICANT"] else "muted")

    w = D.event_weekly(rows(), e)
    t1, t2, t3 = st.tabs(["Sales vs. counterfactual", "Price", "Store price tiers"])
    fig = go.Figure()
    fig.add_bar(x=w["REL_WEEK"], y=w["actual"], name="Actual",
                marker_color=[ORANGE if v else "#cfcdc8" for v in w["in_event"]])
    fig.add_scatter(x=w["REL_WEEK"], y=w["counterfactual"], name="Without sale", mode="lines+markers",
                    line=dict(color=BLUE, width=3))
    fig.update_xaxes(title="Weeks relative to promotion start", dtick=1)
    fig.update_yaxes(title="Units, all stores")
    T.chart(fig, t1, 400, "Actual vs. counterfactual sales")
    fig = go.Figure()
    fig.add_scatter(x=w["REL_WEEK"], y=w["price"], name="Median shelf price", mode="lines+markers",
                    line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=w["REL_WEEK"], y=w["regular_price"], name="Regular price", mode="lines",
                    line=dict(color=GREY, dash="dash"))
    fig.update_xaxes(title="Weeks relative to promotion start", dtick=1)
    fig.update_yaxes(title="$ per unit")
    T.chart(fig, t2, 360, "Shelf price around the event")
    tier = D.event_by_tier(rows(), stores(), e).reset_index()
    t3.caption("The same chain price is a deeper discount in high-price zones — this variation identifies "
               "cannibalization in Phase 7.")
    T.table(tier, t3, hide_index=True, column_config={
        "PRICE_TIER": "Price tier", "stores": CC.NumberColumn("Stores"),
        "actual": CC.NumberColumn("Actual units", format="%.0f"),
        "counterfactual": CC.NumberColumn("Without promotion", format="%.0f"),
        "lift_pct": CC.NumberColumn("Lift", format="%+.0f%%"),
        "mean_depth_%": CC.NumberColumn("Mean depth", format="%.1f%%")})


# ----------------------------------------------------------------------------
# Page 3 — Cannibalization
# ----------------------------------------------------------------------------
def page_cannibalization() -> None:
    s = scoped()
    by_brand, mat = diversion()
    T.hero("Where do the extra units come from?", "Separating new category sales from units stolen from other "
           f"{U()} on the same shelf.", "Cannibalization", "causal", filter_chips(s))
    if s.empty:
        return no_events()
    gross, can = s["GROSS_LIFT_UNITS"].clip(lower=0).sum(), s["CANNIBALIZED_UNITS"].sum()
    T.kpi_row([
        T.kpi("Extra promoted units", f"{gross / 1e6:,.2f}M", "gross lift of promoted products", "neutral"),
        T.kpi(f"Taken from other {U()}", f"{can / 1e6:,.2f}M", f"<b>{can / gross * 100:.1f}%</b> of the extra units",
              "bad"),
        T.kpi(f"Net new {U()} units", f"{(gross - can) / 1e6:,.2f}M", "category growth for the store", "good"),
    ])
    t1, t2, t3 = st.tabs(["Overview", "Brand-to-brand matrix", "Method"])
    left, right = t1.columns(2)
    fig = go.Figure(go.Bar(x=["Extra units", "Taken from others", "Net new"], y=[gross, -can, gross - can],
                           marker_color=[ORANGE, RED, BLUE],
                           text=[f"{v / 1e6:,.2f}M" for v in [gross, -can, gross - can]], textposition="outside",
                           cliponaxis=False))
    fig.update_yaxes(title="Units")
    T.chart(fig, left, 380, "Where the extra units came from")
    b = by_brand.copy()
    fig = go.Figure(go.Bar(y=b.index, x=-b["coef"] * 100, orientation="h", marker_color=BLUE,
                           error_x=dict(type="data", array=b["se"] * 1.645 * 100, color=INK_2)))
    fig.update_xaxes(title=f"Other {U()} units lost per 100 extra promoted units (90% CI)")
    T.chart(fig, right, 380, "By promoted brand")
    t1.caption("Brand-level diversion uses all events (not affected by the sidebar filters).")
    if mat is not None:
        fig = go.Figure(go.Heatmap(z=mat.values * 100, x=mat.columns, y=mat.index,
                                   colorscale=[[0, RED], [0.5, "#f4f3f0"], [1, BLUE]], zmid=0,
                                   text=np.round(mat.values * 100, 2), texttemplate="%{text}",
                                   colorbar=dict(title="units / 100", thickness=12)))
        fig.update_xaxes(title="Brand losing (−) or gaining (+) units")
        fig.update_yaxes(title="Brand promoted")
        T.chart(fig, t2, 460, "Who loses to whom (per 100 extra promoted units)")
    else:
        T.empty_state("Matrix not available", "Run Phase 7 to create diversion_matrix_brands.csv.", t2)
    if D.CAT == "tna":
        example = "($0.79 → $0.59 = 25%) than in cheap ones ($0.68 → $0.59 = 13%)"
        result = "Result: ≈ 4 of 100 cans"
    else:
        example = "(about 21% off on average in high-price stores vs 18% in cub-fighter stores)"
        result = "Result: ≈ 0 of 100 (not significant)"
    T.rec_cards([
        dict(title="Compare stores in the same week", tone="neutral", tag="Removes busy weeks & big stores",
             body="Holidays and seasons lift <b>all</b> stores; big stores always sell more. Comparing stores "
                  "<b>within the same week</b> removes both."),
        dict(title="Same sale, different discount", tone="neutral", tag="Natural experiment",
             body=f"One chain price is a <b>bigger discount in expensive stores</b> {example}."),
        dict(title=f"Watch the other {U()}", tone="neutral", tag=result,
             body=f"Where the sale felt bigger, do the <b>non-promoted {U()}</b> sell less? The extra drop is the "
                  "cannibalization."),
    ], t3, cols=3)
    T.kpi_row([T.kpi("Future test", "passed", "a sale 2 weeks later has no effect today", "good"),
               T.kpi("Net new units come from", "3 sources", "eating more · stocking up · other shops", "neutral",
                     "Store-level data cannot tell these apart.")], t3)
    with t3.expander("Technical description (for specialists)"):
        st.markdown("Store × week panel with store and week fixed effects (two-way demeaning). The promoted "
                    "products' extra sales are instrumented by the **expected discount volume** (Σ counterfactual "
                    "units × depth, and Σ counterfactual units) — the same chain promotion is a deeper discount in "
                    f"high-price zones. Outcome: sales of **non-promoted ('bystander') {U()}**. 2SLS with standard "
                    "errors clustered by week; lead test on promotions 2 weeks ahead; bystander estimate scaled by "
                    f"{'2.17' if D.CAT == 'tna' else '5.68'} to the whole assortment.")


# ----------------------------------------------------------------------------
# Page 4 — Profit impact
# ----------------------------------------------------------------------------
def page_profit() -> None:
    s = scoped()
    T.hero("From sales lift to gross profit", "Two cost views: the retailer pays the whole discount, or the cost "
           "recorded in the data (which includes manufacturer deal funding).", "Profit impact", "causal",
           filter_chips(s))
    if s.empty:
        return no_events()
    with st.container(border=True):
        c = st.columns([2, 1, 2])
        depths = c[0].multiselect("Discount depth", DEPTH_ORDER, default=DEPTH_ORDER)
        lent = c[1].selectbox("Season", ["All", "Lent", "not Lent"])
        view = c[2].radio("Cost view", ["Retailer pays discount", "Recorded cost"], horizontal=True)
    col = "NET_GP_IMPACT_PRE" if view.startswith("Retailer") else "NET_GP_IMPACT_REC"
    f = s[s["DEPTH_BUCKET"].isin(depths)]
    if lent != "All":
        f = f[f["LENT_LABEL"].eq(lent)]
    if f.empty:
        return T.empty_state("No events match these filters", "Select at least one depth or another season.")
    net, disc = f[col].sum(), f["DISCOUNT_COST"].sum()
    T.kpi_row([
        T.kpi("Events", f"{len(f):,}", f"{f['PRODUCT_ID'].nunique()} products", "neutral"),
        T.kpi("Net gross-profit impact", D.money(net), view.lower(), tone(net)),
        T.kpi("Events that raised GP", f"{(f[col] > 0).mean() * 100:.0f}%", "share of events", "neutral"),
        T.kpi("GP per $1 discount", f"{net / disc:+.2f}" if disc else "—", f"on {D.money(disc)} of discounts",
              tone(net)),
    ])
    t1, t2 = st.tabs(["Profit bridge", "Event table"])
    steps = [("GP lift (promoted)", f["GROSS_GP_LIFT_PRE"].sum()), ("After-event weeks", f["POST_GP_EFFECT"].sum()),
             ("Cannibalized", -f["CANNIB_GP_LOSS"].sum())]
    measure, xs, ys = ["relative"] * 3 + ["total"], [a for a, _ in steps] + ["Net, retailer pays"], \
        [b for _, b in steps] + [0]
    if view == "Recorded cost":
        measure = ["relative"] * 4 + ["total"]
        xs, ys = xs[:3] + ["Trade-deal funding", "Net, recorded cost"], ys[:3] + [f["TRADE_FUNDING_EST"].sum(), 0]
    fig = go.Figure(go.Waterfall(x=xs, y=ys, measure=measure, increasing=dict(marker_color=AQUA),
                                 decreasing=dict(marker_color=RED), totals=dict(marker_color=BLUE),
                                 connector=dict(line=dict(color="#cfcdc8")),
                                 text=[D.money(v) if m != "total" else D.money(sum(ys)) for v, m in zip(ys, measure)],
                                 textposition="outside"))
    fig.update_yaxes(title="Gross profit")
    T.money_axis(fig)
    left, right = t1.columns(2)
    T.chart(fig, left, 420, "Profit bridge")
    fig = go.Figure()
    for b in sorted(f["BRAND_GROUP"].unique()):
        g = f[f["BRAND_GROUP"].eq(b)]
        fig.add_scatter(x=(1 + g["LIFT_PCT"] / 100).clip(lower=0.2), y=g[col], mode="markers", name=b,
                        marker=dict(color=T.BRAND_COLORS.get(b, GREY), size=8, opacity=0.75,
                                    line=dict(color="white", width=0.5)),
                        text=g["LABEL"], hovertemplate="%{text}<br>sales ×%{x:.1f} normal<br>GP %{y:$,.0f}<extra></extra>")
    fig.add_hline(y=0, line_color=INK_2)
    fig.add_vline(x=1, line_color=T.GRID, line_dash="dot")
    fig.update_xaxes(title="Sales during the promotion (× normal sales, log scale)", type="log",
                     tickvals=[0.5, 1, 2, 5, 10, 30, 100], ticktext=["×0.5", "×1", "×2", "×5", "×10", "×30", "×100"])
    fig.update_yaxes(title="Net gross-profit impact")
    T.money_axis(fig)
    T.chart(fig, right, 420, "More units ≠ more profit")

    show = f[["LABEL", "BRAND_GROUP", "PROMO_KIND", "MEAN_DEPTH", "LIFT_PCT", "CANNIBALIZED_UNITS", "DISCOUNT_COST",
              "NET_GP_IMPACT_PRE", "NET_GP_IMPACT_REC", "LIFT_SIGNIFICANT"]].sort_values(col).copy()
    show["MEAN_DEPTH"] *= 100
    T.table(show, t2, hide_index=True, height=440, column_config={
        "LABEL": CC.TextColumn("Event", width="large"), "BRAND_GROUP": "Brand", "PROMO_KIND": "Type",
        "MEAN_DEPTH": CC.NumberColumn("Depth", format="%.0f%%"),
        "LIFT_PCT": CC.NumberColumn("Lift", format="%+.0f%%"),
        "CANNIBALIZED_UNITS": CC.NumberColumn("Cannibalized units", format="%.0f"),
        "DISCOUNT_COST": CC.NumberColumn("Discount cost", format="$%.0f"),
        "NET_GP_IMPACT_PRE": CC.NumberColumn("Net GP · retailer pays", format="$%.0f"),
        "NET_GP_IMPACT_REC": CC.NumberColumn("Net GP · recorded", format="$%.0f"),
        "LIFT_SIGNIFICANT": CC.CheckboxColumn("Significant")})
    t2.download_button("⬇ Download filtered events (CSV)", f.to_csv(index=False).encode(), f"promoguard_events_{D.CAT}.csv",
                       "text/csv")


# ----------------------------------------------------------------------------
# Page 4b — Post-event report card
# ----------------------------------------------------------------------------
CLASS_COLORS = {"Winner": AQUA, "Funded win": BLUE, "Volume trap": YELLOW, "Loser": RED}
CLASS_TONES = {"Winner": "good", "Funded win": "neutral", "Volume trap": "warn", "Loser": "bad"}
CLASS_HELP = {"Winner": "Raised gross profit even when Dominick's pays the whole discount.",
              "Funded win": "Profitable only because the manufacturer funded the deal.",
              "Volume trap": "Significant sales lift, but gross profit fell in both cost views.",
              "Loser": "No clear lift and lower gross profit."}


@st.cache_data
def _report_card(cat: str) -> pd.DataFrame:
    c = D.classify_events(_events(cat))
    c["YEAR"] = c["START_DATE"].dt.year
    return c


def report_card() -> pd.DataFrame:
    return _report_card(D.CAT)


def page_reportcard() -> None:
    s = scoped()
    T.hero("Post-event report card", "Every past promotion sorted into Winners, Funded wins, Volume traps and "
           "Losers — with a verdict for the next calendar.", "Report card", "causal", filter_chips(s))
    if s.empty:
        return no_events()
    c = report_card()
    c = c[c["EVENT_ID"].isin(s["EVENT_ID"])]
    g = c.groupby("CLASS").agg(n=("EVENT_ID", "size"), pre=("NET_GP_IMPACT_PRE", "sum"),
                               rec=("NET_GP_IMPACT_REC", "sum")).reindex(D.CLASSES).fillna(0)
    T.kpi_row([T.kpi(k, f"{int(g.loc[k, 'n']):,}", f"{g.loc[k, 'n'] / len(c) * 100:.0f}% of events · "
                     f"<b>{D.money(g.loc[k, 'pre'])}</b> GP (retailer pays)", CLASS_TONES[k], CLASS_HELP[k])
               for k in D.CLASSES])
    bad = g.loc[["Funded win", "Volume trap", "Loser"], "pre"].sum()
    T.callout(f"<b>{g.loc['Winner', 'n'] / len(c) * 100:.0f}% of events are winners</b> and earn "
              f"{D.money(g.loc['Winner', 'pre'])}, but the other groups lose {D.money(-bad)} when Dominick's pays the "
              f"discount — the portfolio nets <b>{D.money(g['pre'].sum())}</b>. Cutting volume traps and losers is "
              f"worth more than finding new winners.", "neutral")

    t1, t2, t3, t4 = st.tabs(["Portfolio map", "By brand & depth", "Funding check", "Event verdicts"])
    # --- portfolio map: retailer-pays GP vs recorded GP ---
    lim_x = np.nanpercentile(c["NET_GP_IMPACT_PRE"].abs(), 97)
    lim_y = np.nanpercentile(c["NET_GP_IMPACT_REC"].abs(), 97)
    fig = go.Figure()
    for k in D.CLASSES:
        d = c[c["CLASS"].eq(k)]
        if d.empty:
            continue
        fig.add_scatter(x=d["NET_GP_IMPACT_PRE"].clip(-lim_x, lim_x), y=d["NET_GP_IMPACT_REC"].clip(-lim_y, lim_y),
                        mode="markers", name=k, text=d["LABEL"], customdata=d[["VERDICT"]],
                        marker=dict(color=CLASS_COLORS[k], size=8, opacity=0.75, line=dict(color="white", width=0.5)),
                        hovertemplate="%{text}<br>retailer pays %{x:$,.0f} · recorded %{y:$,.0f}"
                                      "<br><b>%{customdata[0]}</b><extra></extra>")
    fig.add_hline(y=0, line_color=INK_2)
    fig.add_vline(x=0, line_color=INK_2)
    for x, y, t, xa in [(0.98, -0.95, "Winners →", "right"), (-0.98, 0.95, "← Funded wins", "left"),
                        (-0.98, -0.95, "← Traps & losers", "left")]:
        fig.add_annotation(x=x * lim_x, y=y * lim_y, text=f"<b>{t}</b>", showarrow=False, xanchor=xa,
                           bgcolor="rgba(255,255,255,0.85)", font=dict(color=INK_2, size=13))
    fig.update_xaxes(title="Net GP impact — retailer pays the discount (clipped)", zeroline=False)
    fig.update_yaxes(title="Net GP impact — recorded cost (clipped)", zeroline=False)
    T.money_axis(fig, "x")
    T.money_axis(fig, "y")
    T.chart(fig, t1, 520, "Portfolio map")
    t1.caption("Points right of the vertical line made money without manufacturer help; points above the horizontal "
               "line made money once trade funding is counted. Axes clipped at the 97th percentile.")

    # --- mix by brand and by depth ---
    left, right = t2.columns(2)
    for where, by, title, order in [(left, "BRAND_GROUP", "Mix by brand", None),
                                    (right, "DEPTH_BUCKET", "Mix by discount depth", DEPTH_ORDER)]:
        m = pd.crosstab(c[by], c["CLASS"], normalize="index").reindex(columns=D.CLASSES, fill_value=0) * 100
        if order:
            m = m.reindex([o for o in order if o in m.index])
        fig = go.Figure([go.Bar(y=m.index, x=m[k], name=k, orientation="h", marker_color=CLASS_COLORS[k],
                                hovertemplate=f"{k}: %{{x:.0f}}%<extra></extra>") for k in D.CLASSES if m[k].any()])
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="% of events", range=[0, 100])
        T.chart(fig, where, 380, title)

    # --- funding check: estimated funding vs break-even funding ---
    f = c[c["DISCOUNT_COST"].ge(50) & c["FUNDED_SHARE"].notna() & c["BREAK_EVEN_SHARE"].notna()]
    fig = go.Figure()
    for k in D.CLASSES:
        d = f[f["CLASS"].eq(k)]
        if d.empty:
            continue
        fig.add_scatter(x=d["BREAK_EVEN_SHARE"].clip(0, 1.5) * 100, y=d["FUNDED_SHARE"].clip(-0.5, 1.5) * 100,
                        mode="markers", name=k, text=d["LABEL"],
                        marker=dict(color=CLASS_COLORS[k], size=8, opacity=0.75, line=dict(color="white", width=0.5)),
                        hovertemplate="%{text}<br>needed %{x:.0f}% · funded %{y:.0f}%<extra></extra>")
    fig.add_scatter(x=[0, 150], y=[0, 150], mode="lines", line=dict(color=INK_2, dash="dash"), name="enough line")
    fig.update_xaxes(title="Funding needed to break even (% of discount)")
    fig.update_yaxes(title="Estimated manufacturer funding (% of discount)")
    T.chart(fig, t3, 480, "Was the manufacturer funding enough?")
    short = f[f["BREAK_EVEN_SHARE"].gt(0)]
    if len(short):
        ok = (short["FUNDED_SHARE"] >= short["BREAK_EVEN_SHARE"]).mean() * 100
        t3.caption(f"Points above the dashed line had enough funding. Of the {len(short)} events that needed funding "
                   f"to break even, {ok:.0f}% received enough. Events with discounts under $50 are not shown. "
                   "Funding is inferred from the recorded cost (Phase 8), not observed.")

    # --- verdict table ---
    pick = t4.multiselect("Show groups", D.CLASSES, default=["Volume trap", "Loser"])
    v = c[c["CLASS"].isin(pick)].sort_values("NET_GP_IMPACT_PRE")
    v = v.assign(MEAN_DEPTH=v["MEAN_DEPTH"] * 100)
    T.table(v[["LABEL", "CLASS", "VERDICT", "REASON", "EVIDENCE", "MEAN_DEPTH", "LIFT_PCT", "NET_GP_IMPACT_PRE",
               "NET_GP_IMPACT_REC"]], t4, hide_index=True, height=440, column_config={
        "LABEL": CC.TextColumn("Event", width="medium"), "CLASS": "Group", "VERDICT": "Verdict",
        "REASON": CC.TextColumn("Why", width="large"), "EVIDENCE": "Evidence",
        "MEAN_DEPTH": CC.NumberColumn("Depth", format="%.0f%%"), "LIFT_PCT": CC.NumberColumn("Lift", format="%+.0f%%"),
        "NET_GP_IMPACT_PRE": CC.NumberColumn("Net GP · retailer pays", format="$%.0f"),
        "NET_GP_IMPACT_REC": CC.NumberColumn("Net GP · recorded", format="$%.0f")})
    t4.caption("Evidence = 'clear' when the 90% range of the retailer-pays gross-profit impact has a single sign.")
    t4.download_button("⬇ Download report card (CSV)", c.to_csv(index=False).encode(), f"promoguard_report_card_{D.CAT}.csv",
                       "text/csv")


# ----------------------------------------------------------------------------
# Page 4c — Compare categories (same pipeline, two product categories)
# ----------------------------------------------------------------------------
CAT_COLORS = {"tna": BLUE, "sdr": ORANGE}


@st.cache_data(show_spinner="Comparing categories…")
def _category_summary(cat: str) -> dict:
    ev = _events(cat)
    s = ev[ev["IN_SCOPE"]]
    k = D.kpis(ev)
    k["gp_per_dollar_pre"] = k["net_gp_pre"] / k["discount_cost"] if k["discount_cost"] else np.nan
    k["gp_per_dollar_rec"] = k["net_gp_rec"] / k["discount_cost"] if k["discount_cost"] else np.nan
    k["break_even_share"] = max(0.0, -k["gp_per_dollar_pre"]) if pd.notna(k["gp_per_dollar_pre"]) else np.nan
    k["by_depth"] = D.pooled_by(s, "DEPTH_BUCKET").reindex(DEPTH_ORDER).dropna(subset=["events"])
    k["classes"] = _report_card(cat)["CLASS"].value_counts(normalize=True).reindex(D.CLASSES).fillna(0) * 100
    sim = _simulator(cat)
    top = int(sim.products.sort_values("TOTAL_UNITS", ascending=False).index[0])
    best = {}
    for tf in (0.0, 0.5):
        t = sim.scenario_table(top, duration=2, lent=False, trade_funding=tf).iloc[1:]
        b = t.loc[t["net_gp_impact"].idxmax()]
        best[tf] = (b["scenario"], b["net_gp_impact"])
    k["top_product"], k["best"] = sim.products.loc[top, "DESCRIP"], best
    k["years"] = f"{int(s['YEAR'].min())}–{int(s['YEAR'].max())}"
    return k


def page_compare() -> None:
    cats = D.available_categories()
    T.hero("Same method, different answers", "The PromoGuard pipeline run unchanged on two product categories. "
           "The numbers show that the right promotion depends on the product.", "Compare categories", "causal",
           [D.CATEGORIES[c]["name"] for c in cats] + ["all in-scope events", "sidebar filters not applied"])
    if len(cats) < 2:
        T.empty_state("Only one category available", "Run the pipeline with <code>--cat sdr</code> (Phases 2–9) "
                      "to add soft drinks.")
        return
    K = {c: _category_summary(c) for c in cats}
    name = {c: D.CATEGORIES[c]["name"] for c in cats}

    T.kpi_row([T.kpi(name[c], f"{K[c]['events']:,} events", f"{K[c]['products']} products · {K[c]['years']} · "
                     f"<b>{D.money(K[c]['net_gp_pre'])}</b> net GP if the retailer pays", tone(K[c]["net_gp_pre"]))
               for c in cats])

    metrics = [
        ("Promotion events (in scope)", "events", lambda v: f"{v:,}"),
        ("Typical sales lift (median event)", "median_lift_pct", pct),
        ("Pooled sales lift", "pooled_lift_pct", pct),
        ("Events with a significant lift", "share_significant", lambda v: f"{v:.0f}%"),
        ("Extra units taken from other products", "cannibalization_pct", lambda v: f"{v:.1f}%"),
        ("Net gross profit · retailer pays", "net_gp_pre", D.money),
        ("Net gross profit · recorded cost", "net_gp_rec", D.money),
        ("GP per $1 of discount · retailer pays", "gp_per_dollar_pre", lambda v: f"{v:+.2f}"),
        ("Events that raised GP · retailer pays", "share_profitable_pre", lambda v: f"{v:.0f}%"),
        ("Supplier share needed to break even (all events)", "break_even_share", lambda v: f"{v:.0%}"),
    ]
    tab = pd.DataFrame({name[c]: [f(K[c][key]) for _, key, f in metrics] for c in cats},
                       index=[m for m, _, _ in metrics])
    T.section("Side by side", "Causal estimates under the stated assumptions; gross profit, not net profit.", "causal")
    T.table(tab.reset_index().rename(columns={"index": "Measure"}), hide_index=True)

    T.section("Depth decides profit — differently", "Pooled lift and gross profit per $1 of discount, by depth.",
              "causal")
    left, right = st.columns(2)
    fig = go.Figure()
    for c in cats:
        b = K[c]["by_depth"]
        fig.add_bar(x=b.index, y=b["lift_pct"], name=name[c], marker_color=CAT_COLORS.get(c, GREY),
                    text=[pct(v) for v in b["lift_pct"]], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Pooled sales lift (%)")
    T.chart(fig, left, 380, "Sales lift by discount depth")
    fig = go.Figure()
    for c in cats:
        b = K[c]["by_depth"]
        fig.add_bar(x=b.index, y=b["gp_per_discount_dollar_pre"], name=name[c], marker_color=CAT_COLORS.get(c, GREY),
                    text=[f"{v:+.2f}" for v in b["gp_per_discount_dollar_pre"]], textposition="outside",
                    cliponaxis=False)
    fig.add_hline(y=0, line_color=INK_2)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="$ gross profit per $1 discount (retailer pays)")
    T.chart(fig, right, 380, "Gross profit per $1 of discount")

    T.section("How the promotions turned out", "Share of events in each report-card group.", "causal")
    fig = go.Figure([go.Bar(y=[name[c] for c in cats], x=[K[c]["classes"][k] for c in cats], name=k,
                            orientation="h", marker_color=CLASS_COLORS[k],
                            text=[f"{K[c]['classes'][k]:.0f}%" for c in cats], textposition="inside",
                            hovertemplate=f"{k}: %{{x:.0f}}%<extra></extra>") for k in D.CLASSES])
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="% of events", range=[0, 100])
    T.chart(fig, height=260, title="Winners, funded wins, volume traps and losers")

    T.section("What this means", "Generated from the numbers above.")
    cards = []
    for c in cats:
        k = K[c]
        bd = k["by_depth"]
        best_depth = bd["gp_per_discount_dollar_pre"].idxmax() if len(bd) else "—"
        (s0, g0), (s5, g5) = k["best"][0.0], k["best"][0.5]
        good = k["net_gp_pre"] > 0 or k["gp_per_dollar_pre"] > -0.05
        cards.append(dict(
            title=f"{name[c]}: " + ("promotions roughly pay for themselves" if good else "promotions lose money"),
            tone="good" if good else "bad", tag="Causal estimate + simulation",
            body=f"Sales {pct(k['pooled_lift_pct'])} during promotions, {k['cannibalization_pct']:.1f}% taken from "
                 f"other products. Retailer pays: <b>{k['gp_per_dollar_pre']:+.2f}</b> per $1 of discount; best depth "
                 f"<b>{best_depth}</b>. Top product ({T.esc(str(k['top_product']))}): best plan "
                 f"<b>{T.esc(s0)}</b> ({D.money(g0)}) when the retailer pays, <b>{T.esc(s5)}</b> ({D.money(g5)}) "
                 "when the supplier pays half."))
    T.rec_cards(cards, cols=len(cards))
    T.callout("<b>Same tool, different advice.</b> A promotion rule that works for one category can lose money in "
              "another: the shelf margin, how strongly buyers react and how much the supplier pays decide the answer. "
              "Each category must be measured on its own.", "neutral")
    st.caption("Both categories: Dominick's Finer Foods, Chicago, same stores and weeks, same promotion definition, "
               "same models and tests (Phases 2–9). The comparison uses all in-scope events of each category.")


@st.cache_data(show_spinner="Computing break-even funding for every product…")
def _negotiation_sheet(cat: str, depth: float, duration: int, lent: bool, deal: bool) -> pd.DataFrame:
    return _simulator(cat).negotiation_sheet(depth, duration=duration, lent=lent, deal_coded=deal)


def negotiation_sheet(depth: float, duration: int, lent: bool, deal: bool) -> pd.DataFrame:
    return _negotiation_sheet(D.CAT, depth, duration, lent, deal)


# ----------------------------------------------------------------------------
# Page 5 — What-if simulator
# ----------------------------------------------------------------------------
def page_whatif() -> None:
    sim = simulator()
    s_all = events()[events()["IN_SCOPE"]]
    n_ev = len(s_all)
    deal_share = (s_all["SHARE_DEAL_CODED"] >= 0.5).mean() * 100
    xd = sim.extrapolation_depth
    rare = (s_all["MEAN_DEPTH"] > xd).mean() * 100
    T.hero("Plan the next promotion", "Test depth, duration, timing and manufacturer funding before committing. "
           f"Estimates come from a model trained on {n_ev:,} past {U()} promotions.", "What-if simulator", "sim",
           [CATNAME(), f"based on {n_ev:,} past promotions", "monotonic in depth"])
    plist = sim.product_list()
    plist["LABEL"] = plist["DESCRIP"].str.strip() + " · " + plist["SIZE"].astype(str) + " · $" + \
        plist["REG_PRICE"].map("{:.2f}".format)
    with st.container(border=True):
        c = cols([3, 2, 1, 1])
        label = c[0].selectbox("Product", plist["LABEL"].tolist())
        pid = int(plist.loc[plist["LABEL"].eq(label), "PRODUCT_ID"].iloc[0])
        depth = c[1].slider("Discount depth (%)", 0, 40 if D.CAT == "tna" else 50, 15, step=5) / 100
        duration = c[2].selectbox("Weeks", [1, 2, 3, 4], index=1)
        lent = c[3].toggle("During Lent", value=False)
        c = cols([3, 4])
        trade = c[0].slider("Share of the discount paid by the manufacturer (%)", 0, 100, 0, step=10) / 100
        c[1].caption("Manufacturer share 0% = conservative view (retailer pays everything). Promotions are "
                     f"simulated as flagged deals in the shop system, as {deal_share:.0f}% of past promotions were.")
        deal = True   # most past events carried a deal code; the flag is not a controllable lever (see Method)
    r = sim.simulate(pid, depth, duration=duration, lent=lent, deal_coded=deal, trade_funding=trade)
    if r["extrapolation"]:
        T.callout(f"⚠ Discounts deeper than {xd:.0%} were rare in the historical data (≈ {rare:.0f}% of events): "
                  "this estimate is an <b>extrapolation</b>.", "warn")
    gpd = r["gp_per_discount_dollar"]
    T.kpi_row([
        T.kpi("Expected units", f"{r['units']:,.0f}", f"<b>{pct(r['lift_pct'])}</b> vs. no promotion", "neutral"),
        T.kpi(f"Taken from other {U()}", f"{r['cannibalized_units']:,.0f}", "cannibalized units", "neutral"),
        T.kpi("Net GP impact", D.money(r["net_gp_impact"]), "vs. no promotion", tone(r["net_gp_impact"])),
        T.kpi("GP per $1 discount", "—" if np.isnan(gpd) else f"{gpd:+.2f}", "", tone(gpd)),
    ])
    tab = sim.scenario_table(pid, duration=duration, lent=lent, trade_funding=trade, deal_coded=deal)
    best = tab.iloc[1:].loc[tab.iloc[1:]["net_gp_impact"].idxmax()]
    if best["net_gp_impact"] > 0:
        T.callout(f"<b>Recommended option:</b> {T.esc(best['scenario'])} — expected net gross-profit impact "
                  f"<b>{D.money(best['net_gp_impact'])}</b>.", "good")
    else:
        T.callout("<b>Recommended option: do not discount.</b> No tested depth raises gross profit under these "
                  "assumptions — try a higher manufacturer share or a shallower discount.", "bad")
    t1, t2, t3, t4 = st.tabs(["Scenario comparison", "Scenario table", "Break-even funding",
                              "Supplier negotiation sheet"])
    break_even_tab(sim, pid, depth, duration, lent, deal, trade, t3)
    negotiation_tab(depth, duration, lent, deal, t4)
    left, right = t1.columns(2)
    is_base = tab["scenario"].eq("no promotion")
    fig = go.Figure(go.Bar(x=tab["scenario"], y=tab["units"], marker_color=np.where(is_base, GREY, BLUE)))
    fig.update_yaxes(title="Units")
    T.chart(fig, left, 380, "Expected units by scenario")
    fig = go.Figure()
    fig.add_scatter(x=tab["scenario"], y=tab["net_gp_impact"], mode="lines+markers", line=dict(color=BLUE, width=3),
                    name="Expected")
    fig.add_hline(y=0, line_color=INK_2)
    fig.add_annotation(x=best["scenario"], y=best["net_gp_impact"], text="best", showarrow=True, arrowhead=2,
                       font=dict(color=T.INK))
    fig.update_yaxes(title="Net gross-profit impact")
    T.money_axis(fig)
    T.chart(fig, right, 380, "Net gross-profit impact by scenario")
    T.table(tab.drop(columns=["extrapolation", "units_low", "units_high", "net_gp_impact_low", "net_gp_impact_high"]), t2, hide_index=True, column_config={
        "scenario": "Scenario", "promo_price": CC.NumberColumn("Price", format="$%.2f"),
        "units": CC.NumberColumn("Units", format="%.0f"), "lift_pct": CC.NumberColumn("Lift", format="%+.0f%%"),
        "incremental_units": CC.NumberColumn("Incremental", format="%.0f"),
        "cannibalized_units": CC.NumberColumn("Cannibalized", format="%.0f"),
        "net_category_units": CC.NumberColumn("Net category", format="%.0f"),
        "revenue": CC.NumberColumn("Revenue", format="$%.0f"),
        "net_gp_impact": CC.NumberColumn("Net GP", format="$%.0f"),
        "discount_cost": CC.NumberColumn("Discount cost", format="$%.0f"),
        "gp_per_discount_dollar": CC.NumberColumn("GP / $1 discount", format="%+.2f")})
    st.caption(f"{CATNAME()}: {len(sim.products)} products actively sold at the end of the data. Assumes future "
               "promotions behave like Dominick's 1989–1997 promotions. Gross profit, not net profit.")


def break_even_tab(sim, pid, depth, duration, lent, deal, trade, where) -> None:
    cur = sim.break_even(pid, depth, duration=duration, lent=lent, deal_coded=deal) if depth > 0 else None
    if cur is None or np.isnan(cur["break_even_share"]):
        T.callout("A promotion without a price cut has no discount, so there is nothing for the manufacturer "
                  "to fund. Choose a depth above 0%.", "muted", where)
    elif cur["break_even_share"] == 0:
        T.callout(f"At <b>{depth:.0%} off</b> this promotion breaks even <b>without any manufacturer funding</b> "
                  f"(expected {D.money(cur['net_gp_retailer_pays'])} when Dominick's pays everything).", "good", where)
    elif cur["reachable"]:
        T.callout(f"At <b>{depth:.0%} off</b> the manufacturer must fund at least <b>{cur['break_even_share']:.0%}</b> "
                  f"of the discount for Dominick's to "
                  f"break even — about <b>{D.money(cur['break_even_share'] * cur['discount_cost'])}</b>. "
                  f"Current slider: {trade:.0%}.", "good" if trade >= cur["break_even_share"] else "warn", where)
    else:
        T.callout(f"At <b>{depth:.0%} off</b> the promotion loses money <b>even if the manufacturer pays the whole "
                  f"discount</b> (needed {cur['break_even_share']:.0%}). Choose a shallower discount.", "bad", where)
    curve = sim.break_even_curve(pid, duration=duration, lent=lent, deal_coded=deal)
    x = [f"{d:.0%}" for d in curve["depth"]]
    fig = go.Figure()
    fig.add_scatter(x=x, y=curve["break_even_share"].clip(upper=1.5) * 100, mode="lines+markers", name="Break-even",
                    line=dict(color=BLUE, width=3), hovertemplate="%{x} off: %{y:.0f}% of the discount<extra></extra>")
    fig.add_hline(y=100, line_color=RED, line_dash="dash", annotation_text="manufacturer pays everything",
                  annotation_position="top left")
    fig.add_hline(y=trade * 100, line_color=AQUA, line_dash="dot", annotation_text="your slider",
                  annotation_position="bottom right")
    fig.update_xaxes(title="Discount depth")
    fig.update_yaxes(title="Manufacturer funding needed (% of discount)", range=[-5, 155])
    T.chart(fig, where, 400, "Funding needed to break even, by depth")
    where.caption("Net gross profit rises one-for-one with funding, so break-even = −(net GP when the retailer pays) "
                  "÷ discount cost. Values above 100% cannot break even. Capped at 150% in the chart.")


def negotiation_tab(depth, duration, lent, deal, where) -> None:
    if depth <= 0:
        T.callout("Choose a discount depth above 0% to build the negotiation sheet.", "muted", where)
        return
    ns = negotiation_sheet(depth, duration, lent, deal).copy()
    n_free = int((ns["break_even_share"] == 0).sum())
    n_never = int((~ns["reachable"] & ns["break_even_share"].notna()).sum())
    T.kpi_row([T.kpi("Products", f"{len(ns)}", f"{depth:.0%} off · {duration} week(s)", "neutral"),
               T.kpi("Break even unfunded", f"{n_free}", "no manufacturer money needed", "good"),
               T.kpi("Need funding", f"{len(ns) - n_free - n_never}", "negotiate before promoting", "warn"),
               T.kpi("Cannot break even", f"{n_never}", "even at 100% funding", "bad")], where)
    ns["label"] = ns["description"].str.strip() + " · " + ns["SIZE"].astype(str)
    d = ns.sort_values("break_even_share")
    fig = go.Figure(go.Bar(y=d["label"], x=d["break_even_share"].clip(upper=1.5) * 100, orientation="h",
                           marker_color=np.select([d["break_even_share"].eq(0), d["reachable"]], [AQUA, YELLOW], RED),
                           hovertemplate="%{y}: %{x:.0f}% of the discount<extra></extra>"))
    fig.add_vline(x=100, line_color=RED, line_dash="dash")
    fig.update_xaxes(title="Manufacturer funding needed (% of discount, capped at 150%)")
    fig.update_yaxes(autorange="reversed")
    T.chart(fig, where, max(380, 18 * len(d) + 80), f"Break-even funding at {depth:.0%} off")
    out = d[["description", "BRAND", "SIZE", "REG_PRICE", "net_gp_retailer_pays", "discount_cost", "break_even_share",
             ]].copy()
    for col in ["break_even_share"]:
        out[col] *= 100
    T.table(out, where, hide_index=True, column_config={
        "description": "Product", "BRAND": "Brand", "SIZE": "Size", "REG_PRICE": CC.NumberColumn("Regular", format="$%.2f"),
        "net_gp_retailer_pays": CC.NumberColumn("Net GP · retailer pays", format="$%.0f"),
        "discount_cost": CC.NumberColumn("Discount cost", format="$%.0f"),
        "break_even_share": CC.NumberColumn("Funding needed", format="%.0f%%")})
    where.download_button("⬇ Download negotiation sheet (CSV)", out.to_csv(index=False).encode(),
                          f"promoguard_negotiation_{D.CAT}_{int(depth * 100)}pct.csv", "text/csv")


# ----------------------------------------------------------------------------
# Page 5b — Personalized discounts by customer group (Phase 10)
# ----------------------------------------------------------------------------
GROUP_COLORS = {"Affluent professionals": BLUE, "Older, budget-conscious": ORANGE, "Urban low-income": RED,
                "Young families": AQUA}


@st.cache_data
def groups_data():
    return D.load_groups()


def store_map(stores: pd.DataFrame, where) -> None:
    """Store locations by group. A plain longitude/latitude chart: no map tiles, so it always renders."""
    fig = go.Figure()
    for g, t in stores.groupby("GROUP"):
        fig.add_scatter(x=t["lon"], y=t["lat"], mode="markers", name=g,
                        marker=dict(size=12, color=GROUP_COLORS.get(g, GREY), opacity=0.85,
                                    line=dict(color="white", width=1)),
                        text=t["city"] + " · store " + t["STORE"].astype(str),
                        hovertemplate="%{text}<extra>" + g + "</extra>")
    lake = stores["lon"].max() + 0.02
    fig.add_annotation(x=lake, y=stores["lat"].mean(), text="Lake Michigan →", showarrow=False, xanchor="left",
                       font=dict(color=BLUE, size=12))
    fig.update_xaxes(title="West ← → East", showticklabels=False, showgrid=False, zeroline=False,
                     range=[stores["lon"].min() - 0.03, lake + 0.2])
    fig.update_yaxes(title="South ← → North", showticklabels=False, showgrid=False, zeroline=False,
                     scaleanchor="x", scaleratio=1.35)
    T.chart(fig, where, 560, "Where each customer group shops (Chicago area)")


def page_groups() -> None:
    T.hero("Personalized discounts by customer group", "Stores are grouped by the customers who live around them "
           "(1990 census). For each group: how strongly it reacts to discounts, which products to promote and "
           "where promotions pay most.", "Customer groups", "causal",
           ["4 customer groups", "83 stores", "no individual customer data"])
    if D.CAT != "tna":
        T.callout(f"<b>Customer groups were built for canned tuna.</b> Switch the product category in the sidebar to "
                  f"<b>Canned tuna</b> to see them. (The personalized plan was computed for tuna only.)", "muted")
        return
    gd = groups_data()
    if gd is None:
        T.empty_state("Customer groups not computed yet", "Download <b>demo(stata).zip</b> from the Kilts Center into "
                      "<code>data/raw/dominicks/demo/</code>, then run <code>python src/analysis/10_customer_groups.py"
                      "</code>.")
        return
    prof, resp, plan, stores_ = gd["profiles"], gd["response"], gd["plan"], gd["stores"]
    plan = plan.assign(GP_PER_STORE=plan["BEST_GP"] / plan["STORES"])
    best_store = plan[plan["BEST_GP"] > 0].groupby("GROUP")["GP_PER_STORE"].max()
    m15 = resp[resp["DEPTH"].round(2).eq(0.15)].set_index("GROUP")["MULTIPLIER"]
    order = list(prof.index)
    T.kpi_row([T.kpi(g, f"{int(prof.loc[g, 'stores'])} stores",
                     f"median income <b>${prof.loc[g, 'income'] / 1e3:,.0f}k</b> · reacts "
                     f"<b>{(m15.get(g, 1) - 1) * 100:+.0f}%</b> vs average at 15% off · best promo "
                     f"<b>${best_store.get(g, 0):,.0f}</b>/store",
                     "warn" if m15.get(g, 1) > 1.02 else ("neutral" if m15.get(g, 1) > 0.98 else "muted"))
               for g in order])
    uni, per = plan.groupby("PRODUCT_ID")[["UNIFORM_GP", "BEST_GP"]].sum().sum()
    top_g = best_store.idxmax()
    low_g = best_store.idxmin()
    T.callout(f"<b>Personalize WHERE and WHAT, not HOW MUCH.</b> The best discount is about 15% in every group "
              f"(personalizing the depth adds only {(per / uni - 1) * 100:+.1f}% gross profit). But the same "
              f"promotion earns <b>${best_store[top_g]:,.0f} per store</b> in <b>{T.esc(top_g)}</b> stores vs "
              f"${best_store[low_g]:,.0f} in {T.esc(low_g)} stores, and each group prefers different products.",
              "neutral")

    t1, t2, t3, t4 = st.tabs(["Who are the groups", "Who reacts to discounts", "Recommended plan by group",
                              "Where to promote"])
    store_map(stores_, t1)
    right = t1
    show = prof.copy()
    show["income"] = show["income"] / 1e3
    for c in ["educ", "age60", "poverty", "workwom", "shpavid"]:
        show[c] = show[c] * 100
    T.table(show.reset_index(), right, hide_index=True, column_config={
        "GROUP": "Group", "stores": CC.NumberColumn("Stores"),
        "income": CC.NumberColumn("Median income", format="$%.0fk"),
        "educ": CC.NumberColumn("College grads", format="%.0f%%"),
        "age60": CC.NumberColumn("Age 60+", format="%.0f%%"),
        "hsizeavg": CC.NumberColumn("Household size", format="%.2f"),
        "poverty": CC.NumberColumn("Income < $15k", format="%.0f%%"),
        "workwom": CC.NumberColumn("Working women", format="%.0f%%"),
        "shpavid": CC.NumberColumn("Avid shoppers", format="%.0f%%")})
    right.caption("Groups found by k-means on 7 census variables (best of 3–6 groups by silhouette). "
                  "Ethnicity is deliberately not used.")

    fig = go.Figure()
    for g in order:
        t = resp[resp["GROUP"].eq(g)]
        fig.add_scatter(x=t["DEPTH"] * 100, y=(t["MULTIPLIER"] - 1) * 100, mode="lines+markers", name=g,
                        line=dict(color=GROUP_COLORS.get(g, GREY), width=3),
                        error_y=dict(type="data", symmetric=False, array=(t["MULT_HI90"] - t["MULTIPLIER"]) * 100,
                                     arrayminus=(t["MULTIPLIER"] - t["MULT_LO90"]) * 100, thickness=1),
                        hovertemplate="%{x:.0f}% off: %{y:+.1f}% vs average<extra>" + g + "</extra>")
    fig.add_hline(y=0, line_color=INK_2)
    fig.update_xaxes(title="Discount (%)")
    fig.update_yaxes(title="Extra sales vs. the average store (%)", ticksuffix="%")
    T.chart(fig, t2, 430, "How strongly each group reacts to the same discount")
    t2.caption("Compared inside the same promotion (same product, same weeks, same chain price), so busy weeks and "
               "product differences cancel out. Bars = 90% range from 300 bootstrap samples.")

    g_pick = t3.selectbox("Customer group", order)
    p = plan[plan["GROUP"].eq(g_pick)].sort_values("BEST_GP", ascending=False).copy()
    p["BEST_DEPTH"] *= 100
    p["UNIFORM_DEPTH"] *= 100
    top = p[p["BEST_GP"] > 0].head(10)
    if len(top):
        fig = go.Figure(go.Bar(y=top["NAME"], x=top["BEST_GP"], orientation="h",
                               marker_color=GROUP_COLORS.get(g_pick, BLUE),
                               text=[f"{d:.0f}% off · ${v:,.0f}" for d, v in zip(top["BEST_DEPTH"], top["BEST_GP"])],
                               textposition="inside", insidetextanchor="end", textfont=dict(color="white"),
                               hovertemplate="%{y}<br>%{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Gross profit of a 2-week promotion in this group's stores")
        T.money_axis(fig, "x")
        T.chart(fig, t3, 420, f"Top products to promote for {g_pick}")
    T.table(p[["NAME", "REG_PRICE", "BEST_DEPTH", "BEST_GP", "GP_PER_STORE", "UNIFORM_DEPTH"]],
            t3, hide_index=True, column_config={
                "NAME": CC.TextColumn("Product", width="large"),
                "REG_PRICE": CC.NumberColumn("Regular price here", format="$%.2f"),
                "BEST_DEPTH": CC.NumberColumn("Best discount", format="%.0f%%"),
                "BEST_GP": CC.NumberColumn("Gross profit (2 weeks)", format="$%.0f"),
                "GP_PER_STORE": CC.NumberColumn("Per store", format="$%.0f"),
                "UNIFORM_DEPTH": CC.NumberColumn("Chain-wide best", format="%.0f%%")})
    t3.caption("Simulation: Phase 9 product response × this group's sensitivity, this group's regular price and "
               "sales volume; retailer pays the discount. 0% = better not to promote here.")

    b = best_store.reindex(order)
    fig = go.Figure(go.Bar(x=b.index, y=b.values, marker_color=[GROUP_COLORS.get(g, GREY) for g in b.index],
                           text=[f"${v:,.0f}" for v in b.values], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Gross profit per store, best promotion")
    T.money_axis(fig)
    T.chart(fig, t4, 400, "Where the same promotion pays most")
    t4.caption("Best single promotion in each group, divided by the group's number of stores (2 weeks).")
    st.caption("Store-level analysis: groups describe the customers around each store, not individual shoppers. "
               "Group differences are associations with measured promotion response.")


# ----------------------------------------------------------------------------
# Page 6 — Explainability
# ----------------------------------------------------------------------------
def page_explain() -> None:
    sh = shap()
    s = scoped()
    T.hero("What drives promotion lift?", "Shapley values of the promotion-response model: which design and "
           "context factors push an event's lift up or down.", "Explainability", "model", filter_chips(s))
    t1, t2 = st.tabs(["Global drivers", "Explain one event"])
    g = D.shap_global(sh)
    fig = go.Figure(go.Bar(x=g.values[:12][::-1], y=[LABELS.get(i, i) for i in g.index[:12]][::-1], orientation="h",
                           marker_color=BLUE))
    fig.update_xaxes(title="Mean |SHAP value| (effect on log lift)")
    T.chart(fig, t1, 460, f"Global importance ({len(sh):,} events)")
    m = s.merge(sh[["EVENT_ID", "MODEL_LIFT_PCT", "SHAP_BASE"]], on="EVENT_ID")
    if m.empty:
        T.empty_state("No events match the filters", "Widen the sidebar filters.", t2)
    else:
        label = t2.selectbox("Event (sorted from worst to best gross profit)",
                             m.sort_values("NET_GP_IMPACT_PRE")["LABEL"].tolist())
        e = m[m["LABEL"].eq(label)].iloc[0]
        contrib = D.shap_long(sh, int(e["EVENT_ID"])).head(10)
        T.kpi_row([T.kpi("Measured lift", pct(e["LIFT_PCT"]), "causal estimate (Phase 6)", "neutral"),
                   T.kpi("Model lift", pct(e["MODEL_LIFT_PCT"]), "promotion-response model", "neutral"),
                   T.kpi("Average event", pct(np.exp(e["SHAP_BASE"]) * 100 - 100), "model baseline", "muted")], t2)
        fig = go.Figure(go.Bar(x=contrib["shap"][::-1], y=[LABELS.get(f, f) for f in contrib["feature"]][::-1],
                               orientation="h",
                               marker_color=[AQUA if v > 0 else RED for v in contrib["shap"][::-1]]))
        fig.update_xaxes(title="Contribution to log lift vs. the average event (+ raises lift)")
        T.chart(fig, t2, 420, "Why this event differs from the average")
    st.caption("SHAP values explain the model: associations learned from past events, not separately identified "
               "causal effects of each factor.")


# ----------------------------------------------------------------------------
# Page 7 — Model performance
# ----------------------------------------------------------------------------
def page_models() -> None:
    T.hero("Can we trust the numbers?", "Out-of-sample accuracy of both models and the falsification tests "
           "behind every causal estimate.", "Model performance & validity", "model",
           ["time-aware validation", "tested on unseen years"])
    v = D.validity_numbers(events())
    T.kpi_row([
        T.kpi("Pre-period error", "—" if pd.isna(v["pre_period_error_pct"]) else f"≈ {v['pre_period_error_pct']:+.0f}%",
              "4 weeks before events", "good"),
        T.kpi("Lead test", "not significant", "future promotions → no effect today", "good"),
        T.kpi("Baseline model error", "—" if pd.isna(v["baseline_wape_pct"]) else f"≈ {v['baseline_wape_pct']:.0f}%",
              "on weeks it never saw (validation 1994–95)", "good",
              "WAPE of the selected baseline model at product-week level, validation weeks."),
    ])
    t1, t2, t3 = st.tabs(["Baseline model", "Promotion-response model", "Diagnostic figures"])
    comp = D.load_model_comparison()
    if comp is not None:
        cols = [c for c in ["WAPE_units", "WAPE_product_week", "BIAS_units", "BIAS_lent", "R2_log"] if c in comp.columns]
        t1.caption("Counterfactual model — validation year 1994, clean regular weeks. Lower WAPE is better.")
        T.table(comp[cols].round(3), t1)
    else:
        T.empty_state("Comparison not found", "Run Phase 5 to create model_comparison_validation.csv.", t1)
    try:
        t2.caption(f"Leave-one-year-out cross-validation on {v['events']:,} events.")
        T.table(simulator_bundle_cv(), t2)
    except Exception:  # noqa: BLE001
        T.empty_state("Model not found", "Run Phase 9.", t2)
    figs = [(5, "actual_vs_baseline_test.png", "Baseline calibration on unseen weeks"),
            (7, "binscatter_substitution.png", "Cannibalization: within-week variation")]
    found = [(D.figure_path(p, n), cap) for p, n, cap in figs if D.figure_path(p, n)]
    if found:
        cols = t3.columns(len(found))
        for c, (p, cap) in zip(cols, found):
            c.image(str(p), caption=cap)
    else:
        T.empty_state("No figures yet", "Run Phases 5–7 to create the diagnostic figures.", t3)


# ----------------------------------------------------------------------------
# Page 8 — Method & limitations
# ----------------------------------------------------------------------------
def page_method() -> None:
    T.hero("Method, assumptions and limitations", "How every number in this dashboard was produced — and where "
           "it can be wrong.", "Methodology", None, ["9 pipeline phases", "Dominick's 1989–1997"])
    steps = ["Clean the sales data", "Find the promotions", "Learn normal sales", "Predict sales without promotion",
             "Measure extra sales", "Measure cannibalization", "Measure profit", "Explain what drives lift",
             "What-if simulator"]
    T.section("Pipeline")
    st.markdown('<div class="pg-chips" style="margin-bottom:1rem">' + "".join(
        f'<span class="pg-pill" style="background:#e8f1fb;color:#1d5aa3">{i}. {T.esc(x)}</span>'
        for i, x in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)
    T.section("Key assumptions")
    T.rec_cards([
        dict(title="Nothing else changed during the sale", tone="neutral", tag="Supported by the pre-period test",
             body="The model's 'normal sales' line shows what would have happened without the promotion — no other "
                  "hidden event boosted sales at the same time."),
        dict(title=f"Other {U()} react only to the sale", tone="neutral", tag="Supported by the lead test",
             body=f"In a given week, the discount changes other {U()} sales only by pulling buyers to the promoted "
                  "product — nothing else."),
        dict(title="Cost = average buying price", tone="neutral", tag="Supplier help is estimated",
             body=f"Profit uses the average price the shop paid for the {D.info()['items']}. Supplier help is the "
                  "drop in that price during the sale."),
    ])
    T.section("Limitations")
    T.rec_cards([
        dict(title="Stock-outs are invisible", tone="warn", tag="Data", body="When a week shows no sales, we "
             "can't tell if the shelf was empty or nobody wanted it."),
        dict(title="Hidden chain promotions", tone="warn", tag="Data", body="Some normal-looking weeks were "
             "really promotions (display or ad, no price cut). We removed them when teaching the model."),
        dict(title="Data gap", tone="warn", tag="Data", body=D.info()["gap_text"]),
        dict(title="Gross, not net, profit", tone="warn", tag="Scope", body="We don't count salaries, ads or shelf "
             "fees. Some new buyers may come from other supermarkets."),
        dict(title="Simulations are estimates", tone="warn", tag="Model", body=f"Discounts above "
             f"{simulator().extrapolation_depth:.0%} were rare, so guesses there are less safe. The future may differ "
             "from 1989–1997."),
    ], cols=5)
    if D.CAT != "tna":
        T.rec_cards([dict(title="Soft drinks bring shoppers in", tone="warn", tag="Not measured",
                          body="Stores may accept losses on soda sales because cheap Coke and Pepsi bring shoppers "
                               "who also buy other things. This data covers soft drinks only, so that basket effect "
                               "is not measured.")], cols=1)


# ----------------------------------------------------------------------------
# Page 0 — AI Promotion Analyst (chatbot)
# ----------------------------------------------------------------------------
SUGGESTED = [
    "What is the best discount for StarKist chunk light tuna 6 oz?",
    "Should we run a 30% discount on Pepsi 2-litre bottles?",
    "Do promotions pay off more for tuna or for soft drinks?",
    "Which were the 3 worst soft-drink promotions and why?",
    "Which products should we promote in young-family stores?",
    "How much must the supplier pay for a 30% discount on Bumble Bee chunk light in water?",
]


@st.cache_resource(show_spinner=False)
def analyst(key: str):
    from src.agent.analyst import PromotionAnalyst
    return PromotionAnalyst(api_key=key)


def md_safe(text: str) -> str:
    """Show $ as a dollar sign: Streamlit renders text between two $ as a LaTeX formula."""
    return (text or "").replace("\\$", "$").replace("$", "\\$")


def show_steps(steps: list[dict], where) -> None:
    if not steps:
        return
    with where.expander(f"How I got this answer · {len(steps)} tool call(s)"):
        for i, s_ in enumerate(steps, 1):
            args = ", ".join(f"{k}={v!r}" for k, v in (s_["input"] or {}).items())
            st.markdown(f"**{i}. `{s_['tool']}`**({args})")
            st.json(s_["output"], expanded=False)


def page_chat() -> None:
    from src.agent.analyst import get_api_key
    T.hero("Ask the AI Promotion Analyst", "Ask any question about the promotions in plain language. The AI agent "
           "chooses the right analysis tools, runs them on the PromoGuard results and explains the answer.",
           "AI analyst", None, ["Claude LLM agent", "15 analysis tools", "canned tuna + soft drinks",
                                "answers only from computed results"])
    key = get_api_key()
    if not key:
        T.callout("<b>Setup needed:</b> add your Claude API key. Create the file <code>.streamlit/secrets.toml</code> "
                  "in the project folder with one line: <code>ANTHROPIC_API_KEY = \"sk-ant-...\"</code>, "
                  "then restart the dashboard.", "warn")
        return
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    chat = st.session_state["chat"]

    question = None
    if not chat:
        T.section("Try a question")
        cols_ = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols_[i % 2].button(q, key=f"sugg{i}"):
                question = q
    for m in chat:
        with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/smart_toy:"):
            st.markdown(md_safe(m["content"]))
            if m["role"] == "assistant":
                show_steps(m.get("steps", []), st)
    typed = st.chat_input("Ask about promotions, products, discounts, profit…")
    question = typed or question
    if question:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(md_safe(question))
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            with st.spinner("Thinking and running analysis tools…"):
                history = [{"role": m["role"], "content": m["content"]} for m in chat[-8:]]
                reply = analyst(key).ask(question, history)
            if reply.error:
                st.error(f"The AI service returned an error: {reply.error}")
                return
            st.markdown(md_safe(reply.answer))
            show_steps(reply.steps, st)
        chat += [{"role": "user", "content": question},
                 {"role": "assistant", "content": reply.answer, "steps": reply.steps}]
    if chat and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()
    st.caption("Answers are generated by an AI model from PromoGuard's computed results; every number comes from a "
               "tool call you can inspect. Gross profit, not net profit.")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
PAGES = {
    "AI analyst": page_chat,
    "Executive overview": page_overview,
    "Promotion analysis": page_promotion,
    "Cannibalization": page_cannibalization,
    "Profit impact": page_profit,
    "Report card": page_reportcard,
    "Compare categories": page_compare,
    "What-if simulator": page_whatif,
    "Customer groups": page_groups,
    "Explainability": page_explain,
    "Model performance": page_models,
    "Method & limitations": page_method,
}
NAV = {
    "Ask AI": [("AI analyst", ":material/smart_toy:")],
    "Results": [("Executive overview", ":material/dashboard:"), ("Promotion analysis", ":material/campaign:"),
                ("Cannibalization", ":material/swap_horiz:"), ("Profit impact", ":material/payments:"),
                ("Report card", ":material/fact_check:"), ("Compare categories", ":material/compare_arrows:")],
    "Decide": [("What-if simulator", ":material/science:"), ("Customer groups", ":material/groups:")],
    "Trust": [("Explainability", ":material/psychology:"), ("Model performance", ":material/verified:"),
              ("Method & limitations", ":material/menu_book:")],
}


def main() -> None:
    T.inject_css()
    T.logo()
    category_switch()
    missing = D.missing_files()
    if missing:
        T.hero("Pipeline outputs missing", "Run the phase scripts first, then reload this page.", "Setup")
        st.code("\n".join(missing))
        st.stop()
    pg = st.navigation({sec: [st.Page(PAGES[name], title=name, icon=icon, url_path=name.lower().replace(" ", "-")
                                      .replace("&", "and"), default=name == "Executive overview")
                              for name, icon in items] for sec, items in NAV.items()})
    global_filters()
    st.sidebar.markdown("---")
    T.side_note("<b>Reading the badges</b><br>● Observed data · ● Model estimate · ● Causal estimate (under "
                "assumptions) · ● Simulation")
    T.side_note("<br>Data: Kilts Center for Marketing, Chicago Booth (Dominick's Finer Foods). Academic use only.")
    pg.run()
    T.footer()


main()
