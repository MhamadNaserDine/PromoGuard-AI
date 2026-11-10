"""
PromoGuard AI — dashboard visual system.

Location: dashboard/theme.py

Everything about LOOK lives here (CSS, header, KPI cards, recommendation cards,
chart styling), so app.py only contains page logic.
Only standard Streamlit + Plotly are used: no extra packages to install.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

# ---- palette (colour-vision-deficiency validated, same as the report figures) ----
BLUE, ORANGE, AQUA, YELLOW, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#8a8984"
NAVY = "#0f2a4a"
INK, INK_2, GRID, SURFACE, CARD = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#ffffff"
BRAND_COLORS = {"StarKist": BLUE, "Chicken of the Sea": ORANGE, "Bumble Bee": AQUA, "Dominick's": YELLOW,
                "Coca-Cola Co.": RED, "PepsiCo": BLUE, "7-Up": AQUA,
                "Other": GREY}
TONES = {"good": AQUA, "bad": RED, "neutral": BLUE, "warn": YELLOW, "muted": GREY}

# What kind of number a page shows: the brief requires keeping these separate.
PROVENANCE = {
    "observed": ("Observed data", "#e8f1fb", "#1d5aa3"),
    "model": ("Model estimate", "#fdf0e9", "#a8461d"),
    "causal": ("Causal estimate · under stated assumptions", "#e6f6ef", "#11734f"),
    "sim": ("Simulation · not a guarantee", "#fdf5e1", "#8a5d00"),
}

# Streamlit renamed `use_container_width` to `width="stretch"`; support both.
try:
    _V = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:  # noqa: BLE001
    _V = (1, 0)
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
FONT = "Source Sans Pro, Source Sans 3, Segoe UI, sans-serif"   # Streamlit's own font
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}

CSS = """
<style>
.stApp { background: #fcfcfb; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: -0.01em; color: #0b0b0b; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #f4f3f0; border-right: 1px solid #e4e3df; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.pg-brand { display: flex; gap: .7rem; align-items: center; padding: .35rem 0 .9rem 0;
            border-bottom: 1px solid #e4e3df; margin-bottom: .6rem; }
.pg-logo { width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center;
           background: linear-gradient(135deg, #0f2a4a 0%, #2a78d6 100%); color: #fff; font-weight: 800;
           font-size: 15px; letter-spacing: .02em; box-shadow: 0 4px 12px rgba(42,120,214,.25); }
.pg-brand-name { font-weight: 800; font-size: 1.05rem; color: #0b0b0b; line-height: 1.1; }
.pg-brand-tag { font-size: .74rem; color: #52514e; line-height: 1.25; margin-top: 2px; }
.pg-side-label { font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                 color: #8a8984; margin: 1rem 0 .3rem 0; }
.pg-side-note { font-size: .72rem; color: #6b6a66; line-height: 1.45; }

[data-testid="stSidebarHeader"] { padding-bottom: .4rem; }
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] { height: 2.6rem !important; max-width: 100%; }
[data-testid="stSidebarNav"] { border-bottom: 1px solid #e4e3df; padding-bottom: .6rem; }
section[data-testid="stSidebar"] [data-baseweb="tag"] { background: #e8f1fb !important; color: #1d5aa3 !important;
    font-size: .78rem; height: 1.55rem; }
section[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #1d5aa3 !important; }

/* ---------- page header ---------- */
.pg-hero { background: linear-gradient(120deg, #0f2a4a 0%, #1b4f8a 55%, #2a78d6 100%); color: #fff;
           border-radius: 18px; padding: 1.5rem 1.8rem 1.35rem 1.8rem; margin-bottom: 1.3rem;
           box-shadow: 0 10px 30px rgba(15,42,74,.18); position: relative; overflow: hidden; }
.pg-hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px;
                 border-radius: 50%; background: rgba(255,255,255,.06); }
.pg-eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .78; }
.pg-title { font-size: 1.85rem; font-weight: 800; margin: .25rem 0 .35rem 0; line-height: 1.15; color: #fff; }
.pg-sub { font-size: .98rem; opacity: .9; max-width: 900px; }
.pg-chips { margin-top: .85rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.pg-chip { display: inline-flex; align-items: center; gap: .35rem; font-size: .74rem; font-weight: 600;
           padding: .25rem .65rem; border-radius: 999px; background: rgba(255,255,255,.14);
           border: 1px solid rgba(255,255,255,.22); color: #fff; }
.pg-pill { display: inline-block; font-size: .72rem; font-weight: 700; padding: .2rem .6rem; border-radius: 999px; }

/* ---------- KPI cards ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .85rem; margin: .2rem 0 1.1rem 0; }
.kpi { background: #fff; border: 1px solid #e4e3df; border-radius: 14px; padding: .95rem 1.05rem .9rem 1.05rem;
       border-top: 4px solid var(--tone); box-shadow: 0 1px 2px rgba(0,0,0,.03); }
.kpi-label { font-size: .74rem; font-weight: 600; color: #52514e; text-transform: uppercase; letter-spacing: .05em;
             display: flex; justify-content: space-between; gap: .4rem; }
.kpi-help { cursor: help; color: #8a8984; font-weight: 700; }
.kpi-value { font-size: 1.75rem; font-weight: 800; color: #0b0b0b; margin-top: .3rem; line-height: 1.1;
             font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: .8rem; color: #52514e; margin-top: .3rem; }
.kpi-sub b { color: var(--tone); }

/* ---------- section titles ---------- */
.pg-section { margin: 1.4rem 0 .5rem 0; }
.pg-section-title { font-size: 1.12rem; font-weight: 750; color: #0b0b0b; }
.pg-section-cap { font-size: .86rem; color: #52514e; margin-top: .1rem; }

/* ---------- recommendation / insight cards ---------- */
.rec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .85rem; margin-bottom: 1rem; }
@media (max-width: 900px) { .rec-grid { grid-template-columns: 1fr !important; } }
.rec { background: #fff; border: 1px solid #e4e3df; border-radius: 14px; padding: 1rem 1.1rem; position: relative; }
.rec-top { display: flex; align-items: center; gap: .55rem; margin-bottom: .45rem; }
.rec-num { width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; color: #fff;
           font-weight: 800; font-size: .8rem; background: var(--tone); flex-shrink: 0; }
.rec-title { font-weight: 750; color: #0b0b0b; font-size: .98rem; line-height: 1.25; }
.rec-body { font-size: .87rem; color: #3a3936; line-height: 1.5; }
.rec-tag { margin-top: .6rem; font-size: .7rem; font-weight: 700; color: var(--tone); text-transform: uppercase;
           letter-spacing: .06em; }

/* ---------- callouts, empty state, footer ---------- */
.pg-callout { border-radius: 12px; padding: .85rem 1rem; font-size: .9rem; line-height: 1.5; margin: .4rem 0 1rem 0;
              border: 1px solid; }
.pg-empty { text-align: center; padding: 2.4rem 1rem; border: 1.5px dashed #d6d4cf; border-radius: 16px;
            color: #52514e; background: #fff; }
.pg-empty-title { font-weight: 750; color: #0b0b0b; font-size: 1.05rem; margin-bottom: .3rem; }
.pg-footer { margin-top: 2.5rem; padding-top: .9rem; border-top: 1px solid #e4e3df; font-size: .74rem; color: #8a8984;
             display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; }

/* ---------- native widgets ---------- */
[data-testid="stMetric"] { background: #fff; border: 1px solid #e4e3df; border-radius: 14px; padding: .8rem 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid #e4e3df; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: .55rem .9rem; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background: #fff; border: 1px solid #e4e3df;
    border-radius: 14px; padding: .35rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
.stDownloadButton button, .stButton button { border-radius: 10px; font-weight: 600; }
</style>
"""


def esc(x) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


ASSETS = __import__("pathlib").Path(__file__).resolve().parent / "assets"


def logo() -> None:
    """Logo above the page menu (st.logo); falls back to an HTML block on old Streamlit versions."""
    if hasattr(st, "logo"):
        try:
            st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))
        except TypeError:
            st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"))
        return
    sidebar_brand()


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="pg-brand"><div class="pg-logo">PG</div><div><div class="pg-brand-name">PromoGuard AI</div>'
        '<div class="pg-brand-tag">Causal decision intelligence for retail promotions</div></div></div>',
        unsafe_allow_html=True)


def side_label(text: str) -> None:
    st.sidebar.markdown(f'<div class="pg-side-label">{esc(text)}</div>', unsafe_allow_html=True)


def side_note(text: str) -> None:
    st.sidebar.markdown(f'<div class="pg-side-note">{text}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str, kind: str | None = None, chips: list[str] | None = None) -> None:
    items = []
    if kind:
        items.append(f'<span class="pg-chip">● {esc(PROVENANCE[kind][0])}</span>')
    items += [f'<span class="pg-chip">{esc(c)}</span>' for c in (chips or [])]
    st.markdown(
        f'<div class="pg-hero"><div class="pg-eyebrow">{esc(eyebrow)}</div><div class="pg-title">{esc(title)}</div>'
        f'<div class="pg-sub">{subtitle}</div><div class="pg-chips">{"".join(items)}</div></div>',
        unsafe_allow_html=True)


def pill(kind: str) -> str:
    label, bg, fg = PROVENANCE[kind]
    return f'<span class="pg-pill" style="background:{bg};color:{fg}">{esc(label)}</span>'


def kpi(label: str, value: str, sub: str = "", tone: str = "neutral", help: str | None = None) -> dict:
    return {"label": label, "value": value, "sub": sub, "tone": tone, "help": help}


def kpi_row(cards: list[dict], where=None) -> None:
    where = where or st
    out = []
    for c in cards:
        tip = f'<span class="kpi-help" title="{esc(c["help"])}">ⓘ</span>' if c.get("help") else ""
        out.append(f'<div class="kpi" style="--tone:{TONES.get(c["tone"], BLUE)}">'
                   f'<div class="kpi-label"><span>{esc(c["label"])}</span>{tip}</div>'
                   f'<div class="kpi-value">{esc(c["value"])}</div>'
                   f'<div class="kpi-sub">{c.get("sub", "")}</div></div>')
    where.markdown(f'<div class="kpi-grid">{"".join(out)}</div>', unsafe_allow_html=True)


def section(title: str, caption: str | None = None, kind: str | None = None, where=None) -> None:
    where = where or st
    cap = f'<div class="pg-section-cap">{caption}</div>' if caption else ""
    tag = f" &nbsp;{pill(kind)}" if kind else ""
    where.markdown(f'<div class="pg-section"><div class="pg-section-title">{esc(title)}{tag}</div>{cap}</div>',
                   unsafe_allow_html=True)


def rec_cards(items: list[dict], where=None, cols: int | None = None) -> None:
    """items: dict(title, body (HTML allowed), tag, tone). cols: fixed cards per row (stacks on small screens)."""
    where = where or st
    out = []
    for i, r in enumerate(items, 1):
        out.append(f'<div class="rec" style="--tone:{TONES.get(r.get("tone", "neutral"), BLUE)}">'
                   f'<div class="rec-top"><div class="rec-num">{i}</div><div class="rec-title">{esc(r["title"])}</div></div>'
                   f'<div class="rec-body">{r["body"]}</div>'
                   f'<div class="rec-tag">{esc(r.get("tag", ""))}</div></div>')
    grid = f' style="grid-template-columns:repeat({cols},minmax(0,1fr))"' if cols else ""
    where.markdown(f'<div class="rec-grid"{grid}>{"".join(out)}</div>', unsafe_allow_html=True)


def callout(text: str, tone: str = "neutral", where=None) -> None:
    where = where or st
    col = TONES.get(tone, BLUE)
    where.markdown(f'<div class="pg-callout" style="border-color:{col}55;background:{col}10">{text}</div>',
                   unsafe_allow_html=True)


def empty_state(title: str, body: str, where=None) -> None:
    where = where or st
    where.markdown(f'<div class="pg-empty"><div class="pg-empty-title">{esc(title)}</div>{body}</div>',
                   unsafe_allow_html=True)


def footer() -> None:
    st.markdown('<div class="pg-footer"><span>PromoGuard AI · Causal AI decision intelligence for promotions</span>'
                '<span>Data: Kilts Center for Marketing, Chicago Booth — Dominick\'s Finer Foods · academic use'
                '</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    """Title on its own line, legend on the line below it, plot underneath: they never overlap."""
    shown = [t for t in fig.data if getattr(t, "showlegend", None) is not False and getattr(t, "name", None)]
    has_legend = len(shown) > 1 or any(getattr(t, "showlegend", None) is True for t in fig.data)
    bottom_legend = len(shown) > 3            # many items: legend under the plot so it never hits the title
    top = (34 if title else 12) + (30 if has_legend and not bottom_legend else 0)
    fig.update_layout(
        height=height, margin=dict(l=8, r=30, t=top + 8, b=90 if bottom_legend else 8),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT, color=INK_2, size=12),
        title=dict(text=f"<b>{title}</b>", x=0.01, xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=12), font=dict(color=INK, size=15)) if title else None,
        legend=(dict(orientation="h", yanchor="top", y=-0.2, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                     font=dict(size=12)) if bottom_legend else
                dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                     font=dict(size=12))),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        hovermode="closest", bargap=0.28)
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", automargin=True, title_font=dict(size=12),
                     title_standoff=8)
    fig.update_yaxes(gridcolor="#efeeeb", zerolinecolor="#bdbbb6", linecolor=GRID, automargin=True,
                     title_font=dict(size=12), title_standoff=8)
    return fig


def money_axis(fig: go.Figure, axis: str = "y") -> go.Figure:
    """Format an axis as money: $4k, −$2k, $1.2M."""
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(tickformat="$~s")
    return fig


def chart(fig: go.Figure, where=None, height: int = 380, title: str | None = None) -> None:
    where = where or st
    where.plotly_chart(style(fig, height, title), config=PLOTLY_CONFIG, **WIDE)


def table(df, where=None, **kw) -> None:
    where = where or st
    where.dataframe(df, **WIDE, **kw)
