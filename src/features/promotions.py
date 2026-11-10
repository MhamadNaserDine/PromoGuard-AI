"""
PromoGuard AI — promotion definition.

Location: src/features/promotions.py

Why a constructed definition is needed
--------------------------------------
The deal-code field (SALE) is not a reliable promotion label. The manual says it
"is not set by DFF on a consistent basis", and in the tuna data only ~45% of
Bonus-Buy weeks show a price cut, while 8% of uncoded weeks do show one.
We therefore combine two sources of evidence: the deal code and the observed
price relative to the product's regular price in that store.

Definitions (all at PRODUCT_ID x STORE x WEEK level)
----------------------------------------------------
REG_PRICE   Regular (reference) price = the MODAL unit price of the previous
            REG_WINDOW observed weeks of the same product in the same store
            (ties -> the higher price). Backward-looking only: it never uses the
            current or future weeks. Needs >= REG_MIN_OBS past weeks, else NaN.
            The modal price is the standard reference-price measure in the
            retail-pricing literature because temporary sales do not move it.
DEPTH       1 - UNIT_PRICE / REG_PRICE   (0.20 = 20% below regular)
PRICE_CUT   DEPTH >= CUT_THRESHOLD
TEMPORARY   within the next RETURN_WINDOW observed weeks the price returns to
            >= (1 - CUT_THRESHOLD) x the regular price of the CUT week.
            Cuts that do not return are permanent price changes, not promotions. NOTE: this looks FORWARD. It is
            used only to label past events. It must never be used as a model
            feature (that would be data leakage).
PROMO_TYPE  price_promo   PRICE_CUT and TEMPORARY (with or without a deal code)
            deal_only     deal code B/S/G/C but no price cut (non-price support,
                          e.g. display or feature; the data does not say which)
            permanent_cut price cut that does not return to regular (price change, not a promo)
            regular       everything else with a known REG_PRICE
            no_history    REG_PRICE unknown (first weeks of a series)
PROMO       PROMO_TYPE in {price_promo, deal_only}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REG_WINDOW = 12        # weeks of history used for the modal regular price
REG_MIN_OBS = 4        # minimum history to define a regular price
CUT_THRESHOLD = 0.05   # 5% below regular = price cut
RETURN_WINDOW = 8      # a cut is temporary if price returns within 8 observed weeks

# Strongest deal code kept when barcode versions are combined. "C" (coupon) appears in soft drinks only;
# adding it does not change the relative order of B < S < G used for canned tuna.
DEAL_PRIORITY = {"NONE": 0, "C": 1, "B": 2, "S": 3, "G": 4}
DEAL_FROM_PRIORITY = {v: k for k, v in DEAL_PRIORITY.items()}


def aggregate_to_product(mv: pd.DataFrame) -> pd.DataFrame:
    """
    UPC x store x week  ->  PRODUCT_ID x store x week.
    Several UPCs can share a PRODUCT_ID (barcode changes); in 1% of cases both
    sell in the same week. Units, dollars and gross profit are summed; the unit
    price is the sales-weighted average; the strongest deal code is kept.
    """
    df = mv.assign(_deal=mv["DEAL_CODE"].map(DEAL_PRIORITY).fillna(0).astype(int))
    flags = [c for c in ("NEG_MARGIN_FLAG", "EXTREME_MARGIN_FLAG", "MOVE_OUTLIER_FLAG") if c in df.columns]
    agg = {"MOVE": ("MOVE", "sum"), "DOLLARS": ("DOLLARS", "sum"), "GROSS_PROFIT": ("GROSS_PROFIT", "sum"),
           "_deal": ("_deal", "max"), "N_UPC": ("UPC", "nunique"),
           "WEEK_START": ("WEEK_START", "first"), "HOLIDAY": ("HOLIDAY", "first")}
    agg.update({f: (f, "max") for f in flags})
    out = df.groupby(["PRODUCT_ID", "STORE", "WEEK"], sort=True).agg(**agg).reset_index()
    out["UNIT_PRICE"] = (out["DOLLARS"] / out["MOVE"]).round(2)
    out["MARGIN_PCT"] = out["GROSS_PROFIT"] / out["DOLLARS"] * 100
    out["DEAL_CODE"] = out.pop("_deal").map(DEAL_FROM_PRIORITY)
    return out


def trailing_modal_price(panel: pd.DataFrame, window: int = REG_WINDOW,
                         min_obs: int = REG_MIN_OBS, chunk: int = 200_000) -> np.ndarray:
    """
    Modal UNIT_PRICE over the previous `window` observations of each
    PRODUCT_ID x STORE series (current week excluded). Vectorised.
    Requires `panel` sorted by PRODUCT_ID, STORE, WEEK.
    """
    price = panel["UNIT_PRICE"].to_numpy(dtype=float)
    series = panel.groupby(["PRODUCT_ID", "STORE"], sort=False).ngroup().to_numpy()
    n = len(price)
    idx = np.arange(n)
    series_start = idx - pd.Series(series).groupby(series).cumcount().to_numpy()
    out = np.full(n, np.nan)
    offsets = np.arange(1, window + 1)

    for k in range(0, n, chunk):
        rows = idx[k:k + chunk]
        lags = rows[:, None] - offsets[None, :]
        valid = lags >= series_start[rows][:, None]
        win = np.where(valid, price[np.clip(lags, 0, None)], np.nan)
        counts = (win[:, :, None] == win[:, None, :]).sum(axis=2).astype(float)
        counts[np.isnan(win)] = -1
        score = counts * 1_000 + np.nan_to_num(win, nan=-1.0)   # most frequent, then highest price
        best = np.argmax(score, axis=1)
        mode = win[np.arange(len(rows)), best]
        mode[valid.sum(axis=1) < min_obs] = np.nan
        out[rows] = mode
    return out


def label_promotions(panel: pd.DataFrame) -> pd.DataFrame:
    """Add REG_PRICE, DEPTH, PRICE_CUT, TEMPORARY, PROMO_TYPE, PROMO."""
    p = panel.sort_values(["PRODUCT_ID", "STORE", "WEEK"]).reset_index(drop=True)
    p["REG_PRICE"] = trailing_modal_price(p)
    p["DEPTH"] = 1 - p["UNIT_PRICE"] / p["REG_PRICE"]
    p["PRICE_CUT"] = p["DEPTH"].ge(CUT_THRESHOLD)

    # Compare FUTURE prices with the regular price AT THE TIME OF THE CUT. (Comparing with
    # the future regular price would wrongly call a permanent price cut "temporary", because
    # the modal price itself moves down a few weeks after a permanent cut.)
    threshold_now = ((1 - CUT_THRESHOLD) * p["REG_PRICE"]).to_numpy()
    grp = p.groupby(["PRODUCT_ID", "STORE"])["UNIT_PRICE"]
    returns = np.zeros(len(p), dtype=bool)
    for k in range(1, RETURN_WINDOW + 1):
        future_price = grp.shift(-k).to_numpy()
        returns |= np.nan_to_num(future_price, nan=-1.0) >= threshold_now
    p["TEMPORARY"] = p["PRICE_CUT"] & returns

    has_code = p["DEAL_CODE"].ne("NONE")
    p["PROMO_TYPE"] = np.select(
        [p["REG_PRICE"].isna(), p["PRICE_CUT"] & p["TEMPORARY"], has_code & ~p["PRICE_CUT"],
         p["PRICE_CUT"] & ~p["TEMPORARY"]],
        ["no_history", "price_promo", "deal_only", "permanent_cut"],
        default="regular",
    )
    p["PROMO"] = p["PROMO_TYPE"].isin(["price_promo", "deal_only"])
    return p


def build_events(p: pd.DataFrame, min_store_share: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chain-level promotion EVENTS. Dominick's set promoted prices chainwide (manual,
    Part 6), so a promotion is a product-level decision. A product-week is an event
    week if at least `min_store_share` of the stores selling the product that week
    are on PROMO. Consecutive event weeks of the same product form one event.
    Returns (event table, panel with EVENT_ID; rows outside events get EVENT_ID = -1).
    """
    pw = (p[p["PROMO_TYPE"].ne("no_history")]
          .groupby(["PRODUCT_ID", "WEEK"])
          .agg(STORES=("STORE", "nunique"), PROMO_SHARE=("PROMO", "mean"))
          .reset_index())
    pw = pw[pw["PROMO_SHARE"].ge(min_store_share)].sort_values(["PRODUCT_ID", "WEEK"])
    new_event = pw["PRODUCT_ID"].ne(pw["PRODUCT_ID"].shift()) | pw["WEEK"].diff().ne(1)
    pw["EVENT_ID"] = new_event.cumsum()

    p = p.merge(pw[["PRODUCT_ID", "WEEK", "EVENT_ID"]], on=["PRODUCT_ID", "WEEK"], how="left")
    p["EVENT_ID"] = np.where(p["PROMO"], p["EVENT_ID"], np.nan)
    p["EVENT_ID"] = p["EVENT_ID"].fillna(-1).astype(int)

    ev_rows = p[p["EVENT_ID"].ge(0)]
    events = ev_rows.groupby("EVENT_ID").agg(
        PRODUCT_ID=("PRODUCT_ID", "first"),
        START_WEEK=("WEEK", "min"), END_WEEK=("WEEK", "max"),
        START_DATE=("WEEK_START", "min"),
        STORES=("STORE", "nunique"),
        MEAN_DEPTH=("DEPTH", "mean"),
        SHARE_PRICE_PROMO=("PROMO_TYPE", lambda s: (s == "price_promo").mean()),
        SHARE_DEAL_CODED=("DEAL_CODE", lambda s: (s != "NONE").mean()),
        MAIN_DEAL_CODE=("DEAL_CODE", lambda s: s.value_counts().index[0]),
        UNITS=("MOVE", "sum"), DOLLARS=("DOLLARS", "sum"), GROSS_PROFIT=("GROSS_PROFIT", "sum"),
    ).reset_index()
    events["DURATION_WEEKS"] = events["END_WEEK"] - events["START_WEEK"] + 1
    return events, p
