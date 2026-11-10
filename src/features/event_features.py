"""
PromoGuard AI — event-level features for the promotion-response model.

Location: src/features/event_features.py

One row per chain-level promotion event. Every feature is a property of the
promotion DESIGN or its CONTEXT, known when the promotion is planned:
depth, duration, type, timing (Lent / holiday / month), the product (brand, type,
size, price), its normal sales level, and how busy the promotion calendar is.
The target is the measured lift from Phase 6 (log of actual / counterfactual units).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["BRAND_GROUP", "TUNA_MEAT"]
NUMERIC = [
    "DEPTH", "PRICE_PROMO_SHARE", "DEAL_CODED_SHARE", "DURATION", "LENT", "HOLIDAY_IN_EVENT", "START_MONTH",
    "STORE_COVERAGE", "LOG_BASE_UNITS", "REG_PRICE", "REG_PRICE_PER_OZ", "SIZE_OZ",
    "COMP_OTHER_BRAND_EVENTS", "COMP_SAME_BRAND_EVENTS", "WEEKS_SINCE_LAST_EVENT",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "LOG_LIFT"
MAJOR_BRANDS = ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's"]
ATTRS = ["TUNA_MEAT"]          # product-type columns from the product table (categorical features)
CAT = "tna"

# Category settings (tuna is the default; set_category switches the lists IN PLACE so every
# module that imported them sees the same values)
CATEGORY_SETTINGS = {
    "tna": {"attrs": ["TUNA_MEAT"], "majors": ["StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's"],
            "labels": {"TUNA_MEAT": "Tuna type (light/white)", "SIZE_OZ": "Can size (oz)"}},
    "sdr": {"attrs": ["SD_DIET", "SD_PACK"], "majors": ["Coca-Cola Co.", "PepsiCo", "7-Up", "Dominick's"],
            "labels": {"SD_DIET": "Diet or regular", "SD_PACK": "Package (cans, bottles)",
                       "SIZE_OZ": "Package size (oz)", "LENT": "During Lent"}},
}

LABELS = {   # plain-language names for charts and the dashboard
    "DEPTH": "Discount depth", "PRICE_PROMO_SHARE": "Stores with a price cut (%)",
    "DEAL_CODED_SHARE": "Stores with a deal code (%)", "DURATION": "Duration (weeks)",
    "LENT": "During Lent", "HOLIDAY_IN_EVENT": "Holiday week", "START_MONTH": "Month",
    "STORE_COVERAGE": "Share of stores promoting", "LOG_BASE_UNITS": "Normal sales level (log)",
    "REG_PRICE": "Regular price", "REG_PRICE_PER_OZ": "Regular price per oz", "SIZE_OZ": "Can size (oz)",
    "COMP_OTHER_BRAND_EVENTS": "Competitor promos (same weeks)", "COMP_SAME_BRAND_EVENTS":
        "Same-brand promos (same weeks)", "WEEKS_SINCE_LAST_EVENT": "Weeks since last promotion",
    "BRAND_GROUP": "Brand", "TUNA_MEAT": "Tuna type (light/white)",
}


def set_category(cat: str) -> None:
    """Switch to another product category ('tna' canned tuna, 'sdr' soft drinks)."""
    global CAT
    CAT = cat
    cfg = CATEGORY_SETTINGS[cat]
    ATTRS[:] = cfg["attrs"]
    CATEGORICAL[:] = ["BRAND_GROUP", *ATTRS]
    FEATURES[:] = CATEGORICAL + NUMERIC
    MAJOR_BRANDS[:] = cfg["majors"]
    LABELS.update(cfg["labels"])


def total_oz(df: pd.DataFrame) -> pd.Series:
    """Total ounces in the package: tuna SIZE_OZ is per can (x PACK_COUNT); soft drinks SIZE_OZ is already total."""
    if CAT == "tna":
        return df["SIZE_OZ"] * df["PACK_COUNT"].fillna(1)
    return df["SIZE_OZ"]


def _calendar_load(events: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Average number of OTHER products' chain events running during each event (other / same brand).
    Tuna: brand = product brand. Soft drinks: brand = brand group (all Coca-Cola Co. brands count as one)."""
    e = events[["EVENT_ID", "PRODUCT_ID", "START_WEEK", "END_WEEK"]].copy()
    if CAT == "tna":
        e["BRAND"] = e["PRODUCT_ID"].map(products.set_index("UPC")["BRAND"])
    else:
        e["BRAND"] = events["BRAND_GROUP"].to_numpy()
    rows = [(r.EVENT_ID, r.PRODUCT_ID, r.BRAND, w) for r in e.itertuples(index=False)
            for w in range(r.START_WEEK, r.END_WEEK + 1)]
    ew = pd.DataFrame(rows, columns=["EVENT_ID", "PRODUCT_ID", "BRAND", "WEEK"])
    per_week = ew.groupby("WEEK").size()
    per_week_brand = ew.groupby(["WEEK", "BRAND"]).size()
    ew["ALL"] = ew["WEEK"].map(per_week) - 1
    ew["SAME"] = [per_week_brand.get((w, b), 1) - 1 for w, b in zip(ew["WEEK"], ew["BRAND"])]
    ew["OTHER"] = ew["ALL"] - ew["SAME"]
    return ew.groupby("EVENT_ID")[["OTHER", "SAME"]].mean() \
        .rename(columns={"OTHER": "COMP_OTHER_BRAND_EVENTS", "SAME": "COMP_SAME_BRAND_EVENTS"})


def build_event_features(ev: pd.DataFrame, products: pd.DataFrame, weeks: pd.DataFrame,
                         rows: pd.DataFrame) -> pd.DataFrame:
    """ev: Phase 8 event table (tna_event_profit.csv). rows: Phase 6 row table."""
    e = ev.copy()
    attr = products.drop_duplicates("UPC")[["UPC", *ATTRS, "SIZE_OZ", "PACK_COUNT"]] \
        .rename(columns={"UPC": "PRODUCT_ID"})
    e = e.merge(attr, on="PRODUCT_ID", how="left")
    if CAT == "tna" or "BRAND_GROUP" not in e.columns:
        e["BRAND_GROUP"] = np.where(e["BRAND"].isin(MAJOR_BRANDS), e["BRAND"], "Other")
    # soft drinks: BRAND_GROUP (Coca-Cola Co. / PepsiCo / 7-Up / Dominick's / Other) comes from Phase 6
    for c in ATTRS:
        e[c] = e[c].fillna("unknown")

    e["DEPTH"] = e["MEAN_DEPTH"].clip(lower=0)
    e["PRICE_PROMO_SHARE"] = e["SHARE_PRICE_PROMO"]
    e["DEAL_CODED_SHARE"] = e["SHARE_DEAL_CODED"]
    e["DURATION"] = e["DURATION_WEEKS"]
    e["LENT"] = e["LENT"].astype(float)
    hol = weeks.set_index("WEEK")["HOLIDAY"].notna()
    e["HOLIDAY_IN_EVENT"] = [float(hol.reindex(range(s, t + 1)).fillna(False).any())
                             for s, t in zip(e["START_WEEK"], e["END_WEEK"])]
    e["START_MONTH"] = pd.to_datetime(e["START_DATE"]).dt.month.astype(float)
    e["STORE_COVERAGE"] = (e["STORES"] / e["STORES_SELLING"]).clip(upper=1)
    e["LOG_BASE_UNITS"] = np.log(e["PROMO_CF"] / (e["STORES_SELLING"] * e["DURATION"])).replace(-np.inf, np.nan)

    promo = rows[rows["PHASE"].eq("promo")]
    reg = (promo.assign(w=promo["PRED_CF"].clip(lower=0.01))
           .groupby("EVENT_ID").apply(lambda g: np.average(g["REG_PRICE"].fillna(g["UNIT_PRICE"]), weights=g["w"]),
                                      include_groups=False))
    e["REG_PRICE"] = e["EVENT_ID"].map(reg)
    e["REG_PRICE_PER_OZ"] = e["REG_PRICE"] / total_oz(e).replace(0, np.nan)

    e = e.join(_calendar_load(ev, products), on="EVENT_ID")
    prev_end = (e.sort_values("START_WEEK").groupby("PRODUCT_ID")["END_WEEK"].shift(1))
    e["WEEKS_SINCE_LAST_EVENT"] = (e["START_WEEK"] - prev_end.reindex(e.index)).clip(upper=52).fillna(52)

    e[TARGET] = np.log(e["PROMO_ACTUAL"] / e["PROMO_CF"])
    return e
