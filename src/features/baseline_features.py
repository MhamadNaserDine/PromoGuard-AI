"""
PromoGuard AI — features for the baseline (counterfactual) demand model.

Location: src/features/baseline_features.py

Goal of the baseline model
--------------------------
Predict how many units a product would sell in a store-week WITHOUT its own
promotion. The model is trained on regular-price weeks only. For a promotion
week, its prediction is the counterfactual "sales without promotion".

Leakage rules (every feature must be known without the current week's sales)
-----------------------------------------------------------------------------
* History features use only PREVIOUS weeks (shift(1)) and only REGULAR weeks,
  so past promotion spikes do not inflate the baseline level.
* REG_PRICE is already backward-looking (Phase 3).
* CHAIN_PROMO_SHARE = share of OTHER stores promoting this same product this week.
  Promoted prices are set chainwide, so in some stores the promoted price equals
  their (already low) regular price: the row looks "regular" but the product is
  on display / in the weekly ad and sells far above normal (e.g. product
  4800000024, week 231: 46 "regular" stores at $0.79 selling ~10x normal).
  The feature lets the model see this; the counterfactual sets it to 0.
* Competitor promotion features describe what OTHER products do in the same
  store-week. These are chain decisions announced in advance (weekly ad), not
  outcomes of this product's sales, so they are legitimate inputs.
* Product volume weights are computed on the TRAINING period only.
* TEMPORARY (Phase 3) looks forward and is never used here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HIST_SHORT = 13      # weeks of regular-week history (one quarter)
HIST_LONG = 52       # one year
WEEKS_SINCE_CAP = 52

# Product attribute columns per category (from the Phase 2 product table)
CATEGORY_ATTRS = {
    "tna": ["TUNA_MEAT", "TUNA_STYLE", "TUNA_PACK"],      # canned tuna: light/white, chunk/solid, water/oil
    "sdr": ["SD_DIET", "SD_FLAVOR", "SD_PACK"],          # soft drinks: diet/regular, flavor, package type
}
PRODUCT_ATTRS = list(CATEGORY_ATTRS["tna"])
CATEGORICAL = ["PRODUCT_ID", "STORE", "BRAND", *PRODUCT_ATTRS, "PRICE_TIER"]
NUMERIC = [
    # history (regular weeks only, strictly past)
    "BASE_LOG_13", "BASE_LOG_52", "LAST_REG_LOG", "N_REG_HIST",
    # own promotion history (post-promotion dip / stockpiling)
    "WEEKS_SINCE_PROMO", "PROMOS_LAST_13",
    # price
    "REG_PRICE", "PRICE_INDEX", "REG_PRICE_PER_OZ",
    # product
    "SIZE_OZ", "PACK_COUNT",
    # competition in the same store-week
    "N_OTHER_PROMO", "OTHER_BRAND_PROMO_W", "SAME_BRAND_PROMO_W", "OTHER_MAX_DEPTH",
    # same product promoted in other stores of the chain this week
    "CHAIN_PROMO_SHARE",
    # calendar
    "LENT_WEEK", "LENT_DAYS", "WEEKS_TO_EASTER_C", "WEEK_OF_YEAR",
    "HOL_THANKSGIVING", "HOL_CHRISTMAS", "HOL_NEW_YEAR", "HOL_EASTER", "HOL_SUMMER",
]
TARGET = "LOG_UNITS"
# Note: no raw time-trend feature. Tree models cannot extrapolate a trend beyond the
# training years; the level is carried by the regular-week history features instead.


def set_category(cat: str) -> None:
    """Switch the product attribute features (tuna is the default). Lists are changed in place,
    so every module that imported CATEGORICAL sees the same columns."""
    PRODUCT_ATTRS[:] = CATEGORY_ATTRS[cat]
    CATEGORICAL[:] = ["PRODUCT_ID", "STORE", "BRAND", *PRODUCT_ATTRS, "PRICE_TIER"]


def _group_roll_mean(s: pd.Series, keys: list[pd.Series], window: int) -> pd.Series:
    """Mean of the previous `window` observations within each group (current row excluded)."""
    return s.groupby(keys).transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())


def build_features(panel: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame,
                   weeks: pd.DataFrame, train_end_week: int) -> pd.DataFrame:
    """Return the panel with all model features. Sorted by PRODUCT_ID, STORE, WEEK."""
    d = panel.sort_values(["PRODUCT_ID", "STORE", "WEEK"]).reset_index(drop=True).copy()
    keys = [d["PRODUCT_ID"], d["STORE"]]

    # ---- target -----------------------------------------------------------
    d[TARGET] = np.log1p(d["MOVE"])

    # ---- history from regular weeks only, strictly before the current week ---
    reg_log = d[TARGET].where(d["PROMO_TYPE"].eq("regular"))
    d["BASE_LOG_13"] = _group_roll_mean(reg_log, keys, HIST_SHORT)
    d["BASE_LOG_52"] = _group_roll_mean(reg_log, keys, HIST_LONG)
    d["LAST_REG_LOG"] = reg_log.groupby(keys).transform(lambda x: x.shift(1).ffill())
    d["N_REG_HIST"] = reg_log.notna().astype(int).groupby(keys).transform(lambda x: x.shift(1).cumsum()).fillna(0)

    # ---- own promotion history -----------------------------------------------
    promo_week = d["WEEK"].where(d["PROMO"])
    last_promo = promo_week.groupby(keys).transform(lambda x: x.shift(1).ffill())
    d["WEEKS_SINCE_PROMO"] = (d["WEEK"] - last_promo).clip(upper=WEEKS_SINCE_CAP).fillna(WEEKS_SINCE_CAP)
    d["PROMOS_LAST_13"] = _group_roll_mean(d["PROMO"].astype(float), keys, HIST_SHORT)

    # ---- product & store attributes ------------------------------------------
    attr = products[["UPC", "BRAND", *PRODUCT_ATTRS, "SIZE_OZ", "PACK_COUNT"]] \
        .rename(columns={"UPC": "PRODUCT_ID"})
    d = d.merge(attr, on="PRODUCT_ID", how="left")
    d = d.merge(stores[["STORE", "PRICE_TIER"]], on="STORE", how="left")
    d["PRICE_TIER"] = d["PRICE_TIER"].fillna("unknown")
    for c in [*PRODUCT_ATTRS, "BRAND"]:
        d[c] = d[c].fillna("unknown")

    # ---- price -----------------------------------------------------------------
    d["PRICE_INDEX"] = d["UNIT_PRICE"] / d["REG_PRICE"]
    total_oz = d["SIZE_OZ"].where(d["SIZE_OZ"] > 0)
    d["REG_PRICE_PER_OZ"] = d["REG_PRICE"] / total_oz

    # ---- competition in the same store-week ------------------------------------
    train_units = d.loc[d["WEEK"].le(train_end_week)].groupby("PRODUCT_ID")["MOVE"].sum()
    weight = (train_units / train_units.sum()).rename("VOL_W")
    d = d.merge(weight, left_on="PRODUCT_ID", right_index=True, how="left")
    d["VOL_W"] = d["VOL_W"].fillna(0.0)

    promo = d["PROMO"].astype(int)
    w_promo = d["VOL_W"] * promo
    sw = [d["STORE"], d["WEEK"]]
    swb = [d["STORE"], d["WEEK"], d["BRAND"]]
    d["N_OTHER_PROMO"] = promo.groupby(sw).transform("sum") - promo
    total_w = w_promo.groupby(sw).transform("sum") - w_promo
    brand_w = w_promo.groupby(swb).transform("sum") - w_promo
    d["SAME_BRAND_PROMO_W"] = brand_w
    d["OTHER_BRAND_PROMO_W"] = total_w - brand_w
    depth_promo = d["DEPTH"].where(d["PROMO_TYPE"].eq("price_promo"))
    # max depth among OTHER products: max over the store-week, excluding own row
    top2 = (depth_promo.fillna(-1).groupby(sw)
            .transform(lambda x: np.sort(x.to_numpy())[-2] if len(x) > 1 else -1))
    top1 = depth_promo.fillna(-1).groupby(sw).transform("max")
    own = depth_promo.fillna(-1)
    d["OTHER_MAX_DEPTH"] = np.where(own.eq(top1), top2, top1).clip(min=0)

    # ---- same product promoted elsewhere in the chain this week ---------------------
    pw = d.groupby(["PRODUCT_ID", "WEEK"])["PROMO"].agg(["sum", "size"])
    pw.columns = ["_n_promo", "_n_rows"]
    d = d.join(pw, on=["PRODUCT_ID", "WEEK"])
    others = (d["_n_rows"] - 1).replace(0, np.nan)
    d["CHAIN_PROMO_SHARE"] = ((d["_n_promo"] - d["PROMO"].astype(int)) / others).fillna(0.0)
    d = d.drop(columns=["_n_promo", "_n_rows"])

    # ---- calendar ----------------------------------------------------------------
    w = weeks[["WEEK", "LENT_WEEK", "LENT_DAYS", "WEEKS_TO_EASTER", "WEEK_OF_YEAR", "HOLIDAY", "COVERAGE_OK"]] \
        .rename(columns={"HOLIDAY": "HOLIDAY_W"})
    d = d.merge(w, on="WEEK", how="left")
    d["LENT_WEEK"] = d["LENT_WEEK"].astype(float)
    d["WEEKS_TO_EASTER_C"] = d["WEEKS_TO_EASTER"].clip(-8, 16)
    d["TIME_INDEX"] = d["WEEK"].astype(float)
    hol = d["HOLIDAY_W"].fillna("")
    d["HOL_THANKSGIVING"] = hol.eq("Thanksgiving").astype(float)
    d["HOL_CHRISTMAS"] = hol.eq("Christmas").astype(float)
    d["HOL_NEW_YEAR"] = hol.eq("New-Year").astype(float)
    d["HOL_EASTER"] = hol.eq("Easter").astype(float)
    d["HOL_SUMMER"] = hol.isin(["Memorial Day", "4th of July", "Labor Day"]).astype(float)

    d = d.sort_values(["PRODUCT_ID", "STORE", "WEEK"]).reset_index(drop=True)
    return d


def counterfactual_view(d: pd.DataFrame) -> pd.DataFrame:
    """
    Feature view of the SAME rows under 'no own promotion this week':
    the shelf price is the regular price (PRICE_INDEX = 1). History, competition
    and calendar features are unchanged (they do not depend on this week's own promotion).
    """
    cf = d.copy()
    cf["PRICE_INDEX"] = 1.0
    return cf
