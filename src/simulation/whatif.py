"""
PromoGuard AI — what-if promotion simulator.

Location: src/simulation/whatif.py

Usage
    from src.simulation.whatif import PromoSimulator
    sim = PromoSimulator.load()                       # reads models/promo_response_model.joblib
    sim = PromoSimulator.load(cat="sdr")              # soft drinks: models/sdr_promo_response_model.joblib
    sim.simulate(product_id=8000000245, depth=0.15, duration=2, lent=False)
    sim.scenario_table(8000000245)                    # no promotion / deal only / 5% ... 40%

How a scenario is computed (all MODEL-BASED ESTIMATES, not guarantees)
    normal units   = product's normal weekly sales per store (recent regular-price weeks)
                     x stores selling x duration                          -> CF_UNITS
    lift ratio     = exp(promotion-response model prediction)            (Phase 9 model)
    units          = CF_UNITS x lift ratio                               -> 80% range from CV errors
    incremental    = units - CF_UNITS
    cannibalized   = incremental x brand diversion ratio (Phase 7)
    gross profit   = units x (promo price - unit cost + manufacturer funding per unit)
                     - CF_UNITS x (regular price - unit cost)
                     - cannibalized x average unit margin of the other products in the category
    manufacturer funding per unit = trade_funding x regular price x depth
        trade_funding = 0  -> the retailer pays the whole discount (conservative)
        trade_funding = 1  -> the manufacturer pays the whole discount
    The post-promotion effect is omitted: Phase 6 measured it at about 0.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "promo_response_model.joblib"

DEFAULT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
EXTRAPOLATION_DEPTH = 0.30   # only ~2% of past events were deeper: results beyond this are extrapolations


class PromoSimulator:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.q_lo, self.q_hi = bundle["residual_q10"], bundle["residual_q90"]
        self.products = bundle["products"].set_index("PRODUCT_ID")
        self.products["DESCRIP"] = self.products["DESCRIP"].astype(str).str.strip().str.lstrip("$~#* ").str.strip()
        self.defaults = bundle["defaults"]
        self.category = bundle.get("category", "tna")
        self.cat_cols = bundle.get("categorical", ["BRAND_GROUP", "TUNA_MEAT"])
        # depth beyond which few past events exist (tuna 30%; soft drinks: from their own history)
        self.extrapolation_depth = bundle.get("extrapolation_depth", EXTRAPOLATION_DEPTH)

    @classmethod
    def load(cls, path: Path | None = None, cat: str = "tna") -> "PromoSimulator":
        if path is None:
            path = MODEL_PATH if cat == "tna" else MODEL_PATH.with_name(f"{cat}_promo_response_model.joblib")
        return cls(joblib.load(path))

    def product_list(self) -> pd.DataFrame:
        return self.products.reset_index()[["PRODUCT_ID", "DESCRIP", "BRAND", "SIZE", "REG_PRICE", "UNIT_COST"]]

    def _features(self, p: pd.Series, depth: float, duration: int, lent: bool, deal_coded: bool,
                  month: int | None, comp_other: float | None, comp_same: float | None,
                  store_coverage: float, holiday: bool) -> pd.DataFrame:
        row = {c: p[c] for c in self.cat_cols}            # brand group + product type (tuna or soft drink)
        row.update({
            "DEPTH": depth, "PRICE_PROMO_SHARE": 1.0 if depth >= 0.05 else 0.0,
            "DEAL_CODED_SHARE": 1.0 if deal_coded else 0.0, "DURATION": float(duration),
            "LENT": float(lent), "HOLIDAY_IN_EVENT": float(holiday),
            "START_MONTH": float(month if month is not None else (3 if lent else self.defaults["month"])),
            "STORE_COVERAGE": store_coverage, "LOG_BASE_UNITS": np.log(p["BASE_UNITS_PER_STORE_WEEK"]),
            "REG_PRICE": p["REG_PRICE"], "REG_PRICE_PER_OZ": p["REG_PRICE_PER_OZ"], "SIZE_OZ": p["SIZE_OZ"],
            "COMP_OTHER_BRAND_EVENTS": comp_other if comp_other is not None else self.defaults["comp_other"],
            "COMP_SAME_BRAND_EVENTS": comp_same if comp_same is not None else self.defaults["comp_same"],
            "WEEKS_SINCE_LAST_EVENT": self.defaults["weeks_since"],
        })
        return pd.DataFrame([row])[self.features]

    def simulate(self, product_id: int, depth: float, duration: int = 1, lent: bool = False,
                 deal_coded: bool = True, trade_funding: float = 0.0, month: int | None = None,
                 comp_other: float | None = None, comp_same: float | None = None,
                 store_coverage: float = 0.95, holiday: bool = False, promote: bool = True) -> dict:
        p = self.products.loc[product_id]
        cf_units = p["BASE_UNITS_PER_STORE_WEEK"] * p["STORES_SELLING"] * duration
        reg, cost = p["REG_PRICE"], p["UNIT_COST"]
        if not promote:
            ratio = lo = hi = 1.0
        else:
            x = self._features(p, depth, duration, lent, deal_coded, month, comp_other, comp_same,
                               store_coverage, holiday)
            pred = float(self.model.predict(x)[0])
            ratio, lo, hi = np.exp(pred), np.exp(pred + self.q_lo), np.exp(pred + self.q_hi)

        def outcome(r: float) -> dict:
            units = cf_units * r
            inc = units - cf_units
            cannib = -p["DIVERSION"] * max(inc, 0) if promote else 0.0
            price = reg * (1 - depth) if promote else reg
            funding = trade_funding * reg * depth if promote else 0.0
            gp_promo = units * (price - cost + funding)
            gp_cf = cf_units * (reg - cost)
            net = gp_promo - gp_cf - cannib * p["OTHER_UNIT_GP"]
            disc = units * reg * depth if promote else 0.0
            return {"units": units, "incremental_units": inc, "cannibalized_units": cannib,
                    "net_category_units": inc - cannib, "revenue": units * price,
                    "gross_profit": gp_promo, "net_gp_impact": net, "discount_cost": disc,
                    "gp_per_discount_dollar": net / disc if disc > 0 else np.nan}

        mid, low, high = outcome(ratio), outcome(lo), outcome(hi)
        res = {"product_id": product_id, "description": p["DESCRIP"], "depth": depth if promote else 0.0,
               "duration": duration, "lent": lent, "deal_coded": deal_coded and promote,
               "trade_funding": trade_funding, "promoted": promote,
               "regular_price": reg, "promo_price": reg * (1 - depth) if promote else reg, "unit_cost": cost,
               "cf_units": cf_units, "lift_ratio": ratio, "lift_pct": (ratio - 1) * 100,
               "extrapolation": bool(promote and depth > self.extrapolation_depth)}
        res.update(mid)
        res["units_low"], res["units_high"] = low["units"], high["units"]
        res["net_gp_impact_low"] = min(low["net_gp_impact"], high["net_gp_impact"])
        res["net_gp_impact_high"] = max(low["net_gp_impact"], high["net_gp_impact"])
        res["outcome_low"], res["outcome_high"] = low, high      # raw outcomes at the 10% / 90% unit levels
        return res

    def scenario_table(self, product_id: int, depths: list[float] = DEFAULT_GRID, **kw) -> pd.DataFrame:
        rows = [self.simulate(product_id, 0.0, promote=False, **{k: v for k, v in kw.items()
                                                               if k in ("duration", "lent", "trade_funding")})]
        rows[0]["scenario"] = "no promotion"
        for d in depths:
            r = self.simulate(product_id, d, **kw)
            r["scenario"] = "deal flag, no price cut" if d == 0 else f"{d:.0%} off"
            rows.append(r)
        cols = ["scenario", "promo_price", "units", "units_low", "units_high", "lift_pct", "incremental_units",
                "cannibalized_units", "net_category_units", "revenue", "net_gp_impact", "net_gp_impact_low",
                "net_gp_impact_high", "discount_cost", "gp_per_discount_dollar", "extrapolation"]
        return pd.DataFrame(rows)[cols]

    # ------------------------------------------------------------------
    # Break-even manufacturer funding
    # ------------------------------------------------------------------
    def break_even(self, product_id: int, depth: float, **kw) -> dict:
        """Share of the discount the manufacturer must fund so the retailer's net GP impact is >= 0.

        Net GP is linear in funding: net(f) = net(0) + f x discount_cost, so
        break-even share = max(0, -net(0) / discount_cost). Computed at the expected units and at the
        10% / 90% unit levels (range). A share > 1 means the promotion cannot break even even if the
        manufacturer pays the whole discount.
        """
        kw = {k: v for k, v in kw.items() if k != "trade_funding"}
        r = self.simulate(product_id, depth, trade_funding=0.0, **kw)

        def share(o: dict) -> float:
            d = o["discount_cost"]
            return max(0.0, -o["net_gp_impact"] / d) if d > 0 else np.nan

        mid, a, b = share(r), share(r["outcome_low"]), share(r["outcome_high"])
        return {"product_id": product_id, "description": r["description"], "depth": depth,
                "net_gp_retailer_pays": r["net_gp_impact"], "discount_cost": r["discount_cost"],
                "break_even_share": mid, "break_even_low": np.nanmin([a, b]), "break_even_high": np.nanmax([a, b]),
                "reachable": bool(mid <= 1) if not np.isnan(mid) else False, "extrapolation": r["extrapolation"]}

    def break_even_curve(self, product_id: int, depths: list[float] | None = None, **kw) -> pd.DataFrame:
        depths = depths or [d for d in DEFAULT_GRID if d > 0]
        return pd.DataFrame([self.break_even(product_id, d, **kw) for d in depths])

    def negotiation_sheet(self, depth: float, **kw) -> pd.DataFrame:
        """Break-even funding for every active product at one promotion design."""
        out = pd.DataFrame([self.break_even(pid, depth, **kw) for pid in self.products.index])
        info = self.product_list().rename(columns={"PRODUCT_ID": "product_id"})
        return out.merge(info[["product_id", "BRAND", "SIZE", "REG_PRICE"]], on="product_id", how="left") \
            .sort_values("break_even_share")
