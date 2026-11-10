"""
PromoGuard AI — Phase 10: personalized discounts by customer group.

Location: src/analysis/10_customer_groups.py
Run from the repo root:  python src/analysis/10_customer_groups.py
Needs: data/raw/dominicks/demo/demo.dta (Kilts Center "demo(stata).zip", unzipped or zipped)
       Phase 6 rows and Phase 9 model.

Why customer GROUPS and not individual customers
    Dominick's scanner data is store x week: there are no customer IDs. The store-level
    demographics (1990 census, processed by Market Metrics) describe the customers living
    around each store, so stores can be grouped by the kind of customers they serve.
    Ethnicity / race variables are deliberately NOT used (no pricing by ethnicity).

Steps
    1. Customer groups: k-means (k chosen by silhouette, 3-6) on standardized store
       demographics: income, education, age 60+, household size, poverty, working women,
       avid shoppers. Groups are named from their profile.
    2. Promotion response by group (causal estimate, within-promotion comparison):
       store x event log-lift  y = log(actual / counterfactual)  (Phase 6),
       regressed on group dummies, depth and group x depth, with EVENT fixed effects
       (same product, same weeks, same chain price) and weights = counterfactual units.
       -> response multiplier of each group relative to the chain average, by depth.
       Uncertainty: cluster bootstrap over events (300 draws).
    3. Personalized discount: for every active product and group, the depth (0-40%) that
       maximizes the group's gross-profit impact of a 2-week promotion, using
       Phase 9 product response x group multiplier, the group's own regular price and
       sales volume, chain unit cost and cannibalization (retailer pays the discount).
       Compared with ONE chain-wide depth -> value of personalization.

Outputs
    data/processed/tna_store_groups.csv
    reports/phase10/group_profiles.csv, group_response.csv, group_lift_by_depth.csv,
    personalized_discounts.csv, phase10_report.txt
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.simulation.whatif import EXTRAPOLATION_DEPTH, PromoSimulator  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "dominicks" / "demo"
PROC = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "reports" / "phase10"

FEATURES = ["income", "educ", "age60", "hsizeavg", "poverty", "workwom", "shpavid"]
FEATURE_LABELS = {"income": "Median income ($)", "educ": "College graduates", "age60": "Age 60+",
                  "hsizeavg": "Household size", "poverty": "Income under $15k", "workwom": "Working women",
                  "shpavid": "Avid shoppers"}
DEPTHS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
WEEKS = 2
N_BOOT = 300
SEED = 42


# ----------------------------------------------------------------------------
def load_demo() -> pd.DataFrame:
    dta = RAW / "demo.dta"
    if not dta.exists():
        z = RAW / "demo_stata.zip"
        if not z.exists():
            sys.exit(f"Missing {dta} — download demo(stata).zip from the Kilts Center into {RAW}")
        with zipfile.ZipFile(z) as f:
            f.extract("demo.dta", RAW)
    d = pd.read_stata(dta, convert_categoricals=False)
    d = d[d["store"].notna()].copy()
    d["store"] = d["store"].astype(int)
    d["city"] = d["city"].astype(str).str.strip().str.title()
    d["lat"], d["lon"] = d["lat"] / 1e4, -d["long"] / 1e4        # stored as 419081 -> 41.9081 N, 87.8131 W
    return d.drop_duplicates("store")


def name_groups(centers: pd.DataFrame) -> dict[int, str]:
    """Readable names from cluster centres (z-scores). Assigned greedily, most distinctive first."""
    rules = [("Young families", lambda c: c["hsizeavg"] + 0.5 * c["shpavid"]),
             ("Urban low-income", lambda c: c["poverty"] - c["hsizeavg"]),
             ("Affluent professionals", lambda c: c["income"] + c["educ"]),
             ("Older, budget-conscious", lambda c: c["age60"] - c["income"])]
    names, free = {}, set(centers.index)
    for label, score in rules[: len(centers)]:
        g = max(free, key=lambda i: score(centers.loc[i]))
        names[g] = label
        free.remove(g)
    for i, g in enumerate(sorted(free)):
        names[g] = f"Group {i + 1}"
    return names


def make_groups(demo: pd.DataFrame, stores_in_data: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = demo[demo["store"].isin(stores_in_data)].dropna(subset=FEATURES).copy()
    scaler = StandardScaler().fit(x[FEATURES])
    z = scaler.transform(x[FEATURES])
    sil = {k: silhouette_score(z, KMeans(k, n_init=50, random_state=SEED).fit_predict(z)) for k in range(3, 7)}
    k = max(sil, key=sil.get)
    km = KMeans(k, n_init=50, random_state=SEED).fit(z)
    x["GROUP_ID"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    names = name_groups(centers)
    x["GROUP"] = x["GROUP_ID"].map(names)
    prof = x.groupby("GROUP")[FEATURES].mean()
    prof["income"] = np.exp(prof["income"])                     # log median income -> $
    prof.insert(0, "stores", x.groupby("GROUP").size())
    return x, prof, {"k": k, "silhouette": sil}


# ----------------------------------------------------------------------------
def store_event_table(rows: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    p = rows[rows["PHASE"].eq("promo") & rows["PRED_CF"].gt(0)]
    p = p.merge(groups[["store", "GROUP"]], left_on="STORE", right_on="store")
    se = p.groupby(["GROUP", "EVENT_ID", "STORE"]).agg(A=("MOVE", "sum"), C=("PRED_CF", "sum"),
                                                       D=("DEPTH", "mean")).reset_index()
    se["y"] = np.log((se["A"] + 0.5) / (se["C"] + 0.5))
    se["D"] = se["D"].clip(0, 0.7)
    return se


def response_model(se: pd.DataFrame, ref: str) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Within-event WLS: y ~ group + depth + group x depth, event fixed effects (weighted demeaning)."""
    others = [g for g in sorted(se["GROUP"].unique()) if g != ref]
    X = pd.DataFrame({"D": se["D"]})
    for g in others:
        X[f"G[{g}]"] = (se["GROUP"] == g).astype(float)
        X[f"G[{g}]xD"] = X[f"G[{g}]"] * se["D"]
    w = se["C"].to_numpy()
    ev = se["EVENT_ID"]

    def demean(v: pd.Series) -> pd.Series:
        num = (v * w).groupby(ev).transform("sum")
        den = pd.Series(w, index=v.index).groupby(ev).transform("sum")
        return v - num / den

    Xd, yd, sw = X.apply(demean), demean(se["y"]), np.sqrt(w)
    Xv, yv = Xd.to_numpy() * sw[:, None], yd.to_numpy() * sw
    coef = pd.Series(np.linalg.lstsq(Xv, yv, rcond=None)[0], X.columns)
    rng = np.random.default_rng(SEED)
    events = ev.unique()
    idx = {e: np.flatnonzero(ev.to_numpy() == e) for e in events}
    boot = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[e] for e in rng.choice(events, len(events))])
        boot.append(np.linalg.lstsq(Xv[ii], yv[ii], rcond=None)[0])
    return coef, pd.DataFrame(boot, columns=X.columns), others


def log_multiplier(params: pd.Series, group: str, ref: str, d: float) -> float:
    """log(response of group / response of reference group) at depth d."""
    if group == ref:
        return 0.0
    return params[f"G[{group}]"] + params[f"G[{group}]xD"] * d


def group_multipliers(coef, boot, groups_all, ref, weights) -> pd.DataFrame:
    """Multiplier vs the CHAIN AVERAGE (store-weighted), with 90% bootstrap range."""
    out = []
    for d in DEPTHS:
        def chain_norm(params):
            lm = {g: log_multiplier(params, g, ref, d) for g in groups_all}
            avg = np.log(sum(weights[g] * np.exp(lm[g]) for g in groups_all))
            return {g: lm[g] - avg for g in groups_all}
        pt = chain_norm(coef)
        bs = pd.DataFrame([chain_norm(b) for _, b in boot.iterrows()])
        for g in groups_all:
            lo, hi = np.exp(bs[g].quantile([0.05, 0.95]))
            out.append({"GROUP": g, "DEPTH": d, "MULTIPLIER": np.exp(pt[g]), "MULT_LO90": lo, "MULT_HI90": hi,
                        "SIGNIFICANT": bool(lo > 1 or hi < 1)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
def group_price_volume(rows: pd.DataFrame, groups: pd.DataFrame, products: list[int]) -> pd.DataFrame:
    """Per product x group: regular-price index and normal-volume index vs the chain (last 2 years of normal weeks)."""
    r = rows[rows["PRODUCT_ID"].isin(products) & rows["PHASE"].isin(["none", "pre"])]
    r = r[r["WEEK"] >= r["WEEK"].max() - 104]
    r = r.merge(groups[["store", "GROUP"]], left_on="STORE", right_on="store")
    chain = r.groupby("PRODUCT_ID").agg(P=("REG_PRICE", "median"), V=("MOVE", "mean"))
    g = r.groupby(["PRODUCT_ID", "GROUP"]).agg(P=("REG_PRICE", "median"), V=("MOVE", "mean"),
                                               STORES=("STORE", "nunique")).reset_index()
    g = g.join(chain, on="PRODUCT_ID", rsuffix="_CHAIN")
    g["PRICE_IDX"] = (g["P"] / g["P_CHAIN"]).fillna(1)
    g["VOL_IDX"] = (g["V"] / g["V_CHAIN"]).fillna(1)
    return g[["PRODUCT_ID", "GROUP", "STORES", "PRICE_IDX", "VOL_IDX"]]


def personalize(sim: PromoSimulator, mult: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    m = mult.set_index(["GROUP", "DEPTH"])["MULTIPLIER"]
    out = []
    for pid, p in sim.products.iterrows():
        ratio = {d: sim.simulate(pid, d, duration=WEEKS)["lift_ratio"] for d in DEPTHS}
        uniform = {}
        for _, row in pv[pv["PRODUCT_ID"].eq(pid)].iterrows():
            g = row["GROUP"]
            base = p["BASE_UNITS_PER_STORE_WEEK"] * row["VOL_IDX"] * WEEKS * row["STORES"]
            price, cost = p["REG_PRICE"] * row["PRICE_IDX"], p["UNIT_COST"]
            res = {}
            for d in DEPTHS:
                units = base * ratio[d] * m[(g, d)]
                cannib = -p["DIVERSION"] * max(units - base, 0)
                res[d] = units * (price * (1 - d) - cost) - base * (price - cost) - cannib * p["OTHER_UNIT_GP"]
            uniform[g] = res
            best = max(res, key=res.get)
            out.append({"PRODUCT_ID": pid, "DESCRIP": str(p["DESCRIP"]).strip().lstrip("$~ "), "BRAND": p["BRAND"],
                        "SIZE": p["SIZE"], "GROUP": g, "STORES": int(row["STORES"]),
                        "REG_PRICE": price, "BEST_DEPTH": best if res[best] > 0 else 0.0,
                        "BEST_GP": max(res[best], 0.0), "EXTRAPOLATION": bool(best > EXTRAPOLATION_DEPTH and res[best] > 0),
                        **{f"GP_{int(d * 100)}": v for d, v in res.items()}})
        # one chain-wide depth = best depth for the sum over groups
        if uniform:
            tot = {d: sum(r[d] for r in uniform.values()) for d in DEPTHS}
            dbest = max(tot, key=tot.get)
            for o in out[-len(uniform):]:
                o["UNIFORM_DEPTH"] = dbest if tot[dbest] > 0 else 0.0
                o["UNIFORM_GP"] = o[f"GP_{int(dbest * 100)}"] if tot[dbest] > 0 else 0.0
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading data …")
    rows = pd.read_parquet(PROC / "tna_counterfactual_rows.parquet",
                           columns=["PRODUCT_ID", "STORE", "WEEK", "MOVE", "REG_PRICE", "DEPTH", "PHASE",
                                    "EVENT_ID", "PRED_CF"])
    demo = load_demo()
    groups, prof, info = make_groups(demo, set(rows["STORE"].unique()))
    groups[["store", "name", "city", "zone", "lat", "lon", "GROUP_ID", "GROUP"] + FEATURES].rename(
        columns={"store": "STORE"}).to_csv(PROC / "tna_store_groups.csv", index=False)
    prof.round(4).to_csv(OUT / "group_profiles.csv")
    print(f"  {len(groups)} stores with demographics, k = {info['k']} groups "
          f"(silhouette {', '.join(f'{k}: {v:.3f}' for k, v in info['silhouette'].items())})")

    print("Promotion response by group (within-promotion, event fixed effects, bootstrap) …")
    se = store_event_table(rows, groups)
    ref = "Affluent professionals" if "Affluent professionals" in set(groups["GROUP"]) else sorted(groups["GROUP"])[0]
    coef, boot, _ = response_model(se, ref)
    weights = (groups["GROUP"].value_counts() / len(groups)).to_dict()
    gall = sorted(groups["GROUP"].unique())
    mult = group_multipliers(coef, boot, gall, ref, weights)
    mult.round(4).to_csv(OUT / "group_response.csv", index=False)

    se["DEPTH_BUCKET"] = pd.cut(se["D"], [-1, 0.05, 0.10, 0.20, 1], labels=["<5%", "5-10%", "10-20%", ">=20%"])
    lift = se.groupby(["GROUP", "DEPTH_BUCKET"], observed=True).agg(
        store_events=("y", "size"), actual=("A", "sum"), counterfactual=("C", "sum")).reset_index()
    lift["LIFT_PCT"] = (lift["actual"] / lift["counterfactual"] - 1) * 100
    lift.round(2).to_csv(OUT / "group_lift_by_depth.csv", index=False)

    print("Personalized discount per product and group …")
    sim = PromoSimulator.load()
    pv = group_price_volume(rows, groups, list(sim.products.index))
    pers = personalize(sim, mult, pv)
    pers.round(4).to_csv(OUT / "personalized_discounts.csv", index=False)

    # ---- report ----
    gain = pers.groupby("PRODUCT_ID")[["BEST_GP", "UNIFORM_GP"]].sum()
    lines = ["PHASE 10 — PERSONALIZED DISCOUNTS BY CUSTOMER GROUP", "=" * 55,
             f"Stores with demographics: {len(groups)} · groups: {info['k']} (k-means, best silhouette "
             f"{info['silhouette'][info['k']]:.3f})", "Variables: " + ", ".join(FEATURES) +
             "  (ethnicity deliberately excluded)", "", "GROUP PROFILES (means; income = median $)",
             prof.round(3).to_string(), "",
             f"RESPONSE MULTIPLIER vs chain average (1.00 = average; within-promotion, {N_BOOT} bootstrap draws)",
             mult.pivot(index="GROUP", columns="DEPTH", values="MULTIPLIER").round(3).to_string(), "",
             "Significant at 90% (range excludes 1.00):",
             mult.pivot(index="GROUP", columns="DEPTH", values="SIGNIFICANT").to_string(), "",
             "POOLED LIFT % BY GROUP AND DEPTH (descriptive)",
             lift.pivot(index="GROUP", columns="DEPTH_BUCKET", values="LIFT_PCT").round(0).to_string(), "",
             "PERSONALIZED vs ONE CHAIN-WIDE DISCOUNT (2-week promotion, retailer pays, all active products)",
             f"  chain-wide best depth: total gross-profit impact {gain['UNIFORM_GP'].sum():,.0f} $",
             f"  personalized by group: total gross-profit impact {gain['BEST_GP'].sum():,.0f} $",
             f"  gain from personalization: {gain['BEST_GP'].sum() - gain['UNIFORM_GP'].sum():,.0f} $ "
             f"({(gain['BEST_GP'].sum() / max(gain['UNIFORM_GP'].sum(), 1) - 1) * 100:+.1f}%)", "",
             "Most common recommended depth by group:",
             pers.groupby("GROUP")["BEST_DEPTH"].agg(lambda s: s.mode().iloc[0]).map("{:.0%}".format).to_string(),
             "", "Caveats: group differences are associations between store demographics and measured promotion",
             "response (store groups also differ in other ways). Store-level data — no individual customers."]
    (OUT / "phase10_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
