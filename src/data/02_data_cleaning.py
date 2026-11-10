"""
PromoGuard AI — Phase 2: Data Cleaning
======================================

Location in repo:  PromoGuard-AI/src/data/02_data_cleaning.py
Run from repo root:
    python src/data/02_data_cleaning.py              # canned tuna (default, --cat tna)
    python src/data/02_data_cleaning.py --cat sdr    # soft drinks

Inputs
    data/raw/dominicks/<cat>/upc<cat>.csv
    data/raw/dominicks/<cat>/w<cat>.zip  (or w<cat>.csv)
    data/reference/dominicks_weeks.csv   (week calendar, extracted from the official manual)

Outputs
    data/processed/<cat>_products.csv            one row per UPC + derived attributes + PRODUCT_ID
    data/processed/<cat>_movement_clean.parquet  cleaned weekly UPC x store panel
    reports/phase2/phase2_cleaning_report.txt    (tuna) / phase2_<cat>_cleaning_report.txt

Cleaning rules (each justified in the Phase 1 findings / official manual)
    R1  Drop OK == 0                     manual: "0 for trash"; Kilts "do not use flagged data"
    R2  Drop PRICE == 0 with MOVE == 0   no sales and no recorded price that week. In this file EVERY
                                         zero-sales week has PRICE == 0, so "not stocked" and "stocked but
                                         unsold" cannot be told apart. Without a price these rows cannot
                                         enter a price/demand model. (Limitation, documented.)
    R3  Drop PRICE <= 0 with MOVE > 0    logically invalid (units sold at no price)
    R4  Keep PRICE (truncated), drop *_HEX   Kilts CSV note recommends the truncated values
    R5  UNIT_PRICE = PRICE / QTY ; DOLLARS = PRICE * MOVE / QTY      manual formula
    R6  GROSS_PROFIT = DOLLARS * PROFIT / 100                        PROFIT = gross margin %
    R7  Negative / extreme (>=95%) margins and extreme MOVE values are FLAGGED, not removed
        (they may be real loss-leader promotions / promotion spikes)
    R8  PRODUCT_ID links UPCs that are the same product under a new barcode (see build_product_ids)
    R9  (soft drinks) keep only UPCs inside the category scope: products whose SUBCATEGORY is
        "out_of_scope" (waters, iced teas, mixers, unrelated items by commodity code) are dropped
        before cleaning to save memory. Category-specific product rules:
        tna -> src/features/product_attributes.py ; sdr -> src/features/product_attributes_sdr.py

Nothing in data/raw is modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import product_attributes as attrs_tna  # noqa: E402
from src.features import product_attributes_sdr as attrs_sdr  # noqa: E402

# Category settings: product-rule module, the attributes that must match to link two UPCs,
# and the sub-category that is analysed in the later phases.
CATEGORIES = {
    "tna": {"name": "Canned Tuna", "attrs": attrs_tna, "main": "tuna",
            "link_cols": ["SUBCATEGORY", "BRAND", "PACK_COUNT", "TUNA_MEAT", "TUNA_STYLE", "TUNA_PACK", "TUNA_VARIANT"],
            "describe_cols": ["BRAND", "TUNA_MEAT", "TUNA_STYLE", "TUNA_PACK", "TUNA_VARIANT"]},
    "sdr": {"name": "Soft Drinks (carbonated)", "attrs": attrs_sdr, "main": "csd",
            "link_cols": attrs_sdr.ATTRIBUTE_COLUMNS,
            "describe_cols": ["MAKER", "BRAND", "SD_DIET", "SD_FLAVOR", "SD_PACK"]},
}
CAT = "tna"                       # set from --cat in main()
RAW_DIR = REPO_ROOT / "data" / "raw" / "dominicks" / CAT
REF_WEEKS = REPO_ROOT / "data" / "reference" / "dominicks_weeks.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports" / "phase2"

# Product-linking thresholds (R8)
MAX_SIZE_DIFF_OZ = 0.25      # 6 oz vs 6.12 oz relabel = same can; 3.25 oz vs 3.5 oz = different item
MAX_CONCURRENT_SHARE = 0.20  # if two UPCs sell side by side in >20% of store-weeks, they are separate items

# Outlier flag (R7): robust z-score of log(1+MOVE) within each UPC-store series
OUTLIER_Z = 5.0
OUTLIER_MIN_UNITS = 100

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


# ----------------------------------------------------------------------------
class Tee:
    """Print to screen and to the report file at the same time."""

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


def find_movement_file() -> Path:
    for name in (f"w{CAT}.csv", f"w{CAT}.zip", f"w{CAT}_csv.zip", f"w{CAT}.csv.zip"):
        if (RAW_DIR / name).exists():
            return RAW_DIR / name
    raise FileNotFoundError(f"No movement file w{CAT}.csv / w{CAT}.zip in {RAW_DIR}")


# ----------------------------------------------------------------------------
# Step 1 — row-level cleaning
# ----------------------------------------------------------------------------
def clean_movement(mv: pd.DataFrame) -> pd.DataFrame:
    section("STEP 1 — ROW FILTERS (waterfall)")
    n0 = len(mv)
    log = [("raw rows", n0, None)]

    keep = mv["OK"].eq(1)
    log.append(("R1 drop OK == 0 (flagged as trash by Kilts)", int((~keep).sum()), None))
    mv = mv[keep]

    not_stocked = mv["PRICE"].le(0) & mv["MOVE"].eq(0)
    log.append(("R2 drop PRICE == 0 & MOVE == 0 (no sale, no price)", int(not_stocked.sum()), None))
    mv = mv[~not_stocked]

    invalid = mv["PRICE"].le(0) & mv["MOVE"].gt(0)
    log.append(("R3 drop PRICE <= 0 & MOVE > 0 (invalid)", int(invalid.sum()), None))
    mv = mv[~invalid]

    for label, n, _ in log:
        print(f"  {label:<60s} {n:>12,}")
    print(f"  {'rows kept':<60s} {len(mv):>12,}  ({len(mv) / n0 * 100:.1f}% of raw)")

    # sanity checks on what remains
    assert mv["PRICE"].gt(0).all()
    assert mv["QTY"].ge(1).all()
    assert mv["MOVE"].ge(0).all()
    print(f"\n  Remaining rows with MOVE == 0 and PRICE > 0: {int(mv['MOVE'].eq(0).sum()):,}")
    print("  -> the raw data never records a zero-sales week together with a price.")
    return mv.copy()


# ----------------------------------------------------------------------------
# Step 2 — derived measures
# ----------------------------------------------------------------------------
def add_measures(mv: pd.DataFrame) -> pd.DataFrame:
    section("STEP 2 — DERIVED MEASURES (R4-R6)")
    mv = mv.drop(columns=[c for c in ("PRICE_HEX", "PROFIT_HEX") if c in mv.columns])
    mv["UNIT_PRICE"] = mv["PRICE"] / mv["QTY"]
    mv["DOLLARS"] = mv["PRICE"] * mv["MOVE"] / mv["QTY"]
    mv = mv.rename(columns={"PROFIT": "MARGIN_PCT"})
    mv["GROSS_PROFIT"] = mv["DOLLARS"] * mv["MARGIN_PCT"] / 100.0

    sale = mv["SALE"].astype("string").str.strip().str.upper()
    mv["DEAL_CODE"] = sale.fillna("NONE").replace("", "NONE")
    mv = mv.drop(columns=["SALE", "OK"])

    print("UNIT_PRICE = PRICE / QTY ; DOLLARS = PRICE * MOVE / QTY ; GROSS_PROFIT = DOLLARS * MARGIN_PCT / 100")
    print(f"Total dollar sales  : ${mv['DOLLARS'].sum():,.0f}")
    print(f"Total gross profit  : ${mv['GROSS_PROFIT'].sum():,.0f}  "
          f"(average margin {mv['GROSS_PROFIT'].sum() / mv['DOLLARS'].sum() * 100:.1f}%)")
    print("\nDEAL_CODE counts after cleaning:")
    print(mv["DEAL_CODE"].value_counts().to_string())
    return mv


# ----------------------------------------------------------------------------
# Step 3 — flags (kept rows, marked for later decisions)
# ----------------------------------------------------------------------------
def add_flags(mv: pd.DataFrame) -> pd.DataFrame:
    section("STEP 3 — QUALITY FLAGS (R7: flagged, NOT removed)")
    mv["NEG_MARGIN_FLAG"] = mv["MARGIN_PCT"].lt(0)
    # Margins >= 95% imply a cost of goods near zero: most likely an average-acquisition-cost
    # artefact (manual: AAC drops sharply when buyers stock up on manufacturer deals).
    mv["EXTREME_MARGIN_FLAG"] = mv["MARGIN_PCT"].ge(95)

    logm = np.log1p(mv["MOVE"])
    grp = logm.groupby([mv["UPC"], mv["STORE"]])
    med = grp.transform("median")
    mad = (logm - med).abs().groupby([mv["UPC"], mv["STORE"]]).transform("median")
    robust_z = (logm - med) / (1.4826 * mad.replace(0, np.nan))
    mv["MOVE_OUTLIER_FLAG"] = robust_z.gt(OUTLIER_Z) & mv["MOVE"].ge(OUTLIER_MIN_UNITS)

    n = len(mv)
    print(f"NEG_MARGIN_FLAG   : {int(mv['NEG_MARGIN_FLAG'].sum()):,} rows ({mv['NEG_MARGIN_FLAG'].mean() * 100:.2f}%)")
    print(f"EXTREME_MARGIN_FLAG: {int(mv['EXTREME_MARGIN_FLAG'].sum()):,} rows (MARGIN_PCT >= 95, cost near zero)")
    print(f"MOVE_OUTLIER_FLAG : {int(mv['MOVE_OUTLIER_FLAG'].sum()):,} rows ({mv['MOVE_OUTLIER_FLAG'].sum() / n * 100:.3f}%)"
          f"  [robust z of log(1+MOVE) > {OUTLIER_Z} within UPC-store, and MOVE >= {OUTLIER_MIN_UNITS}]")

    print("\nDeal code mix — all rows vs. negative-margin rows vs. outlier rows (% of rows):")
    mix = pd.DataFrame({
        "all": mv["DEAL_CODE"].value_counts(normalize=True),
        "neg_margin": mv.loc[mv["NEG_MARGIN_FLAG"], "DEAL_CODE"].value_counts(normalize=True),
        "outlier": mv.loc[mv["MOVE_OUTLIER_FLAG"], "DEAL_CODE"].value_counts(normalize=True),
    }).fillna(0).mul(100).round(1)
    print(mix.to_string())

    print("\n15 largest flagged MOVE values:")
    cols = ["UPC", "STORE", "WEEK", "MOVE", "UNIT_PRICE", "MARGIN_PCT", "DEAL_CODE"]
    print(mv.loc[mv["MOVE_OUTLIER_FLAG"], cols].nlargest(15, "MOVE").to_string(index=False))
    return mv


# ----------------------------------------------------------------------------
# Step 4 — product table and PRODUCT_ID (R8)
# ----------------------------------------------------------------------------
def build_product_ids(products: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    """
    The manual: UPCs with the same NITEM are newer versions of the same product,
    but "this scheme is not foolproof". In this file, NITEM links barcode changes
    (e.g. Chicken of the Sea 1780000024 -> 4800000024) but also groups different
    sizes (3.25 oz can vs 3x3 oz multipack).

    Two UPCs get the same PRODUCT_ID only if ALL hold:
      * same NITEM
      * same SUBCATEGORY, BRAND, PACK_COUNT and (for tuna) meat / style / pack / variant
      * SIZE_OZ differs by at most MAX_SIZE_DIFF_OZ
      * they sell side by side in at most MAX_CONCURRENT_SHARE of their store-weeks
    Otherwise the UPC keeps its own PRODUCT_ID (= its UPC).
    """
    section("STEP 4 — PRODUCT TABLE AND PRODUCT_ID (R8)")
    p = products.copy()
    p["PRODUCT_ID"] = p["UPC"]

    active = mv.loc[mv["MOVE"].gt(0), ["UPC", "STORE", "WEEK"]]
    attr_cols = CATEGORIES[CAT]["link_cols"]

    merges = []
    for nitem, grp in p.groupby("NITEM"):
        if len(grp) < 2:
            continue
        # order by volume: the main UPC becomes the PRODUCT_ID
        vol = active[active["UPC"].isin(grp["UPC"])].groupby("UPC").size()
        grp = grp.assign(_vol=grp["UPC"].map(vol).fillna(0)).sort_values("_vol", ascending=False)
        anchor = grp.iloc[0]
        for _, other in grp.iloc[1:].iterrows():
            same_attr = all(str(anchor[c]) == str(other[c]) for c in attr_cols)
            size_ok = (pd.notna(anchor["SIZE_OZ"]) and pd.notna(other["SIZE_OZ"])
                       and abs(anchor["SIZE_OZ"] - other["SIZE_OZ"]) <= MAX_SIZE_DIFF_OZ)
            pair = active[active["UPC"].isin([anchor["UPC"], other["UPC"]])]
            sw = pair.groupby(["STORE", "WEEK"])["UPC"].nunique()
            concurrent = float((sw > 1).mean()) if len(sw) else 0.0
            linked = bool(same_attr and size_ok and concurrent <= MAX_CONCURRENT_SHARE)
            if linked:
                p.loc[p["UPC"] == other["UPC"], "PRODUCT_ID"] = anchor["UPC"]
            merges.append({
                "NITEM": nitem, "ANCHOR_UPC": anchor["UPC"], "UPC": other["UPC"],
                "ANCHOR_DESC": anchor["DESCRIP"], "DESC": other["DESCRIP"],
                "ANCHOR_SIZE": anchor["SIZE"], "SIZE": other["SIZE"],
                "SAME_ATTRIBUTES": same_attr, "SIZE_OK": size_ok,
                "CONCURRENT_SHARE": round(concurrent, 3), "LINKED": linked,
            })

    m = pd.DataFrame(merges)
    print(f"UPC pairs sharing an NITEM: {len(m)} | linked into one PRODUCT_ID: {int(m['LINKED'].sum()) if len(m) else 0}")
    print(f"Distinct UPCs: {p['UPC'].nunique()} -> distinct PRODUCT_IDs: {p['PRODUCT_ID'].nunique()}")
    main = CATEGORIES[CAT]["main"]
    sel = p["SUBCATEGORY"].eq(main)
    print(f"{main}: {int(sel.sum())} UPCs -> {p.loc[sel, 'PRODUCT_ID'].nunique()} PRODUCT_IDs")
    if len(m):
        print("\nAll NITEM pairs and the decision (review this table):")
        print(m.drop(columns=["NITEM"]).to_string(index=False))
    return p.drop(columns=["_vol"], errors="ignore")


def describe_products(p: pd.DataFrame, mv: pd.DataFrame) -> None:
    section("STEP 5 — CATEGORY COMPOSITION (cleaned data)")
    units = mv.groupby("UPC")["MOVE"].sum()
    dollars = mv.groupby("UPC")["DOLLARS"].sum()
    q = p.assign(UNITS=p["UPC"].map(units).fillna(0), DOLLARS=p["UPC"].map(dollars).fillna(0))
    comp = q.groupby("SUBCATEGORY").agg(UPCs=("UPC", "size"), PRODUCTS=("PRODUCT_ID", "nunique"),
                                        UNITS=("UNITS", "sum"), DOLLARS=("DOLLARS", "sum"))
    comp["UNITS_%"] = (comp["UNITS"] / comp["UNITS"].sum() * 100).round(1)
    comp["DOLLARS_%"] = (comp["DOLLARS"] / comp["DOLLARS"].sum() * 100).round(1)
    print(comp.sort_values("UNITS", ascending=False).to_string())

    main = CATEGORIES[CAT]["main"]
    t = q[q["SUBCATEGORY"].eq(main)]
    for col in CATEGORIES[CAT]["describe_cols"]:
        share = (t.groupby(col)["UNITS"].sum() / t["UNITS"].sum() * 100).round(1).sort_values(ascending=False)
        print(f"\n{main} units by {col} (%):")
        print(share.head(25).to_string())


# ----------------------------------------------------------------------------
# Step 6 — calendar
# ----------------------------------------------------------------------------
def add_calendar(mv: pd.DataFrame) -> pd.DataFrame:
    section("STEP 6 — CALENDAR (official week table from the manual)")
    weeks = pd.read_csv(REF_WEEKS, parse_dates=["START", "END"])
    mv = mv.merge(weeks.rename(columns={"START": "WEEK_START", "END": "WEEK_END", "EVENT": "HOLIDAY"}),
                  on="WEEK", how="left", validate="many_to_one")
    assert mv["WEEK_START"].notna().all(), "Some weeks are missing from the calendar"
    print(f"Date range: {mv['WEEK_START'].min().date()} -> {mv['WEEK_END'].max().date()}")
    print("Rows in holiday weeks:")
    print(mv["HOLIDAY"].value_counts().to_string())
    return mv


# ----------------------------------------------------------------------------
def read_movement(path: Path, keep_upcs: set | None) -> pd.DataFrame:
    """Read the movement file in chunks with compact types; optionally keep only some UPCs (R9)."""
    dtypes = {"STORE": "int16", "UPC": "int64", "WEEK": "int16", "MOVE": "int32", "QTY": "int16",
              "PRICE": "float64", "PROFIT": "float64", "OK": "int8", "SALE": "string"}
    parts = []
    reader = pd.read_csv(path, chunksize=2_000_000, low_memory=False, dtype={"SALE": "string"},
                         usecols=lambda c: c.strip().upper() not in ("PRICE_HEX", "PROFIT_HEX"))
    for ch in reader:
        ch.columns = [c.strip().upper() for c in ch.columns]
        if keep_upcs is not None:
            ch = ch[ch["UPC"].isin(keep_upcs)]
        parts.append(ch.astype({k: v for k, v in dtypes.items() if k in ch.columns}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    global CAT, RAW_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", choices=sorted(CATEGORIES), help="Dominick's category acronym")
    parser.add_argument("--movement", default=None, help="optional path to a movement file (e.g. a test sample)")
    args = parser.parse_args()
    CAT = args.cat
    RAW_DIR = REPO_ROOT / "data" / "raw" / "dominicks" / CAT
    cfg = CATEGORIES[CAT]
    report = "phase2_cleaning_report.txt" if CAT == "tna" else f"phase2_{CAT}_cleaning_report.txt"
    tee = Tee(REPORT_DIR / report)
    sys.stdout = tee
    try:
        section(f"PROMOGUARD AI — PHASE 2 DATA CLEANING — category '{CAT}' ({cfg['name']})")
        mv_path = Path(args.movement) if args.movement else find_movement_file()
        print(f"Movement file: {mv_path}")
        upc = pd.read_csv(RAW_DIR / f"upc{CAT}.csv", encoding="latin-1")
        upc.columns = [c.strip().upper() for c in upc.columns]
        products = cfg["attrs"].build_product_table(upc)

        keep = None
        if (products["SUBCATEGORY"] == "out_of_scope").any():
            keep = set(products.loc[products["SUBCATEGORY"] != "out_of_scope", "UPC"])
            section("R9 — CATEGORY SCOPE")
            print(products.groupby("SUBCATEGORY").size().rename("UPCs").to_string())
            print(f"Keeping {len(keep)} of {len(products)} UPCs (SUBCATEGORY != 'out_of_scope')")
            products = products[products["UPC"].isin(keep)].reset_index(drop=True)
        mv = read_movement(mv_path, keep)
        print(f"Movement rows read (in scope): {len(mv):,}")

        mv = clean_movement(mv)
        mv = add_measures(mv)
        mv = add_flags(mv)

        products = build_product_ids(products, mv)
        describe_products(products, mv)

        mv = mv.merge(products[["UPC", "PRODUCT_ID", "SUBCATEGORY"]], on="UPC", how="left", validate="many_to_one")
        mv = add_calendar(mv)

        cols = ["STORE", "UPC", "PRODUCT_ID", "SUBCATEGORY", "WEEK", "WEEK_START", "WEEK_END", "HOLIDAY",
                "MOVE", "QTY", "PRICE", "UNIT_PRICE", "DOLLARS", "MARGIN_PCT", "GROSS_PROFIT",
                "DEAL_CODE", "NEG_MARGIN_FLAG", "EXTREME_MARGIN_FLAG", "MOVE_OUTLIER_FLAG"]
        mv = mv[cols].sort_values(["UPC", "STORE", "WEEK"]).reset_index(drop=True)
        assert not mv.duplicated(["UPC", "STORE", "WEEK"]).any()

        section("OUTPUTS")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        prod_path = OUT_DIR / f"{CAT}_products.csv"
        mv_out = OUT_DIR / f"{CAT}_movement_clean.parquet"
        products.to_csv(prod_path, index=False)
        mv.to_parquet(mv_out, index=False)
        print(f"{prod_path.relative_to(REPO_ROOT)}  ({len(products)} products)")
        print(f"{mv_out.relative_to(REPO_ROOT)}  ({len(mv):,} rows x {mv.shape[1]} columns)")
        print("\nColumns:", list(mv.columns))
        print("\nSample rows:")
        print(mv.sample(5, random_state=7).to_string(index=False))
        section("DONE")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
