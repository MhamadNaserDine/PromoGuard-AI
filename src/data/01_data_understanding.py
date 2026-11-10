"""
PromoGuard AI — Phase 1: Dataset Setup and Data Understanding
=============================================================

Location in repo:  PromoGuard-AI/src/data/01_data_understanding.py
Run from repo root:
    python src/data/01_data_understanding.py            # default category: tna (Canned Tuna)
    python src/data/01_data_understanding.py --cat cer  # any other Dominick's category

Expected raw files (as downloaded from the Kilts Center, CSV versions):
    data/raw/dominicks/<cat>/upc<cat>.csv
    data/raw/dominicks/<cat>/w<cat>.csv   (or the downloaded .zip — pandas reads it directly)

What this script does: DESCRIBES the raw data only. It does not clean, drop,
or transform anything. Every printed result is also saved to
    reports/phase1/phase1_<cat>_report.txt
so you can send the whole file back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "dominicks"
REPORT_DIR = REPO_ROOT / "reports" / "phase1"

# Variables documented for Dominick's files. We CHECK for them; we never assume.
EXPECTED_MOVE_COLS = ["UPC", "STORE", "WEEK", "MOVE", "QTY", "PRICE",
                      "SALE", "PROFIT", "OK", "PRICE_HEX", "PROFIT_HEX"]
EXPECTED_UPC_COLS = ["COM_CODE", "UPC", "DESCRIP", "SIZE", "CASE", "NITEM"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 60)


# ----------------------------------------------------------------------------
# Output helper: print to screen AND to the report file
# ----------------------------------------------------------------------------
class Tee:
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
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def has(df: pd.DataFrame, *cols: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [SKIPPED] column(s) not found: {missing}")
        return False
    return True


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_file(folder: Path, stem: str) -> Path:
    """Find <stem>.csv or <stem>.zip / <stem>_csv.zip inside folder."""
    candidates = (sorted(folder.glob(f"{stem}.csv")) + sorted(folder.glob(f"{stem}*.zip"))
                  + sorted(folder.glob(f"{stem.upper()}.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{stem}.csv' or '{stem}*.zip' in {folder}\n"
            f"Files present: {[p.name for p in folder.glob('*')]}"
        )
    return candidates[0]


def load_csv(path: Path) -> pd.DataFrame:
    # latin-1 avoids crashes on old product descriptions with non-UTF-8 bytes
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def basic_overview(df: pd.DataFrame, name: str, expected: list[str]) -> None:
    section(f"{name}: SHAPE, COLUMNS, TYPES")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
    print("\nColumns and dtypes:")
    print(df.dtypes.to_string())
    print(f"\nDocumented columns NOT found : {[c for c in expected if c not in df.columns]}")
    print(f"Columns found but not expected: {[c for c in df.columns if c not in expected]}")

    section(f"{name}: SAMPLE ROWS (first 5 + random 5)")
    print(df.head(5).to_string())
    print()
    print(df.sample(min(5, len(df)), random_state=42).to_string())

    section(f"{name}: MISSING VALUES")
    miss = df.isna().sum()
    out = pd.DataFrame({"n_missing": miss, "pct_missing": (miss / len(df) * 100).round(3)})
    print(out.to_string())
    # blank strings are a separate case from NaN
    obj_cols = df.select_dtypes(exclude=["number", "bool"]).columns
    for c in obj_cols:
        n_blank = (df[c].astype(str).str.strip() == "").sum()
        if n_blank:
            print(f"  '{c}': {n_blank:,} blank strings (not counted as NaN)")

    section(f"{name}: DUPLICATES")
    print(f"Fully duplicated rows: {df.duplicated().sum():,}")


def movement_report(mv: pd.DataFrame) -> None:
    section("MOVEMENT: KEY DUPLICATES (UPC, STORE, WEEK)")
    if has(mv, "UPC", "STORE", "WEEK"):
        dup = mv.duplicated(["UPC", "STORE", "WEEK"], keep=False)
        print(f"Rows sharing a (UPC, STORE, WEEK) key: {dup.sum():,}")
        if dup.any():
            print(mv[dup].sort_values(["UPC", "STORE", "WEEK"]).head(10).to_string())

    section("MOVEMENT: PRODUCTS, STORES, WEEKS")
    if has(mv, "UPC", "STORE", "WEEK"):
        print(f"Unique UPCs  : {mv['UPC'].nunique():,}")
        print(f"Unique stores: {mv['STORE'].nunique():,}")
        print(f"Unique weeks : {mv['WEEK'].nunique():,}  (min {mv['WEEK'].min()}, max {mv['WEEK'].max()})")
        all_weeks = set(range(int(mv['WEEK'].min()), int(mv['WEEK'].max()) + 1))
        gaps = sorted(all_weeks - set(mv["WEEK"].unique()))
        print(f"Weeks missing inside the range: {len(gaps)} -> {gaps[:40]}{' ...' if len(gaps) > 40 else ''}")
        print("\nStore list:", sorted(mv["STORE"].unique().tolist()))

        per_store = mv.groupby("STORE")["WEEK"].agg(["min", "max", "nunique"])
        print("\nWeeks observed per store (describe):")
        print(per_store["nunique"].describe().round(1).to_string())

        upc_weeks = mv.groupby("UPC")["WEEK"].nunique()
        print("\nWeeks observed per UPC (describe):")
        print(upc_weeks.describe().round(1).to_string())

        panel = mv.groupby(["UPC", "STORE"]).size()
        print(f"\nUPC-store series: {len(panel):,}; rows per series (describe):")
        print(panel.describe().round(1).to_string())

        rows_week = mv.groupby("WEEK").size()
        print("\nRows per week (describe):")
        print(rows_week.describe().round(1).to_string())

    section("MOVEMENT: OK FLAG (data-quality flag)")
    if has(mv, "OK"):
        print(mv["OK"].value_counts(dropna=False).to_string())

    section("MOVEMENT: QTY (bundle size, e.g. 3 for $1)")
    if has(mv, "QTY"):
        print(mv["QTY"].value_counts(dropna=False).head(15).to_string())

    section("MOVEMENT: PRICE (shelf price for QTY units)")
    if has(mv, "PRICE"):
        print(mv["PRICE"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
        print(f"\nPRICE == 0 : {(mv['PRICE'] == 0).sum():,}")
        print(f"PRICE <  0 : {(mv['PRICE'] < 0).sum():,}")
        print(f"PRICE NaN  : {mv['PRICE'].isna().sum():,}")
    if has(mv, "PRICE", "QTY"):
        q = mv["QTY"].replace(0, np.nan)
        unit_price = mv["PRICE"] / q
        print("\nUnit price = PRICE / QTY (describe, PRICE > 0 only):")
        print(unit_price[mv["PRICE"] > 0].describe(percentiles=[.01, .5, .99]).round(3).to_string())

    section("MOVEMENT: MOVE (units sold)")
    if has(mv, "MOVE"):
        print(mv["MOVE"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99, .999]).round(2).to_string())
        print(f"\nMOVE == 0 : {(mv['MOVE'] == 0).sum():,} ({(mv['MOVE'] == 0).mean()*100:.2f}%)")
        print(f"MOVE <  0 : {(mv['MOVE'] < 0).sum():,}")
        print(f"Non-integer MOVE values: {(mv['MOVE'] % 1 != 0).sum():,}")
    if has(mv, "MOVE", "PRICE"):
        print(f"MOVE > 0 but PRICE <= 0: {((mv['MOVE'] > 0) & (mv['PRICE'] <= 0)).sum():,}")
        print(f"MOVE == 0 and PRICE > 0: {((mv['MOVE'] == 0) & (mv['PRICE'] > 0)).sum():,}")
    if has(mv, "MOVE", "PRICE", "QTY"):
        q = mv["QTY"].replace(0, np.nan)
        dollars = mv["PRICE"] * mv["MOVE"] / q
        print(f"\nTotal dollar sales (PRICE*MOVE/QTY, PRICE>0): ${dollars[mv['PRICE'] > 0].sum():,.0f}")

    section("MOVEMENT: PROFIT (documented as gross margin, %)")
    if has(mv, "PROFIT"):
        print(mv["PROFIT"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
        print(f"\nPROFIT <  0   : {(mv['PROFIT'] < 0).sum():,}")
        print(f"PROFIT == 0   : {(mv['PROFIT'] == 0).sum():,}")
        print(f"PROFIT >= 100 : {(mv['PROFIT'] >= 100).sum():,}")
        print(f"PROFIT <= -100: {(mv['PROFIT'] <= -100).sum():,}")

    section("MOVEMENT: SALE (deal code)")
    if has(mv, "SALE"):
        sale = mv["SALE"].astype(str).str.strip().replace({"nan": "<missing>", "": "<blank>"})
        vc = sale.value_counts(dropna=False)
        print(pd.DataFrame({"n": vc, "pct": (vc / len(mv) * 100).round(3)}).to_string())

        if has(mv, "MOVE", "PRICE", "QTY", "PROFIT"):
            tmp = mv.assign(SALE_CODE=sale, UNIT_PRICE=mv["PRICE"] / mv["QTY"].replace(0, np.nan))
            tmp = tmp[tmp["PRICE"] > 0]
            print("\nMean / median by deal code (PRICE > 0 rows):")
            print(tmp.groupby("SALE_CODE")[["MOVE", "UNIT_PRICE", "PROFIT"]]
                  .agg(["mean", "median"]).round(3).to_string())

            # Descriptive comparison: deal code vs. observed price drop.
            # Reference = median unit price of the same UPC-store series.
            # (Descriptive only — the real promotion definition comes in Phase 2.)
            ref = tmp.groupby(["UPC", "STORE"])["UNIT_PRICE"].transform("median")
            drop_pct = 1 - tmp["UNIT_PRICE"] / ref
            tmp["PRICE_DROP_5PCT"] = drop_pct >= 0.05
            print("\nDeal code vs. price >=5% below its UPC-store median (row % within code):")
            print((pd.crosstab(tmp["SALE_CODE"], tmp["PRICE_DROP_5PCT"], normalize="index") * 100)
                  .round(2).to_string())
            print("\nCounts:")
            print(pd.crosstab(tmp["SALE_CODE"], tmp["PRICE_DROP_5PCT"], margins=True).to_string())

        if has(mv, "WEEK"):
            flagged = mv["SALE"].astype(str).str.strip().isin(["B", "C", "S"])
            share = flagged.groupby(mv["WEEK"]).mean() * 100
            print("\nShare of rows with a B/C/S deal code, by week (describe):")
            print(share.describe().round(2).to_string())
            print("Weeks with 0% flagged:", share[share == 0].index.tolist()[:40])


def upc_report(upc: pd.DataFrame) -> None:
    section("UPC FILE: PRODUCTS")
    if has(upc, "UPC"):
        print(f"Unique UPCs: {upc['UPC'].nunique():,} (rows: {len(upc):,})")
        print(f"Duplicated UPC values: {upc['UPC'].duplicated().sum():,}")
    if has(upc, "COM_CODE"):
        print("\nCOM_CODE (commodity code) counts:")
        print(upc["COM_CODE"].value_counts(dropna=False).head(30).to_string())
    if has(upc, "SIZE"):
        print("\nTop 25 SIZE values:")
        print(upc["SIZE"].astype(str).str.strip().value_counts().head(25).to_string())
    if has(upc, "DESCRIP"):
        print("\n40 random product descriptions:")
        print(upc["DESCRIP"].dropna().sample(min(40, upc["DESCRIP"].notna().sum()),
                                             random_state=1).to_string())
        first_word = upc["DESCRIP"].astype(str).str.strip().str.split().str[0]
        print("\nMost common first word of DESCRIP (rough brand hint, NOT a brand field):")
        print(first_word.value_counts().head(25).to_string())
    for c in ["CASE", "NITEM"]:
        if c in upc.columns:
            print(f"\n{c} (describe):")
            print(upc[c].describe().round(2).to_string())


def join_report(mv: pd.DataFrame, upc: pd.DataFrame) -> None:
    section("JOIN CHECK: MOVEMENT <-> UPC on UPC")
    if not (has(mv, "UPC") and has(upc, "UPC")):
        return
    mv_upcs, upc_upcs = set(mv["UPC"].unique()), set(upc["UPC"].unique())
    print(f"UPCs in movement: {len(mv_upcs):,} | in UPC file: {len(upc_upcs):,}")
    print(f"In movement but NOT in UPC file: {len(mv_upcs - upc_upcs):,}")
    print(f"In UPC file but NOT in movement: {len(upc_upcs - mv_upcs):,}")
    rows_unmatched = (~mv["UPC"].isin(upc_upcs)).sum()
    print(f"Movement rows without product info: {rows_unmatched:,}")

    if has(mv, "MOVE") and "DESCRIP" in upc.columns:
        top = (mv.groupby("UPC")["MOVE"].sum().sort_values(ascending=False).head(20)
               .rename("TOTAL_UNITS").reset_index()
               .merge(upc[[c for c in ["UPC", "DESCRIP", "SIZE", "COM_CODE"] if c in upc.columns]],
                      on="UPC", how="left"))
        print("\nTop 20 UPCs by total units sold:")
        print(top.to_string(index=False))
        total = mv["MOVE"].sum()
        top_share = mv.groupby("UPC")["MOVE"].sum().sort_values(ascending=False)
        for k in (10, 20, 50):
            print(f"Top {k} UPCs = {top_share.head(k).sum() / total * 100:.1f}% of units")


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", default="tna", help="Dominick's category acronym (default: tna)")
    args = parser.parse_args()
    cat = args.cat.lower()

    folder = RAW_DIR / cat
    tee = Tee(REPORT_DIR / f"phase1_{cat}_report.txt")
    sys.stdout = tee
    try:
        section(f"PROMOGUARD AI — PHASE 1 DATA UNDERSTANDING — category '{cat}'")
        upc_path = find_file(folder, f"upc{cat}")
        mv_path = find_file(folder, f"w{cat}")
        print(f"UPC file     : {upc_path}")
        print(f"Movement file: {mv_path}")

        upc = load_csv(upc_path)
        mv = load_csv(mv_path)

        basic_overview(upc, "UPC FILE", EXPECTED_UPC_COLS)
        upc_report(upc)
        basic_overview(mv, "MOVEMENT FILE", EXPECTED_MOVE_COLS)
        movement_report(mv)
        join_report(mv, upc)

        section("DONE")
        print(f"Report saved to: {REPORT_DIR / f'phase1_{cat}_report.txt'}")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
