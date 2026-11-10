"""
PromoGuard AI — product attributes parsed from the Dominick's UPC file.

Location: src/features/product_attributes.py

Why this exists
---------------
All 278 products in the canned-tuna file share one commodity code (COM_CODE = 329),
and the file mixes tuna with salmon, sardines, shellfish, caviar, etc. COM_CODE
therefore cannot tell us which products compete with each other.

The only product information available is DESCRIP (truncated to 20 characters)
and SIZE. This module derives attributes from those two fields with explicit,
documented keyword rules. Nothing is invented: when a description does not
reveal an attribute, the value is "unknown".

The rules are deliberately simple and reviewable. The output table is saved so
every classification can be checked by hand.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Description prefixes documented in the manual (Part 3, remark 5)
#   ~  discontinued item      #  sold in Combo stores only
#   <  trial size (manual: "not accurate")      $ and *  mean nothing
# ---------------------------------------------------------------------------
PREFIX_CHARS = "~#<$*"

# ---------------------------------------------------------------------------
# Sub-category rules — checked IN ORDER, first match wins.
# Non-tuna groups are checked first so that e.g. "TUNA MIX-IN" seasoning
# packets or "SALMON CAVIAR" are not classified as tuna.
# ---------------------------------------------------------------------------
SUBCATEGORY_RULES: list[tuple[str, str]] = [
    ("tuna_seasoning",  r"TUNA MIX|MIX-IN"),                      # 0.05 oz flavour packets
    ("caviar_roe",      r"CAVI|LUMP ?F|LUMPFS|ICELAND LUM|WHTFSH|WHITEFISH"),
    ("salmon",          r"SALM|SOCKEY|KETA|PINK|RED-RED|TRADITIONAL (PIN|RED)|ALASKA RE"),
    ("sardine_herring", r"SARD|BRISL|BRSL|BRSTL|KIPPER|2 LAYER|OIL FISH|NORWEIGEN|MACK|E-Z OPEN MUSTA"),
    ("anchovy",         r"ANCH|ROLLED AN$"),
    ("shellfish",       r"CLAM|BABY ?CL?$|BABY C\b|OYST|OYS\b|SMOKE$|SHRIMP|SHR\b|SHP\b|CRAB|CRA$|CRB|LEG C$|MUSSEL|LOBS"
                        r"|SNAIL|DEVEIN|DEVEND|DEVINED|IMPORTED (TIN|BROK)|TINY IMPORT|DOMESTIC SMA|DOM MED|CHICKEN/SEA MINCED|WHOLE BOILED"),
    ("other_non_tuna",  r"PATE|CAPE|CAPERS|BOUILL|FROG"),
    ("tuna",            r"TUNA|TUN\b|TUN$|ALBACORE|TONGOL"),
]

# Tuna products whose description does not contain the word TUNA, recognised by
# a tuna brand + a tuna style word (e.g. "BUM BEE/WATER CHK LI").
TUNA_BRAND_RE = r"^(STAR ?K|STARKIST|STARKSIT|STRKST|SK\b|BUM BEE|BUMBLE BEE|C ?O ?S\b|C\.O\.S|COS\b|HH\b|DOM\b|GEISHA|3-DIAMOND|P\.S\.)"
TUNA_STYLE_RE = r"CHK|CHUNK|CHNK|CNK|CH LT|CK LT|SOLID|SLD|WHT|WHITE|LIGHT|LITE|LGHT|LGT|DIET|PRIME|LUNCH|SLEEVE|HICKORY|REDUCED|L/S|LOW SLT"

# ---------------------------------------------------------------------------
# Brand rules (first match wins). Dominick's private label appears as "DOM" and
# "HH"; both share Dominick's UPC manufacturer prefix 38281.
# ---------------------------------------------------------------------------
BRAND_RULES: list[tuple[str, str]] = [
    ("StarKist",           r"^(STAR ?K|STARKIST|STARKSIT|STRKST|SK\b)"),
    ("Bumble Bee",         r"^(BUM BEE|BUMBLE BEE|BB\b)"),
    ("Chicken of the Sea", r"^(C ?O ?S\b|C\.O\.S|COS\b|CHICKEN/SEA)"),
    ("Dominick's",         r"^(DOM\b|HH\b)"),
    ("Geisha",             r"^GEISHA"),
    ("3-Diamond",          r"^3-DIAMOND"),
    ("P.S.",               r"^P\.S\."),
    ("Pillar Rock",        r"^PILLAR ROCK"),
    ("King Oscar",         r"^(KING OSCAR|KNG OSCR)"),
    ("Polar",              r"^POLAR"),
    ("Orleans",            r"^ORLEANS"),
    ("Reese",              r"^(REESE|RESE\b|R SPICY)"),
    ("Romanoff",           r"^(ROMANOFF|ROMANOF|ROM\b|ROM\.)"),
    ("Royal",              r"^ROYAL"),
    ("Booth",              r"^BOOTH"),
    ("Brunswick",          r"^BRUNSWICK"),
    ("Port Clyde",         r"^PORT CLYDE"),
    ("Beach Cliff",        r"^BEACH ?CLIFF"),
    ("Snow's",             r"^SNOW'?S"),
    ("Doxsee",             r"^DOXEE"),
]


def _clean_desc(desc: str) -> str:
    return str(desc).upper().lstrip(PREFIX_CHARS).strip()


def _first_match(text: str, rules: list[tuple[str, str]], default: str) -> str:
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return default


def classify_subcategory(desc_clean: str) -> str:
    sub = _first_match(desc_clean, SUBCATEGORY_RULES, default="unclassified")
    if sub == "unclassified" and re.search(TUNA_BRAND_RE, desc_clean) and re.search(TUNA_STYLE_RE, desc_clean):
        sub = "tuna"
    return sub


def tuna_meat(desc_clean: str) -> str:
    """white (albacore) vs light tuna. 'unknown' if not stated."""
    if re.search(r"WHITE|WHT|WH\.|ALBACORE|SOLID WHIT", desc_clean):
        return "white"
    if re.search(r"LIGHT|LITE|LGHT|LGT|LIGH|\bLT\b|\bLI$|TONGOL", desc_clean):
        return "light"
    return "unknown"


def tuna_pack(desc_clean: str) -> str:
    """water vs oil pack. Descriptions are truncated, so often 'unknown'."""
    if re.search(r"WATER|WATE|\bWA\b|/WA|/W\b|- WA|SWTR| W$", desc_clean):   # " W$": truncated "W(ATER)"
        return "water"
    if re.search(r"OIL|\bOI\b|/O\b| O$", desc_clean):                       # " O$": truncated "O(IL)"
        return "oil"
    return "unknown"


def tuna_style(desc_clean: str) -> str:
    """solid vs chunk."""
    if re.search(r"SOLID|SLD|SOLI$", desc_clean):
        return "solid"
    if re.search(r"CHUNK|CHK|CHNK|CNK|CH LT|CK LT|CHUNK", desc_clean):
        return "chunk"
    return "unknown"


def tuna_variant(desc_clean: str) -> str:
    """Special formulations visible in the description."""
    if re.search(r"DIET|L/S|LOW SLT|LS/LF|REDUCED", desc_clean):
        return "diet_low_sodium"
    if re.search(r"SALA|PRE-MIX|HICKORY", desc_clean):
        return "flavoured_kit"
    return "regular"


def parse_size(size: str) -> tuple[float, int]:
    """
    Return (total ounces, pack count) from the SIZE field.
      "6 OZ" -> (6.0, 1); "6.12 O" -> (6.12, 1); "3/3 OZ" -> (9.0, 3);
      "3 CT", "3 EA" -> (nan, 3); unparseable -> (nan, 1)
    SIZE is 6 characters wide in the source, so units are often truncated ("6.12 O").
    """
    s = str(size).upper().strip()
    m = re.match(r"^(\d+)\s*/\s*(\d*\.?\d+)", s)            # multipack "3/3 OZ", "3/3.2O"
    if m:
        n, each = int(m.group(1)), float(m.group(2))
        return n * each, n
    m = re.match(r"^(\d+)\s*(CT|EA)\b", s)
    if m:
        return np.nan, int(m.group(1))
    m = re.match(r"^(\d*\.?\d+)\s*(OZ|O\b|O$|OZ\.)?", s)
    if m and m.group(1):
        return float(m.group(1)), 1
    return np.nan, 1


def build_product_table(upc: pd.DataFrame) -> pd.DataFrame:
    """Return one row per UPC with all derived attributes."""
    df = upc.copy()
    raw = df["DESCRIP"].astype(str)
    clean = raw.map(_clean_desc)

    df["DESC_CLEAN"] = clean
    df["DISCONTINUED_FLAG"] = raw.str.strip().str.startswith("~")
    df["COMBO_ONLY_FLAG"] = raw.str.contains("#", regex=False)
    df["SUBCATEGORY"] = clean.map(classify_subcategory)
    df["BRAND"] = clean.map(lambda d: _first_match(d, BRAND_RULES, default="other"))

    is_tuna = df["SUBCATEGORY"].eq("tuna")
    for col, fn in [("TUNA_MEAT", tuna_meat), ("TUNA_PACK", tuna_pack),
                    ("TUNA_STYLE", tuna_style), ("TUNA_VARIANT", tuna_variant)]:
        df[col] = np.where(is_tuna, clean.map(fn), pd.NA)

    size_parsed = df["SIZE"].map(parse_size)
    df["SIZE_OZ"] = size_parsed.map(lambda t: t[0])
    df["PACK_COUNT"] = size_parsed.map(lambda t: t[1])
    return df
