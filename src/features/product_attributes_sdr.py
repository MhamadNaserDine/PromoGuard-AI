"""
PromoGuard AI — product attributes for SOFT DRINKS (Dominick's category 'sdr').

Location: src/features/product_attributes_sdr.py

Scope
-----
The soft-drinks file mixes carbonated soft drinks with waters, iced teas, juice drinks,
mixers and a few unrelated items. Dominick's commodity codes (COM_CODE) mostly encode the
PACKAGE type. Phase 1 review of the descriptions in each code:

    227  Dominick's / generic sodas        235  24-packs of cans     236  12-packs of cans
    237  6-packs of cans                   238  2-litre (and 1L/3L) bottles
    239  multi-packs of bottles            241  single bottles       224  one 12-pack (Diet 7-Up)
    233  waters & seltzers   234  iced teas & juice drinks   240  mixers (tonic, club soda)
    225  mixed specialty drinks            61 / 201  single unrelated items

CSD_COM_CODES below are the carbonated-soft-drink codes. Inside them a few iced teas,
lemonades and punches remain (e.g. Lipton cans); they get SUBCATEGORY != "csd".
Only SUBCATEGORY == "csd" is analysed, like SUBCATEGORY == "tuna" for canned tuna.

Attributes are derived from DESCRIP (20 characters, abbreviated) and SIZE with explicit
keyword rules; when a description does not reveal an attribute the value is "unknown".
Nothing is invented. The product table is saved so every classification can be checked.

Columns produced (in addition to the UPC file columns)
    DESC_CLEAN, DISCONTINUED_FLAG, COMBO_ONLY_FLAG, SUBCATEGORY, BRAND, MAKER,
    SD_DIET (diet / regular), SD_FLAVOR, SD_PACK, SIZE_OZ (total ounces), PACK_COUNT, UNIT_OZ
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

PREFIX_CHARS = "~#<$*"           # manual: ~ discontinued, # combo stores only, < trial size, $ * meaningless
CSD_COM_CODES = {224, 227, 235, 236, 237, 238, 239, 241}
LITRE_OZ = 33.814

# Non-carbonated drinks that sit inside the CSD package codes (first match wins)
NON_CSD_RULES: list[tuple[str, str]] = [
    ("iced_tea",       r"LIPTON|LPTN|TETLEY|TETELY|NESTEA|ICED TEA|\bTEA\b|SUNTEA|SUN TEA"),
    ("lemonade_punch", r"COUNTRY ?TIME|LEMONADE|LEMADE|HAW+A?I+A?N|PUNCH|MINUTE MAID|MIN MAID|M MAID|\bJUICE\b"
                       r"|SUNNY DELIGHT"),
    ("water",          r"\bWATER\b|\bWODA\b|SELTZ|CLUB SODA|TONIC"),
    ("sports_drink",   r"ENDURO|ALL ?SPORT|GATORADE"),
    ("variety_or_code", r"^B$|SINGLE SERVE UPC|ALL SELECT|CASE CODE|CASE 12OZ|VARIETY|FUN PACK"),
]

# Brand rules — checked in order on the cleaned description, first match wins
BRAND_RULES: list[tuple[str, str]] = [
    ("Dominick's",      r"^DOM\b|^DOMINICK|^HOLY COW|^GENERIC|^PS\b|^WORLD CLASS|^WRLD|^WLDWD|^WLWD"),
    ("Crystal Pepsi",   r"CRYSTAL"),
    ("Diet Coke",       r"DIET COKE|DT COKE|COKE DIET|COKE DT"),
    ("Coca-Cola",       r"COKE|COCA|CLASSIC"),
    ("Pepsi",           r"PEPSI|WILD CHERRY PEP"),
    ("Mountain Dew",    r"MOUNTA|MOUTAIN|MTN ?DE|\bMT\.? ?D|\bNT DE|\bDEW\b"),
    ("Sprite",          r"SPRITE"),
    ("7-Up",            r"7 ?-? ?UP|SEVEN ?[-/]? ?UP"),
    ("Dr Pepper",       r"DR\.? ?PEP"),
    ("Mr Pibb",         r"PIBB"),
    ("Mello Yello",     r"MELLO"),
    ("Fresca",          r"FRESCA"),
    ("Tab",             r"\bTAB\b"),
    ("Slice",           r"SLIC"),
    ("Mug",             r"\bMUG\b"),
    ("RC Cola",         r"\bR\.? ?C\.?\b|ROYAL CROWN"),
    ("Diet Rite",       r"DIET RITE"),
    ("Nehi",            r"NEHI"),
    ("A&W",             r"A ?& ?W|\bA W\b"),
    ("Barq's",          r"BARQ"),
    ("Canada Dry",      r"CANADA DRY|CAN DRY|CANADA D"),
    ("Schweppes",       r"SCHW"),
    ("Seagram's",       r"SEAGRAM|SGRMS"),
    ("Sunkist",         r"SUNKIST"),
    ("Crush",           r"CRUSH"),
    ("Squirt",          r"SQUIRT"),
    ("Vernors",         r"VERNOR"),
    ("Hires",           r"HIRES"),
    ("Dad's",           r"DAD'?S"),
    ("Canfield's",      r"CANFI?E?L?D|CANFILED"),
    ("Old Town",        r"OLD TOWN|OLD TYME"),
    ("IBC",             r"I\.?B\.?C"),
    ("Welch's",         r"WELCH"),
    ("Corr's",          r"\bCORR"),
    ("Big Red",         r"BIG RED"),
    ("Upper 10",        r"UPPER"),
    ("Goya",            r"GOYA"),
    ("Nesbitt's",       r"NESBITT"),
    ("Naturale 90",     r"NATURALE|NAT 90"),
    ("Green River",     r"GREEN RIVER"),
]
COKE_BRANDS = {"Coca-Cola", "Diet Coke", "Sprite", "Mr Pibb", "Mello Yello", "Fresca", "Tab"}
PEPSI_BRANDS = {"Pepsi", "Crystal Pepsi", "Mountain Dew", "Slice", "Mug"}

FLAVOR_RULES: list[tuple[str, str]] = [
    ("root_beer",   r"ROOT|RT ?BEER|\bR/B\b|A ?& ?W|BARQ|\bMUG\b|HIRES|DAD'?S|SARSAPAR"),
    ("cream",       r"CREAM|CREME|BIG RED|RED POP"),
    ("ginger_ale",  r"GING|GNGR|VERNOR|\bGI$|\bGIN\b|\bGN\b|CRAN-ALE"),
    ("pepper",      r"DR\.? ?PEP|PIBB|PEPPER"),
    ("cola",        r"COLA|COKE|PEPSI|\bR\.? ?C\.?\b|\bTAB\b|DIET RITE$|CLASSIC|\bCO$|DT CO\b"),
    ("citrus",      r"MOUNTA|MOUTAIN|MTN|\bMT\.? ?D|\bNT DE|\bDEW\b|MELLO|SQUIRT|FRESCA|CITRUS|GRAPEFR|\bTING\b"
                    r"|KICK|50/50"),
    ("lemon_lime",  r"SPRITE|7 ?-? ?UP|SEVEN|LEMON[- ]?LIME|LEM[- ]LIME|UPPER|LEMON|LIME|\bLMN\b"),
    ("orange",      r"ORANGE|ORNG|SUNKIST|CRUSH|MANDARIN|ORENZADA|MANDARYNDA|TANGERIN"),
    ("other_fruit", r"GRAP|CHERRY|CHER\b|STRAW|BERRY|PEACH|\bPEA\b|PINEAPP|FRUIT|PUNCH|MANGO|KIWI|LEMONIADA|RASP|RSPB"
                    r"|APPLE|TROPIC|PASSION|CRAN|BLACKCH|\bBLK\b|BANANA|CONCORD|COCO"),
]


def _clean_desc(desc: str) -> str:
    s = str(desc).upper().strip()
    s = re.sub(r"^[^A-Z0-9]+", "", s.lstrip(PREFIX_CHARS))      # also drops stray non-ASCII prefixes (e.g. 'å')
    return s.strip()


def _first_match(text: str, rules: list[tuple[str, str]], default: str) -> str:
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return default


def classify_subcategory(desc: str, com_code: int) -> str:
    if int(com_code) not in CSD_COM_CODES:
        return "out_of_scope"
    return _first_match(desc, NON_CSD_RULES, default="csd")


def diet(desc: str) -> str:
    if re.search(r"DIET|\bDT\b|SUGAR ?FREE|\bS/F\b|\bTAB\b|FRESCA|\bLITE\b|NUTRA", desc):
        return "diet"
    return "regular"


def maker(brand: str) -> str:
    if brand in COKE_BRANDS:
        return "Coca-Cola Co."
    if brand in PEPSI_BRANDS:
        return "PepsiCo"
    if brand == "Dominick's":
        return "Private label"
    return "Other"


def parse_size(size: str) -> tuple[float, int]:
    """(total ounces, pack count). '24/12O' -> (288, 24); '2 LT' -> (67.6, 1); '6 CT' -> (nan, 6)."""
    s = str(size).upper().strip()
    m = re.match(r"^(\d+)\s*/\s*(\d*\.?\d+)", s)                       # multipack "24/12O", "6/11.5"
    if m:
        n, each = int(m.group(1)), float(m.group(2))
        return n * each, n
    m = re.match(r"^(\d*\.?\d+)\s*(LT|LITE|LTRE|LITER|LTR)\b", s) or re.match(r"^(\d*\.?\d+)\s*LT", s)
    if m:
        return float(m.group(1)) * LITRE_OZ, 1
    m = re.match(r"^(\d+)\s*(CT|EA|PK|PAC|PACK)\b", s)                 # "24 PK", "6 CT" — count only
    if m:
        return np.nan, int(m.group(1))
    m = re.match(r"^(\d*\.?\d+)\s*(OZ|O\b|O$)", s)
    if m:
        return float(m.group(1)), 1
    return np.nan, 1


def pack_format(total_oz: float, count: int) -> str:
    """Package type from the parsed size (consistent across commodity codes)."""
    unit = total_oz / count if count and pd.notna(total_oz) else np.nan
    if count >= 20:
        return "can_24pk"
    if count >= 10:
        return "can_12pk"
    if count >= 4:
        return "can_6pk" if (pd.isna(unit) or unit <= 12.5) else "bottle_multipack"
    if count == 1 and pd.notna(total_oz):
        return "bottle_large" if total_oz >= 30 else "single"
    return "unknown"


def build_product_table(upc: pd.DataFrame) -> pd.DataFrame:
    """One row per UPC with derived attributes (soft drinks)."""
    df = upc.copy()
    raw = df["DESCRIP"].astype(str)
    clean = raw.map(_clean_desc)
    df["DESC_CLEAN"] = clean
    df["DISCONTINUED_FLAG"] = raw.str.strip().str.startswith("~")
    df["COMBO_ONLY_FLAG"] = raw.str.contains("#", regex=False)
    df["SUBCATEGORY"] = [classify_subcategory(d, c) for d, c in zip(clean, df["COM_CODE"])]
    df["BRAND"] = clean.map(lambda d: _first_match(d, BRAND_RULES, default="Other"))
    df["MAKER"] = df["BRAND"].map(maker)
    df["SD_DIET"] = clean.map(diet)
    df["SD_FLAVOR"] = clean.map(lambda d: _first_match(d, FLAVOR_RULES, default="unknown"))
    parsed = df["SIZE"].map(parse_size)
    df["SIZE_OZ"] = parsed.map(lambda t: t[0])
    df["PACK_COUNT"] = parsed.map(lambda t: t[1])
    df["UNIT_OZ"] = df["SIZE_OZ"] / df["PACK_COUNT"]
    df["SD_PACK"] = [pack_format(o, n) for o, n in zip(df["SIZE_OZ"], df["PACK_COUNT"])]
    # Dominick's private-label single 12 oz cans (code 227, 'DOM COLA 12 OZ') are sold as singles
    return df


ATTRIBUTE_COLUMNS = ["SUBCATEGORY", "BRAND", "PACK_COUNT", "SD_DIET", "SD_FLAVOR", "SD_PACK"]
