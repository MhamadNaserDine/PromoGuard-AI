# PromoGuard AI — Phase 1 Findings (Canned Tuna, `tna`)

Source: `reports/phase1/phase1_tna_report.txt` (output of `src/data/01_data_understanding.py`)
and the official *Dominick's Data Manual and Codebook* (Kilts Center), `docs/`.

## 1. Structure — matches the official documentation
| Item | Result |
|---|---|
| UPC file | 278 products, 6 columns (`COM_CODE, UPC, DESCRIP, SIZE, CASE, NITEM`), no missing values |
| Movement file | 3,763,229 rows, 11 columns (`STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE, PROFIT, OK, PRICE_HEX, PROFIT_HEX`) |
| Row count vs. manual | 3,763,229 = manual's stated count (exact match) |
| Key `(UPC, STORE, WEEK)` | unique — no duplicates |
| Join on `UPC` | 278 / 278 matched, 0 orphan rows |
| Stores / weeks | 93 stores; weeks 1–399 (week 1 = 14 Sep 1989); 24 weeks missing chain-wide |

## 2. Data-quality issues (to handle in Phase 2 — not yet changed)
1. **`OK = 0`: 77,965 rows (2.1%).** Manual: "1 for valid data, 0 for trash". → Exclude.
2. **`PRICE = 0`: 1,359,671 rows (36%).** These rows also have `MOVE = 0` and `PROFIT = 0`.
   They are weeks where the product was not sold/stocked, *not* weeks with zero demand at a price.
   → Must not be treated as "zero sales" observations in demand models.
3. **452 rows with `MOVE > 0` but `PRICE = 0`.** Logically invalid. → Exclude.
4. **Bundles (`QTY > 1`): 11,085 rows (0.3%).** `PRICE` is the bundle price.
   Unit price = `PRICE / QTY`; dollar sales = `PRICE × MOVE / QTY` (manual formula).
5. **Extreme `MOVE` values:** median 2, 99th pct 125, max 19,456 units/week. → Outlier review in Phase 2.
6. **Store coverage is uneven:** most stores have all 375 weeks; some have as few as 22.
7. **Missing weeks:** 262–265, 278–279, 284–285, 314–317, 352–357, 384–387, 392–393 (chain-wide gaps).
   Lags and rolling features must respect these gaps.

## 3. Promotion field (`SALE`) — NOT a reliable label on its own
| Code | Rows | Share | Meaning (manual) |
|---|---|---|---|
| missing | 3,325,958 | 88.4% | no code recorded |
| `B` | 417,998 | 11.1% | Bonus Buy |
| `S` | 17,949 | 0.5% | simple price reduction |
| `G` | 1,324 | 0.04% | **undocumented** (manual lists only B, C, S) |
| `C` | 0 | 0% | coupon — never appears in this category |

- Manual: the code "is not set by DFF on a consistent basis — if not set, there might still be a promotion".
- Only **51%** of `B` rows show a price ≥5% below the product-store median price; `S`: 78%; `G`: 96%.
  So "Bonus Buy" is often not a price cut (possibly display/feature support).
- `G` rows: very high volume (median 63 units) and **negative average margin (−2.3%)** → looks like loss-leader deals.
- Week 219 (Nov 1993) has no deal code at all — likely a recording gap.

**Consequence:** the promotion definition in Phase 2 must combine the deal code with observed
price reductions, and be documented and reproducible.

## 4. Profit (`PROFIT`)
- Manual: gross margin in percent (25.3 = 25.3 cents per dollar of sales).
- Cost basis is **average acquisition cost (AAC)**, not replacement cost. Manufacturer deals can make
  AAC drop temporarily, so margins during/after promotions may be distorted.
- 11,173 rows with negative margin.
- **Terminology for the project:** "estimated gross-profit impact" — never "net profit".
- Gross profit per row = `PRICE × MOVE / QTY × PROFIT / 100`.

## 5. Category composition — important for cannibalization
- All 278 products share one commodity code (329), so `COM_CODE` cannot define related products.
- The category mixes tuna, salmon, sardines, clams, oysters, crab, caviar, anchovies, and more.
- Demand is concentrated in tuna: the **top 10 UPCs (all tuna) = 48% of units**; top 50 = 81%.
- Main brands visible in `DESCRIP`: StarKist, Bumble Bee, Chicken of the Sea (COS), Dominick's own
  label (DOM/HH), Geisha, 3-Diamond. Descriptions are truncated to 20 characters and spelled
  inconsistently (e.g. "STAR K", "STAR KIST", "STARKIST", "STARKSIT").
- Description prefixes (manual): `~` discontinued, `#` combo stores only, `<` trial size; `$` and `*` mean nothing.

**Proposed scope:** focus the core analysis on the canned **tuna** sub-category, with product attributes
(brand, light/white, water/oil, size) parsed from `DESCRIP` to define substitutes.

## 6. Causal-identification notes
- Manual: randomized experiments ran in "more than 25 categories", but does **not** list which ones.
  We must not assume tuna was experimental.
- Manual: 16 pricing zones in 4 price tiers (Cub-Fighter, low, medium, high); regular prices vary by zone,
  but **promoted prices are chainwide**. Promotions are therefore set centrally, not per store — relevant
  for treatment assignment and confounding. The store→zone table is in the manual (stores opened after 1992 have none).
