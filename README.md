# PromoGuard AI

**Causal AI decision intelligence for retail promotions.** PromoGuard measures what a promotion really did:
how many *extra* units it sold, how many it took from other products (cannibalization), and whether it
made or lost **gross profit**. It then explains why, and simulates the next promotion before it runs.
An AI analyst (a Claude LLM agent with 15 analysis tools) answers questions in plain language.

Data: Dominick's Finer Foods (Chicago, 1989–1997), weekly store-level scanner data from the
Kilts Center for Marketing, Chicago Booth. The same pipeline is run on **two product categories**:
**canned tuna** and **carbonated soft drinks**.

> Scope: product and store level. There is no individual customer data. Personalization is done by
> **store customer group** (built from census data around each store), never by individual shopper.

---

## Main results

| | Canned tuna | Soft drinks |
|---|---|---|
| Promotion events analysed | 1,068 | 22,469 |
| Typical sales lift (median event) | +50% | +174% |
| Pooled sales lift | +218% | +291% |
| Extra units taken from other products (cannibalization) | 4.1% | 4.5% |
| Net gross profit · retailer pays the discount | ≈ $0 (+$2.5k) | **−$56.7M** |
| Net gross profit · recorded cost (incl. supplier funding) | +$849k | −$9.3M |
| Gross profit per $1 of discount · retailer pays | +$0.00 | −$0.51 |
| Events that raised gross profit · retailer pays | 62% | 19% |
| Best discount depth (profit per $1) | < 5% (+$1.25) | < 5% (−$0.12) |

**Same tool, different advice.**

- **Canned tuna:** promotions roughly pay for themselves. Shallow and medium discounts (up to about 15%)
  make money, and very deep ones lose it.
- **Soft drinks:** sales react strongly, but margins are thin (about 8.5%), so almost every price cut loses
  gross profit. A soft-drink sale only breaks even if the supplier pays at least half the discount.
  Deals with no price cut lose the least.

**Model quality**

- Baseline (no-promotion) model error on unseen weeks: 13% for tuna and 21% for soft drinks
  (WAPE, product-week level).
- Pre-period check: +4% for tuna and +0.3% for soft drinks.
- Lead test: passed for both categories.

**AI analyst evaluation:** **26 / 26** questions with known correct answers (100%):

- 16 tuna questions, 9 soft-drink questions and 1 comparison question
- correct tool, correct number and correct product category in every answer
- mean answer time 5.2 s

All figures are estimates under the assumptions listed below. Gross profit is margin on the average
acquisition cost, not net profit.

---

## How it works (10 phases)

| Phase | Script | What it does |
|---|---|---|
| 1 | `src/data/01_data_understanding.py` | Inspect the raw files, columns and coverage |
| 2 | `src/data/02_data_cleaning.py` | Clean sales rows; derive product attributes (brand, type, size) |
| 3 | `src/data/03_promotion_definition.py` | Regular price = 12-week modal price; promotion = ≥ 5% below it or a deal code; chain events (≥ 50% of stores) |
| 4 | `src/analysis/04_eda.py` | Seasonality, holidays/Lent, brand shares, post-promotion dips, data gaps |
| 5 | `src/models/05_baseline_model.py` | Learn normal sales without promotion (Random Forest vs ridge, gradient boosting, XGBoost; time-based validation) |
| 6 | `src/analysis/06_incremental_sales.py` | Cross-fitted counterfactual → lift, post-promotion dip, placebo windows |
| 7 | `src/analysis/07_cannibalization.py` | Store × week panel, two-way fixed effects, instrumental variables (expected discount volume), lead test |
| 8 | `src/analysis/08_profit_impact.py` | Gross-profit impact in two cost views; inferred supplier (trade) funding |
| 9 | `src/models/09_explain_and_simulate.py` | Monotonic gradient-boosting response model, SHAP explanations, what-if simulator |
| 10 | `src/analysis/10_customer_groups.py` | Store customer groups (k-means on census data) and personalized discount plan (tuna) |

The **dashboard** (`dashboard/app.py`, Streamlit) shows everything, with a category switch on every page. It has
12 pages, in four groups:

- **Ask AI:** AI analyst
- **Results:** Executive overview, Promotion analysis, Cannibalization, Profit impact, Report card,
  Compare categories
- **Decide:** What-if simulator (with break-even funding and a supplier negotiation sheet), Customer groups
- **Trust:** Explainability, Model performance, Method & limitations

The **AI analyst** (`src/agent/`) is a tool-using LLM agent (ReAct style).

- Claude chooses tools, the tools return computed results, and Claude explains them. It never invents numbers.
- Every tool call is shown under the answer.
- `src/agent/evaluate_agent.py` scores the agent against ground truth computed by the same tools.

---

## Setup

### 1. Python

Use **one Python 3.12 interpreter** for everything: pipeline, dashboard and evaluation. Saved models must
be loaded with the same scikit-learn version that created them (this project: **1.8.0**).

```
pip install -r requirements.txt
```

### 2. Data (not included: Kilts Center licence, academic use only)

Download from the Kilts Center Dominick's page and place the files like this:

```
data/raw/dominicks/tna/upctna.csv      canned tuna: product (UPC) file
data/raw/dominicks/tna/wtna.zip        canned tuna: weekly movement file
data/raw/dominicks/sdr/upcsdr.csv      soft drinks: product file
data/raw/dominicks/sdr/wsdr.zip        soft drinks: weekly movement file
data/raw/dominicks/demo/demo.dta       store demographics (only for phase 10)
```

The store and week reference tables (`data/reference/`) come from the Dominick's manual and are included.

### 3. Claude API key (only for the AI analyst)

Create `.streamlit/secrets.toml` with one line:

```
ANTHROPIC_API_KEY = "sk-ant-..."
```

This file is listed in `.gitignore` and must never be shared or committed.

---

## Run

```
python run_pipeline.py --list            # show the phases
python run_pipeline.py                   # canned tuna, phases 1-10  (under 1 h)
python run_pipeline.py --cat sdr         # soft drinks, phases 1-9   (~2 h; 7.5 million rows)
python run_pipeline.py --cat sdr --from 6   # resume from a phase
python run_pipeline.py --eval            # also evaluate the AI analyst (needs the API key)

python -m streamlit run dashboard/app.py # open the dashboard
python -m src.agent.evaluate_agent       # AI analyst evaluation only
```

Each phase writes a text report and figures to `reports/phaseN/`; soft-drink outputs are in
`reports/phaseN/sdr/`. Intermediate tables go to `data/processed/` and models to `models/`.

---

## Project structure

```
PromoGuard-AI/
├── run_pipeline.py          one-command pipeline
├── requirements.txt
├── dashboard/               Streamlit app (app.py), data layer (data.py), theme (theme.py)
├── src/
│   ├── data/                phases 1-3
│   ├── analysis/            phases 4, 6, 7, 8, 10
│   ├── models/              phases 5, 9 + counterfactual cross-fitting, SHAP
│   ├── features/            product attributes (tuna, soft drinks), baseline and event features
│   ├── simulation/          what-if simulator
│   └── agent/               AI analyst: tools, agent loop, evaluation
├── data/reference/          store and week tables (from the Dominick's manual)
├── reports/                 phase reports, figures, AI evaluation
└── docs/                    Dominick's manual and codebook
```

---

## Assumptions and limitations

- **Counterfactual:** no hidden demand shock coincides with a promotion. Supported by the pre-period check
  and placebo windows.
- **Cannibalization:** within a week, store-to-store differences in discount depth affect other products
  only through the promoted product. Supported by the lead test.
- **Cost:** the recorded average acquisition cost. Supplier (trade) funding is *inferred* from the drop in that
  cost, not observed.
- **Data:**
  - Stock-outs cannot be seen.
  - There are data gaps: tuna Jan–Oct 1996, soft drinks Jul–Sep 1994.
  - Hidden chain promotions (display or ad without a price cut) are excluded from model training.
- **Profit:** gross profit, not net profit. Labour, advertising and slotting fees are not included.
- **Basket effect:** cheap soft drinks may bring shoppers who also buy other categories. This basket effect
  is not measured, because the data covers one category at a time.
- **Simulations:** estimates. Discounts deeper than most past events (30% for tuna, 45% for soft drinks)
  are flagged as extrapolations.

---

*Data: James M. Kilts Center for Marketing, University of Chicago Booth School of Business —
Dominick's Finer Foods database. Used for academic purposes only.*
