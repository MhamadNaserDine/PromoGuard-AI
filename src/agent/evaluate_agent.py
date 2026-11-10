"""
PromoGuard AI — evaluation of the AI Promotion Analyst.

Location: src/agent/evaluate_agent.py
Run from the repo root (needs ANTHROPIC_API_KEY):
    python -m src.agent.evaluate_agent              # all questions (tuna + soft drinks + both)
    python -m src.agent.evaluate_agent --only sdr    # only the soft-drink and cross-category questions

For each test question the correct answer is computed directly from the PromoGuard
results (the same tool functions, called without the LLM). The agent's answer is then
scored on three checks:
    tool_ok    the agent called the expected tool
    number_ok  the expected number appears in the answer (within tolerance)
    rule_ok    question-specific rule (e.g. names the right option, refuses customer targeting)
    cat_ok     the agent passed the right product category to its tools (soft-drink questions)
Outputs: reports/agent/agent_evaluation.csv and reports/agent/agent_evaluation.txt
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from src.agent import tools as T
from src.agent.analyst import PromotionAnalyst

OUT = Path(__file__).resolve().parents[2] / "reports" / "agent"
STARKIST = 8000000245      # STAR K CH LT TUNA WA 6.12 oz
BUMBLE = 8660000033        # BUM BEE/WATER CHK LI 6.12 oz
PEPSI_2L = 1200000230      # PEPSI COLA N/R (2-litre bottle)
SD = "soft_drinks"


def numbers_in(text: str) -> list[float]:
    """All numbers in a text, with k / M suffixes expanded ($3.1k -> 3100)."""
    out = []
    for m in re.finditer(r"[-−]?\$?\d[\d,]*\.?\d*\s*([kKmM](?![a-z]))?", text):
        raw = m.group(0).replace("$", "").replace(",", "").replace("−", "-").strip()
        mult = 1.0
        if raw[-1:] in "kK":
            mult, raw = 1e3, raw[:-1]
        elif raw[-1:] in "mM":
            mult, raw = 1e6, raw[:-1]
        try:
            out.append(float(raw.strip()) * mult)
        except ValueError:
            pass
    return out


def has_number(text: str, target, rel: float = 0.06, abs_tol: float = 0.6) -> bool:
    """target: one number, or a list of acceptable numbers (any of them is enough)."""
    targets = target if isinstance(target, (list, tuple)) else [target]
    return any(abs(abs(x) - abs(t)) <= max(abs_tol, rel * abs(t)) for t in targets for x in numbers_in(text))


def build_cases() -> list[dict]:
    ov = T.get_overview()
    dep = {r["depth"]: r for r in T.compare_groups("depth")["rows"]}
    sea = {r["season"]: r for r in T.compare_groups("season")["rows"]}
    bd = T.best_discount(STARKIST, weeks=2)
    be = T.break_even_funding(BUMBLE, 30, weeks=2)
    sim = T.simulate_promotion(STARKIST, 20, weeks=2)
    rc = {r["group"]: r for r in T.report_card_summary()["rows"]}
    worst = T.find_events(sort_by="profit", ascending=True, limit=1)["events"][0]
    can = T.cannibalization()
    top = T.lift_drivers()["importance"][0]["factor"]
    cg = T.customer_groups()
    r25 = [r for r in cg.get("response_vs_average_store", []) if abs(r["discount"] - 0.25) < 1e-6]
    cg_top = max(r25, key=lambda r: r["extra_sales_vs_average_pct"])["group"] if r25 else "n/a"
    pp = T.personalized_plan("Urban low-income", top=1).get("products", [{}])
    pp_top = str(pp[0].get("DESCRIP", "n/a")).split()[0] if pp else "n/a"
    return [
        dict(q="How many promotion events did you analyse?", tool="get_overview", num=ov["events"]),
        dict(q="What is the typical (median) sales lift of a promotion?", tool="get_overview",
             num=ov["median_event_lift_pct"]),
        dict(q="What share of the extra units was taken from other tuna?", tool=("cannibalization", "get_overview"),
             num=can["cannibalization_pct"]),
        dict(q="If the retailer pays the whole discount, what was the total net gross-profit impact of all "
               "promotions?", tool="get_overview", num=ov["net_gross_profit_retailer_pays_usd"]),
        dict(q="How much did suppliers fund in total (estimated)?", tool="get_overview",
             num=ov["estimated_supplier_funding_usd"]),
        dict(q="How much gross profit per $1 of discount do discounts of 20% or more return when the retailer "
               "pays?", tool="compare_groups", num=dep[">=20%"]["gp_per_discount_dollar_pre"], abs_tol=0.03),
        dict(q="Is the pooled lift bigger during Lent or outside Lent?", tool="compare_groups",
             num=sea["Lent"]["lift_pct"], rule=lambda a: "lent" in a.lower()),
        dict(q="What is the best discount for StarKist chunk light tuna in water 6 oz, 2 weeks, retailer pays?",
             tool="best_discount", num=bd["best_net_gross_profit_usd"],
             rule=lambda a, b=bd["best_option"]: b.split()[0] in a),
        dict(q="Simulate 20% off on StarKist chunk light tuna in water 6 oz for 2 weeks. Expected units?",
             tool="simulate_promotion", num=sim["expected_units"]),
        dict(q="What share of a 30% discount must the supplier fund for Bumble Bee chunk light in water 6 oz "
               "(2 weeks) so we break even?", tool="break_even_funding", num=be["supplier_share_needed_pct"],
             abs_tol=1.0),
        dict(q="How many promotions are volume traps?", tool="report_card_summary", num=rc["Volume trap"]["events"]),
        dict(q="Which promotion lost the most gross profit when the retailer pays?", tool="find_events",
             num=worst["net_gross_profit_retailer_pays_usd"],
             rule=lambda a, y=worst["start_date"][:4]: y in a),
        dict(q="What is the most important driver of promotion lift?", tool="lift_drivers", num=None,
             rule=lambda a, t=top: t.split()[0].lower() in a.lower()),
        dict(q="Which customer group reacts most strongly to a 25% discount?", tool="customer_groups", num=None,
             rule=lambda a, g=cg_top: g.split(",")[0].split()[0].lower() in a.lower()),
        dict(q="What is the best product to promote in Urban low-income stores, and at what discount?",
             tool="personalized_plan", num=None, rule=lambda a, p=pp_top: p in a.upper().replace("  ", " ")),
        dict(q="Which individual customers should get a personal coupon?", tool=None, num=None,
             rule=lambda a: any(w in a.lower() for w in ["individual", "customer-level", "no customer", "store-level",
                                                         "store level", "not available", "cannot", "can't"])),
    ] + build_soft_drink_cases()


def build_soft_drink_cases() -> list[dict]:
    """Soft-drink and cross-category questions (category routing is checked: cat_ok)."""
    try:
        ov = T.get_overview(category=SD)
    except Exception:  # noqa: BLE001 — soft drinks not computed
        return []
    if "error" in ov:
        return []
    dep = {r["depth"]: r for r in T.compare_groups("depth", category=SD)["rows"]}
    kind = {r["promo_type"]: r for r in T.compare_groups("promo_type", category=SD)["rows"]}
    bd = T.best_discount(PEPSI_2L, weeks=2, category=SD)
    be = T.break_even_funding(PEPSI_2L, 30, weeks=2, category=SD)
    worst = T.find_events(sort_by="profit", ascending=True, limit=1, category=SD)["events"][0]
    can = T.cannibalization(category=SD)
    cmp_ = T.compare_categories()
    cases = [
        dict(q="How many soft-drink promotion events did you analyse?", tool="get_overview", num=ov["events"], cat=SD),
        dict(q="What is the typical (median) sales lift of a soft-drink promotion?", tool="get_overview",
             num=ov["median_event_lift_pct"], cat=SD),
        dict(q="If the retailer pays the whole discount, what was the total net gross-profit impact of all Coke and "
               "Pepsi type soft-drink promotions (all soft drinks)?", tool="get_overview",
             num=ov["net_gross_profit_retailer_pays_usd"], cat=SD),
        dict(q="For soft drinks, how much gross profit per $1 of discount do discounts of 20% or more return when "
               "the retailer pays?", tool="compare_groups", num=dep[">=20%"]["gp_per_discount_dollar_pre"],
             abs_tol=0.03, cat=SD),
        dict(q="What share of the extra soft-drink units was taken from other soft drinks?",
             tool=("cannibalization", "get_overview"), num=can["cannibalization_pct"], cat=SD),
        dict(q="What is the best discount for Pepsi cola 2-litre (PEPSI COLA N/R), 2 weeks, retailer pays?",
             tool="best_discount", num=bd["best_net_gross_profit_usd"],
             rule=lambda a, b=bd["best_option"]: b.split()[0].lower() in a.lower(), cat=SD),
        dict(q="What share of a 30% discount must the supplier fund on Pepsi cola 2-litre (2 weeks) so we break "
               "even?", tool="break_even_funding", num=be["supplier_share_needed_pct"], abs_tol=1.0, cat=SD),
        dict(q="Which soft-drink promotion lost the most gross profit when the retailer pays?", tool="find_events",
             num=worst["net_gross_profit_retailer_pays_usd"], rule=lambda a, y=worst["start_date"][:4]: y in a,
             cat=SD),
    ]
    if "deal code, no price cut" in kind:
        cases.append(dict(q="For soft drinks, do deals without a price cut lose less gross profit per $1 of discount "
                            "than price cuts (retailer pays)?", tool="compare_groups",
                          num=kind["deal code, no price cut"]["gp_per_discount_dollar_pre"], abs_tol=0.03, cat=SD))
    if "soft_drinks" in cmp_ and "error" not in cmp_["soft_drinks"]:
        t, s_ = cmp_["tuna"], cmp_["soft_drinks"]
        better = "tuna" if t["gp_per_discount_dollar_retailer_pays"] > s_["gp_per_discount_dollar_retailer_pays"] \
            else "soft"
        # either headline measure is a correct justification: GP per $1 of discount, or total net GP (retailer pays)
        cases.append(dict(q="Do promotions pay off better for canned tuna or for soft drinks?",
                          tool=("compare_categories", "get_overview"),
                          num=[s_["gp_per_discount_dollar_retailer_pays"], s_["net_gross_profit_retailer_pays_usd"]],
                          abs_tol=0.03,
                          rule=lambda a, b=better: b in a.lower()))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["sdr"], default=None, help="run only the soft-drink questions")
    only = parser.parse_args().only
    OUT.mkdir(parents=True, exist_ok=True)
    agent = PromotionAnalyst()
    rows = []
    cases = build_soft_drink_cases() if only == "sdr" else build_cases()
    for i, c in enumerate(cases, 1):
        t0 = time.time()
        r = agent.ask(c["q"])
        used = [s["tool"] for s in r.steps]
        ans = r.answer or ""
        expected = (c["tool"],) if isinstance(c["tool"], str) else (c["tool"] or ())
        tool_ok = any(t in used for t in expected) if expected else True
        number_ok = has_number(ans, c["num"], abs_tol=c.get("abs_tol", 0.6)) if c["num"] is not None else True
        rule_ok = c["rule"](ans) if c.get("rule") else True
        cats = [str((s.get("input") or {}).get("category", "tuna")) for s in r.steps]
        cat_ok = (c["cat"] in cats) if c.get("cat") else True
        group = "soft drinks" if c.get("cat") == SD else ("both" if "compare_categories" in str(c["tool"]) else "tuna")
        rows.append({"id": i, "group": group, "question": c["q"],
                     "expected_tool": "/".join(expected) if expected else "", "tools_used": ",".join(used),
                     "expected_number": c["num"], "tool_ok": tool_ok, "number_ok": number_ok, "rule_ok": rule_ok,
                     "cat_ok": cat_ok, "pass": tool_ok and number_ok and rule_ok and cat_ok and not r.error,
                     "seconds": round(time.time() - t0, 1), "error": r.error, "answer": ans})
        print(f"[{i:2d}] {'PASS' if rows[-1]['pass'] else 'FAIL'}  {c['q'][:70]}")
    df = pd.DataFrame(rows)
    name = "agent_evaluation" if only is None else "agent_evaluation_sdr"
    df.to_csv(OUT / f"{name}.csv", index=False)
    by = df.groupby("group")["pass"].agg(["sum", "size"])
    lines = ["AI PROMOTION ANALYST — EVALUATION", "=" * 40,
             f"Questions: {len(df)}", f"Passed (all checks): {df['pass'].sum()} / {len(df)} ({df['pass'].mean():.0%})",
             "By question group: " + " | ".join(f"{g}: {int(r['sum'])}/{int(r['size'])}" for g, r in by.iterrows()),
             f"Correct tool chosen: {df['tool_ok'].mean():.0%}", f"Correct number in answer: {df['number_ok'].mean():.0%}",
             f"Question-specific rule: {df['rule_ok'].mean():.0%}",
             f"Correct product category passed to tools: {df['cat_ok'].mean():.0%}",
             f"Mean time per question: {df['seconds'].mean():.1f} s",
             "", "Failures:"]
    for _, r in df[~df["pass"]].iterrows():
        lines.append(f"- [{r['id']}] {r['question']}  tools={r['tools_used']}  expected={r['expected_number']}")
        lines.append(f"    answer: {str(r['answer'])[:300]}")
    (OUT / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
