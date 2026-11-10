"""
PromoGuard AI — the AI Promotion Analyst (tool-using LLM agent).

Location: src/agent/analyst.py

How it works (ReAct-style tool use)
    1. The user asks a question in plain language.
    2. Claude decides which tool(s) to call (src/agent/tools.py) and with which arguments.
    3. The tools return numbers computed by the PromoGuard pipeline (never invented).
    4. Claude reads the results, may call more tools, then writes a short answer that
       says what kind of number each one is (causal estimate / model / simulation).
Every tool call is recorded in `steps`, so the dashboard can show "how I got this answer".

Setup
    pip install anthropic
    API key: environment variable ANTHROPIC_API_KEY, or .streamlit/secrets.toml
             (ANTHROPIC_API_KEY = "sk-ant-...")
    Model:   PROMOGUARD_MODEL (default claude-sonnet-5-5)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.agent.tools import anthropic_tools, run_tool

DEFAULT_MODEL = os.environ.get("PROMOGUARD_MODEL", "claude-sonnet-5-5")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are the PromoGuard AI Promotion Analyst, an assistant for a retail category manager.
The data: Dominick's Finer Foods (Chicago), 1989-1997, store-level weekly scanner data (no individual customers),
two product categories analysed with the same method:
  - canned tuna (category="tuna", 1,068 chain promotion events)
  - carbonated soft drinks (category="soft_drinks", 22,469 chain promotion events; Coke, Pepsi, 7-Up, Dominick's...)

Rules
- Use the tools for every number. Never invent or estimate numbers yourself. If no tool can answer, say so.
- Choose the category from the question (Coke, Pepsi, soda, cola, 7-Up, Sprite... = soft_drinks; StarKist,
  Bumble Bee, Chicken of the Sea, tuna = tuna) and pass it to every tool. If the question does not say,
  use tuna and say so in one short phrase. For questions about both, use compare_categories.
- For simulations, first find the product_id with list_products in the same category (names are abbreviated).
- Customer groups (customer_groups, personalized_plan) exist for canned tuna only.
- Say what kind of number you give: observed data, causal estimate (under stated assumptions),
  model estimate, or simulation (not a guarantee).
- Profit means GROSS profit (margin on average acquisition cost), never net profit.
- Two cost views exist: "retailer pays the whole discount" (conservative) and "recorded cost"
  (includes supplier/trade funding, which is estimated, not observed). Name the view you use.
- Mention an extrapolation warning when a tool returns extrapolation_warning = true (rare discount depth).
- The project is product/store level. Personalization is by STORE CUSTOMER GROUP (customer_groups,
  personalized_plan), never by individual customer: refuse individual-customer targeting.
- Answer in simple, clear English, short: 2-6 sentences or a small table, then one actionable
  recommendation when relevant. Round numbers sensibly ($3.1k, +50%).
"""


@dataclass
class AgentReply:
    answer: str
    steps: list[dict] = field(default_factory=list)   # [{"tool", "input", "output"}]
    error: str | None = None


def get_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:                               # Streamlit secrets, when running inside the dashboard
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001
        return None


class PromotionAnalyst:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or get_api_key())
        self.client, self.model = client, model
        self.tools = anthropic_tools()

    def ask(self, question: str, history: list[dict] | None = None) -> AgentReply:
        """history: previous turns as [{"role": "user"|"assistant", "content": str}] (text only)."""
        messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
        messages.append({"role": "user", "content": question})
        steps: list[dict] = []
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = self.client.messages.create(model=self.model, max_tokens=1500, system=SYSTEM_PROMPT,
                                                   tools=self.tools, messages=messages)
                if resp.stop_reason != "tool_use":
                    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
                    return AgentReply(text or "(no answer)", steps)
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if getattr(b, "type", "") != "tool_use":
                        continue
                    out = run_tool(b.name, b.input)
                    steps.append({"tool": b.name, "input": b.input, "output": out})
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out, default=str)[:20000]})
                messages.append({"role": "user", "content": results})
            return AgentReply("I needed too many steps for this question. Please ask something more specific.", steps)
        except Exception as exc:  # noqa: BLE001
            return AgentReply("", steps, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":     # quick terminal test:  python -m src.agent.analyst "your question"
    import sys
    q = " ".join(sys.argv[1:]) or "What is the best discount for StarKist chunk light tuna 6 oz?"
    r = PromotionAnalyst().ask(q)
    for s in r.steps:
        print(f"[tool] {s['tool']}({s['input']})")
    print(r.error or r.answer)
