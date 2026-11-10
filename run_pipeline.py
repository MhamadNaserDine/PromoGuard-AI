"""
PromoGuard AI — run the whole pipeline with one command.

Location: run_pipeline.py (repo root)

Usage (from the repo root, with the same Python you use for the dashboard):
    python run_pipeline.py                      # canned tuna, all phases (1-10)
    python run_pipeline.py --cat sdr            # carbonated soft drinks, phases 1-9
    python run_pipeline.py --cat all            # both categories
    python run_pipeline.py --cat sdr --from 5   # start at phase 5 (earlier outputs already exist)
    python run_pipeline.py --cat tna --only 9   # one phase only
    python run_pipeline.py --eval               # also run the AI analyst evaluation (needs the API key)
    python run_pipeline.py --list               # show the phases and stop

Each phase is a separate script that writes its own report in reports/phaseN/.
The runner stops at the first phase that fails and prints a timing summary.
Approximate run times on a laptop: tuna under 1 hour; soft drinks about 2 hours
(phases 5-7 are the slowest: 7.5 million rows).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RAW = REPO / "data" / "raw" / "dominicks"

# (phase, script, description, categories it applies to)
PHASES = [
    (1, "src/data/01_data_understanding.py", "Data understanding (raw files, columns, coverage)", ("tna", "sdr")),
    (2, "src/data/02_data_cleaning.py", "Data cleaning + product attributes", ("tna", "sdr")),
    (3, "src/data/03_promotion_definition.py", "Promotion definition + chain events", ("tna", "sdr")),
    (4, "src/analysis/04_eda.py", "Exploratory analysis + calendar features", ("tna", "sdr")),
    (5, "src/models/05_baseline_model.py", "Baseline (no-promotion) demand model", ("tna", "sdr")),
    (6, "src/analysis/06_incremental_sales.py", "Incremental sales (lift) per event", ("tna", "sdr")),
    (7, "src/analysis/07_cannibalization.py", "Cannibalization (fixed effects + IV)", ("tna", "sdr")),
    (8, "src/analysis/08_profit_impact.py", "Gross-profit impact (two cost views)", ("tna", "sdr")),
    (9, "src/models/09_explain_and_simulate.py", "Response model, SHAP, what-if simulator", ("tna", "sdr")),
    (10, "src/analysis/10_customer_groups.py", "Store customer groups (census) — tuna only", ("tna",)),
]
CAT_NAMES = {"tna": "canned tuna", "sdr": "carbonated soft drinks"}
RAW_FILES = {"tna": ["upctna.csv", "wtna.zip"], "sdr": ["upcsdr.csv", "wsdr.zip"]}


def check_raw(cat: str) -> list[str]:
    """Raw Kilts Center files needed for phases 1-2 (movement file may also be unzipped: wtna.csv)."""
    missing = []
    for f in RAW_FILES[cat]:
        p = RAW / cat / f
        if not p.exists() and not p.with_suffix(".csv").exists():
            missing.append(str(p.relative_to(REPO)))
    return missing


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the PromoGuard AI pipeline.")
    ap.add_argument("--cat", default="tna", choices=["tna", "sdr", "all"], help="product category (default tna)")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first phase to run (default 1)")
    ap.add_argument("--to", dest="end", type=int, default=10, help="last phase to run (default 10)")
    ap.add_argument("--only", type=int, default=None, help="run this phase only")
    ap.add_argument("--eval", action="store_true", help="run the AI analyst evaluation at the end (API key needed)")
    ap.add_argument("--list", action="store_true", help="list the phases and exit")
    a = ap.parse_args()

    if a.list:
        for n, script, desc, cats in PHASES:
            print(f"  {n:>2}  {desc:<48} {script}  [{', '.join(cats)}]")
        return
    start, end = (a.only, a.only) if a.only else (a.start, a.end)
    cats = ["tna", "sdr"] if a.cat == "all" else [a.cat]

    timings = []
    for cat in cats:
        print("=" * 90 + f"\nPROMOGUARD AI PIPELINE — {CAT_NAMES[cat]} (--cat {cat}) — phases {start}-{end}\n" + "=" * 90)
        if start <= 2:
            miss = check_raw(cat)
            if miss:
                print("Missing raw data (download from the Kilts Center, see README):\n  " + "\n  ".join(miss))
                sys.exit(1)
        for n, script, desc, applies in PHASES:
            if not (start <= n <= end) or cat not in applies:
                continue
            cmd = [sys.executable, script] + (["--cat", cat] if n != 10 else [])
            t0 = time.time()
            code = run(cmd)
            dt = time.time() - t0
            timings.append((cat, n, desc, dt, code))
            if code != 0:
                print(f"\nPhase {n} ({desc}) FAILED for {cat} — fix the error above, then resume with:\n"
                      f"    python run_pipeline.py --cat {cat} --from {n}")
                summary(timings)
                sys.exit(code)
    if a.eval:
        t0 = time.time()
        code = run([sys.executable, "-m", "src.agent.evaluate_agent"])
        timings.append(("both", 11, "AI analyst evaluation", time.time() - t0, code))
    summary(timings)


def summary(timings: list) -> None:
    print("\n" + "=" * 90 + "\nSUMMARY\n" + "=" * 90)
    for cat, n, desc, dt, code in timings:
        print(f"  {cat:<4} phase {n:>2}  {desc:<48} {dt / 60:6.1f} min  {'OK' if code == 0 else 'FAILED'}")
    print(f"  total {sum(t[3] for t in timings) / 60:.1f} min")
    print("\nNext: python -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
