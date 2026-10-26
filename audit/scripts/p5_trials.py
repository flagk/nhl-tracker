"""Phase 5: count every model variant, setting and threshold the project has ever tried that could have influenced a reported number.

The count is a LOWER BOUND: it reads what the code and committed configuration still contain (grids, strategy lists, thresholds). Manual experiments that left no trace in the
repository (notebook sessions, ablations described in prose) cannot be counted and are listed as unknown. Writes audit/data/trial_register.csv and prints the total.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from nhlbet.models.base import make_zoo  # noqa: E402
from nhlbet.models.tune import GRIDS  # noqa: E402
from nhlbet.risk.parlays import PARLAY_STRATEGIES  # noqa: E402
from nhlbet.risk.shadow import PROP_STRATEGIES, STRATEGIES  # noqa: E402

pol = yaml.safe_load((ROOT / "config/policy.yaml").read_text())
sel = json.loads((ROOT / "data/models/feature_selection.json").read_text())
rows = []


def add(family, n, source, note=""):
    rows.append({"family": family, "variants": n, "source": source, "note": note})


add("base models in the zoo", len(make_zoo(sel["kept"], sel["kept"])), "nhlbet/models/base.py make_zoo", "home_rate, elo, logistic, rf, lgbm, xgb, lgbm_all")
add("hyper-parameter grid points (tuned by time-series CV)", sum(len(g) for _, g in GRIDS.values()), "nhlbet/models/tune.py GRIDS", "one chosen per model")
add("ensemble heads and calibration maps", 2 + 4, "nhlbet/models/walkforward.py, calibration.py", "stack, weighted average; platt, isotonic, online platt, online isotonic (beta added by the audit, not counted in the original work)")
add("candidate features screened for the 14 kept", len(sel["kept"]) + len(sel["dropped"]), "data/models/feature_selection.json", f"{len(sel['kept'])} kept, {len(sel['dropped'])} dropped")
add("builder settings (half-life, shrinkage, window, Elo K, home advantage, regression)", 6, "nhlbet/features/builder.py BuilderConfig", "fields exist; the number of values tried is not recorded, counted once each")
add("goals-model regularisation grid", 9, "scripts/goals_backtest.py", "alpha grid 0.3 .. 300")
add("props shrinkage grid and dispersions", 9 + 7, "nhlbet/models/props.py, scripts/props_backtest.py")
add("paper-trading game strategies", len(STRATEGIES), "nhlbet/risk/shadow.py STRATEGIES")
add("paper-trading prop strategies", len(PROP_STRATEGIES), "nhlbet/risk/shadow.py PROP_STRATEGIES")
add("paper-trading parlay strategies", len(PARLAY_STRATEGIES), "nhlbet/risk/parlays.py")
add("edge thresholds examined in reports", len(pol["evaluation"]["edge_thresholds"]), "config/policy.yaml evaluation.edge_thresholds")
add("de-vig methods", 3, "nhlbet/odds/math.py", "proportional, power, Shin")
add("settlement rule sets", len(pol["settlement"]["rulesets"]), "config/policy.yaml settlement.rulesets")
add("model-selection candidates compared in Phase 4 (hardening)", 9, "reports/model_comparison_phase4.md")
add("walk-forward step lengths and minimum training sizes tried", 2, "scripts/backtest.py defaults", "recorded values only")
add("markets reported (h2h, totals, spreads, 4 player props, parlays)", 8, "reports/evaluation.md")
add("UNKNOWN manual experiments", 0, "not recoverable", "README history, ablations in prose, anything run interactively and not committed")
total = sum(r["variants"] for r in rows)
out = ROOT / "audit/data"
out.mkdir(parents=True, exist_ok=True)
with (out / "trial_register.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "variants", "source", "note"]); w.writeheader(); w.writerows(rows)
    w.writerow({"family": "TOTAL (sum, lower bound)", "variants": total, "source": "", "note": "used as the number of trials in the deflated Sharpe ratio; the strategy matrix used for PBO and the reality check has fewer columns (only paper strategies have return series)"})
print(total)
