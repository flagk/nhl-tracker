"""Monte Carlo risk-of-ruin for the current staking policy.

    python scripts/risk_sim.py [--edge 0.03] [--stake-pct 0.01] [--bets 500] [--bankroll 1000]

Uses the bet log's realised (stake, odds, probability) template when >= 30 bets exist, otherwise an illustrative template.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nhlbet.risk.montecarlo import default_template, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--edge", type=float, default=0.03)
ap.add_argument("--decimal", type=float, default=1.95)
ap.add_argument("--stake-pct", type=float, default=0.01)
ap.add_argument("--bets", type=int, default=500)
ap.add_argument("--paths", type=int, default=5000)
ap.add_argument("--bankroll", type=float, default=1000.0)
a = ap.parse_args()
t = default_template(a.edge, a.decimal, a.stake_pct)
pd.set_option("display.width", 200)
res = simulate(t, a.paths, a.bets, bankroll0=a.bankroll)
print(f"template: edge {a.edge:.1%} at ~{a.decimal} odds, stake {a.stake_pct:.1%} of bankroll, {a.bets} bets, {a.paths} paths, start ${a.bankroll:,.0f}")
print("skill = share of the model's claimed edge that is real (0 = none, 1 = all)")
print(res.round(3).to_string())
