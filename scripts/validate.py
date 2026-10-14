"""Re-run the honest evaluation from the committed files only:  python scripts/validate.py  -> reports/VALIDATION.json (+ a readable summary on stdout)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.report.validation import (backtest_by_period, log_integrity, market_summary, pick_table, random_pick_baseline, required_n,  # noqa: E402
                                      score_vs_market, split_table, summarize_picks)

ROOT = Path(".")
gp = pd.read_csv(ROOT / "data/export/games_predictions.csv")
paper = pd.read_csv(ROOT / "data/export/paper_trading.csv")
res: dict = {"data_through": str(gp.game_date.max()), "games_with_results": int(gp.home_win.notna().sum())}

res["game_level"] = score_vs_market(gp)
res["picks"], res["splits"], res["random_baseline"] = {}, {}, {}
for strat in ("every_game", "flat_model_side", "market_favorite", "home_ml_control"):
    d = pick_table(gp, paper, strat)
    res["picks"][strat] = summarize_picks(d)
    if strat in ("every_game", "flat_model_side"):
        res["splits"][strat] = {k: split_table(d, k).round(4).to_dict("records") for k in ("month", "side", "role", "odds_bucket")}
        res["random_baseline"][strat] = random_pick_baseline(d, gp)
    if strat == "every_game":
        sd = res["picks"][strat].get("sd_unit_return", np.nan)
        csd = res["picks"][strat].get("clv", {}).get("sd", np.nan)
        res["required_bets"] = {"roi": {f"{e:.0%}": required_n(e, sd) for e in (0.01, 0.02, 0.03, 0.05)},
                                "clv_ev": {f"{e:.1%}": required_n(e, csd) for e in (0.005, 0.01, 0.02)}}

res["by_market"] = market_summary(paper)                      # the model's picks only
res["controls_by_market"] = market_summary(paper, controls=True)   # no-skill controls: the price of the bookmaker margin alone
res["log_integrity"] = log_integrity(ROOT / "data/logs", ROOT / "site/archive")

wf = ROOT / "reports/walkforward_predictions.csv"
if wf.exists():
    P = pd.read_csv(wf, index_col=0)
    res["backtest_by_season"] = {m: backtest_by_period(P, m).round(4).to_dict("records") for m in ("logistic", "stack")}
Path("reports/VALIDATION.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float)[:6000])
