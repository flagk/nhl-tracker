"""Walk-forward backtest of the goals model (totals and puck line) vs base rates -> reports/GOALS.md.

    python scripts/goals_backtest.py --eval-start 2021-10-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd

from nhlbet.config import load_builder_config
from nhlbet.data.store import Store
from nhlbet.features.builder import build_features
from nhlbet.models.goals import add_goal_targets, goal_feature_columns
from nhlbet.models.goals_eval import calibration_params, evaluate, goal_rate_table, online_calibrate_goals, walk_forward_goals

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--eval-start", default="2021-10-01")
ap.add_argument("--step-days", type=int, default=14)
ap.add_argument("--alphas", nargs="*", type=float, default=[10.0, 50.0, 200.0])
ap.add_argument("--wf", default="reports/walkforward_predictions.csv")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
store = Store(a.db)
F = build_features(store, load_builder_config())
F = F[(F.game_type == 2) & F.home_score.notna()]
F = add_goal_targets(F, store.df("SELECT game_id, home_score, away_score, last_period FROM games"))
feats = goal_feature_columns(F[F.game_date < pd.Timestamp(a.eval_start)])
p_home = None
if Path(a.wf).exists():
    w = pd.read_csv(a.wf, index_col=0)
    p_home = w["stack__online"].dropna() if "stack__online" in w else (w["stack"] if "stack" in w else None)

res, best = {}, None
for alpha in a.alphas:
    P = walk_forward_goals(F, feats, a.eval_start, a.step_days, alpha=alpha, p_home=p_home)
    if P.empty:
        sys.exit("not enough data for a goals backtest")
    rate = goal_rate_table(P)
    score = float(rate["diff"].mean())                         # mean Poisson NLL difference vs base (negative = better)
    res[alpha] = (P, evaluate(P), rate, score)
    if best is None or score < res[best][3]:
        best = alpha
P, ev, rate, _ = res[best]
cal = calibration_params(P)
Pc = online_calibrate_goals(P, a.step_days)
evc = evaluate(Pc, calibrated=True)
Path(a.out).mkdir(exist_ok=True)
P.to_csv(Path(a.out) / "goals_walkforward.csv")
md = ["# Goals model: walk-forward backtest (totals and puck line)", "",
      "> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; "
      "whether it beats the *market* is measured by the paper-trading bets going forward.", "",
      f"{len(P):,} out-of-sample games from {P.game_date.min().date()} to {P.game_date.max().date()}; retrained every {a.step_days} days on strictly earlier games; "
      f"{len(feats)} candidate features; regularisation alpha chosen: **{best}** (of {a.alphas}).", "",
      "## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)", "", rate.round(4).to_markdown(index=False), "",
      "## Totals and puck line vs base rates (log loss; negative diff = model better)", "", ev.round(4).to_markdown(index=False), "",
      "## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)", "",
      (evc.round(4).to_markdown(index=False) if len(evc) else "Not enough out-of-sample predictions to calibrate."), "",
      "`lin_slope` is the coefficient of outcome on the model's logit: about 1 means calibrated, well below 1 means overconfident, near 0 means no signal. "
      "The maps used in production (logit p' = a + b*logit p): " + json.dumps({k: {"a": round(v["a"], 3), "b": round(v["b"], 3)} for k, v in cal.items()}) + ".", "",
      f"Mean predicted total {P.p_total_mean.mean():.2f} vs actual {P.tot.mean():.2f}; predicted tie-after-60 rate {P.p_tie.mean():.3f} vs actual {(P.hr == P.ar).mean():.3f}.", "",
      "## Alpha comparison (mean Poisson NLL diff)", ""] + [f"- alpha {k}: {v[3]:+.5f}" for k, v in res.items()]
Path(a.out, "GOALS.md").write_text("\n".join(md) + "\n")
Path(a.out, "goals_model.json").write_text(json.dumps({"alpha": best, "features_n": len(feats)}))
Path(a.out, "goals_calibration.json").write_text(json.dumps(cal))
print("\n".join(md))
