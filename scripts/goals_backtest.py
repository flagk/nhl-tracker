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
from nhlbet.models.goals_eval import calibration_params, evaluate, goal_rate_table, joint_table, online_calibrate_goals, pit_table, totals_pmf_table, walk_forward_goals
from nhlbet.settle import rules_for

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--eval-start", default="2021-10-01")
ap.add_argument("--step-days", type=int, default=14)
ap.add_argument("--alphas", nargs="*", type=float, default=[0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])      # wide enough to bracket the optimum: a value on the edge of the grid is not a result
ap.add_argument("--select-end", default="2024-06-30", help="the regularisation is chosen on games up to this date and the headline numbers are scored after it")
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
    rate = goal_rate_table(P[P.game_date <= pd.Timestamp(a.select_end)])      # alpha is chosen on games up to the selection date only
    score = float(rate["diff"].mean())                         # mean Poisson NLL difference vs base (negative = better)
    res[alpha] = (P, evaluate(P), rate, score)
    if best is None or score < res[best][3]:
        best = alpha
P, ev, rate, _ = res[best]
pinned = bool(best in (min(a.alphas), max(a.alphas)))
rate = goal_rate_table(P)
cal = calibration_params(P)
Pc = online_calibrate_goals(P, a.step_days)
evc = evaluate(Pc, calibrated=True)
Path(a.out).mkdir(exist_ok=True)
P.to_csv(Path(a.out) / "goals_walkforward.csv")
md = ["# Goals model: walk-forward backtest (totals and puck line)", "",
      "> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; "
      "whether it beats the *market* is measured by the paper-trading bets going forward.", "",
      f"{len(P):,} out-of-sample games from {P.game_date.min().date()} to {P.game_date.max().date()}; retrained every {a.step_days} days on strictly earlier games; "
      f"{len(feats)} candidate features; regularisation alpha chosen on games up to {a.select_end}: **{best}** (of {a.alphas})" + ("; **this is the edge of the grid, so it is not used in production**" if pinned else "") + ".", "",
      "## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)", "", rate.round(4).to_markdown(index=False), "",
      "## Which joint score distribution? (out-of-sample log-likelihood of the observed regulation score; higher is better)", "", joint_table(P).round(4).to_markdown(index=False), "",
      "## Totals and puck line vs base rates (log loss; negative diff = model better)", "", ev.round(4).to_markdown(index=False), "",
      f"**Scored only after the selection date ({a.select_end}), when the regularisation was already fixed:**", "", evaluate(P[P.game_date > pd.Timestamp(a.select_end)]).round(4).to_markdown(index=False), "",
      f"## Totals as a whole distribution (shootout rule `{rules_for()['ruleset']}`: a shootout adds one goal to the total = {rules_for()['shootout_in_total']})", "",
      "Each forecast is scored on the observed game total under the CURRENT rule. `log_score` is higher-is-better, `rps` (ranked probability score) lower-is-better; the interval is for the "
      "paired RPS difference to the training base rate, resampling whole days. 'previous model' is the same goal model with the shootout goal excluded, i.e. what the pipeline "
      "published before the rule was made explicit.", "", totals_pmf_table(P).round(5).to_markdown(index=False), "",
      "Randomised probability-integral-transform of the totals forecast (a calibrated forecast fills the bins equally):", "", pit_table(P).round(4).to_markdown(index=False), "",
      "## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)", "",
      (evc.round(4).to_markdown(index=False) if len(evc) else "Not enough out-of-sample predictions to calibrate."), "",
      "`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. "
      "The maps used in production (logit p' = a + b*logit p): " + json.dumps({k: {"a": round(v["a"], 3), "b": round(v["b"], 3), "c": round(v.get("c", 0.0), 3)} for k, v in cal.items()}) + ". "
      "`cal_slope_se` is its standard error: totals probabilities barely vary, so a slope of 0.5 versus 0.8 is within noise.", "",
      f"Mean predicted total {P.p_total_mean.mean():.2f} vs actual {P.tot.mean():.2f}; predicted tie-after-60 rate {P.p_tie.mean():.3f} vs actual {(P.hr == P.ar).mean():.3f}.", "",
      "## Alpha comparison (mean Poisson NLL diff)", ""] + [f"- alpha {k}: {v[3]:+.5f}" for k, v in res.items()]
Path(a.out, "GOALS.md").write_text("\n".join(md) + "\n")
Path(a.out, "goals_model.json").write_text(json.dumps({"alpha": best, "features_n": len(feats), "grid": a.alphas, "selected_on_games_through": a.select_end, "pinned_to_grid_edge": pinned}))
Path(a.out, "goals_calibration.json").write_text(json.dumps({**cal, "_rule": rules_for()["ruleset"]}))     # the maps are only valid for the shootout rule they were fitted under
print("\n".join(md))
