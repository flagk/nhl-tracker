"""Walk-forward backtest of the player shots-on-goal model vs each player's own shrunk rate -> reports/PROPS.md

    python scripts/props_backtest.py --eval-start 2022-10-01
Run `python -m nhlbet.data.ingest --reparse-skaters` first (the props workflow does) so every game has shots on goal.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd

from nhlbet.data.store import Store
from nhlbet.features.players import add_asof_features, load_player_games, team_allowance
from nhlbet.models.props_eval import calibration_table, evaluate, walk_forward_props

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--eval-start", default="2022-10-01")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
store = Store(a.db)
pg = load_player_games(store)
if len(pg) < 20000:
    sys.exit(f"only {len(pg)} player-games with shots on goal: run `python -m nhlbet.data.ingest --reparse-skaters` first")
feats = add_asof_features(pg, team_allowance(store))
P = walk_forward_props(feats, a.eval_start)
if P.empty:
    sys.exit("not enough data for a props backtest")
ev = evaluate(P)
cal = calibration_table(P)
md = ["# Player shots on goal: walk-forward backtest", "",
      "> Research/education only. There are no historical sportsbook prop lines here, so this tests the model against each player's own shrunk shot rate and checks calibration at typical lines; "
      "whether it beats the *market* is measured by the paper-trading bets going forward.", "",
      f"{len(P):,} out-of-sample player-games ({P.player_id.nunique():,} players) from {P.game_date.min().date()} to {P.game_date.max().date()}; retrained every 28 days on strictly earlier games; players need {5}+ prior games.", "",
      "## Model vs the player's own rate (negative diff = model better)", "", ev.round(4).to_markdown(index=False), "",
      "## Calibration of P(over 2.5 shots)", "", cal.round(4).to_markdown(index=False), "",
      f"Average shots: predicted {P.lam.mean():.3f}, actual {P.sog.mean():.3f}."]
Path(a.out).mkdir(exist_ok=True)
Path(a.out, "PROPS.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
