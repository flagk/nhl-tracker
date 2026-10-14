"""Walk-forward backtest of the player count models (shots, points, assists, anytime goals) vs each player's own shrunk rate -> reports/PROPS.md

    python scripts/props_backtest.py --eval-start 2022-10-01

For every stat it scores the full count PMF (negative-binomial log-likelihood and ranked probability score, not just win rate), the over probability at
each realistic line, and the reliability of P(over) line by line. It also tests how much to shrink a player's own rate toward his position's average
(pseudo-games 3 / 6 / 12 / 25 / 50): the value is CHOSEN on games before --select-end and then SCORED on later games, and the choice is written to
reports/props_shrink.json for the live model. Run `python -m nhlbet.data.ingest --reparse-skaters` first (the props workflow does).
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

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.features.players import add_asof_features, load_player_games, team_allowance
from nhlbet.models.props import SHRINK_K
from nhlbet.models.props_eval import LINES_BY_STAT, _pmf_matrix, calibration_table, evaluate, walk_forward_props

GRID = (3.0, 6.0, 12.0, 25.0, 50.0)
NAMES = {"sog": "Player shots on goal", "points": "Player points (goals + assists)", "assists": "Player assists", "goals": "Player goals (anytime scorer = over 0.5)"}

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--eval-start", default="2022-10-01")
ap.add_argument("--select-end", default="2024-06-30", help="the shrinkage is chosen on games up to this date and scored on later games")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
store = Store(a.db)
pg = load_player_games(store)
if len(pg) < 20000:
    sys.exit(f"only {len(pg)} player-games with shots on goal: run `python -m nhlbet.data.ingest --reparse-skaters` first")
feats = add_asof_features(pg, team_allowance(store))


def nll(P: pd.DataFrame, kmax: int | None = None) -> np.ndarray:
    pm = _pmf_matrix(P.lam, P.k) if kmax is None else _pmf_matrix(P.lam, P.k, kmax)
    return -np.log(np.maximum(pm[np.arange(len(P)), np.minimum(P.y.to_numpy(int), pm.shape[1] - 1)], 1e-12))


md = ["# Player count models: walk-forward backtest", "",
      "> Research/education only. There are no historical sportsbook prop lines here, so this tests each model against the player's own shrunk rate and checks calibration at typical lines; "
      "whether it beats the *market* is measured by the paper-trading bets going forward. `n_to_confirm` is the number of player-games a significance test would need at the observed effect "
      "size; where `n` is smaller the verdict reads \"insufficient evidence\".", ""]
chosen: dict[str, float] = {}
for stat in ("sog", "points", "assists", "goals"):
    sens = {}
    for k in GRID:
        P = walk_forward_props(feats, a.eval_start, stat=stat, shrink_k=k)
        if P.empty:
            continue
        sens[k] = P
    if not sens:
        continue
    cut = pd.Timestamp(a.select_end)
    sel = {k: float(nll(P[P.game_date <= cut]).mean()) for k, P in sens.items() if (P.game_date <= cut).sum() > 1000}
    k_best = min(sel, key=sel.get) if sel else SHRINK_K
    chosen[stat] = k_best
    P = sens[k_best]
    after = {k: float(nll(P2[P2.game_date > cut]).mean()) for k, P2 in sens.items() if (P2.game_date > cut).sum() > 1000}
    md += [f"## {NAMES[stat]}", "",
           f"{len(P):,} out-of-sample player-games ({P.player_id.nunique():,} players) from {P.game_date.min().date()} to {P.game_date.max().date()}; retrained every 28 days on strictly earlier games; "
           "players need 5+ prior games.", "",
           f"**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to {a.select_end}: **{k_best:g}**; "
           "scored on later games:", "",
           pd.DataFrame({"shrink_k": list(sens), f"NLL up to {a.select_end}": [sel.get(k, np.nan) for k in sens], "NLL after": [after.get(k, np.nan) for k in sens]}).round(5).to_markdown(index=False), "",
           "**Model vs the player's own rate** (negative diff = model better):", "", evaluate(P, stat).round(4).to_markdown(index=False), ""]
    for line in LINES_BY_STAT[stat]:
        md += [f"**Reliability of P(over {line:g}), by predicted-probability bin:**", "", calibration_table(P, line).round(4).to_markdown(index=False), ""]
    md += [f"Average {stat}: predicted {P.lam.mean():.3f}, actual {P.y.mean():.3f}.", ""]
# ---- goalie saves: research model only (see nhlbet/models/saves.py); the goalie's save-percentage prior is chosen the same way (before --select-end, scored after) ----
try:
    from nhlbet.models.saves import goalie_starts, walk_forward_saves
    gs = goalie_starts(store)
    if len(gs) > 3000:
        sens = {k: walk_forward_saves(gs, a.eval_start, sv_prior=k) for k in (200.0, 400.0, 800.0, 1600.0)}
        sens = {k: v for k, v in sens.items() if not v.empty}
        cut = pd.Timestamp(a.select_end)
        selv = {k: float(nll(v[v.game_date <= cut], 90).mean()) for k, v in sens.items() if (v.game_date <= cut).sum() > 500}
        kb = min(selv, key=selv.get) if selv else 400.0
        Ps = sens[kb]
        aft = {k: float(nll(v[v.game_date > cut], 90).mean()) for k, v in sens.items() if (v.game_date > cut).sum() > 500}
        md += ["## Goalie saves (research model, not priced live)", "",
               f"{len(Ps):,} goalie starts ({Ps.player_id.nunique():,} goalies) from {Ps.game_date.min().date()} to {Ps.game_date.max().date()}. Saves = shots against thinned by the goalie's shrunk save percentage; "
               "a goalie who is pulled has fewer shots against in the history, which is how the risk of a short night enters the distribution. Settlement of a goalie who starts but is replaced differs by book, "
               "so nothing is priced or staked from this model until a book's rule and live prices are verified.", "",
               f"**Save-percentage prior sensitivity** (pseudo-shots of league-average goaltending). Chosen on starts up to {a.select_end}: **{kb:g}**; scored after:", "",
               pd.DataFrame({"sv_prior": list(sens), f"NLL up to {a.select_end}": [selv.get(k, np.nan) for k in sens], "NLL after": [aft.get(k, np.nan) for k in sens]}).round(5).to_markdown(index=False), "",
               "**Model vs the goalie's own average:**", "", evaluate(Ps, "saves").round(4).to_markdown(index=False), ""]
        for line in LINES_BY_STAT["saves"]:
            md += [f"**Reliability of P(over {line:g} saves):**", "", calibration_table(Ps, line).round(4).to_markdown(index=False), ""]
        md += [f"Average saves: predicted {Ps.lam.mean():.2f}, actual {Ps.y.mean():.2f}.", ""]
except Exception as exc:                                  # the saves section must never block the other stats
    md += [f"_Goalie saves backtest unavailable: {exc}_", ""]
Path(a.out).mkdir(exist_ok=True)
Path(a.out, "PROPS.md").write_text("\n".join(md) + "\n")
Path(a.out, "props_shrink.json").write_text(json.dumps({"shrink_k": chosen, "selected_on_games_through": a.select_end}, indent=1))
print("\n".join(md))
