"""Null worlds for pipeline validation: outcomes shuffled within each season, so team strength carries no information about results while every feature
pipeline step still runs on realistic data. A pipeline that finds skill here is leaking."""
from __future__ import annotations

import numpy as np
import pandas as pd


def shuffle_outcomes(tables: dict[str, pd.DataFrame], seed: int = 0) -> dict[str, pd.DataFrame]:
    """Copy of ``tables`` where each game's result (scores, overtime/shootout flag, winner) is reassigned to another game of the same season, and the team-level
    goal columns follow the new score. Stats that are not results (shots, penalties, goalie workload) stay with their game."""
    rng = np.random.default_rng(seed)
    t = {k: v.copy() for k, v in tables.items()}
    g = t["games"].reset_index(drop=True)
    cols = ["home_score", "away_score", "last_period", "home_win"]
    for _, idx in g.groupby("season").groups.items():
        idx = np.asarray(list(idx))
        perm = rng.permutation(idx)
        g.loc[idx, cols] = g.loc[perm, cols].to_numpy()
    t["games"] = g
    new = g[["game_id", "home_score", "away_score"]].rename(columns={"home_score": "_hs", "away_score": "_as"})
    tg = t["team_game"].merge(new, on="game_id", how="left")
    tg["goals"] = np.where(tg.is_home == 1, tg._hs, tg._as)
    tg["goals_against"] = np.where(tg.is_home == 1, tg._as, tg._hs)
    t["team_game"] = tg.drop(columns=["_hs", "_as"])
    return t
