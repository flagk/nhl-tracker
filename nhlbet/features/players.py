"""Leak-free, as-of-game player shot features (shots on goal per game) for the player-prop model.

A player plays at most once per day, so every per-player statistic is computed from that player's *earlier* rows only (``shift(1)``).
Opponent shot allowance is the same idea per team, divided by the league average of strictly earlier dates. Rows with ``sog`` NaN
(tonight's players) get features from history only, which is exactly how live predictions are produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.data.store import Store

HALFLIFE = 8          # games, exponentially weighted shot rate
ALLOW_WINDOW = 10     # games, opponent shots-allowed window


def load_player_games(store: Store) -> pd.DataFrame:
    """One row per dressed skater per completed regular-season game: shots on goal, ice time, venue and opponent."""
    df = store.df("""
        SELECT s.game_id, g.game_date, s.player_id, s.name, s.team, s.position, s.toi_sec, s.sog, s.points,
               CASE WHEN s.team = g.home THEN 1 ELSE 0 END AS is_home,
               CASE WHEN s.team = g.home THEN g.away ELSE g.home END AS opp
        FROM skater_game s JOIN games g ON g.game_id = s.game_id
        WHERE g.game_type = 2 AND g.source = 'nhl_api' AND s.sog IS NOT NULL AND s.player_id IS NOT NULL""")
    df["game_date"] = pd.to_datetime(df.game_date)
    return df


def team_allowance(store: Store) -> pd.DataFrame:
    """(team, game_date) -> ``allow`` = mean shots against over the team's previous games, ``lg`` = league mean shots against per team-game
    over strictly earlier dates. Their ratio says how generous a defence is."""
    tg = store.df("""SELECT t.game_id, t.team, g.game_date, t.sog_against FROM team_game t JOIN games g ON g.game_id = t.game_id
                     WHERE g.game_type = 2 AND g.source = 'nhl_api' AND t.sog_against IS NOT NULL""")
    if tg.empty:
        return pd.DataFrame(columns=["team", "game_date", "allow", "lg"])
    tg["game_date"] = pd.to_datetime(tg.game_date)
    tg = tg.sort_values(["team", "game_date"])
    tg["allow"] = tg.groupby("team").sog_against.transform(lambda s: s.shift(1).rolling(ALLOW_WINDOW, min_periods=3).mean())
    daily = tg.groupby("game_date").sog_against.agg(["sum", "count"]).sort_index()
    cum = daily.cumsum().shift(1)
    lg = (cum["sum"] / cum["count"]).rename("lg")
    return tg[["team", "game_date", "allow"]].merge(lg, left_on="game_date", right_index=True, how="left")


def add_asof_features(pg: pd.DataFrame, allowance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add n_prev, ewm_sog, l10_sog, mean_sog, toi_l10, rest_days, pos (F/D) and opp_factor, all from earlier rows only."""
    pg = pg.sort_values(["player_id", "game_date", "game_id"]).copy()
    g = pg.groupby("player_id", sort=False)
    pg["n_prev"] = g.cumcount()
    prev = g.sog.shift(1)
    pg["mean_sog"] = prev.groupby(pg.player_id).expanding().mean().reset_index(level=0, drop=True)
    pg["ewm_sog"] = g.sog.transform(lambda s: s.ewm(halflife=HALFLIFE, adjust=True, ignore_na=True).mean().shift(1))
    pg["l10_sog"] = g.sog.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    if "points" in pg:                                    # same recipe for points (goals + assists), used by the points prop model
        pg["ewm_points"] = g.points.transform(lambda s: s.ewm(halflife=HALFLIFE, adjust=True, ignore_na=True).mean().shift(1))
    pg["toi_l10"] = g.toi_sec.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    pg["rest_days"] = g.game_date.diff().dt.days.clip(1, 10)
    pg["pos"] = np.where(pg.position == "D", "D", "F")
    pg["opp_factor"] = 1.0
    if allowance is not None and len(allowance):
        m = pg.merge(allowance.rename(columns={"team": "opp"}), on=["opp", "game_date"], how="left")
        f = (m["allow"] / m["lg"]).to_numpy()
        pg["opp_factor"] = np.where(np.isfinite(f), np.clip(f, 0.7, 1.4), 1.0)
    return pg
