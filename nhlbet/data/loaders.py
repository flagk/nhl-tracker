"""Load clean, de-duplicated tables from the store for feature building."""
from __future__ import annotations

import pandas as pd

from nhlbet.data.store import Store


def season_of(date: pd.Series) -> pd.Series:
    """NHL season id = calendar year in which the season started (Aug 1 boundary)."""
    d = pd.to_datetime(date)
    return (d.dt.year - (d.dt.month < 8).astype(int)).astype(int)


def load_games(store: Store, include_unplayed: bool = True) -> pd.DataFrame:
    """All games, chronologically ordered, one row per real game.

    A legacy-CSV row is dropped whenever an API game exists with the same (date, home, away).
    """
    g = store.df("SELECT * FROM games WHERE game_type IN (2,3)")
    if g.empty:
        return g
    g["game_date"] = pd.to_datetime(g["game_date"])
    api_keys = set(zip(g.loc[g.source == "nhl_api", "game_date"], g.loc[g.source == "nhl_api", "home"],
                       g.loc[g.source == "nhl_api", "away"]))
    dup = (g.source == "legacy_csv") & pd.Series([k in api_keys for k in zip(g.game_date, g.home, g.away)], index=g.index)
    g = g[~dup]
    g = g[g.home != g.away]
    if not include_unplayed:
        g = g[g.home_score.notna()]
    derived = season_of(g.game_date)
    # the API's own season id (e.g. 20192020 -> 2019) is authoritative: the 2019-20 bubble playoffs were played Aug-Sep 2020, which a
    # date rule would wrongly file under the 2020-21 season. Legacy CSV rows carry no season, so they fall back to the date rule.
    api = pd.to_numeric(g.season, errors="coerce")
    api_year = api.where(api < 10000, api // 10000)              # accept 20192020 (API) or 2019 (plain start year)
    g["season"] = api_year.where(api.notna(), derived).astype(int)
    g["start_utc"] = pd.to_datetime(g.start_utc, utc=True, errors="coerce")
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_tables(store: Store) -> dict[str, pd.DataFrame]:
    games = load_games(store)
    ids = set(games.game_id)
    out = {"games": games}
    for t in ("team_game", "goalie_game", "skater_game"):
        df = store.df(f"SELECT * FROM {t}")
        out[t] = df[df.game_id.isin(ids)].reset_index(drop=True)
    return out
