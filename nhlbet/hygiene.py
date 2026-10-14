"""Data hygiene: remove anything that was logged for a game *after* it started.

A run that happens after puck drop (a delayed scheduled job, a manual re-run) sees in-play odds and must not log them as pre-game market data,
and must not replace the earlier pre-game decision for that game. ``build_slate`` now skips started games and the odds readers ignore in-play
quotes; this purge cleans rows written before those guards existed, and is idempotent.
"""
from __future__ import annotations

import logging

import pandas as pd

from nhlbet.data.store import Store

log = logging.getLogger(__name__)

RUN_TABLES = ("recommendations", "shadow_bets", "alt_quotes")


def purge_inplay(store: Store) -> dict[str, int]:
    """Delete run rows whose run time is after the game's start and consensus rows captured after the start. Returns rows removed per table."""
    removed: dict[str, int] = {}
    with store.tx() as c:
        for t in RUN_TABLES:
            cur = c.execute(f"DELETE FROM {t} WHERE EXISTS (SELECT 1 FROM games g WHERE g.game_id = {t}.game_id AND g.start_utc IS NOT NULL "
                            f"AND julianday({t}.run_at) > julianday(g.start_utc))")
            removed[t] = cur.rowcount
        cur = c.execute("DELETE FROM odds_consensus WHERE EXISTS (SELECT 1 FROM games g WHERE g.game_id = odds_consensus.game_id AND g.start_utc IS NOT NULL "
                        "AND julianday(odds_consensus.captured_at) > julianday(g.start_utc))")
        removed["odds_consensus"] = cur.rowcount
    removed = {k: v for k, v in removed.items() if v}
    if removed:
        log.warning("purged in-play data logged after puck drop: %s", removed)
    return removed


MAX_DATA_AGE_DAYS = 3


def data_freshness(store: Store, date: str, refresh_error: str | None = None, max_age_days: int = MAX_DATA_AGE_DAYS) -> tuple[bool, str]:
    """(ok, message). Picks must never be published on stale inputs.

    Stale = the NHL data refresh failed AND the newest completed regular-season game is more than ``max_age_days`` before ``date`` while games are
    scheduled for ``date`` (a quiet off-season is not stale: there is nothing to pick). A failed refresh with recent cached data is allowed, with a note.
    """
    last = store.df("SELECT MAX(game_date) AS d FROM games WHERE game_type = 2 AND home_score IS NOT NULL AND source = 'nhl_api'")
    today = store.df("SELECT COUNT(*) AS n FROM games WHERE game_date = ? AND game_type IN (2, 3)", [date])
    newest = last.d.iloc[0] if len(last) else None
    if not refresh_error:
        return True, ""
    if newest is None:
        return (False, f"data refresh failed ({refresh_error}) and there is no completed-game history: no picks published") if int(today.n.iloc[0]) else (True, "")
    age = (pd.Timestamp(date) - pd.Timestamp(newest)).days
    if age > max_age_days and int(today.n.iloc[0]):
        return False, f"data refresh failed ({refresh_error}) and the newest completed game is {age} days old: picks withheld"
    return True, ""


def validate_inputs(store: Store) -> list[str]:
    """Schema and sanity checks on the game table the model is about to use. Returns problems (empty = fine); the daily job withholds picks when there are any.

    Checks: the needed columns exist, game ids and (date, home, away) are unique among API games, final scores are non-negative integers with a decided
    winner, ``home_win`` agrees with the score, and no completed game lacks a result flag.
    """
    problems = []
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(games)")}
    missing = {"game_id", "game_date", "home", "away", "home_score", "away_score", "home_win", "status", "game_type"} - cols
    if missing:
        return [f"games table is missing columns: {sorted(missing)}"]
    g = store.df("SELECT game_id, game_date, home, away, home_score, away_score, home_win, status, source FROM games")
    if g.empty:
        return []
    if g.game_id.duplicated().any():
        problems.append(f"{int(g.game_id.duplicated().sum())} duplicate game ids")
    api = g[g.source == "nhl_api"]
    dup = api.duplicated(["game_date", "home", "away"]).sum()
    if dup:
        problems.append(f"{int(dup)} duplicate (date, home, away) games")
    fin = api[api.home_score.notna() & api.away_score.notna()]
    if ((fin.home_score < 0) | (fin.away_score < 0)).any():
        problems.append("negative scores")
    if (fin.home_score == fin.away_score).any():
        problems.append(f"{int((fin.home_score == fin.away_score).sum())} completed games with equal scores")
    bad = fin[(fin.home_win.isna()) | (fin.home_win != (fin.home_score > fin.away_score).astype(int))]
    if len(bad):
        problems.append(f"{len(bad)} completed games where home_win disagrees with the score")
    if g.game_date.isna().any():
        problems.append("games without a date")
    return problems
