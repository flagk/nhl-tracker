"""As-of observations: event time and knowledge time stored separately.

``event_time``     when the thing happened (the game's puck drop).
``knowledge_time`` when the repository first held this version of the value (the fetch time of the snapshot it was parsed from).

A feature for a game on date E may use a past game's statistics only in the version known at the decision time of date E. If the provider later corrects a shot
location, the corrected version has a later knowledge time and is invisible to every decision made before it was fetched.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable

from nhlbet.data.parsers import TEAM_STAT_COLS, parse_play_by_play, team_game_rows
from nhlbet.data.xg import XGModel
from nhlbet.pit.snapshots import SnapshotStore


def team_observations(store: SnapshotStore, game: dict, xgm: XGModel | None = None) -> list[dict]:
    """One row per (version of the play-by-play, team): the parsed team stats with ``event_time`` and ``knowledge_time``.

    ``game``: game_id, home, away, home_score, away_score, start_utc (the result as known when the play-by-play was final).
    """
    out = []
    for v in store.versions("pbp", game["game_id"]):
        pp = parse_play_by_play(store.load(v), xgm)
        for r in team_game_rows(game, pp):
            out.append({**r, "event_time": game.get("start_utc"), "knowledge_time": v.knowledge_time, "snapshot": v.sha256})
    return out


def make_adv_asof(obs: list[dict], decision_hour_utc: float = 13.0) -> Callable[[int, str, int], dict]:
    """Resolver ``f(game_id, team, day_ordinal) -> dict`` of advanced stats as known at that day's decision time (default 13:00 UTC, about the morning run).
    Unknown at that time -> ``{}`` (the feature builder reads that as missing, exactly as for a game never ingested)."""
    by: dict[tuple[int, str], list[tuple[str, dict]]] = defaultdict(list)
    for r in obs:
        by[(r["game_id"], r["team"])].append((r["knowledge_time"], {c: r.get(c) for c in TEAM_STAT_COLS}))
    for k in by:
        by[k].sort(key=lambda x: x[0])

    def resolve(game_id: int, team: str, day: int) -> dict:
        t = (datetime.fromordinal(day).replace(tzinfo=timezone.utc) + timedelta(hours=decision_hour_utc)).isoformat(timespec="seconds")
        best = {}
        for kt, stats in by.get((game_id, team), []):
            if kt <= t:
                best = stats
            else:
                break
        return best
    return resolve


def revision_report(obs: list[dict]) -> dict:
    """How often, and by how much, a later fetch of the same game changed the team statistics (the size of the point-in-time problem)."""
    by = defaultdict(list)
    for r in obs:
        by[(r["game_id"], r["team"])].append(r)
    multi = {k: sorted(v, key=lambda x: x["knowledge_time"]) for k, v in by.items() if len({x["snapshot"] for x in v}) > 1}
    deltas = defaultdict(list)
    for rows in multi.values():
        a, b = rows[0], rows[-1]
        for c in TEAM_STAT_COLS:
            x, y = a.get(c), b.get(c)
            if x is not None and y is not None and x == x and y == y and x != y:
                deltas[c].append(abs(y - x))
    return {"team_games": len(by), "team_games_with_revisions": len(multi),
            "columns_changed": {c: {"n": len(d), "mean_abs_change": sum(d) / len(d)} for c, d in sorted(deltas.items())}}
