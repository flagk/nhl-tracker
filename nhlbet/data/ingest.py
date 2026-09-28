"""Idempotent ingestion: NHL API -> local cache (raw JSON) -> SQLite.

Re-running is always safe: raw responses are cached on disk, rows are upserted on their natural
keys, and games whose detail was already ingested successfully (``ingest_log``) are skipped.

    python -m nhlbet.data.ingest --seasons 20232024 20242025 20252026
    python -m nhlbet.data.ingest --update            # last 7 days + upcoming slate
    python -m nhlbet.data.ingest --legacy-csv data/history/nhl_history.csv
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nhlbet.data.client import NHLAPIError, NHLClient
from nhlbet.data.parsers import (FINAL_STATES, parse_boxscore, parse_play_by_play, parse_schedule_games,
                                 team_game_rows)
from nhlbet.data.store import Store
from nhlbet.data.xg import XGModel
from nhlbet.teams import TEAMS, canon

log = logging.getLogger("nhlbet.ingest")
XG_PATH = Path("data/manual/xg_model.json")


def season_teams(season: int) -> list[str]:
    """Teams that existed in ``season`` (e.g. 20232024). ARI became UTA in 2024-25; SEA from 2021-22."""
    start = season // 10000
    return sorted(t for t in TEAMS if not (t == "ARI" and start >= 2024) and not (t == "UTA" and start < 2024))


def ingest_schedule(client: NHLClient, store: Store, season: int) -> int:
    """Upsert every regular-season/playoff game of ``season`` from the 32 club schedules."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for team in season_teams(season):
        try:
            payload = client.club_schedule_season(team, season)
        except NHLAPIError as e:
            log.error("schedule failed for %s %s: %s", team, season, e)
            continue
        rows = parse_schedule_games(payload)
        for r in rows:
            r["updated_at"] = now
        n += store.upsert("games", rows, ["game_id"])
    log.info("season %s: %d game rows upserted", season, n)
    return n


def ingest_day(client: NHLClient, store: Store, day: str) -> int:
    payload = client.schedule(day)
    rows = parse_schedule_games(payload)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        r["updated_at"] = now
    return store.upsert("games", rows, ["game_id"])


def ingest_game_detail(client: NHLClient, store: Store, game: dict, xgm: XGModel) -> None:
    gid = game["game_id"]
    box = client.boxscore(gid)
    pbp = client.play_by_play(gid)
    pp = parse_play_by_play(pbp, xgm)
    bx = parse_boxscore(box)
    for g in bx["goalies"]:
        g["xg_faced"] = pp["goalie_xg"].get(g["player_id"])
    store.upsert("team_game", team_game_rows(game, pp), ["game_id", "team"])
    store.upsert("goalie_game", bx["goalies"], ["game_id", "player_id"])
    store.upsert("skater_game", bx["skaters"], ["game_id", "player_id"])
    store.upsert("shots", pp["shots"], ["game_id", "event_id"])


def ingest_details(client: NHLClient, store: Store, limit: int | None = None) -> tuple[int, int]:
    """Fetch boxscore + play-by-play for every final game not yet ingested. Returns (ok, failed)."""
    xgm = XGModel.load(XG_PATH)
    done = store.known_final_games()
    todo = store.df("SELECT * FROM games WHERE status IN ('OFF','FINAL') AND source='nhl_api' "
                    "AND home_score IS NOT NULL ORDER BY game_date, game_id")
    todo = todo[~todo.game_id.isin(done)]
    if limit:
        todo = todo.head(limit)
    ok = bad = 0
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    for i, g in enumerate(todo.to_dict("records"), 1):
        try:
            ingest_game_detail(client, store, g, xgm)
            store.upsert("ingest_log", [{"game_id": g["game_id"], "stage": "detail", "ok": 1, "note": None, "ts": now()}], ["game_id"])
            ok += 1
        except Exception as e:  # keep going: one bad game must not stop a backfill
            log.error("detail failed for game %s: %s", g["game_id"], e)
            store.upsert("ingest_log", [{"game_id": g["game_id"], "stage": "detail", "ok": 0, "note": str(e)[:200], "ts": now()}], ["game_id"])
            bad += 1
        if i % 100 == 0:
            log.info("details: %d/%d (ok=%d failed=%d, network calls=%d)", i, len(todo), ok, bad, client.network_calls)
    return ok, bad


def import_legacy_csv(store: Store, path: str | Path) -> int:
    """Load the legacy score-only CSV as a fallback data source (``source='legacy_csv'``).

    Synthetic negative ids keep it apart from API games; ``load_games`` drops a legacy row as soon
    as the API supplies the same (date, home, away) game.
    """
    df = pd.read_csv(path)
    games, tg = [], []
    for i, r in enumerate(df.itertuples(index=False), 1):
        gid = -i
        home, away = canon(r.Home), canon(r.Away)
        games.append({"game_id": gid, "season": None, "game_type": 2, "game_date": str(r.Date)[:10],
                      "start_utc": None, "home": home, "away": away, "home_score": int(r.HomeScore),
                      "away_score": int(r.AwayScore), "status": "FINAL", "last_period": None,
                      "home_win": int(r.HomeScore > r.AwayScore), "source": "legacy_csv", "updated_at": None})
        for side, t, o, gf, ga in (("h", home, away, r.HomeScore, r.AwayScore), ("a", away, home, r.AwayScore, r.HomeScore)):
            tg.append({"game_id": gid, "team": t, "opp": o, "is_home": int(side == "h"), "goals": int(gf), "goals_against": int(ga)})
    store.upsert("games", games, ["game_id"])
    store.upsert("team_game", tg, ["game_id", "team"])
    return len(games)


def current_season(today: date | None = None) -> int:
    d = today or date.today()
    y = d.year if d.month >= 9 else d.year - 1
    return y * 10000 + y + 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/nhl.db")
    ap.add_argument("--seasons", nargs="*", type=int, help="e.g. 20232024 20242025")
    ap.add_argument("--update", action="store_true", help="refresh current season schedule + new finals")
    ap.add_argument("--legacy-csv")
    ap.add_argument("--limit", type=int, help="max games to fetch detail for (debug)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store, client = Store(a.db), NHLClient()
    if a.legacy_csv:
        log.info("imported %d legacy games", import_legacy_csv(store, a.legacy_csv))
    seasons = a.seasons or ([current_season()] if a.update else [])
    for s in seasons:
        ingest_schedule(client, store, s)
    if a.update:
        for k in range(-7, 3):
            try:
                ingest_day(client, store, (date.today() + timedelta(days=k)).isoformat())
            except NHLAPIError as e:
                log.warning("day schedule failed: %s", e)
    if seasons or a.update:
        ok, bad = ingest_details(client, store, a.limit)
        log.info("detail ingest finished: ok=%d failed=%d", ok, bad)
        if bad and not ok:
            raise SystemExit("all detail fetches failed; check network access to api-web.nhle.com")


if __name__ == "__main__":
    main()
