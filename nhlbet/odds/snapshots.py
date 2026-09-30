"""Persist odds observations and link them to games. Snapshots are append-only: the opening line and the
closing line (for CLV) are both just rows in this table at different ``captured_at`` times."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from nhlbet.data.store import Store
from nhlbet.odds.client import OddsFetch
from nhlbet.teams import team_from_name

log = logging.getLogger(__name__)


def parse_events(events: list[dict], captured_at: str) -> list[dict]:
    """Flatten Odds API v4 events -> one row per (book, market, outcome). Unknown teams are skipped and logged."""
    rows = []
    for ev in events:
        home, away = team_from_name(ev.get("home_team", "")), team_from_name(ev.get("away_team", ""))
        if not home or not away:
            log.warning("skipping event with unknown team(s): %s @ %s", ev.get("away_team"), ev.get("home_team"))
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    price = oc.get("price")
                    if price is None or price <= 1:
                        continue
                    name = oc.get("name", "")
                    side = home if team_from_name(name) == home else away if team_from_name(name) == away else name  # Over/Under stay as-is
                    rows.append({"captured_at": captured_at, "event_id": ev["id"], "commence_time": ev.get("commence_time"),
                                 "home": home, "away": away, "book": bk.get("key"), "market": mk.get("key"),
                                 "outcome": side, "point": float(oc.get("point") or 0.0), "price": float(price),
                                 "book_updated": mk.get("last_update") or bk.get("last_update"), "game_id": None})
    return rows


def link_games(store: Store) -> int:
    """Fill ``game_id`` on unlinked snapshots by matching (home, away) and start time / date. Returns rows linked."""
    snaps = store.df("SELECT DISTINCT event_id, home, away, commence_time FROM odds_snapshots WHERE game_id IS NULL")
    if snaps.empty:
        return 0
    games = store.df("SELECT game_id, game_date, home, away, start_utc FROM games WHERE game_type IN (2,3) AND source='nhl_api'")
    n = 0
    for e in snaps.itertuples(index=False):
        ct = pd.to_datetime(e.commence_time, utc=True, errors="coerce")
        if pd.isna(ct):
            continue
        cand = games[(games.home == e.home) & (games.away == e.away)]
        cand = cand[(pd.to_datetime(cand.game_date) >= (ct - timedelta(days=1)).tz_localize(None).normalize())
                    & (pd.to_datetime(cand.game_date) <= ct.tz_localize(None).normalize())]
        if len(cand) == 0:
            continue
        # prefer the closest start time when several (doubleheaders don't happen in the NHL, so usually one)
        st = pd.to_datetime(cand.start_utc, utc=True, errors="coerce")
        gid = int(cand.loc[(st - ct).abs().fillna(pd.Timedelta(days=9)).idxmin(), "game_id"])
        with store.tx() as c:
            c.execute("UPDATE odds_snapshots SET game_id=? WHERE event_id=? AND game_id IS NULL", (gid, e.event_id))
        n += 1
    return n


def record_fetch(store: Store, fetch: OddsFetch, market_note: str = "") -> int:
    """Store a fetch (skipping re-storage of identical stale/cached data) and log quota state."""
    rows = parse_events(fetch.events, fetch.captured_at)
    n = store.upsert("odds_snapshots", rows, ["captured_at", "event_id", "book", "market", "outcome", "point"]) if rows else 0
    # A capture is logged once, with how it was really obtained. Re-reading the cached response later (same captured_at) must not
    # overwrite a 'live' row with 'cache', or the audit trail would lose the fact that this capture was a real API call.
    exists = store.df("SELECT 1 FROM odds_fetch_log WHERE captured_at=?", [fetch.captured_at]).shape[0]
    if not exists or fetch.source == "live":
        store.upsert("odds_fetch_log", [{"captured_at": fetch.captured_at, "ok": 1, "source": fetch.source, "remaining": fetch.remaining,
                                         "used": fetch.used, "events": len({r["event_id"] for r in rows}), "note": market_note}],
                     ["captured_at"])
    link_games(store)
    return n
