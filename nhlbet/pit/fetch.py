"""Fetch raw NHL payloads into the snapshot store, with their fetch time, and probe endpoints before trusting them.

Endpoints are NOT assumed: ``probe`` records, for each candidate URL, the HTTP status and the top-level shape of the answer. Ingestion fails closed when a payload does not have
the keys the parsers need. api-web.nhle.com and api.nhle.com are unreachable from the authoring sandbox, so the probe runs in GitHub Actions (``audit-realdata.yml``) and its
result is committed as ``audit/data/endpoint_probe.json``; the shift-chart endpoint in particular is community-documented only, not official.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

from nhlbet.pit.snapshots import SnapshotStore

WEB = "https://api-web.nhle.com"
STATS = "https://api.nhle.com/stats/rest/en"

# kind -> (URL template, keys that must exist at the top level of the JSON)
ENDPOINTS = {
    "pbp": (WEB + "/v1/gamecenter/{gid}/play-by-play", ("plays", "homeTeam", "awayTeam")),
    "boxscore": (WEB + "/v1/gamecenter/{gid}/boxscore", ("playerByGameStats", "homeTeam", "awayTeam")),
    "shifts": (STATS + "/shiftcharts?cayenneExp=gameId={gid}", ("data",)),
}
SCHEDULE = WEB + "/v1/schedule/{date}"


class SchemaError(RuntimeError):
    pass


def http_get(url: str, timeout: float = 30.0) -> tuple[int, object]:
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "nhl-tracker-audit/1.0"})
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


def check_shape(kind: str, payload) -> None:
    need = ENDPOINTS[kind][1]
    if not isinstance(payload, dict) or any(k not in payload for k in need):
        raise SchemaError(f"{kind} payload lacks required keys {need}: got {sorted(payload)[:12] if isinstance(payload, dict) else type(payload).__name__}")


def fetch_game(store: SnapshotStore, game_id: int, get: Callable = http_get, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
               kinds=("pbp", "boxscore", "shifts"), pause: float = 0.25, sleep: Callable = time.sleep) -> dict[str, str]:
    """Fetch each kind for one game and store it as a new version. Returns {kind: 'stored <sha>' | 'http <status>' | 'schema: ...'} (failures are reported, never stored as data)."""
    out = {}
    for kind in kinds:
        url = ENDPOINTS[kind][0].format(gid=game_id)
        t = now()
        status, payload = get(url)
        if status != 200 or payload is None:
            out[kind] = f"http {status}"
            continue
        try:
            check_shape(kind, payload)
        except SchemaError as e:
            out[kind] = f"schema: {e}"
            continue
        v = store.put(kind, game_id, payload, t, url, status)
        out[kind] = f"stored {v.sha256[:12]}"
        sleep(pause)
    return out


def probe(get: Callable = http_get, game_id: int | None = None, date: str | None = None) -> dict:
    """Status and top-level shape of each endpoint for one real game (the newest final game on ``date``, or ``game_id``). Nothing is stored as data."""
    res = {"probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoints": {}}
    if game_id is None:
        status, sched = get(SCHEDULE.format(date=date or datetime.now(timezone.utc).date().isoformat()))
        res["endpoints"]["schedule"] = {"url": SCHEDULE.format(date=date or "today"), "status": status, "top_level_keys": sorted(sched) if isinstance(sched, dict) else None}
        games = [g for d in (sched or {}).get("gameWeek", []) for g in d.get("games", []) if g.get("gameState") in ("OFF", "FINAL")] if isinstance(sched, dict) else []
        game_id = games[-1]["id"] if games else None
    res["game_id"] = game_id
    if game_id is None:
        res["note"] = "no final game found to probe"
        return res
    for kind, (tpl, need) in ENDPOINTS.items():
        url = tpl.format(gid=game_id)
        status, payload = get(url)
        keys = sorted(payload) if isinstance(payload, dict) else None
        res["endpoints"][kind] = {"url": url, "status": status, "top_level_keys": keys, "has_required_keys": bool(keys and all(k in keys for k in need)),
                                  "required": list(need)}
    return res
