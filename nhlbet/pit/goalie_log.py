"""Append-only log of goalie confirmation events with the time each was first seen.

``goalie_confirmations`` keeps one row per (date, team) and overwrites it when the confirmed goalie changes, which loses the earlier confirmation and its time. This log keeps every
distinct (date, team, goalie) once, with ``first_seen``, so "who was confirmed at decision time" stays answerable after a late change. Lives under data/logs, so the hash-chain ledger covers it.
"""
from __future__ import annotations

import json
from pathlib import Path

LOG = "data/logs/goalie_events/log.jsonl"


def append_events(store, path: str | Path = LOG) -> int:
    p = Path(path)
    seen = set()
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                seen.add((e["game_date"], e["team"], e["player_id"], e["name"]))
    rows = store.df("SELECT game_date, team, player_id, name, source, captured_at FROM goalie_confirmations ORDER BY captured_at")
    new = []
    for r in rows.itertuples(index=False):
        pid = None if r.player_id != r.player_id or r.player_id is None else int(r.player_id)
        k = (r.game_date, r.team, pid, r.name)
        if k in seen:
            continue
        seen.add(k)
        new.append({"game_date": r.game_date, "team": r.team, "player_id": pid, "name": r.name, "source": r.source, "first_seen": r.captured_at})
    if new:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for e in new:
                f.write(json.dumps(e, sort_keys=True) + "\n")
    return len(new)


def confirmed_as_of(path: str | Path, game_date: str, team: str, t: str) -> dict | None:
    """The latest confirmation for (date, team) first seen at or before ``t`` (ISO strings, UTC)."""
    p = Path(path)
    if not p.exists():
        return None
    best = None
    for ln in p.read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            if e["game_date"] == game_date and e["team"] == team and e["first_seen"] <= t and (best is None or e["first_seen"] >= best["first_seen"]):
                best = e
    return best
