"""Archive raw NHL payloads as immutable, timestamped snapshots (point-in-time store).

    python scripts/pit_ingest.py probe                      # verify endpoints, write audit/data/endpoint_probe.json
    python scripts/pit_ingest.py games --days 3             # fetch every final game of the last N days (new versions, never overwrites)
    python scripts/pit_ingest.py verify                     # hash-check every stored object

Objects go under ``audit/snapshots/objects`` (not committed; pushed with DVC to a PRIVATE remote). The manifest ``audit/snapshots/manifest.jsonl`` (hashes and fetch times) is committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nhlbet.pit.fetch import SCHEDULE, fetch_game, http_get, probe  # noqa: E402
from nhlbet.pit.snapshots import SnapshotStore  # noqa: E402

STORE = ROOT / "audit" / "snapshots"


def final_games(day: str, get=http_get) -> list[int]:
    status, sched = get(SCHEDULE.format(date=day))
    if status != 200 or not isinstance(sched, dict):
        return []
    return [g["id"] for d in sched.get("gameWeek", []) if d.get("date") == day for g in d.get("games", []) if g.get("gameState") in ("OFF", "FINAL") and g.get("gameType") in (2, 3)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "games", "verify"])
    ap.add_argument("--days", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        res = probe()
        out = ROOT / "audit" / "data" / "endpoint_probe.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        bad = [k for k, v in res["endpoints"].items() if v.get("status") != 200 or v.get("has_required_keys") is False]
        return 1 if bad else 0
    store = SnapshotStore(STORE)
    if a.cmd == "verify":
        p = store.verify()
        print("\n".join(p) or f"{len(store.manifest())} manifest lines verified")
        return 1 if p else 0
    today = datetime.now(timezone.utc).date()
    n = bad = 0
    for i in range(a.days):
        day = (today - timedelta(days=i)).isoformat()
        for gid in final_games(day):
            r = fetch_game(store, gid)
            n += 1
            bad += any(not v.startswith("stored") for v in r.values())
            print(day, gid, r)
    print(f"{n} games, {bad} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
