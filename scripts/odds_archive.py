"""Archive one timestamped raw odds snapshot (all books, h2h) into the point-in-time store. Off unless the policy enables it.

Raw per-book quotes must never reach the public repository, so objects are written under ``audit/snapshots/objects`` (git-ignored) and the workflow pushes them with DVC to a PRIVATE
remote; without that remote the workflow refuses to run. Cost: one Odds API credit per call (h2h, one region). The plan has 500 credits a month, so a capture every few minutes is not
affordable: see ``audit.odds_archive`` in config/policy.yaml and audit/PHASE2_PIT.md for the arithmetic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nhlbet.pit.snapshots import SnapshotStore  # noqa: E402
from nhlbet.policy import policy_get  # noqa: E402


def main(client=None, store: SnapshotStore | None = None, now=None) -> int:
    if not policy_get("audit", "odds_archive", "enabled", default=False):
        print("odds archive disabled in config/policy.yaml (audit.odds_archive.enabled): nothing fetched, no credit spent")
        return 0
    store = store or SnapshotStore(ROOT / "audit" / "snapshots")
    min_gap = float(policy_get("audit", "odds_archive", "min_interval_min", default=10))
    floor = int(policy_get("audit", "odds_archive", "min_credits_remaining", default=150))
    last = [v for v in store.manifest() if v.kind == "odds"]
    now = now or datetime.now(timezone.utc)
    if last and (now - datetime.fromisoformat(last[-1].knowledge_time)).total_seconds() < min_gap * 60:
        print("last archived snapshot is newer than the minimum interval: skipped")
        return 0
    if client is None:
        from nhlbet.odds.client import OddsClient
        client = OddsClient()
    rem = client.remaining_credits()
    if rem is not None and rem < floor:
        print(f"only {rem} credits left (archive floor {floor}): skipped to protect the live pipeline")
        return 0
    f = client.fetch_odds(("h2h",), force=True)
    if f.source != "live":
        print(f"not a live capture (source={f.source}): not archived")
        return 1
    v = store.put("odds", f.captured_at[:10], {"events": f.events, "remaining": f.remaining, "used": f.used}, f.captured_at, "the-odds-api h2h")
    print(f"archived {v.sha256[:12]} at {v.knowledge_time}, {f.remaining} credits left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
