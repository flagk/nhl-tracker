"""Step 1 of the pre-game review: pick the wave, compute odds movement in code, write the agent's input. Deterministic; no LLM.

    python scripts/review_prepare.py [--force] [--no-fetch] [--now 2026-10-14T21:50:00+00:00]

Writes agent/work/review_input.json (what the agent sees) and agent/work/context.json (what the apply step needs), and emits ``run`` / ``agent_needed`` /
``run_id`` to $GITHUB_OUTPUT. Fresh prices cost 3 odds credits and are skipped when ODDS_API_KEY is missing or few credits remain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.gate import review_target  # noqa: E402
from nhlbet.review.apply import prepare_input  # noqa: E402
from nhlbet.review.odds_check import fetch_current, odds_facts  # noqa: E402


def out(**kv) -> None:
    print(" ".join(f"{k}={v}" for k, v in kv.items()))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for k, v in kv.items():
                f.write(f"{k}={v}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--work", default="agent/work")
    ap.add_argument("--force", action="store_true", help="manual run: the first wave that has not started, whatever the clock says")
    ap.add_argument("--no-fetch", action="store_true", help="do not buy fresh prices (odds movement stays 'unconfirmed')")
    ap.add_argument("--now", help="ISO time (tests)")
    a = ap.parse_args()
    now = datetime.fromisoformat(a.now) if a.now else datetime.now(timezone.utc)
    root, work = Path(a.root), Path(a.root) / a.work
    t = review_target(now, root, force=a.force)
    if not t.get("key"):
        out(run="false", agent_needed="false", reason=t["reason"].replace(" ", "_")[:80])
        return 0
    cand = json.loads((root / "data" / "review" / "picks_candidate.json").read_text())
    wave_ids = set(t["game_ids"])
    wave_picks = [p for p in cand["picks"] if p["game_id"] in wave_ids]
    if a.no_fetch:
        events, captured, remaining, note = None, None, None, "fetching was switched off for this run"
    else:
        events, captured, remaining, note = fetch_current(root, now)
    facts = odds_facts(wave_picks, cand["games"], events, captured, cand["min_edge"], note)
    inp = prepare_input(cand, wave_ids, facts, now)
    work.mkdir(parents=True, exist_ok=True)
    (work / "picks_review.json").unlink(missing_ok=True)                   # a stale answer from an earlier run must never be mistaken for this one
    (work / "review_input.json").write_text(json.dumps(inp, indent=1, allow_nan=False))
    (work / "context.json").write_text(json.dumps({"run_id": t["key"], "wave_start": t["start"].isoformat(), "wave_game_ids": sorted(wave_ids), "candidate_id": cand["candidate_id"],
                                                    "prepared_at": now.isoformat(timespec="seconds"), "facts": facts, "odds": {"captured_at": captured, "credits_remaining": remaining, "note": note},
                                                    "n_wave_picks": len(wave_picks), "n_for_agent": len(inp["picks"])}, indent=1, allow_nan=False))
    out(run="true", agent_needed=str(bool(inp["picks"])).lower(), run_id=t["key"], n_picks=len(inp["picks"]), reason=t["reason"].replace(" ", "_")[:80])
    return 0


if __name__ == "__main__":
    sys.exit(main())
