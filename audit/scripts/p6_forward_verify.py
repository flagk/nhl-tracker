"""Phase 6(b): verify the forward ledger. Every prediction (one row per game per run in data/logs/recommendations) must have been COMMITTED to git before that game's puck drop.

The ledger is data/logs/recommendations (model probability, no-vig market probability, best price, decision, stake, odds capture time), data/logs/odds_consensus (the market
snapshots), data/logs/goalie_events, hash-chained in data/logs/chain.jsonl and mirrored (never forced) to the log-backup branch. A row counts as verified only when the commit that
first added its file is dated before the game's start. Trust assumption, stated plainly: the commit date is written by the machine that commits (the Actions runner), so this proves
"committed before the game" against anyone without push rights, not against the repository owner; an external timestamp (for example OpenTimestamps) would close that gap and is not built.

    python audit/scripts/p6_forward_verify.py [--since 2026-10-27] [--db data/nhl.db]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def first_commit_time(repo: Path, rel: str) -> pd.Timestamp | None:
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%cI", "--", rel], cwd=repo, capture_output=True, text=True).stdout.split()
    return pd.Timestamp(out[-1]).tz_convert("UTC") if out else None


def verify(repo: Path, starts: dict[int, pd.Timestamp], since: str | None = None) -> dict:
    files = sorted((repo / "data" / "logs" / "recommendations").glob("*.csv"))
    rows, per_file = [], []
    for f in files:
        rel = str(f.relative_to(repo))
        ct = first_commit_time(repo, rel)
        d = pd.read_csv(f)
        if since:
            d = d[d.game_date >= since]
        for r in d.itertuples():
            st = starts.get(int(r.game_id))
            run = pd.Timestamp(r.run_at).tz_convert("UTC") if pd.notna(r.run_at) else None
            status = ("no_commit_time" if ct is None else "no_start_time" if st is None else "verified_before_puck_drop" if ct < st else "LATE_commit_after_puck_drop")
            rows.append({"file": rel, "game_id": int(r.game_id), "run_at": None if run is None else run.isoformat(), "status": status, "commit_minus_run_min": None if ct is None or run is None else (ct - run).total_seconds() / 60,
                         "start_minus_commit_min": None if ct is None or st is None else (st - ct).total_seconds() / 60})
        per_file.append(rel)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"predictions": 0, "note": "no recommendation rows in the period"}
    last = df.dropna(subset=["run_at"]).sort_values("run_at").groupby("game_id").tail(1)             # the decision that counts for a game is its last run before the game
    c = df.status.value_counts().to_dict()
    ok = df[df.status == "verified_before_puck_drop"]
    return {"predictions": int(len(df)), "files": len(per_file), "status_counts": {k: int(v) for k, v in c.items()},
            "share_verified": float(len(ok) / len(df)), "median_minutes_between_commit_and_run": None if df.commit_minus_run_min.dropna().empty else float(df.commit_minus_run_min.median()),
            "median_minutes_before_puck_drop_when_verified": None if ok.empty else float(ok.start_minus_commit_min.median()),
            "last_run_verified_game_ids": sorted(int(g) for g in last[last.status == "verified_before_puck_drop"].game_id),
            "late_examples": df[df.status == "LATE_commit_after_puck_drop"].head(10).to_dict("records")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--repo", default=str(ROOT), help="checkout whose git history holds the logs (the default branch, where the daily job commits)")
    ap.add_argument("--db", default=str(ROOT / "data" / "nhl.db"))
    ap.add_argument("--out", default=str(ROOT / "audit/results/forward_verification.json"))
    a = ap.parse_args()
    from nhlbet.data.store import Store
    g = Store(a.db).df("SELECT game_id, start_utc FROM games WHERE start_utc IS NOT NULL")
    starts = {int(r.game_id): pd.Timestamp(r.start_utc).tz_convert("UTC") for r in g.itertuples()}
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "since": a.since, "games_with_start_time": len(starts), **verify(Path(a.repo).resolve(), starts, a.since)}
    from nhlbet.ops import chain
    res["hash_chain_problems"] = chain.verify(Path(a.repo).resolve())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("late_examples", "last_run_verified_game_ids")}, indent=1, default=float))
    return 1 if res.get("status_counts", {}).get("LATE_commit_after_puck_drop") else 0


if __name__ == "__main__":
    sys.exit(main())
