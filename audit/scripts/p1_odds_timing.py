"""Phase 1: when are the odds used for decisions and for 'closing line value' actually captured, relative to puck drop?

Reads only the committed public-safe logs (data/logs). Writes audit/data/odds_timing.json. No network.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.report.betlog import restore_logs  # noqa: E402


def q(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), **{f"p{p}": round(float(np.percentile(x, p)), 1) for p in (5, 25, 50, 75, 95)}, "max": round(float(x.max()), 1)} if len(x) else {"n": 0}


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        st = Store(str(Path(d) / "t.db"))
        restore_logs(st, str(ROOT / "data" / "logs"))
        g = Store(str(ROOT / "data" / "nhl.db")).df("SELECT game_id, start_utc, game_date FROM games WHERE start_utc IS NOT NULL")   # the real database exists only in the Actions cache
        rec = st.df("SELECT game_id, run_at, run_type, action FROM recommendations")
        cons = st.df("SELECT game_id, captured_at, n_books FROM odds_consensus")
    # games table may be empty in a bare restore: use start times from the recommendation logs' own game ids via the consensus capture range
    out = {"source": "data/logs (committed, public-safe)", "recommendation_rows": int(len(rec)), "consensus_rows": int(len(cons)), "games_rows": int(len(g))}
    if g.empty:
        out["note"] = "no game start times in data/nhl.db here (the full database exists only in the Actions cache): run via the audit-realdata workflow"
        Path(ROOT / "audit/data/odds_timing.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1)); return
    g["start"] = pd.to_datetime(g.start_utc, utc=True, errors="coerce")
    r = rec.merge(g[["game_id", "start"]], on="game_id")
    r["run"] = pd.to_datetime(r.run_at, utc=True)
    r["lead_min"] = (r.start - r.run).dt.total_seconds() / 60
    out["decision_lead_minutes_before_start_all_runs"] = q(r.lead_min)
    out["decision_lead_by_run_type"] = {k: q(v.lead_min) for k, v in r.groupby("run_type")}
    c = cons.merge(g[["game_id", "start"]], on="game_id")
    c["cap"] = pd.to_datetime(c.captured_at, utc=True)
    c["lead_min"] = (c.start - c.cap).dt.total_seconds() / 60
    c = c[c.lead_min >= 0]
    last = c.sort_values("cap").groupby("game_id").tail(1)
    first = c.sort_values("cap").groupby("game_id").head(1)
    out["captures_per_game"] = q(c.groupby("game_id").size())
    out["last_capture_before_start_minutes"] = q(last.lead_min)
    out["first_capture_before_start_minutes"] = q(first.lead_min)
    out["share_games_whose_last_capture_is_over_60_min_before_start"] = round(float((last.lead_min > 60).mean()), 3)
    out["share_games_whose_last_capture_is_over_15_min_before_start"] = round(float((last.lead_min > 15).mean()), 3)
    Path(ROOT / "audit/data/odds_timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
