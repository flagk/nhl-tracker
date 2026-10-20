"""Baseline snapshot from committed files only (no database, no network): headline metrics, per-market live record, and a hash manifest of the pick logs.

    python scripts/baseline.py [--out reports/baseline] [--label before]

Run at the commit BEFORE a change and again after it; reports/phase1_leakage.md compares the two. Nothing here is tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="reports/baseline")
ap.add_argument("--label", default="before")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


res: dict = {"label": a.label}
# 1. the legacy headline (the README claim), recomputed from its committed per-bet file
leg = pd.read_csv("audit/legacy_predictions.csv")
w = int((leg.result == "WIN").sum())
res["legacy_headline"] = {"bets": int(len(leg)), "wins": w, "win_rate": w / len(leg), "roi_at_even_money": 2 * w / len(leg) - 1, "home_pick_share": float((leg.prediction == leg.home).mean()),
                          "note": "even-money payouts, no odds anywhere in the data: an arithmetic artefact (see AUDIT.md). Re-running the legacy code is logged in docs/PROGRESS.md."}
# 2. current system: walk-forward predictive metrics (no market prices exist for the backtest)
wf = pd.read_csv("reports/walkforward_predictions.csv", index_col=0, parse_dates=["game_date"])
ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
y = wf.y.to_numpy(float)
res["walk_forward"] = {"n": int(len(wf)), "from": str(wf.game_date.min().date()), "to": str(wf.game_date.max().date()), "home_win_rate": float(y.mean()),
                       "log_loss": {c: ll(wf[c].to_numpy(float), y) for c in ("home_rate", "elo", "logistic", "stack", "stack__platt") if c in wf}}
# 3. live record per market, from the committed logs and exports
try:
    from nhlbet.report.validation import run_validation
    v = run_validation(".")
    res["live"] = {k: v.get(k) for k in ("data_through", "games_with_results", "game_level", "picks", "by_market", "controls_by_market", "log_integrity", "required_bets")}
except Exception as e:                                           # a baseline must still be written if one section fails
    res["live_error"] = repr(e)
# 4. hash manifest of every pick / bet log
files = sorted(p for p in Path("data/logs").rglob("*") if p.is_file())
manifest = {str(p): sha(p) for p in files}
res["pick_logs"] = {"files": len(files), "tree_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                    "paper_trading_csv_sha256": sha(Path("data/export/paper_trading.csv")) if Path("data/export/paper_trading.csv").exists() else None}
(out / f"{a.label}.json").write_text(json.dumps(res, indent=1, default=float))
(out / f"{a.label}_log_manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({k: res[k] for k in ("legacy_headline", "walk_forward", "pick_logs")}, indent=1, default=float))
