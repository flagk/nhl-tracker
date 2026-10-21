"""Phase 1 checks on the REAL database (run in CI, where the database cache lives): truncation invariance, availability guard + sentinels, outcome-shuffle null.

    python scripts/phase1_realdata.py [--db data/nhl.db]

Writes reports/phase1_realdata.json and .md; exits 1 if any check fails or the database is too small to mean anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from nhlbet.config import load_selected_features  # noqa: E402
from nhlbet.data.loaders import load_tables  # noqa: E402
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.validity import realdata  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--truncate-at", default="2025-03-01")
ap.add_argument("--first-test", default="2024-10-01")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
store = Store(a.db)
tables = load_tables(store)
n = int(tables["games"].home_score.notna().sum())
if n < 5000:
    sys.exit(f"only {n} completed games in the database: restore the cache (or run the backfill workflow) before this check can mean anything")
feats = load_selected_features()
res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "completed_games": n, "selected_features": feats}
res["truncation_invariance"] = realdata.truncation_invariance(tables, feats, pd.Timestamp(a.truncate_at), a.first_test)
res["availability_and_sentinel"] = realdata.availability_and_sentinel(tables, feats)
res["shuffle_null"] = realdata.shuffle_null(tables, feats, a.first_test)
try:
    res["settlement_sensitivity"] = realdata.settlement_sensitivity(store)
except Exception as e:                       # informative only
    res["settlement_sensitivity"] = {"error": repr(e)}
res["pass"] = all(res[k]["pass"] for k in ("truncation_invariance", "availability_and_sentinel", "shuffle_null"))
out = Path(a.out)
out.mkdir(exist_ok=True)
(out / "phase1_realdata.json").write_text(json.dumps(res, indent=1, default=float))
t, s, z = res["truncation_invariance"], res["availability_and_sentinel"], res["shuffle_null"]
(out / "phase1_realdata.md").write_text("\n".join([
    "# Phase 1 checks on the real database", "", f"Run {res['run_at']}; {n:,} completed games; features: {', '.join(feats)}.", "",
    f"**Truncation invariance** (data cut at {t['truncated_at']}): {t['games_compared']:,} games compared; max difference in any model, the stack or any calibrator "
    f"{t['max_abs_diff_models_stack_calibrators']:.2e}; weighted average {t['max_abs_diff_weighted_average']:.2e} (solver noise). **{'PASS' if t['pass'] else 'FAIL'}**", "",
    f"**Availability guard**: {s['features_checked']} features clean = {s['clean']}; planted outcome copies caught: {s['sentinel_caught']}. **{'PASS' if s['pass'] else 'FAIL'}**", "",
    "**Outcome-shuffle null** (results reassigned within each season; a clean pipeline finds no skill):", "",
    f"- real world: {z['real']['n']:,} out-of-sample games, AUC elo {z['real']['auc_elo']:.3f}, logistic {z['real']['auc_logistic']:.3f}", *[
        f"- shuffled seed {r['seed']}: AUC elo {r['auc_elo']:.3f}, logistic {r['auc_logistic']:.3f}, log-loss gain over the base rate {r['logloss_gain_vs_base_rate']:+.4f}, "
        f"ROI against a fair market with a 4.5% margin {r['roi_vs_fair_market_with_4.5pct_margin'] * 100:+.1f}%" for r in z["shuffled"]],
    f"- **{'PASS' if z['pass'] else 'FAIL'}**", "",
    f"**Settlement sensitivity** (does the shootout rule matter for totals paper bets?): {res['settlement_sensitivity']}", ""]))
print((out / "phase1_realdata.md").read_text())
sys.exit(0 if res["pass"] else 1)
