"""Run the daily job.

    python scripts/daily.py --run-type morning      # refresh data, retrain if new games, fetch odds, report
    python scripts/daily.py --run-type late         # after starting goalies are confirmed: refresh odds + report

Environment: ODDS_API_KEY (optional; without it odds are disabled), BANKROLL (default 1000).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from nhlbet.logging_setup import setup_logging
from nhlbet.pipeline import run_daily

ap = argparse.ArgumentParser()
ap.add_argument("--run-type", choices=["morning", "late"], default="morning")
ap.add_argument("--date", help="YYYY-MM-DD (default: today, US/Eastern)")
ap.add_argument("--bankroll", type=float, default=float(os.environ.get("BANKROLL", 1000)))
ap.add_argument("--fixed-bankroll", action="store_true", help="size stakes from --bankroll instead of bankroll + realised profit")
ap.add_argument("--no-refresh", action="store_true")
ap.add_argument("--no-odds", action="store_true")
ap.add_argument("--retrain", dest="retrain", action="store_true", default=None)
ap.add_argument("--no-retrain", dest="retrain", action="store_false")
ap.add_argument("--db", default="data/nhl.db")
a = ap.parse_args()
setup_logging(log_file="logs/nhlbet.log")
try:
    r = run_daily(a.date, a.run_type, a.db, a.bankroll, a.fixed_bankroll, not a.no_refresh, not a.no_odds, a.retrain)
except Exception as e:  # noqa: BLE001
    print(f"FATAL: {e}", file=sys.stderr)
    sys.exit(1)
print(r)
