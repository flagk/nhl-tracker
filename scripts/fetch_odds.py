"""Fetch current NHL odds from The Odds API, store a timestamped snapshot, print quota.

    export ODDS_API_KEY=...           # never commit this; use a GitHub Actions secret in CI
    python scripts/fetch_odds.py [--markets h2h spreads totals] [--regions us]

Puck line / totals are stored for the record but the model only bets moneylines (it predicts win probability).
Exit code 0 even when serving stale data (flagged in the log); 2 on hard failure.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nhlbet.data.store import Store
from nhlbet.odds.client import OddsAPIError, OddsClient
from nhlbet.odds.snapshots import record_fetch

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--markets", nargs="+", default=["h2h"])
ap.add_argument("--regions", default="us")
ap.add_argument("--force", action="store_true", help="ignore the short-lived cache")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
try:
    f = OddsClient().fetch_odds(a.markets, a.regions, force=a.force)
except OddsAPIError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(2)
store = Store(a.db)
n = record_fetch(store, f, ",".join(a.markets))
print(f"{len(f.events)} events, {n} price rows stored | source={f.source}{' (STALE)' if f.stale else ''} | "
      f"credits remaining={f.remaining} used={f.used}")
