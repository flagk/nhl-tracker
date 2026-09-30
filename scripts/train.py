"""Daily retrain: rebuild features, retrain only if there are new completed games (or --force), drift-check, register.

    python scripts/train.py [--force]

Exit code 3 if drift status is ALERT (so CI can flag it) - the model is still saved and registered.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from nhlbet.train import retrain

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--force", action="store_true")
ap.add_argument("--live-log", default=None)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
r = retrain(a.db, a.force, live_log=a.live_log)
print({k: v for k, v in r.items() if k != "drift"}, "| drift:", r["drift"].get("status"), r["drift"].get("performance", {}).get("reasons", ""))
sys.exit(3 if r["drift"].get("status") == "ALERT" else 0)
