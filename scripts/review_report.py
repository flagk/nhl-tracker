"""Grade the logged pre-game reviews and write reports/REVIEW.md + REVIEW.json (needs the database for results).

    python scripts/review_report.py [--db data/nhl.db]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.review.evaluate import write_review_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
a = ap.parse_args()
rep = write_review_report(Store(a.db))
print("no reviewed picks logged yet" if rep is None else Path("reports/REVIEW.md").read_text())
