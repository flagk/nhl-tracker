"""Run feature selection on the *selection window only* and save the result.

    python scripts/select_features.py --db data/nhl.db --selection-end 2024-06-30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nhlbet.analysis.importance import select_features
from nhlbet.data.store import Store
from nhlbet.features.builder import build_features

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--selection-end", default="2024-06-30", help="only games up to this date are used to choose features")
ap.add_argument("--out", default="data/models/feature_selection.json")
ap.add_argument("--report", default="reports/feature_importance.csv")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
F = build_features(Store(a.db))
F = F[F.home_score.notna() & (F.game_date <= pd.Timestamp(a.selection_end)) & (F.game_type == 2)]
print(f"selection window: {len(F)} games, {F.game_date.min().date()} .. {F.game_date.max().date()}")
res = select_features(F)
res.save(a.out, {"selection_end": a.selection_end, "n_games": len(F)})
Path(a.report).parent.mkdir(parents=True, exist_ok=True)
res.report.to_csv(a.report)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(res.report.round(5).to_string())
print(f"\nkept {len(res.kept)}: {res.kept}")
print(f"dropped {len(res.dropped)}")
