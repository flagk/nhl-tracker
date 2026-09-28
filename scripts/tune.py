"""Tune Elo + model hyper-parameters on the TUNING window only (time-series CV).

    python scripts/tune.py --tune-end 2024-06-30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd

from nhlbet.data.loaders import load_tables
from nhlbet.data.store import Store
from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.models.tune import GRIDS, save_hyperparams, tune_elo, tune_model

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--tune-end", default="2024-06-30")
ap.add_argument("--features", default="data/models/feature_selection.json")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

tables = load_tables(Store(a.db))
elo = tune_elo(tables, a.tune_end)
print("Elo grid (best 5):\n", elo["table"].head(5).round(5).to_string(index=False))
print("chosen Elo:", elo["params"])
results = {"elo": elo}
cfg = BuilderConfig(**elo["params"])
F = FeatureBuilder(cfg).build(tables)
F = F[(F.game_type == 2) & F.home_score.notna() & (F.game_date <= pd.Timestamp(a.tune_end))]
kept = json.loads(Path(a.features).read_text())["kept"]
print(f"\ntuning window: {len(F)} games through {a.tune_end}; features: {kept}")
for name in GRIDS:
    r = tune_model(name, kept, F)
    results[name] = r
    print(f"\n{name}: chosen {r['params']}  cv_logloss={r['cv_logloss']:.4f} (se {r['se']:.4f})")
    print(r["table"].sort_values("cv_logloss").head(3).round(4).to_string(index=False))
save_hyperparams(results, meta={"tune_end": a.tune_end, "n_games": len(F), "method": "expanding-window time-series CV, one-SE rule"})
print("\nsaved data/models/hyperparams.json")
