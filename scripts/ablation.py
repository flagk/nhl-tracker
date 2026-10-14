"""Ablations, parameter sensitivity and per-season stability under the production walk-forward protocol -> reports/ABLATION.md

    python scripts/ablation.py --eval-start 2024-10-01
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd

from nhlbet.analysis.ablation import group_ablation, period_stability, regularisation_sensitivity
from nhlbet.config import load_builder_config, load_selected_features
from nhlbet.data.store import Store
from nhlbet.features.builder import build_features

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--eval-start", default="2024-10-01")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
F = build_features(Store(a.db), load_builder_config())
F = F[(F.game_type == 2) & F.home_score.notna()]
feats = [f for f in load_selected_features() if f in F.columns]
abl = group_ablation(F, feats, a.eval_start)
sens = regularisation_sensitivity(F, feats, a.eval_start)
per = period_stability(F, feats, a.eval_start)
md = ["# Ablations and parameter sensitivity", "",
      "> Research/education only. Same walk-forward protocol as the headline backtest (retrained every 28 days on strictly earlier games; the features and C were chosen on games before 2024-07-01). "
      "`cost_vs_full` is the out-of-sample log-loss increase from removing that group (positive = the group helps); the interval is a week-clustered paired bootstrap.", "",
      f"{len(F[F.game_date >= pd.Timestamp(a.eval_start)]):,} evaluation games from {a.eval_start}.", "",
      "## Remove one feature group at a time", "", abl.round(4).to_markdown(index=False), "",
      "## Regularisation strength (logistic C; tuned value 0.003)", "", sens.round(4).to_markdown(index=False), "",
      "## Stability by season (logistic minus Elo, negative = better than Elo)", "", per.round(4).to_markdown(index=False), ""]
Path(a.out).mkdir(exist_ok=True)
Path(a.out, "ABLATION.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
