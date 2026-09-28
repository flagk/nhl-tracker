"""Walk-forward backtest of the full model stack (all models on identical folds) + report.

    python scripts/backtest.py --eval-start 2024-10-01

Tuning / feature selection used data through 2024-06-30 only; everything from --eval-start on is
genuinely out-of-sample for those choices. Retraining happens every --step-days using only earlier games.
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

from nhlbet.analysis.importance import candidate_columns
from nhlbet.config import load_builder_config, load_selected_features
from nhlbet.data.store import Store
from nhlbet.features.builder import build_features
from nhlbet.models.base import make_zoo
from nhlbet.models.calibration import online_calibrate, plot_reliability
from nhlbet.models.evaluate import logloss_ci, metrics_table, paired_logloss_diff
from nhlbet.models.walkforward import walk_forward

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--eval-start", default="2024-10-01")
ap.add_argument("--tune-end", default="2024-06-30")
ap.add_argument("--step-days", type=int, default=14)
ap.add_argument("--out", default="reports")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
out = Path(a.out); out.mkdir(exist_ok=True)

F = build_features(Store(a.db), load_builder_config())
F = F[(F.game_type == 2) & F.home_score.notna()]
kept = load_selected_features()
zoo = make_zoo(kept, candidate_columns(F[F.game_date <= pd.Timestamp(a.tune_end)]))
P = walk_forward(F, zoo, a.eval_start, a.step_days)
P.to_csv(out / "walkforward_predictions.csv")

raw = ["home_rate", "elo", "logistic", "rf", "lgbm", "xgb", "lgbm_all", "stack", "wavg"]
cols = raw + [f"{m}__platt" for m in raw] + [f"{m}__isotonic" for m in ("stack", "wavg", "logistic", "lgbm", "elo")]
M = metrics_table(P, cols)
M["ll_ci_low"], M["ll_ci_high"] = zip(*[logloss_ci(P, c) for c in M.index])
M = M.sort_values("log_loss")
pd.set_option("display.width", 220)
print(M.round(4).to_string())

pairs = []
for m in ("logistic", "lgbm", "xgb", "rf", "stack", "wavg", "stack__platt", "wavg__platt", "elo"):
    for base in ("elo", "home_rate"):
        if m != base:
            pairs.append(paired_logloss_diff(P, m, base))
D = pd.DataFrame(pairs)
print("\nPaired log-loss difference (negative = first model better; week-cluster bootstrap):")
print(D.round(4).to_string(index=False))

# online calibration: fit on the model's own earlier out-of-sample predictions (post-hoc design choice, see docs/RESULTS.md)
for m in ("elo", "logistic", "rf", "lgbm", "xgb", "stack", "wavg"):
    P[f"{m}__online"] = online_calibrate(P, m)
P.to_csv(out / "walkforward_predictions.csv")
S = P[P.stack__online.notna()]
sub_cols = ["home_rate", "elo", "elo__online", "logistic", "logistic__online", "lgbm__online", "rf__online", "xgb__online",
            "stack", "stack__platt", "stack__online", "wavg", "wavg__online"]
M2 = metrics_table(S, sub_cols).sort_values("log_loss")
print(f"\nSame-game comparison incl. online calibration (n={len(S)}, {S.game_date.min().date()}..{S.game_date.max().date()}):")
print(M2.round(4).to_string())
D2 = pd.DataFrame([paired_logloss_diff(S, "stack__online", b) for b in ("elo", "elo__online", "home_rate", "stack__platt")])
print(D2.round(4).to_string(index=False))

best_cal = ["elo", "stack__platt", "stack__online"]
plot_reliability({m: (S[m], S.y) for m in best_cal}, str(out / "reliability.png"))

def md(df, fmt="{:.4f}"):
    return df.to_markdown(floatfmt=".4f") if hasattr(df, "to_markdown") else df.to_string()

try:
    import tabulate  # noqa: F401
except ImportError:
    md = lambda df, fmt=None: "```\n" + df.round(4).to_string() + "\n```"  # noqa: E731

(out / "model_comparison.md").write_text(f"""# Walk-forward model comparison

Out-of-sample games: **{len(P)}** ({P.game_date.min().date()} to {P.game_date.max().date()}), regular season only.
Retrained every {a.step_days} days on strictly earlier games; stackers/calibrators learned from inner
time-series out-of-fold predictions. Features/hyper-parameters chosen on data through {a.tune_end} only.
Baseline for a coin flip: log loss 0.6931, Brier 0.2500.

{md(M)}

## Paired log-loss differences (first minus second; negative = first is better)

{md(D.set_index(['a', 'b']))}

## With online calibration (same games for every column, n={len(S)})

{md(M2)}

Paired: `stack__online` vs others

{md(D2.set_index(['a', 'b']))}

`cal_slope` < 1 means over-confident, > 1 under-confident. ECE = expected calibration error (10 quantile bins).
![reliability](reliability.png)
""")
json.dump({"n": len(P), "start": str(P.game_date.min().date()), "end": str(P.game_date.max().date()),
           "metrics": M.round(5).reset_index().to_dict("records"), "paired": D.round(5).to_dict("records"),
           "online_subset": {"n": len(S), "metrics": M2.round(5).reset_index().to_dict("records"), "paired": D2.round(5).to_dict("records")}},
          open(out / "model_comparison.json", "w"), indent=1)
