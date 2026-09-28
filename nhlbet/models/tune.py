"""Hyper-parameter tuning with time-series cross-validation ONLY (expanding window, cut at date boundaries).

Choice rule: paired one-standard-error rule. Per-fold log losses are compared *against the best config on
the same folds* (fold-to-fold variation is shared by all configs and would otherwise swamp the comparison);
among configs whose mean paired difference is within one standard error of zero, take the *simplest*
(grids are ordered simple -> complex). With ~1-3k games complex models mostly fit noise, so we prefer regularisation.
Tune on data BEFORE the evaluation window; never on the evaluation window itself.
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from nhlbet.models.base import HYPERPARAMS_PATH, LGBM, XGB, Logistic, RandomForest
from nhlbet.models.calibration import log_loss_
from nhlbet.splits import expanding_folds

log = logging.getLogger(__name__)

GRIDS: dict[str, tuple[Callable, list[dict]]] = {
    "logistic": (Logistic, [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]),
    "rf": (RandomForest, [{"n_estimators": 300, "min_samples_leaf": l, "max_features": mf}
                          for l, mf in itertools.product((80, 40, 20), (0.3, 0.6))]),
    "lgbm": (LGBM, [{"n_estimators": n, "learning_rate": 0.03, "num_leaves": nl, "min_child_samples": mc, "reg_lambda": rl}
                    for n, nl, mc, rl in itertools.product((60, 120, 240), (3, 5), (120, 60), (30.0, 5.0))]),
    "xgb": (XGB, [{"n_estimators": n, "learning_rate": 0.03, "max_depth": d, "min_child_weight": mc, "reg_lambda": rl}
                  for n, d, mc, rl in itertools.product((60, 120, 240), (2, 3), (30, 10), (30.0, 5.0))]),
}


def cv_losses(factory: Callable, params: dict, X: pd.DataFrame, y: np.ndarray, dates: pd.Series, n_folds: int = 5) -> np.ndarray:
    folds = expanding_folds(dates, n_folds, 0.4)
    return np.array([log_loss_(factory(**params).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), y[te]) for tr, te in folds])


def tune_model(name: str, features: list[str], F: pd.DataFrame, n_folds: int = 5) -> dict:
    """Return {'params', 'cv_logloss', 'se', 'table'} using only rows of ``F`` (pass the tuning window)."""
    cls, grid = GRIDS[name]
    F = F.sort_values("game_date").reset_index(drop=True)
    y, dates = F.home_win.astype(int).to_numpy(), F.game_date
    losses = np.array([cv_losses(lambda **p: cls(features, **p), params, F, y, dates, n_folds) for params in grid])
    mean = losses.mean(1)
    best = int(mean.argmin())
    diff = losses - losses[best]                                   # paired differences vs best, per fold
    se = diff.std(1, ddof=1) / np.sqrt(diff.shape[1])
    ok = np.flatnonzero(diff.mean(1) <= se)                        # indistinguishable from the best
    chosen = int(ok[0])                                            # simplest first
    t = pd.DataFrame(grid).assign(cv_logloss=mean, paired_se=se)
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in grid[chosen].items()}
    return {"params": params, "cv_logloss": float(mean[chosen]), "se": float(se[chosen]), "table": t}


def tune_elo(games_tables: dict, tune_end: str, burn_in: int = 500) -> dict:
    """Pick Elo (k, home advantage, season regression) by pre-game log loss on the tuning window.

    Elo is an online model - every prediction is made before the game's result is folded in - so its
    sequential log loss is already a valid out-of-sample score. Burn-in games are excluded.
    """
    from nhlbet.features.builder import BuilderConfig, FeatureBuilder

    rows = []
    seasons = {(d.year - (d.month < 8)) for d in games_tables["games"].game_date if d <= pd.Timestamp(tune_end)}
    regs = (0.2, 0.3, 0.5) if len(seasons) > 1 else (0.3,)       # regression only acts at a season boundary
    for k, ha, rg in itertools.product((3.0, 4.5, 6.0, 8.0, 10.0, 12.0), (15.0, 25.0, 35.0, 45.0), regs):
        F = FeatureBuilder(BuilderConfig(elo_k=k, elo_home_adv=ha, elo_regress=rg)).build(games_tables)
        F = F[(F.game_type == 2) & F.home_score.notna() & (F.game_date <= pd.Timestamp(tune_end))].sort_values("game_date")
        F = F.iloc[burn_in:]
        rows.append({"k": k, "home_adv": ha, "regress": rg, "logloss": log_loss_(F.elo_prob, F.home_win.astype(int)), "n": len(F)})
    t = pd.DataFrame(rows).sort_values("logloss")
    near = t[t.logloss <= t.logloss.iloc[0] + 0.0005].copy()          # statistically indistinguishable from the best
    near["dist"] = (near.k / 6.0 - 1) ** 2 + (near.home_adv / 35.0 - 1) ** 2   # prefer conventional values
    best = near.sort_values("dist").iloc[0]
    return {"params": {"elo_k": float(best.k), "elo_home_adv": float(best.home_adv), "elo_regress": float(best.regress)},
            "logloss": float(best.logloss), "table": t}


def save_hyperparams(results: dict[str, dict], path: str | Path = HYPERPARAMS_PATH, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {k: v["params"] for k, v in results.items()}
    out["_meta"] = meta or {}
    Path(path).write_text(json.dumps(out, indent=2))
