"""Walk-forward evaluation of the whole model stack on identical folds.

For each block (default 14 days) every model is retrained using ONLY games dated before the block.
Stacking weights and calibrators are learned from *inner* out-of-fold predictions generated inside that
training set with expanding time-series folds - so nothing about the block, or any later date, can
influence a prediction made for it. There are no random splits anywhere.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nhlbet.models.base import Factory
from nhlbet.models.calibration import Calibrator
from nhlbet.models.ensemble import Stacker, WeightedAverage
from nhlbet.splits import expanding_folds, walk_forward_windows

log = logging.getLogger(__name__)
STACK_INPUTS = ("elo", "logistic", "rf", "lgbm", "xgb")


def fit_stack(zoo: dict[str, Factory], X: pd.DataFrame, y: np.ndarray, dates: pd.Series, inner_folds: int = 4,
              stack_inputs: Sequence[str] = STACK_INPUTS, methods: Sequence[str] = ("platt", "isotonic")):
    """Fit every base model on (X, y) plus stacker/averager/calibrators learned from inner OOF predictions.

    Returns (fitted_models, stacker, wavg, calibrators) where calibrators maps output-name -> {method: Calibrator}.
    """
    inputs = [m for m in stack_inputs if m in zoo]
    folds = expanding_folds(dates, inner_folds, 0.4)
    oof = {m: np.full(len(X), np.nan) for m in zoo}
    for tr, te in folds:
        for m, fac in zoo.items():
            oof[m][te] = fac().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    mask = ~np.isnan(oof[inputs[0]])
    O = pd.DataFrame({m: oof[m][mask] for m in zoo})
    yo = y[mask]
    stack = Stacker().fit(O[inputs], yo)
    wavg = WeightedAverage().fit(O[inputs], yo)
    oof_out = {**{m: O[m].to_numpy() for m in zoo}, "stack": stack.predict(O[inputs]), "wavg": wavg.predict(O[inputs])}
    cals = {name: {meth: Calibrator(meth).fit(p, yo) for meth in methods} for name, p in oof_out.items()}
    fitted = {m: fac().fit(X, y) for m, fac in zoo.items()}
    return fitted, stack, wavg, cals


def predict_stack(fitted, stack, wavg, cals, X: pd.DataFrame, stack_inputs: Sequence[str] = STACK_INPUTS) -> pd.DataFrame:
    inputs = [m for m in stack_inputs if m in fitted]
    raw = {m: f.predict(X) for m, f in fitted.items()}
    P = pd.DataFrame({m: raw[m] for m in inputs})
    raw["stack"], raw["wavg"] = stack.predict(P), wavg.predict(P)
    out = {}
    for name, p in raw.items():
        out[name] = p
        for meth, cal in cals[name].items():
            out[f"{name}__{meth}"] = cal.predict(p)
    return pd.DataFrame(out, index=X.index)


def _block(F, y, dates, zoo, tr, te, inner_folds):
    fitted, stack, wavg, cals = fit_stack(zoo, F.iloc[tr], y[tr], dates.iloc[tr].reset_index(drop=True), inner_folds)
    return predict_stack(fitted, stack, wavg, cals, F.iloc[te])


def walk_forward(F: pd.DataFrame, zoo: dict[str, Factory], first_test: str, step_days: int = 14, min_train: int = 300,
                 inner_folds: int = 4, n_jobs: int = -1) -> pd.DataFrame:
    """Run the full stack over all blocks. ``F`` needs ``home_win`` and ``game_date``; returns OOS predictions per game."""
    F = F[F.home_win.notna()].sort_values(["game_date"]).copy()
    y = F.home_win.astype(int).to_numpy()
    dates = F.game_date.reset_index(drop=True)
    wins = list(walk_forward_windows(dates, first_test, step_days, min_train))
    log.info("walk-forward: %d blocks, %d test games", len(wins), sum(len(te) for _, te in wins))
    parts = Parallel(n_jobs=n_jobs)(delayed(_block)(F, y, dates, zoo, tr, te, inner_folds) for tr, te in wins)
    out = pd.concat(parts)
    out.insert(0, "y", F.loc[out.index, "home_win"].astype(int))
    out.insert(0, "game_date", F.loc[out.index, "game_date"])
    return out
