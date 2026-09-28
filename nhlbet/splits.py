"""Time-ordered splitting utilities. There are deliberately no random splits anywhere in this project.

A calendar date is never split across train and test, and the test block always starts strictly
after the last training date, so no same-day information can cross the boundary.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def walk_forward_windows(dates: pd.Series, first_test: str | pd.Timestamp, step_days: int = 14,
                         min_train: int = 300) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional index arrays.

    Test block k covers ``[first_test + k*step, first_test + (k+1)*step)``; training is *every* game
    dated strictly before the block starts (expanding window) - i.e. retraining happens only on data
    that existed before the block's first prediction date.
    """
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    start, end = pd.Timestamp(first_test), d.max()
    while start <= end:
        stop = start + pd.Timedelta(days=step_days)
        train = np.flatnonzero((d < start).to_numpy())
        test = np.flatnonzero(((d >= start) & (d < stop)).to_numpy())
        if len(test) and len(train) >= min_train:
            yield train, test
        start = stop


def expanding_folds(dates: pd.Series, n_folds: int = 5, min_train_frac: float = 0.4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds cut at date boundaries (for tuning / selection inside a training window)."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    uniq = np.sort(d.unique())
    first = int(len(uniq) * min_train_frac)
    cuts = np.linspace(first, len(uniq), n_folds + 1).astype(int)
    folds = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        t0, t1 = uniq[a], (uniq[b] if b < len(uniq) else None)
        train = np.flatnonzero((d < t0).to_numpy())
        test = np.flatnonzero(((d >= t0) & (d < t1)).to_numpy()) if t1 is not None else np.flatnonzero((d >= t0).to_numpy())
        if len(train) and len(test):
            folds.append((train, test))
    return folds
