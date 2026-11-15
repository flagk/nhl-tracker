"""Feature-availability guard: runs before every model fit (walk-forward and production).

Three independent checks, each of which raises ``LeakageError``:
1. a model input may not be an outcome column (scores, results, goal targets) or a closing-line column;
2. every input must exist in the frame;
3. no input may be (nearly) a copy of the outcome: |correlation| above ``MAX_ABS_CORR`` with the label, or an exact copy / complement. A real pre-game feature in hockey
   has |correlation| with the winner of a few percent; anything near 1 is a leak, whatever it is called.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.features.market import assert_no_closing_columns

OUTCOME_COLUMNS = frozenset({"y", "home_win", "home_score", "away_score", "hr", "ar", "ot", "so", "tot", "mar", "last_period", "result", "won", "profit"})
OUTCOME_PREFIXES = ("tgt_", "target", "label", "outcome")
MAX_ABS_CORR = 0.98
MIN_ROWS = 30


class LeakageError(ValueError):
    """A model input is, or contains, information from the future."""


def assert_features_clean(X: pd.DataFrame, y, columns, max_abs_corr: float = MAX_ABS_CORR) -> None:
    cols = sorted(set(columns))
    banned = [c for c in cols if c in OUTCOME_COLUMNS or c.lower().startswith(OUTCOME_PREFIXES)]
    if banned:
        raise LeakageError(f"outcome columns used as model inputs: {banned}")
    try:
        assert_no_closing_columns(cols)
    except ValueError as e:
        raise LeakageError(str(e)) from e
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise LeakageError(f"model inputs missing from the frame: {missing}")
    yv = np.asarray(y, float)
    bad = []
    for c in cols:
        x = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(yv)
        if m.sum() < MIN_ROWS or np.std(x[m]) == 0 or np.std(yv[m]) == 0:
            continue
        if np.array_equal(x[m], yv[m]) or np.array_equal(x[m], 1.0 - yv[m]) or abs(np.corrcoef(x[m], yv[m])[0, 1]) > max_abs_corr:
            bad.append(c)
    if bad:
        raise LeakageError(f"model inputs that reproduce the outcome (|corr| > {max_abs_corr} or an exact copy): {bad}")
