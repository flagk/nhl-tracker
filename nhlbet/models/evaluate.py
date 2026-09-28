"""Scoring tables and paired significance tests for walk-forward predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nhlbet.models.calibration import brier, calibration_slope_intercept, ece, log_loss_

EPS = 1e-6


def metrics_table(P: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = P.y.to_numpy()
    rows = []
    for c in cols:
        p = P[c].to_numpy()
        slope, icpt = calibration_slope_intercept(p, y)
        rows.append({"model": c, "n": len(p), "log_loss": log_loss_(p, y), "brier": brier(p, y),
                     "accuracy": float(np.mean((p > 0.5) == y)), "auc": roc_auc_score(y, p),
                     "ece": ece(p, y), "cal_slope": slope})
    return pd.DataFrame(rows).set_index("model")


def _week_groups(dates: pd.Series) -> list[np.ndarray]:
    wk = pd.to_datetime(dates).dt.to_period("W").astype(str).to_numpy()
    return [np.flatnonzero(wk == w) for w in np.unique(wk)]


def paired_logloss_diff(P: pd.DataFrame, a: str, b: str, B: int = 4000, seed: int = 0) -> dict:
    """Per-game log-loss(a) - log-loss(b). Negative = a is better. Cluster (week) bootstrap CI and one-sided p."""
    y = P.y.to_numpy()
    la = -(y * np.log(np.clip(P[a], EPS, 1)) + (1 - y) * np.log(np.clip(1 - P[a], EPS, 1)))
    lb = -(y * np.log(np.clip(P[b], EPS, 1)) + (1 - y) * np.log(np.clip(1 - P[b], EPS, 1)))
    d = (la - lb).to_numpy()
    groups = _week_groups(P.game_date)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"a": a, "b": b, "diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_not_better": float(np.mean(means >= 0))}


def logloss_ci(P: pd.DataFrame, col: str, B: int = 3000, seed: int = 0) -> tuple[float, float]:
    y = P.y.to_numpy(); p = np.clip(P[col].to_numpy(), EPS, 1 - EPS)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    groups = _week_groups(P.game_date); rng = np.random.default_rng(seed)
    m = [l[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
