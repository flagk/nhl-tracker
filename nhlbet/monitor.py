"""Drift checks run by the daily retrain: is the model getting worse, or is the data changing?

Performance drift compares the *recent* log loss of the deployed model to its own historical distribution
(rolling windows of the walk-forward backtest) and to the Elo / home-rate baselines. Feature drift uses the
Population Stability Index (PSI). Status: OK < WARN < ALERT; ALERT means "do not trust new recommendations
until reviewed" and is surfaced in the daily report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.models.calibration import calibration_slope_intercept, log_loss_

EPS = 1e-6


def psi(ref, new, bins: int = 10) -> float:
    ref, new = pd.Series(ref).dropna(), pd.Series(new).dropna()
    if len(ref) < 50 or len(new) < 30 or ref.nunique() < 3:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(new, edges)[0] / len(new)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def feature_drift(ref: pd.DataFrame, new: pd.DataFrame, cols: list[str], warn: float = 0.2) -> dict:
    scores = {c: psi(ref[c], new[c]) for c in cols if c in ref and c in new}
    flagged = {c: round(v, 3) for c, v in scores.items() if v == v and v > warn}
    return {"psi": {c: (round(v, 3) if v == v else None) for c, v in scores.items()}, "flagged": flagged,
            "status": "WARN" if flagged else "OK"}


def _ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def performance_drift(hist: pd.DataFrame, p_col: str, baseline_col: str | None = "elo", window: int = 200,
                      min_ref: int = 400, seed: int = 0) -> dict:
    """``hist``: chronologically ordered resolved predictions with ``y``, ``p_col`` (and optionally ``baseline_col``).

    Reference = distribution of rolling-``window`` mean log loss over everything *before* the latest window.
    """
    hist = hist.sort_values("game_date") if "game_date" in hist else hist
    if len(hist) < min_ref + window:
        return {"status": "INSUFFICIENT_DATA", "n": len(hist), "reasons": [f"need >= {min_ref + window} resolved predictions"]}
    l = _ll(hist[p_col], hist.y)
    recent, ref = l[-window:], l[:-window]
    roll = pd.Series(ref).rolling(window).mean().dropna()
    z = float((recent.mean() - roll.mean()) / max(roll.std(ddof=1), 1e-9))
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(recent, len(recent)).mean() for _ in range(2000)])
    slope, _ = calibration_slope_intercept(hist[p_col].to_numpy()[-window:], hist.y.to_numpy()[-window:])
    reasons, status = [], "OK"
    if recent.mean() > 0.6931 and np.percentile(boot, 5) > 0.6931:
        status = "ALERT"; reasons.append(f"recent log loss {recent.mean():.4f} is significantly worse than a coin flip (0.6931)")
    if baseline_col and baseline_col in hist:
        d = _ll(hist[baseline_col], hist.y)[-window:] - recent
        se = d.std(ddof=1) / np.sqrt(len(d))
        if d.mean() > 2 * se and d.mean() > 0:
            status = "ALERT"; reasons.append(f"recent log loss worse than {baseline_col} baseline by {d.mean():.4f} (>2 SE)")
    if z > 2 and status == "OK":
        status = "WARN"; reasons.append(f"recent log loss {recent.mean():.4f} is {z:.1f} sd above its historical rolling mean {roll.mean():.4f}")
    if slope < 0.5 and status == "OK":
        status = "WARN"; reasons.append(f"calibration slope {slope:.2f} (<0.5): probabilities are over-confident")
    return {"status": status, "n": len(hist), "recent_logloss": float(recent.mean()), "reference_logloss": float(roll.mean()),
            "z": z, "calibration_slope": slope, "reasons": reasons}
