"""Probability calibration and calibration/scoring metrics.

Bet sizing is only as good as the probabilities, so calibration is a first-class citizen:
Platt (logistic on the logit) is the default because it needs few points; isotonic is available and
compared, but it overfits with ~1-2k calibration rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nhlbet.models.base import EPS, expit, logit


class Calibrator:
    def __init__(self, method: str = "platt") -> None:
        if method not in ("platt", "isotonic", "none"):
            raise ValueError(method)
        self.method = method

    def fit(self, p, y) -> "Calibrator":
        p, y = np.asarray(p, float), np.asarray(y, int)
        if self.method == "platt":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m_ = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "none":
            return p
        if self.method == "platt":
            return self.m_.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        return np.clip(self.m_.predict(p), EPS, 1 - EPS)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss_(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Quantile bins (equal counts) of predicted probability vs observed frequency."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = pd.qcut(df.p, bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), obs=("y", "mean"))
    return t.reset_index(drop=True)


def ece(p, y, bins: int = 10) -> float:
    t = reliability_table(p, y, bins)
    return float(np.sum(t.n * (t.mean_p - t.obs).abs()) / t.n.sum())


def calibration_slope_intercept(p, y) -> tuple[float, float]:
    """Slope of a logistic recalibration of the logit. 1.0 is perfect; <1 = overconfident; >1 = underconfident."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y, int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def plot_reliability(curves: dict[str, tuple], path: str, bins: int = 10, title: str = "Reliability (out-of-sample, walk-forward)") -> None:
    """curves: name -> (p, y). Saves a reliability diagram with a prediction histogram underneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.5, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0.3, 0.75], [0.3, 0.75], "k--", lw=1, label="perfect")
    for name, (p, y) in curves.items():
        t = reliability_table(p, y, bins)
        ax.plot(t.mean_p, t.obs, "o-", ms=4, label=f"{name} (ECE {ece(p, y, bins):.3f})")
        ax2.hist(p, bins=30, alpha=0.4, label=name)
    ax.set_ylabel("observed home-win frequency"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2.set_xlabel("predicted P(home win)"); ax2.set_ylabel("games")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def online_calibrate(P: pd.DataFrame, col: str, method: str = "platt", min_history: int = 300, block_days: int = 14) -> pd.Series:
    """Calibrate a model's walk-forward predictions using ONLY its own earlier out-of-sample predictions.

    For each ``block_days`` block, a calibrator is fit on all previously *resolved* predictions (dates strictly
    before the block) and applied to the block. Blocks with fewer than ``min_history`` resolved predictions
    are returned as NaN (callers fall back to the inner-OOF calibrated column). This mirrors production, where
    the calibrator is fit on the logged, resolved predictions of the deployed model - a distribution that
    matches future predictions better than inner cross-validation folds do.
    """
    P = P.sort_values("game_date")
    dates = pd.to_datetime(P["game_date"])
    block = (dates - dates.min()).dt.days // block_days
    out = pd.Series(np.nan, index=P.index)
    for b in sorted(block.unique()):
        past = dates < dates[block == b].min()
        if past.sum() >= min_history:
            cal = Calibrator(method).fit(P.loc[past, col], P.loc[past, "y"])
            out[block == b] = cal.predict(P.loc[block == b, col])
    return out
