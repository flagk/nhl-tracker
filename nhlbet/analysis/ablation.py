"""Ablations and parameter sensitivity under the SAME walk-forward protocol as the headline numbers.

Every variant is retrained on strictly earlier games only (``walk_forward``), scored on the same out-of-sample games, and compared with the full model by a
paired, week-clustered bootstrap, so a feature group "mattering" means its removal costs out-of-sample log loss by more than noise, not that it
looked important in-sample. Only the regularised logistic model is used (it is the best single model and fast); the point is to see which inputs earn their place and
whether the result is fragile to the one hyper-parameter that matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.features.builder import feature_group_map
from nhlbet.models.base import EloOnly, HomeRate, Logistic
from nhlbet.models.evaluate import paired_logloss_diff
from nhlbet.models.walkforward import walk_forward


def _zoo(features: list[str], C: float | None = None):
    params = {} if C is None else {"C": C}
    return {"home_rate": lambda: HomeRate(), "elo": lambda: EloOnly(), "logistic": lambda: Logistic(features, **params)}


def _run(F: pd.DataFrame, features: list[str], first_test: str, step_days: int, C: float | None = None) -> pd.DataFrame:
    return walk_forward(F, _zoo(features, C), first_test, step_days, min_train=300, inner_folds=3, n_jobs=1)


def group_ablation(F: pd.DataFrame, features: list[str], first_test: str, step_days: int = 28) -> pd.DataFrame:
    """Out-of-sample log loss with each feature GROUP removed, versus all the selected features. Positive ``cost`` = removing the group hurts."""
    gmap = feature_group_map(features)
    groups = sorted(set(gmap.values()))
    full = _run(F, features, first_test, step_days)
    rows = [{"variant": "all selected features", "n_features": len(features), "log_loss": float(_ll(full, "logistic")), "cost_vs_full": 0.0, "ci_low": 0.0, "ci_high": 0.0}]
    for g in groups:
        keep = [f for f in features if gmap.get(f) != g]
        if not keep or len(keep) == len(features):
            continue
        P = _run(F, keep, first_test, step_days)
        d = paired_logloss_diff(_align(full, P, "logistic"), "logistic_b", "logistic")                     # variant minus full
        rows.append({"variant": f"without {g} ({len(features) - len(keep)} features)", "n_features": len(keep), "log_loss": float(_ll(P, "logistic")),
                     "cost_vs_full": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
    for base in ("elo", "home_rate"):
        d = paired_logloss_diff(_align(full, full, "logistic", base), "logistic_b", "logistic")
        rows.append({"variant": f"{base} only", "n_features": 0, "log_loss": float(_ll(full, base)), "cost_vs_full": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
    out = pd.DataFrame(rows)
    out["verdict"] = np.where(out.ci_low > 0, "group earns its place", np.where(out.ci_high < 0, "group hurts", "not distinguishable from noise"))
    out.loc[0, "verdict"] = "reference"
    return out


def regularisation_sensitivity(F: pd.DataFrame, features: list[str], first_test: str, step_days: int = 28, grid=(0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)) -> pd.DataFrame:
    """Log loss across the logistic regularisation strength (the tuned value is 0.003): a flat curve means the result does not hinge on that choice."""
    rows = []
    for C in grid:
        P = _run(F, features, first_test, step_days, C)
        rows.append({"C": C, "log_loss": float(_ll(P, "logistic")), "vs_elo": float(_ll(P, "logistic") - _ll(P, "elo")), "vs_home_rate": float(_ll(P, "logistic") - _ll(P, "home_rate"))})
    return pd.DataFrame(rows)


def period_stability(F: pd.DataFrame, features: list[str], first_test: str, step_days: int = 28, min_n: int = 200) -> pd.DataFrame:
    """Logistic minus Elo and minus the home rate, per season: a model that only worked in one stretch cannot hide in the pooled number."""
    P = _run(F, features, first_test, step_days)
    d = pd.to_datetime(P.game_date)
    season = np.where(d.dt.month >= 8, d.dt.year, d.dt.year - 1)
    rows = []
    for s in np.unique(season):
        Q = P[season == s]
        if len(Q) < min_n:
            continue
        r = paired_logloss_diff(Q, "logistic", "elo", B=1500)
        rows.append({"season": f"{s}-{str(s + 1)[2:]}", "n": len(Q), "logistic": float(_ll(Q, "logistic")), "elo": float(_ll(Q, "elo")), "home_rate": float(_ll(Q, "home_rate")),
                     "logistic_minus_elo": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def _ll(P: pd.DataFrame, col: str) -> float:
    p, y = np.clip(P[col].to_numpy(float), 1e-6, 1 - 1e-6), P.y.to_numpy(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _align(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str | None = None) -> pd.DataFrame:
    """Frame with the columns ``logistic`` (from ``a``) and ``logistic_b`` (``col_b`` of ``b``) on the games both cover."""
    idx = a.index.intersection(b.index)
    out = a.loc[idx, ["game_date", "y"]].copy()
    out["logistic"] = a.loc[idx, col_a]
    out["logistic_b"] = b.loc[idx, col_b or col_a]
    return out
