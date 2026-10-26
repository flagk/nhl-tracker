"""Placebo tests with positive controls.

* ``time_shift_placebo``: re-time a team-level feature by k games. k > 0 uses stale values (skill should fade), k < 0 uses FUTURE values (skill jumps: the positive control that
  shows the evaluation would see a leak if there were one). The honest curve peaks at k = 0 and falls off on the stale side; a pipeline whose k = 0 point already sits on the
  leaky side of the curve (AUC far above the stale neighbours' trend) has a problem.
* ``adversarial_validation``: a classifier tries to tell earlier rows from later rows by their features alone. AUC near 0.5 = the test period looks like the training period;
  high AUC = covariate shift (the model is being asked about a different world).
* ``shuffled_market_placebo``: the ceiling an evaluation flaw could manufacture (the model against prices unrelated to the game), and how much of the model's information the real price already held.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _boot_auc(y, x, B=300, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.asarray(y), np.asarray(x)
    a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], x[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def time_shift_placebo(F: pd.DataFrame, feature: str = "gd_l10", shifts=(-2, -1, 0, 1, 2, 5, 10)) -> pd.DataFrame:
    """``F``: builder output with h_<feature>, a_<feature>, home, away, game_date, home_win. Signed AUC of (home - away) of the k-game-shifted pre-game feature."""
    F = F[F.home_win.notna()].sort_values(["game_date"], kind="stable")
    rows = []
    seq: dict[str, list] = {}
    for gid, r in zip(F.index, F.itertuples()):
        seq.setdefault(r.home, []).append((gid, getattr(r, f"h_{feature}")))
        seq.setdefault(r.away, []).append((gid, getattr(r, f"a_{feature}")))
    pos = {t: {gid: i for i, (gid, _) in enumerate(s)} for t, s in seq.items()}
    y = F.home_win.astype(int).to_numpy()
    for k in shifts:
        def val(team, gid):
            i = pos[team][gid] - k
            s = seq[team]
            return s[i][1] if 0 <= i < len(s) else np.nan
        x = np.array([val(h, g) - val(a, g) for g, h, a in zip(F.index, F.home, F.away)], float)
        m = np.isfinite(x)
        auc = roc_auc_score(y[m], x[m]) if m.sum() > 30 and len(np.unique(y[m])) == 2 else float("nan")
        lo, hi = _boot_auc(y[m], x[m]) if m.sum() > 30 else (float("nan"), float("nan"))
        rows.append({"shift_games": k, "meaning": "FUTURE value" if k < 0 else "current (as built)" if k == 0 else "stale value", "n": int(m.sum()), "auc": auc, "auc_lo": lo, "auc_hi": hi})
    return pd.DataFrame(rows)


def adversarial_validation(F: pd.DataFrame, features: list[str], split_date, seed: int = 0, folds: int = 5) -> dict:
    d = pd.to_datetime(F.game_date)
    X = F[features].astype(float)
    y = (d >= pd.Timestamp(split_date)).astype(int).to_numpy()
    if y.sum() < 50 or (1 - y).sum() < 50:
        return {"split": str(pd.Timestamp(split_date).date()), "auc": float("nan"), "note": "too few rows on one side"}
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))
    from sklearn.inspection import permutation_importance
    clf.fit(X, y)
    imp = permutation_importance(clf, X, y, n_repeats=3, random_state=seed, scoring="roc_auc")
    top = sorted(zip(features, imp.importances_mean), key=lambda t: -t[1])[:5]
    return {"split": str(pd.Timestamp(split_date).date()), "n_train_side": int((1 - y).sum()), "n_test_side": int(y.sum()), "auc": auc, "top_features": [(f, round(float(v), 4)) for f, v in top],
            "reading": "no detectable shift" if auc < 0.6 else "moderate shift" if auc < 0.75 else "strong shift: the test period is distinguishable from training"}


def shuffled_market_placebo(p_model, p_market, y, edge: float = 0.03, margin: float = 0.045, n_perm: int = 500, seed: int = 0) -> dict:
    """What the same model "earns" against prices that carry no information about the game.

    Bets are chosen as the policy would (home if p_model - p_market >= edge, away if p_market - p_model >= edge) and settled at fair odds less ``margin``. The selection is
    repeated with the market probabilities permuted across games. A model with any real information profits hugely against uninformative prices, so this number is a CEILING
    on what an evaluation flaw (a price that does not belong to its game) could manufacture, and the gap between it and the real-market ROI is how much of the model's
    information the market already contained. A real-market ROI that matched the shuffled one would mean the market is uninformative (or the pairing is broken); one near
    ``-margin`` with a large shuffled ROI is what an efficient market looks like.
    """
    pm, mk, y = np.asarray(p_model, float), np.asarray(p_market, float), np.asarray(y, float)

    def roi(m):
        home, away = (pm - m) >= edge, (m - pm) >= edge
        profit = np.concatenate([np.where(y[home] == 1, (1 - margin) / m[home] - 1, -1.0), np.where(y[away] == 0, (1 - margin) / (1 - m[away]) - 1, -1.0)])
        return (float(profit.mean()), int(len(profit))) if len(profit) >= 30 else (float("nan"), int(len(profit)))
    actual, n = roi(mk)
    rng = np.random.default_rng(seed)
    sims = np.array([roi(mk[rng.permutation(len(mk))])[0] for _ in range(n_perm)])
    sims = sims[np.isfinite(sims)]
    return {"n_bets": n, "actual_roi": actual, "shuffled_roi_mean": float(sims.mean()) if len(sims) else float("nan"),
            "shuffled_roi_95": [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))] if len(sims) else [float("nan")] * 2,
            "information_already_in_the_price": (float(1 - (actual + margin) / (sims.mean() + margin)) if len(sims) and np.isfinite(actual) and sims.mean() + margin > 0 else float("nan"))}
