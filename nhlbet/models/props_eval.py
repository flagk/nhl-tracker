"""Walk-forward evaluation of the shots-on-goal model against the player's own shrunk rate. Retrains on strictly earlier games only.

There are no historical prop lines in this project, so the test is on shot counts and on over/under hit rates at typical lines
(calibration), not against the market. The market test is the paper trading going forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from nhlbet.models.evaluate import _week_groups
from nhlbet.models.props import MIN_GAMES, ShotsModel, baseline_lam
from nhlbet.splits import walk_forward_windows

LINES = (1.5, 2.5, 3.5)
EPS = 1e-6


def walk_forward_props(feats: pd.DataFrame, first_test: str, step_days: int = 28, min_train: int = 5000) -> pd.DataFrame:
    """``feats``: output of ``add_asof_features`` (needs ``opp_factor``). Returns out-of-sample rows with the model and baseline expectations."""
    f = feats.sort_values("game_date").reset_index(drop=True)
    dates = f.game_date
    parts = []
    for tr, te in walk_forward_windows(dates, first_test, step_days, min_train):
        m = ShotsModel().fit(f.iloc[tr])
        t = f.iloc[te]
        t = t[t.n_prev >= MIN_GAMES]
        if t.empty:
            continue
        out = t[["game_id", "game_date", "player_id", "pos", "sog"]].copy()
        out["lam"] = m.lam(t)
        out["base"] = baseline_lam(t, m.pos_mean_)
        out["k"] = m.dispersion_
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _p_over(lam, line, k):
    lam = np.asarray(lam, float)
    kk = np.asarray(k, float)
    r = 1.0 / np.maximum(kk, 1e-9)
    nb = 1.0 - nbinom.cdf(np.floor(line), r, r / (r + lam))
    return np.where(kk <= 1e-9, 1.0 - poisson.cdf(np.floor(line), lam), nb)


def _boot(d: np.ndarray, dates: pd.Series, B: int = 1500, seed: int = 0) -> tuple[float, float]:
    groups = _week_groups(dates)
    rng = np.random.default_rng(seed)
    means = [d[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(P: pd.DataFrame) -> pd.DataFrame:
    """One row for shot counts (Poisson NLL) and one per line (log loss of P(over)); negative diff = the model beats the player's own rate."""
    rows = []
    nll_m, nll_b = -poisson.logpmf(P.sog, P.lam), -poisson.logpmf(P.sog, P.base)
    d = np.asarray(nll_m - nll_b, float)
    lo, hi = _boot(d, P.game_date)
    rows.append({"what": "shots per player-game (Poisson NLL)", "n": len(P), "model": float(nll_m.mean()), "baseline": float(nll_b.mean()), "diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "hit_rate": np.nan, "mean_p": np.nan})
    for line in LINES:
        sel = P[(P.lam > line * 0.45) & (P.lam < line * 1.8)]                       # players for whom this line is a realistic prop
        if len(sel) < 200:
            continue
        y = (sel.sog > line).astype(float).to_numpy()
        pm, pb = np.clip(_p_over(sel.lam, line, sel.k), EPS, 1 - EPS), np.clip(_p_over(sel.base, line, sel.k), EPS, 1 - EPS)
        ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
        dd = ll(pm) - ll(pb)
        lo, hi = _boot(dd, sel.game_date)
        rows.append({"what": f"over {line} (log loss)", "n": len(sel), "model": float(ll(pm).mean()), "baseline": float(ll(pb).mean()), "diff": float(dd.mean()), "ci_low": lo, "ci_high": hi,
                     "hit_rate": float(y.mean()), "mean_p": float(pm.mean())})
    return pd.DataFrame(rows)


def calibration_table(P: pd.DataFrame, line: float = 2.5, bins: int = 8) -> pd.DataFrame:
    sel = P[(P.lam > line * 0.45) & (P.lam < line * 1.8)].copy()
    sel["p"] = _p_over(sel.lam, line, sel.k)
    sel["y"] = (sel.sog > line).astype(float)
    sel["bin"] = pd.qcut(sel.p, bins, duplicates="drop")
    t = sel.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index(drop=True)
    return t
