"""Shots-on-goal model for player props (over/under lines).

Each player's shots in a game are negative-binomial. The mean comes from a Poisson regression on: the player's own shrunk recent shot rate
(shrunk toward his position's average, more when he has few games), how generous tonight's defence has been, home ice, recent ice time and rest.
All inputs are as-of the game date (see ``nhlbet.features.players``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

SHRINK_K = 6.0                 # pseudo-games of position-average shooting blended into a player's own rate
MIN_GAMES = 5                  # no props for players with fewer prior games
DISPERSIONS = (0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.3)
LAM_BOUNDS = (0.01, 8.0)


SHRINK_PATH = "reports/props_shrink.json"


def load_shrink(path: str = SHRINK_PATH) -> dict:
    """Per-stat shrinkage (pseudo-games) chosen by ``scripts/props_backtest.py`` on games before its selection date; empty = the default for every stat."""
    import json
    from pathlib import Path
    p = Path(path)
    try:
        j = json.loads(p.read_text()) if p.exists() else {}
        pinned = {st for st, a in (j.get("grid_audit") or {}).items() if a.get("pinned_to_grid_edge")}
        return {k: float(v) for k, v in (j.get("shrink_k") or {}).items() if k not in pinned}       # a value on the edge of its tuning grid is not a result: use the default instead
    except (ValueError, OSError):
        return {}


class ShotsModel:
    """Despite the name it models any per-game skater count with the same recipe: ``stat`` is 'sog' (shots, the default), 'points', 'assists' or 'goals'."""

    def __init__(self, alpha: float = 1e-3, stat: str = "sog", shrink_k: float | None = None) -> None:
        self.alpha, self.stat = alpha, stat
        self.shrink_k = SHRINK_K if shrink_k is None else float(shrink_k)       # pseudo-games of position-average rate blended into a player's own rate

    def _prep(self, d: pd.DataFrame) -> np.ndarray:
        pos_mean = d.pos.map(self.pos_mean_).fillna(self.pos_mean_["F"]).to_numpy(float)
        n = np.clip(d.n_prev.fillna(0).to_numpy(float), 0, 40)
        own = d[f"ewm_{self.stat}"].fillna(pd.Series(pos_mean, index=d.index)).to_numpy(float)
        rate = (n * own + self.shrink_k * pos_mean) / (n + self.shrink_k)
        toi_ref = d.pos.map(self.toi_ref_).fillna(self.toi_ref_["F"]).to_numpy(float)
        toi_ratio = (d.toi_l10.fillna(pd.Series(toi_ref, index=d.index)).to_numpy(float) / toi_ref).clip(0.4, 1.8)
        return np.column_stack([np.log(np.clip(rate, 0.01, None)), np.log(d.opp_factor.fillna(1.0).to_numpy(float)), d.is_home.to_numpy(float),
                                np.log(toi_ratio), np.minimum(d.rest_days.fillna(3).to_numpy(float), 5.0)])

    def fit(self, d: pd.DataFrame) -> "ShotsModel":
        y_all = self.stat
        d = d[(d.n_prev >= 3) & d[y_all].notna()]
        self.pos_mean_ = d.groupby("pos")[y_all].mean().to_dict() or {"F": 1.9, "D": 1.4}
        self.pos_mean_.setdefault("F", float(d[y_all].mean())); self.pos_mean_.setdefault("D", float(d[y_all].mean()))
        self.toi_ref_ = (d.groupby("pos").toi_sec.mean().to_dict() or {"F": 1000.0, "D": 1300.0})
        self.toi_ref_.setdefault("F", 1000.0); self.toi_ref_.setdefault("D", 1300.0)
        X, y = self._prep(d), d[y_all].to_numpy(float)
        self.reg_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(X, y)
        mu = self.lam(d)
        ll = {k: float(np.sum(np.log(np.maximum(_pmf(y.astype(int), mu, k), 1e-12)))) for k in DISPERSIONS}
        self.dispersion_ = max(ll, key=ll.get)
        return self

    def lam(self, d: pd.DataFrame) -> np.ndarray:
        return np.clip(self.reg_.predict(self._prep(d)), *LAM_BOUNDS)

    def p_over(self, lam, line: float, k: float | None = None) -> np.ndarray:
        """P(shots > line) for half-point lines (a whole-number line would also need a push term)."""
        k = self.dispersion_ if k is None else k
        return 1.0 - _cdf(np.floor(line), np.asarray(lam, float), k)


def over_push_under(model_or_k, lam, line: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P(over), P(push), P(under)) of a count from the full PMF. Half-point lines cannot push; on a whole-number line the stake is returned when the count equals it;
    a quarter line (x.25 / x.75) is half the stake on each adjacent line, so its probabilities are the average of the two legs'."""
    frac = round((float(line) * 4) % 4)
    if frac in (1, 3):                                   # quarter line: two bets of half the stake on the adjacent lines, so average their outcome probabilities
        a, b = over_push_under(model_or_k, lam, line - 0.25), over_push_under(model_or_k, lam, line + 0.25)
        return tuple((x + y) / 2.0 for x, y in zip(a, b))             # type: ignore[return-value]
    k = model_or_k.dispersion_ if hasattr(model_or_k, "dispersion_") else float(model_or_k)
    lam = np.asarray(lam, float)
    over = 1.0 - _cdf(np.floor(line), lam, k)
    under = _cdf(np.ceil(line) - 1.0, lam, k)
    return over, np.clip(1.0 - over - under, 0.0, 1.0), under


def _pmf(x, mu, k):
    mu = np.asarray(mu, float)
    if k <= 1e-9:
        return poisson.pmf(x, mu)
    r = 1.0 / k
    return nbinom.pmf(x, r, r / (r + mu))


def _cdf(x, mu, k):
    if k <= 1e-9:
        return poisson.cdf(x, mu)
    r = 1.0 / k
    return nbinom.cdf(x, r, r / (r + mu))


def baseline_lam(d: pd.DataFrame, pos_mean: dict, stat: str = "sog", shrink_k: float | None = None) -> np.ndarray:
    """The no-model comparison: the player's shrunk own rate only (no opponent, venue, ice time or rest)."""
    k = SHRINK_K if shrink_k is None else float(shrink_k)
    pm = d.pos.map(pos_mean).fillna(float(np.mean(list(pos_mean.values())))).to_numpy(float)
    n = np.clip(d.n_prev.fillna(0).to_numpy(float), 0, 40)
    own = d[f"ewm_{stat}"].fillna(pd.Series(pm, index=d.index)).to_numpy(float)
    return (n * own + k * pm) / (n + k)
