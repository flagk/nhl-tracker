"""Goals model for totals (over/under) and puck-line (spread) probabilities.

Why a separate model: the win-probability stack says who wins, not by how much or how many goals are scored. Totals and spreads
need the *distribution* of the score. Here each team's regulation goals are modelled as negative-binomial (Poisson with a small
overdispersion) with a mean from a regularised Poisson regression on the same leak-free as-of features.

Settlement conventions (match North American sportsbooks):
- **Totals** count regulation + overtime goals but NOT the shootout goal. A game tied after regulation adds one goal if it is decided
  in overtime and none if it goes to a shootout.
- **Puck line** uses the official result: an overtime or shootout winner wins by exactly one goal.
Regulation goals are therefore the official score minus the extra goal of an OT/SO winner; tied-after-regulation games have equal
regulation goals. ``home_win`` from the production moneyline model splits those ties, so all three markets stay consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = 16                    # goals per team considered (P(>15) is negligible)
DISPERSIONS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
LAMBDA_BOUNDS = (0.6, 6.0)


def regulation_goals(games: pd.DataFrame) -> pd.DataFrame:
    """Columns ``hr, ar`` (regulation goals), ``ot`` (decided in overtime), ``so`` (shootout) from official scores + ``last_period``.

    Only meaningful for completed regular-season games; rows with a missing score get NaN.
    """
    hs, as_ = games["home_score"].astype(float), games["away_score"].astype(float)
    lp = games["last_period"].fillna("REG").astype(str)
    extra = lp.isin(["OT", "SO"])
    home_won = hs > as_
    hr = np.where(extra & home_won, hs - 1, hs)
    ar = np.where(extra & ~home_won, as_ - 1, as_)
    return pd.DataFrame({"hr": hr, "ar": ar, "ot": (lp == "OT").to_numpy(), "so": (lp == "SO").to_numpy()}, index=games.index).where(hs.notna() & as_.notna())


def add_goal_targets(F: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach ``hr, ar, ot, so`` to the feature frame (indexed by game_id) using the games table (which carries ``last_period``)."""
    g = games.drop_duplicates("game_id").set_index("game_id")
    sub = g.loc[g.index.intersection(F.index)]
    t = regulation_goals(sub)
    t["tot"] = t.hr + t.ar + t.ot.astype(float)                          # regulation + overtime goals, shootout goal excluded (totals settlement)
    t["mar"] = sub.home_score.astype(float) - sub.away_score.astype(float)   # official margin (an OT/SO winner wins by one): puck-line settlement
    return F.join(t)


def _pmf(mu: np.ndarray, k: float, n: int = GRID) -> np.ndarray:
    """(len(mu), n) probability of 0..n-1 goals; Poisson when k == 0 else negative binomial with variance mu + k*mu^2."""
    x = np.arange(n)[None, :]
    mu = np.asarray(mu, float)[:, None]
    if k <= 1e-9:
        return poisson.pmf(x, mu)
    r = 1.0 / k
    return nbinom.pmf(x, r, r / (r + mu))


class GoalsModel:
    """Fit: Poisson regressions for home and away regulation goals + a dispersion and the overtime split from training data."""

    def __init__(self, features: list[str], alpha: float = 50.0) -> None:
        self.features, self.alpha = list(features), alpha

    def _reg(self):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=500))

    def fit(self, X: pd.DataFrame, hr, ar, ot=None, so=None) -> "GoalsModel":
        hr, ar = np.asarray(hr, float), np.asarray(ar, float)
        self.home_, self.away_ = self._reg().fit(X[self.features], hr), self._reg().fit(X[self.features], ar)
        mh, ma = self._lam(X)
        ll = {k: float(np.sum(np.log(np.maximum(_pmf(mh, k)[np.arange(len(hr)), np.clip(hr, 0, GRID - 1).astype(int)], 1e-12)) +
                              np.log(np.maximum(_pmf(ma, k)[np.arange(len(ar)), np.clip(ar, 0, GRID - 1).astype(int)], 1e-12)))) for k in DISPERSIONS}
        self.dispersion_ = max(ll, key=ll.get)
        tied = hr == ar
        if ot is not None and so is not None and np.sum(tied) > 20:
            o, s = np.asarray(ot, bool)[tied].sum(), np.asarray(so, bool)[tied].sum()
            self.p_ot_ = float(o / (o + s)) if (o + s) > 0 else 0.57
        else:
            self.p_ot_ = 0.57
        self.tie_rate_ = float(np.mean(hr == ar))        # hockey ties after 60 minutes more often than independent scoring implies
        return self

    def _lam(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = LAMBDA_BOUNDS
        return (np.clip(self.home_.predict(X[self.features]), lo, hi), np.clip(self.away_.predict(X[self.features]), lo, hi))

    def predict_lambdas(self, X: pd.DataFrame) -> pd.DataFrame:
        mh, ma = self._lam(X)
        return pd.DataFrame({"lam_home": mh, "lam_away": ma}, index=X.index)

    def distributions(self, X: pd.DataFrame, p_home=None) -> list["ScoreDistribution"]:
        lam = self.predict_lambdas(X)
        ph = np.full(len(X), np.nan) if p_home is None else np.asarray(p_home, float)
        return [ScoreDistribution(a, b, self.dispersion_, self.p_ot_, None if np.isnan(p) else float(p), self.tie_rate_) for a, b, p in zip(lam.lam_home, lam.lam_away, ph)]


@dataclass
class ScoreDistribution:
    """Joint regulation-goal grid for one game plus the market probabilities derived from it."""
    lam_home: float
    lam_away: float
    dispersion: float
    p_ot: float
    p_home_ml: float | None = None       # production moneyline probability; splits regulation ties (falls back to the goals model's own view)
    tie_rate: float | None = None        # target P(tied after regulation); the independent grid under-counts ties, so the diagonal is boosted to match

    def __post_init__(self) -> None:
        ph = _pmf(np.array([self.lam_home]), self.dispersion)[0]
        pa = _pmf(np.array([self.lam_away]), self.dispersion)[0]
        J = np.outer(ph, pa)
        J = J / J.sum()
        if self.tie_rate is not None and 0.0 < self.tie_rate < 1.0:
            t = float(np.trace(J))
            if 0.0 < t < 1.0:
                c = self.tie_rate * (1.0 - t) / (t * (1.0 - self.tie_rate))
                J = J * (1.0 + (c - 1.0) * np.eye(J.shape[0]))
        self.grid = J / J.sum()
        n = self.grid.shape[0]
        i, j = np.indices((n, n))
        self.reg_margin, self.reg_total = (i - j).ravel(), (i + j).ravel()
        self.w = self.grid.ravel()
        self.p_tie = float(self.w[self.reg_margin == 0].sum())
        self.p_home_reg = float(self.w[self.reg_margin > 0].sum())
        if self.p_tie > 0:
            want = (self.p_home_ml if self.p_home_ml is not None else self.p_home_reg + 0.5 * self.p_tie)
            self.s_home_tie = float(np.clip((want - self.p_home_reg) / self.p_tie, 0.0, 1.0))
        else:
            self.s_home_tie = 0.5

    @property
    def p_home(self) -> float:
        return self.p_home_reg + self.p_tie * self.s_home_tie

    def expected_total(self) -> float:
        return float((self.w * (self.reg_total + np.where(self.reg_margin == 0, self.p_ot, 0.0))).sum())

    def total_probs(self, line: float) -> tuple[float, float, float]:
        """(P(over), P(under), P(push)) for a total line, shootout goal excluded."""
        t = self.reg_total.astype(float)
        tied = self.reg_margin == 0
        over = under = push = 0.0
        for add, pr in ((0.0, 1.0 - self.p_ot), (1.0, self.p_ot)):       # tied games: +1 goal only if decided in OT
            tt = np.where(tied, t + add, t)
            wt = np.where(tied, self.w * pr, self.w if add == 0.0 else 0.0)
            over += float(wt[tt > line].sum()); under += float(wt[tt < line].sum()); push += float(wt[tt == line].sum())
        s = over + under + push
        return over / s, under / s, push / s

    def spread_probs(self, home_point: float) -> tuple[float, float, float]:
        """(P(home covers), P(away covers), P(push)) where the home team gets ``home_point`` goals (-1.5 = favourite giving 1.5)."""
        m = self.reg_margin.astype(float)
        tied = self.reg_margin == 0
        cover = push = 0.0
        for fm, pr in ((1.0, self.s_home_tie), (-1.0, 1.0 - self.s_home_tie)):
            mm = np.where(tied, fm, m)
            wt = np.where(tied, self.w * pr, self.w if fm == 1.0 else 0.0)
            cover += float(wt[mm + home_point > 0].sum()); push += float(wt[mm + home_point == 0].sum())
        away = 1.0 - cover - push
        return cover, away, push
