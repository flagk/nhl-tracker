"""Goals model for totals (over/under) and puck-line (spread) probabilities.

Why a separate model: the win-probability stack says who wins, not by how much or how many goals are scored. Totals and spreads
need the *distribution* of the score. Here each team's regulation goals are modelled as negative-binomial (Poisson with a small
overdispersion) with a mean from a regularised Poisson regression on the same leak-free as-of features.

Settlement conventions (match North American sportsbooks):
- **Totals**: a game tied after regulation adds one goal if it is decided in overtime. Whether a shootout adds one as well is the settlement rule in
  config/policy.yaml (``nhlbet.settle.shootout_in_total``; the industry convention is yes, the repo assumed no before Phase 1). Team totals and
  both-teams-to-score always exclude the shootout goal.
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

TARGET_COLS = ("hr", "ar", "ot", "so", "tot", "mar", "home_score", "away_score", "home_win")   # outcomes: never allowed as features
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
    from nhlbet.settle import shootout_in_total
    t["tot"] = t.hr + t.ar + t.ot.astype(float) + (t.so.astype(float) if shootout_in_total() else 0.0)   # totals settlement (shootout goal per the policy's rule set)
    t["mar"] = sub.home_score.astype(float) - sub.away_score.astype(float)   # official margin (an OT/SO winner wins by one): puck-line settlement
    return F.join(t)


def goal_feature_columns(F: pd.DataFrame) -> list[str]:
    """Candidate predictors for the goals model: populated, non-constant as-of features, never the outcome columns added by ``add_goal_targets``."""
    from nhlbet.analysis.importance import candidate_columns
    return candidate_columns(F.drop(columns=[c for c in TARGET_COLS if c in F.columns]))


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

    RECENT = 400                          # games used to re-centre the goal level (league scoring drifts from season to season)

    def __init__(self, features: list[str], alpha: float = 50.0) -> None:
        leaked = sorted(set(features) & set(TARGET_COLS))
        if leaked:
            raise ValueError(f"outcome columns cannot be goals-model features (target leakage): {leaked}")
        self.features, self.alpha = list(features), alpha
        self.cal: dict | None = None          # {"totals": {"a", "b"}, "spreads": {...}}: Platt maps fit on walk-forward predictions (see goals_eval)

    def _reg(self):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=500))

    def fit(self, X: pd.DataFrame, hr, ar, ot=None, so=None) -> "GoalsModel":
        hr, ar = np.asarray(hr, float), np.asarray(ar, float)
        self.home_, self.away_ = self._reg().fit(X[self.features], hr), self._reg().fit(X[self.features], ar)
        self.level_ = (1.0, 1.0)
        mh, ma = self._lam(X)
        k = min(self.RECENT, len(hr))                     # X is in date order: match the model's mean to the most recent games' actual scoring
        self.level_ = (float(np.clip(hr[-k:].mean() / mh[-k:].mean(), 0.9, 1.1)), float(np.clip(ar[-k:].mean() / ma[-k:].mean(), 0.9, 1.1)))
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
        ch, ca = getattr(self, "level_", (1.0, 1.0))
        return (np.clip(ch * self.home_.predict(X[self.features]), lo, hi), np.clip(ca * self.away_.predict(X[self.features]), lo, hi))

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

    def expected_total(self, so_in_total: bool | None = None) -> float:
        """Expected game total under the shootout rule (default: the policy's): a tied game adds a goal in overtime, and in a shootout too when the book counts it."""
        if so_in_total is None:
            from nhlbet.settle import shootout_in_total
            so_in_total = shootout_in_total()
        p_extra = 1.0 if so_in_total else self.p_ot
        return float((self.w * (self.reg_total + np.where(self.reg_margin == 0, p_extra, 0.0))).sum())

    def total_probs(self, line: float, so_in_total: bool | None = None) -> tuple[float, float, float]:
        """(P(over), P(under), P(push)) for a total line. ``so_in_total`` (default: the policy's rule set) says whether a shootout adds a goal to the total.

        Half-point lines cannot push. Whole-number lines have three outcomes (the stake is returned on a push). Quarter lines (x.25 / x.75) are two bets of
        half the stake on the adjacent lines, so the outcome probabilities are the average of the two legs (a half-win or half-loss is half a leg's win or loss,
        which is exactly what averaging the leg probabilities gives for expected value).
        """
        frac = round((float(line) * 4) % 4)
        if frac in (1, 3):
            a, b = self.total_probs(line - 0.25, so_in_total), self.total_probs(line + 0.25, so_in_total)
            return tuple((x + y) / 2.0 for x, y in zip(a, b))                  # type: ignore[return-value]
        if so_in_total is None:
            from nhlbet.settle import shootout_in_total
            so_in_total = shootout_in_total()
        p_extra = 1.0 if so_in_total else self.p_ot                       # tied games: +1 goal if decided in OT, and also in a shootout when the book counts it
        t = self.reg_total.astype(float)
        tied = self.reg_margin == 0
        over = under = push = 0.0
        for add, pr in ((0.0, 1.0 - p_extra), (1.0, p_extra)):
            tt = np.where(tied, t + add, t)
            wt = np.where(tied, self.w * pr, self.w if add == 0.0 else 0.0)
            over += float(wt[tt > line].sum()); under += float(wt[tt < line].sum()); push += float(wt[tt == line].sum())
        s = over + under + push
        return over / s, under / s, push / s

    def total_pmf(self, n: int = 20, so_in_total: bool | None = None) -> np.ndarray:
        """P(game total = k) for k = 0..n-1 (the last cell also holds any mass beyond), under the given shootout rule (default: the policy's). Sums to 1."""
        if so_in_total is None:
            from nhlbet.settle import shootout_in_total
            so_in_total = shootout_in_total()
        p_extra = 1.0 if so_in_total else self.p_ot
        t = self.reg_total.astype(int)
        tied = self.reg_margin == 0
        pmf = np.zeros(n)
        np.add.at(pmf, np.minimum(t, n - 1), np.where(tied, self.w * (1.0 - p_extra), self.w))
        np.add.at(pmf, np.minimum(t + 1, n - 1), np.where(tied, self.w * p_extra, 0.0))
        return pmf / pmf.sum()

    def spread_probs(self, home_point: float) -> tuple[float, float, float]:
        """(P(home covers), P(away covers), P(push)) where the home team gets ``home_point`` goals (-1.5 = favourite giving 1.5)."""
        frac = round((float(home_point) * 4) % 4)
        if frac in (1, 3):                                                   # quarter line: half the stake on each adjacent line
            a, b = self.spread_probs(home_point - 0.25), self.spread_probs(home_point + 0.25)
            return tuple((x + y) / 2.0 for x, y in zip(a, b))                  # type: ignore[return-value]
        m = self.reg_margin.astype(float)
        tied = self.reg_margin == 0
        cover = push = 0.0
        for fm, pr in ((1.0, self.s_home_tie), (-1.0, 1.0 - self.s_home_tie)):
            mm = np.where(tied, fm, m)
            wt = np.where(tied, self.w * pr, self.w if fm == 1.0 else 0.0)
            cover += float(wt[mm + home_point > 0].sum()); push += float(wt[mm + home_point == 0].sum())
        away = 1.0 - cover - push
        return cover, away, push


    # ---- every other market comes from the SAME joint grid, so all prices stay mutually consistent ----
    def _final_goal_cells(self):
        """Yield (weight, home_goals, away_goals) over the official pre-shootout result: regulation goals, plus one for the overtime winner.

        A game tied after regulation is decided in overtime with probability ``p_ot`` (home wins it with probability ``s_home_tie``) and otherwise goes to a
        shootout, whose goal never counts toward totals or team totals.
        """
        n = self.grid.shape[0]
        for i in range(n):
            for j in range(n):
                w = float(self.grid[i, j])
                if w <= 0:
                    continue
                if i == j:
                    yield w * self.p_ot * self.s_home_tie, i + 1, j
                    yield w * self.p_ot * (1.0 - self.s_home_tie), i, j + 1
                    yield w * (1.0 - self.p_ot), i, j
                else:
                    yield w, i, j

    def team_total_probs(self, side: str, line: float) -> tuple[float, float, float]:
        """(P(over), P(under), P(push)) for one team's goal total (regulation + overtime goals, shootout goal excluded)."""
        frac = round((float(line) * 4) % 4)
        if frac in (1, 3):
            a, b = self.team_total_probs(side, line - 0.25), self.team_total_probs(side, line + 0.25)
            return tuple((x + y) / 2.0 for x, y in zip(a, b))                  # type: ignore[return-value]
        over = under = push = 0.0
        for w, h, a in self._final_goal_cells():
            g = h if side == "home" else a
            over += w * (g > line); under += w * (g < line); push += w * (g == line)
        s = over + under + push
        return over / s, under / s, push / s

    def btts_prob(self) -> float:
        """P(both teams score at least once) counting overtime goals but not the shootout goal."""
        tot = sum(w for w, h, a in self._final_goal_cells() if h >= 1 and a >= 1)
        return float(tot / sum(w for w, _, _ in self._final_goal_cells()))


def coherence_issues(d: "ScoreDistribution", tol: float = 1e-9) -> list[str]:
    """Internal-consistency checks across markets that all come from one joint score grid. Empty list = coherent.

    The moneyline equals the puck line at -0.5 (an overtime or shootout winner wins by exactly one), every line's three outcomes sum to one, half-point
    lines never push, over probabilities fall as the line rises, the team totals reproduce the game total's mean, and both-teams-to-score agrees with
    inclusion-exclusion over the team marginals.
    """
    bad = []
    if abs(d.spread_probs(-0.5)[0] - d.p_home) > 1e-7:
        bad.append("moneyline != puck line -0.5")
    prev = 1.0
    for line in (3.5, 4.5, 5.0, 5.25, 5.5, 6.0, 6.5, 7.5):
        o, u, p = d.total_probs(line)
        if abs(o + u + p - 1.0) > 1e-7:
            bad.append(f"total {line}: probabilities do not sum to 1")
        if line % 1 == 0.5 and p > tol:
            bad.append(f"total {line}: a half-point line cannot push")
        if o > prev + 1e-9:
            bad.append(f"total {line}: P(over) rose with the line")
        prev = o
    cells = list(d._final_goal_cells())
    wsum = sum(w for w, _, _ in cells)
    mean_total = sum(w * (h + a) for w, h, a in cells) / wsum
    if abs(mean_total - d.expected_total(so_in_total=False)) > 1e-6:                 # team totals never count the shootout goal
        bad.append("team totals do not reproduce the game total's mean")
    if abs(float((np.arange(40) * d.total_pmf(40)).sum()) - d.expected_total()) > 1e-6:      # the game total under the settlement rule in force
        bad.append("the total's PMF does not reproduce its expected value")
    p_h0 = 1.0 - d.team_total_probs("home", 0.5)[0]
    p_a0 = 1.0 - d.team_total_probs("away", 0.5)[0]
    p_00 = sum(w for w, h, a in cells if h == 0 and a == 0) / wsum
    if abs(d.btts_prob() - (1.0 - p_h0 - p_a0 + p_00)) > 1e-7:
        bad.append("both-teams-to-score disagrees with the team totals")
    return bad
