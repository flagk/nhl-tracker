"""Hierarchical Bayesian challenger: team strength with partial pooling across teams and across seasons.

    logit P(home wins) = h + a[home] - a[away],   a[team, season] = A[team] + B[team, season]
    A ~ N(0, tau^2),  B ~ N(0, sigma^2)   (a team's long-run level, and how far this season departs from it)

Fitted by maximum a posteriori with the Laplace approximation for the predictive distribution (the posterior of the strength difference is Gaussian, so the predicted
probability is E[sigmoid(eta)] ~ sigmoid(mu / sqrt(1 + pi v / 8))). tau and sigma are chosen by out-of-sample log loss on the LAST 20% of the training rows (empirical Bayes
on training data only). A season the model has not seen (the new season) gets B = 0 with extra predictive variance sigma^2, so early-season predictions fall back on the
long-run level instead of last season's final form. Uses no features other than who plays, where, and when: it is the "no engineered features" challenger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nhlbet.models.base import EPS, ProbModel, expit

GRID = [(t, s) for t in (0.08, 0.15, 0.25) for s in (0.05, 0.10, 0.20)]


class HierBT(ProbModel):
    name = "hier_bayes"

    def _index(self, X, fit: bool):
        if fit:
            self.teams_ = {t: i for i, t in enumerate(sorted(set(X.home) | set(X.away)))}
            self.seasons_ = {s: i for i, s in enumerate(sorted(X.season.unique()))}
        T, S = len(self.teams_), len(self.seasons_)
        h = X.home.map(self.teams_).fillna(-1).astype(int).to_numpy()
        a = X.away.map(self.teams_).fillna(-1).astype(int).to_numpy()
        s = X.season.map(self.seasons_).fillna(-1).astype(int).to_numpy()
        return h, a, s, T, S

    @staticmethod
    def _design(h, a, s, T, S):
        n = len(h)
        D = np.zeros((n, 1 + T + T * S))
        D[:, 0] = 1.0
        ok = h >= 0
        D[np.flatnonzero(ok), 1 + h[ok]] += 1.0
        ok = a >= 0
        D[np.flatnonzero(ok), 1 + a[ok]] -= 1.0
        oks = (s >= 0) & (h >= 0)
        D[np.flatnonzero(oks), 1 + T + h[oks] * S + s[oks]] += 1.0
        oks = (s >= 0) & (a >= 0)
        D[np.flatnonzero(oks), 1 + T + a[oks] * S + s[oks]] -= 1.0
        return D

    def _map(self, D, y, tau, sig, T, S, w0=None):
        prior = np.r_[0.0, np.full(T, 1 / tau ** 2), np.full(T * S, 1 / sig ** 2)]
        f = lambda w: (np.sum(np.logaddexp(0, D @ w) - y * (D @ w)) + 0.5 * np.sum(prior * w * w))
        g = lambda w: D.T @ (expit(D @ w) - y) + prior * w
        r = minimize(f, np.zeros(D.shape[1]) if w0 is None else w0, jac=g, method="L-BFGS-B", options={"maxiter": 300})
        return r.x, prior

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        order = np.argsort(pd.to_datetime(X.game_date).to_numpy(), kind="stable")
        X, y = X.iloc[order], y[order]
        h, a, s, T, S = self._index(X, True)
        D = self._design(h, a, s, T, S)
        cut = int(len(y) * 0.8)
        best, best_ll = GRID[0], np.inf
        for tau, sig in GRID:
            w, _ = self._map(D[:cut], y[:cut], tau, sig, T, S)
            p = np.clip(expit(D[cut:] @ w), EPS, 1 - EPS)
            ll = -np.mean(y[cut:] * np.log(p) + (1 - y[cut:]) * np.log(1 - p))
            if ll < best_ll:
                best, best_ll = (tau, sig), ll
        self.tau_, self.sig_ = best
        self.w_, prior = self._map(D, y, *best, T, S)
        p = expit(D @ self.w_)
        H = D.T @ (D * (p * (1 - p))[:, None]) + np.diag(prior + 1e-9)
        self.cov_ = np.linalg.inv(H)
        self.T_, self.S_ = T, S
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        h, a, s, T, S = self._index(X, False)
        D = self._design(h, a, s, T, S)
        mu = D @ self.w_
        v = np.einsum("ij,jk,ik->i", D, self.cov_, D) + np.where(s < 0, 2 * self.sig_ ** 2, 0.0)       # unseen season: both teams' season offsets are unknown
        return np.clip(expit(mu / np.sqrt(1 + np.pi * v / 8)), EPS, 1 - EPS)
