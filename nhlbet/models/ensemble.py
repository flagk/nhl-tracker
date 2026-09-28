"""Ensembles of base-model probabilities: logit-stacking and a non-negative weighted average."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from nhlbet.models.base import EPS, expit, logit
from nhlbet.models.calibration import log_loss_


class Stacker:
    """Logistic regression on the base models' logits (meta-learner). Fit on *out-of-fold* predictions only."""

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def fit(self, P: pd.DataFrame, y) -> "Stacker":
        self.cols_ = list(P.columns)
        self.m_ = LogisticRegression(C=self.C, max_iter=1000).fit(logit(P.to_numpy()), np.asarray(y, int))
        return self

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        return self.m_.predict_proba(logit(P[self.cols_].to_numpy()))[:, 1]

    def weights(self) -> pd.Series:
        return pd.Series(self.m_.coef_[0], index=self.cols_)


class WeightedAverage:
    """Convex combination of probabilities with weights minimising log loss on out-of-fold predictions."""

    def fit(self, P: pd.DataFrame, y) -> "WeightedAverage":
        self.cols_ = list(P.columns)
        A, yv = P.to_numpy(), np.asarray(y, float)
        k = A.shape[1]
        res = minimize(lambda w: log_loss_(A @ w, yv), np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        self.w_ = res.x if res.success else np.full(k, 1 / k)
        return self

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        return np.clip(P[self.cols_].to_numpy() @ self.w_, EPS, 1 - EPS)

    def weights(self) -> pd.Series:
        return pd.Series(self.w_, index=self.cols_)
