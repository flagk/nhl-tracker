"""Model zoo behind one tiny interface: ``fit(X, y)`` and ``predict(X) -> P(home win)``.

Everything takes a feature DataFrame (never a bare array) so column selection is explicit.
Hyper-parameters come from ``data/models/hyperparams.json`` (written by ``nhlbet.models.tune``,
which uses time-series CV only) with conservative defaults.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
HYPERPARAMS_PATH = Path("data/models/hyperparams.json")

DEFAULTS: dict[str, dict] = {
    "logistic": {"C": 0.1},
    "rf": {"n_estimators": 300, "min_samples_leaf": 30, "max_features": 0.5},
    "lgbm": {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 5, "min_child_samples": 60, "reg_lambda": 10.0},
    "xgb": {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 10, "reg_lambda": 10.0},
}


def load_params(name: str, path: str | Path = HYPERPARAMS_PATH) -> dict:
    p = Path(path)
    tuned = json.loads(p.read_text()).get(name, {}) if p.exists() else {}
    return {**DEFAULTS.get(name, {}), **tuned}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


class ProbModel:
    name = "base"

    def __init__(self, features: Sequence[str] | None = None, **params) -> None:
        self.features = list(features) if features is not None else None
        self.params = params

    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features] if self.features is not None else X

    def fit(self, X: pd.DataFrame, y) -> "ProbModel":  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HomeRate(ProbModel):
    """Baseline: predict the training home-win rate for every game."""
    name = "home_rate"

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.rate_)


class EloOnly(ProbModel):
    """Simple Elo-only model: the builder's pre-game Elo home-win probability, unmodified."""
    name = "elo"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.clip(X["elo_prob"].to_numpy(float), EPS, 1 - EPS)


class Logistic(ProbModel):
    name = "logistic"

    def fit(self, X, y):
        self.m_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                LogisticRegression(max_iter=1000, **{**load_params("logistic"), **self.params}))
        self.m_.fit(self._X(X), y)
        return self

    def predict(self, X):
        return self.m_.predict_proba(self._X(X))[:, 1]

    def coefficients(self) -> pd.Series:
        return pd.Series(self.m_[-1].coef_[0], index=self.features)


class RandomForest(ProbModel):
    """The original project's model family, kept as a baseline - but regularised (min_samples_leaf)."""
    name = "rf"

    def fit(self, X, y):
        self.m_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                RandomForestClassifier(random_state=0, n_jobs=1, **{**load_params("rf"), **self.params}))
        self.m_.fit(self._X(X), y)
        return self

    def predict(self, X):
        return self.m_.predict_proba(self._X(X))[:, 1]


class LGBM(ProbModel):
    name = "lgbm"

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        p = {**load_params("lgbm"), **self.params}
        self.m_ = LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1, **p)
        self.m_.fit(self._X(X), y)
        return self

    def predict(self, X):
        return self.m_.predict_proba(self._X(X))[:, 1]


class XGB(ProbModel):
    name = "xgb"

    def fit(self, X, y):
        from xgboost import XGBClassifier
        p = {**load_params("xgb"), **self.params}
        self.m_ = XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=1, verbosity=0, **p)
        self.m_.fit(self._X(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.m_.predict_proba(self._X(X))[:, 1]


Factory = Callable[[], ProbModel]


def make_zoo(features: Sequence[str], all_features: Sequence[str] | None = None) -> dict[str, Factory]:
    """name -> zero-arg factory. ``all_features`` adds an un-selected LightGBM as a selection sanity check."""
    zoo: dict[str, Factory] = {
        "home_rate": lambda: HomeRate(),
        "elo": lambda: EloOnly(),
        "logistic": lambda: Logistic(features),
        "rf": lambda: RandomForest(features),
        "lgbm": lambda: LGBM(features),
        "xgb": lambda: XGB(features),
    }
    if all_features is not None:
        zoo["lgbm_all"] = lambda: LGBM(all_features)
    return zoo
