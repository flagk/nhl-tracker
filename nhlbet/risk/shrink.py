"""Shrink the model's probability toward the market when they disagree a lot.

Why: the market aggregates far more information than this model. A small disagreement can be real edge; a
large one is much more likely a model error (stale feature, missing injury, bad goalie assumption). So trust
in the model *falls* as the disagreement grows:

    w(d) = w0 / (1 + (|d| / d0)^2),   p_adj = p_market + w(d) * (p_model - p_market)

``w0`` (<1) is baseline humility: we have not demonstrated skill against the market. Once enough odds/results
accumulate, ``estimate_trust`` fits the optimal blend out-of-sample so ``w0`` is data-driven, not a guess.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from nhlbet.models.base import logit


def trust_weight(disagreement: float, w0: float = 0.5, d0: float = 0.06) -> float:
    return w0 / (1.0 + (abs(disagreement) / d0) ** 2)


def shrink_to_market(p_model: float, p_market: float, w0: float = 0.5, d0: float = 0.06) -> float:
    d = p_model - p_market
    return float(np.clip(p_market + trust_weight(d, w0, d0) * d, 0.01, 0.99))


def estimate_trust(p_model, p_market, y) -> float:
    """Data-driven ``w0``: coefficient on the model-vs-market logit gap in a logistic regression of outcomes on
    ``logit(p_market)`` (coefficient forced ~1 by fitting the residual) - clipped to [0, 1].

    Needs several hundred resolved games with market prices; returns the weight that maximises out-of-sample log-likelihood
    among blends ``logit(p) = logit(p_market) + w * (logit(p_model) - logit(p_market))``.
    """
    lm, lp = logit(p_market), logit(p_model)
    gap, y = lp - lm, np.asarray(y, int)
    best_w, best_ll = 0.0, -np.inf
    for w in np.linspace(0, 1.5, 61):
        z = lm + w * gap
        ll = np.sum(y * -np.log1p(np.exp(-z)) + (1 - y) * -np.log1p(np.exp(z)))
        if ll > best_ll:
            best_w, best_ll = float(w), ll
    return float(np.clip(best_w, 0.0, 1.0))
