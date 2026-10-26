"""Backtest-overfitting statistics.

``trials``: a matrix with one row per period (e.g. day) and one column per strategy/variant that was ever tried, holding that period's return per unit staked.

* ``pbo``            probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado, Zhu 2014): the share of ways to split the
                     periods into halves in which the variant that was best in-sample ranks below the median out-of-sample.
* ``deflated_sharpe`` the probability that the best variant's true Sharpe ratio is positive after accounting for how many variants were tried, and for skewness and
                     kurtosis (Bailey and Lopez de Prado 2014).
* ``reality_check``  White's (2000) bootstrap reality check for the best variant's mean, with Hansen's (2005) studentised SPA statistic: the probability of seeing a best variant
                     this good if no variant had any edge (stationary bootstrap over periods).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def pbo(trials: pd.DataFrame, splits: int = 16, max_combos: int = 3000, seed: int = 0) -> dict:
    X = trials.to_numpy(float)
    X = np.nan_to_num(X, nan=0.0)
    T, N = X.shape
    S = splits - splits % 2
    blocks = np.array_split(np.arange(T), S)
    combos = list(combinations(range(S), S // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    logits = []
    for c in combos:
        ins = np.concatenate([blocks[i] for i in c]); oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
        mi, mo = X[ins].mean(0), X[oos].mean(0)
        best = int(np.argmax(mi))
        rank = (stats.rankdata(mo)[best]) / (N + 1)              # relative OOS rank of the in-sample winner, in (0, 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "combinations": len(combos), "variants": int(N), "periods": int(T), "median_logit": float(np.median(logits))}


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum of n_trials Sharpe ratios under the null of no skill, given the cross-trial variance of Sharpe ratios."""
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(var_sharpe) * ((1 - EULER) * z(1 - 1 / n_trials) + EULER * z(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(returns: np.ndarray, n_trials: int, var_sharpe_across_trials: float) -> dict:
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    sr0 = expected_max_sharpe(n_trials, var_sharpe_across_trials)
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return {"sharpe_per_period": float(sr), "benchmark_sharpe_from_trials": float(sr0), "n_trials": int(n_trials), "periods": int(n), "skew": float(skew), "kurtosis": float(kurt),
            "dsr": float(stats.norm.cdf(z)), "psr_vs_zero": float(stats.norm.cdf(sr * np.sqrt(n - 1) / denom))}


def _stationary_indices(T: int, B: int, mean_block: int, rng) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((B, T), int)
    idx[:, 0] = rng.integers(0, T, B)
    jump = rng.uniform(size=(B, T)) < p
    new = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], new[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def reality_check(trials: pd.DataFrame, B: int = 2000, mean_block: int = 5, seed: int = 0) -> dict:
    """H0: no variant has a positive expected return. Returns White's reality-check p and Hansen's SPA p (studentised, with the consistent null centring)."""
    X = np.nan_to_num(trials.to_numpy(float), nan=0.0)
    T, N = X.shape
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(T, B, mean_block, rng)
    mean = X.mean(0)
    sd = X.std(0, ddof=1) / np.sqrt(T) + 1e-12
    boot = np.stack([X[idx[b]].mean(0) for b in range(B)])                      # B x N
    v_white = np.sqrt(T) * mean.max()
    p_white = float(((np.sqrt(T) * (boot - mean).max(1)) >= v_white).mean())
    t_obs = float(max(0.0, (mean / sd).max()))
    keep = (mean / sd) >= -np.sqrt(2 * np.log(np.log(max(T, 3))))                   # drop clearly poor variants from the null (Hansen's consistent centring)
    centred = boot - mean * keep
    t_boot = np.maximum(0.0, ((centred) / sd).max(1))
    p_spa = float((t_boot >= t_obs).mean())
    return {"best_variant": str(trials.columns[int(mean.argmax())]), "best_mean_per_period": float(mean.max()), "p_white_reality_check": p_white, "p_hansen_spa": p_spa, "variants": int(N), "periods": int(T)}
