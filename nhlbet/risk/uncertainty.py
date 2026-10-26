"""Bankroll simulation with fractional Kelly under PARAMETER uncertainty.

The model's claim for a bet is p_model; the truth is  p_true = p_market + theta * (p_model - p_market),  where theta is how much of the model's disagreement with the market is
real. theta is not known: it is estimated from resolved games, so each simulated path first draws its own theta from the estimate's sampling distribution (a normal with the
estimate's clustered standard error), and only then draws bets. Stakes are a fraction of the CURRENT bankroll, sized with Kelly on the model's claim (what the live policy
does), scaled by the Kelly fraction and capped per bet. The price is the decision-time price, optionally with a haircut (the best price across books is not always
obtainable at the book you actually use).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_theta(p_model, p_market, y, dates, B: int = 2000, seed: int = 0) -> dict:
    """OLS slope of (y - p_market) on (p_model - p_market) through the origin, with a date-clustered bootstrap standard error."""
    pm, mk, y = np.asarray(p_model, float), np.asarray(p_market, float), np.asarray(y, float)
    x, r = pm - mk, y - mk
    slope = lambda i: float((x[i] * r[i]).sum() / max((x[i] ** 2).sum(), 1e-12))
    days = pd.Series(dates).to_numpy()
    uniq = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(seed)
    boots = np.array([slope(np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])) for _ in range(B)])
    return {"theta": slope(np.arange(len(y))), "se": float(boots.std()), "n": int(len(y)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def kelly_frac(p, dec):
    return np.clip((p * dec - 1) / (dec - 1), 0.0, 1.0)


def simulate_uncertain(template: pd.DataFrame, theta: float, theta_se: float, kelly_fraction: float = 0.25, max_bet_pct: float = 0.02, price_haircut: float = 0.0,
                       n_paths: int = 10000, horizon: int = 500, bankroll0: float = 1000.0, ruin_level: float = 0.5, dd_levels=(0.2, 0.3), seed: int = 0) -> dict:
    """``template``: rows with decimal, p_market, p_model (the bets that cleared the policy). Returns summary statistics over paths."""
    rng = np.random.default_rng(seed)
    T = template[["decimal", "p_market", "p_model"]].to_numpy(float)
    th = rng.normal(theta, theta_se, n_paths)
    idx = rng.integers(0, len(T), (n_paths, horizon))
    dec, pm, pc = T[idx, 0], T[idx, 1], T[idx, 2]
    dec_paid = 1 + (dec - 1) * (1 - price_haircut)
    p_true = np.clip(pm + th[:, None] * (pc - pm), 0.001, 0.999)
    frac = np.minimum(kelly_fraction * kelly_frac(pc, dec_paid), max_bet_pct)             # sized on the model's claim, at the price actually paid
    win = rng.uniform(size=(n_paths, horizon)) < p_true
    bank = np.full(n_paths, bankroll0)
    peak = bank.copy(); maxdd = np.zeros(n_paths); ruined = np.zeros(n_paths, bool)
    for t in range(horizon):
        stake = frac[:, t] * bank
        bank = bank + np.where(win[:, t], stake * (dec_paid[:, t] - 1), -stake)
        peak = np.maximum(peak, bank)
        maxdd = np.maximum(maxdd, (peak - bank) / peak)
        ruined |= bank <= ruin_level * bankroll0
    out = {"paths": n_paths, "horizon": horizon, "kelly_fraction": kelly_fraction, "max_bet_pct": max_bet_pct, "price_haircut": price_haircut, "theta": theta, "theta_se": theta_se,
           "median_final": float(np.median(bank)), "p05_final": float(np.percentile(bank, 5)), "p95_final": float(np.percentile(bank, 95)), "p_loss": float((bank < bankroll0).mean()),
           "p_ruin": float(ruined.mean()), "median_max_drawdown": float(np.median(maxdd)), "p95_max_drawdown": float(np.percentile(maxdd, 95)), "share_paths_theta_positive": float((th > 0).mean())}
    for lv in dd_levels:
        out[f"p_drawdown_{int(lv * 100)}"] = float((maxdd >= lv).mean())
    return out
