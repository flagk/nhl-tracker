"""Monte Carlo risk of ruin under explicit assumptions about how much edge is REAL.

Each simulated path draws bets (with replacement) from a template of (stake fraction, odds, model-adjusted prob,
market prob). The *true* win probability of a bet is  p_true = p_market + skill * (p_adj - p_market):
``skill = 0`` -> the model has no edge at all (only the vig hurts); ``skill = 1`` -> the model is exactly right.
Stakes are re-sized as a fixed fraction of the *current* bankroll, like the live policy. This answers
"what could happen if I follow the policy?", not "what will happen".
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def default_template(edge: float = 0.03, decimal: float = 1.95, stake_pct: float = 0.01, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Illustrative bet template when there is no bet history yet: ``edge`` over the no-vig market at ``decimal`` odds."""
    rng = np.random.default_rng(seed)
    dec = decimal + rng.normal(0, 0.1, n)
    p_mkt = 1 / dec / 1.045                                   # ~4.5% overround removed
    return pd.DataFrame({"stake_pct": stake_pct, "decimal": dec, "p_market": p_mkt, "p_adj": p_mkt + edge})


def simulate(template: pd.DataFrame, n_paths: int = 5000, horizon: int = 500, skills: Sequence[float] = (0.0, 0.5, 1.0),
             bankroll0: float = 1000.0, ruin_level: float = 0.5, dd_levels: Sequence[float] = (0.2, 0.3, 0.5), seed: int = 0) -> pd.DataFrame:
    """One summary row per skill level. ``ruin_level``: 'ruin' = bankroll ever falling to this share of the start."""
    rng = np.random.default_rng(seed)
    T = template[["stake_pct", "decimal", "p_market", "p_adj"]].to_numpy(float)
    rows = []
    for skill in skills:
        idx = rng.integers(0, len(T), (n_paths, horizon))
        stake_pct, dec, pm, pa = (T[idx, k] for k in range(4))
        p_true = np.clip(pm + skill * (pa - pm), 0.001, 0.999)
        wins = rng.random((n_paths, horizon)) < p_true
        bank = np.full(n_paths, bankroll0)
        peak = bank.copy(); min_rel = np.ones(n_paths); max_dd = np.zeros(n_paths)
        for t in range(horizon):
            stake = bank * stake_pct[:, t]
            bank = bank + np.where(wins[:, t], stake * (dec[:, t] - 1), -stake)
            peak = np.maximum(peak, bank)
            max_dd = np.maximum(max_dd, (peak - bank) / peak)
            min_rel = np.minimum(min_rel, bank / bankroll0)
        row = {"skill": skill, "paths": n_paths, "bets": horizon, "median_final": float(np.median(bank)),
               "p05_final": float(np.percentile(bank, 5)), "p95_final": float(np.percentile(bank, 95)),
               "p_loss": float(np.mean(bank < bankroll0)), f"p_ruin_{int(ruin_level * 100)}pct": float(np.mean(min_rel <= ruin_level))}
        for lv in dd_levels:
            row[f"p_maxdd_ge_{int(lv * 100)}"] = float(np.mean(max_dd >= lv))
        row["median_maxdd"] = float(np.median(max_dd))
        rows.append(row)
    return pd.DataFrame(rows).set_index("skill")
