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


SCENARIOS = {"model true": 1.0, "model shrunk halfway to the market": 0.5, "market true (no skill)": 0.0}


def simulate(template: pd.DataFrame, n_paths: int = 5000, horizon: int = 500, skills: Sequence[float] = (0.0, 0.5, 1.0),
             bankroll0: float = 1000.0, ruin_level: float = 0.5, dd_levels: Sequence[float] = (0.2, 0.3, 0.5), seed: int = 0, kill_dd: float | None = None) -> pd.DataFrame:
    """One summary row per skill level. ``ruin_level``: 'ruin' = bankroll ever falling to this share of the start. ``kill_dd``: the policy's kill-switch drawdown.

    The model's claim is ``p_model`` when the template has it (else ``p_adj``); the true win probability is ``p_market + skill * (claim - p_market)``.
    """
    rng = np.random.default_rng(seed)
    claim = "p_model" if "p_model" in template else "p_adj"
    T = template[["stake_pct", "decimal", "p_market", claim]].to_numpy(float)
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
        row["p90_maxdd"], row["p99_maxdd"] = float(np.percentile(max_dd, 90)), float(np.percentile(max_dd, 99))
        if kill_dd is not None:
            row["p_kill_switch"] = float(np.mean(max_dd >= kill_dd))           # the share of paths in which the policy's drawdown kill switch would trip
        rows.append(row)
    return pd.DataFrame(rows).set_index("skill")


def template_from_picks(d: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Bet template from real moneyline picks (columns p_model, p_market, decimal): those clearing the policy's edge floor, staked exactly as the live policy would (shrink toward the
    market, quarter Kelly, per-bet cap)."""
    from nhlbet.risk.kelly import kelly_fraction
    from nhlbet.risk.policy import RiskConfig
    from nhlbet.risk.shrink import shrink_to_market
    cfg = cfg or RiskConfig()
    e = d.p_model - d.p_market
    x = d[(e >= cfg.min_edge) & (e.abs() <= cfg.max_disagreement) & d.decimal.between(1.01, cfg.max_decimal) & (d.p_model >= cfg.min_prob)].copy()
    if x.empty:
        return x
    x["p_adj"] = [shrink_to_market(p, m, cfg.trust_w0, cfg.trust_d0) for p, m in zip(x.p_model, x.p_market)]
    x["stake_pct"] = [min(cfg.kelly_fraction * kelly_fraction(p, dec), cfg.max_bet_pct) for p, dec in zip(x.p_adj, x.decimal)]
    return x[x.stake_pct > 0][["stake_pct", "decimal", "p_market", "p_model", "p_adj"]].reset_index(drop=True)
