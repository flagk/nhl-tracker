"""Closing line value (CLV): did we get a better price than the market's closing consensus?

CLV is the most trustworthy long-run indicator of edge: it needs no results, so it converges far faster than ROI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.odds.math import clv as clv_ev


def closing_consensus(cons: pd.DataFrame, game_id: int, start_utc, min_before_minutes: float = 0.0) -> float | None:
    """No-vig home probability from the LAST consensus snapshot captured no later than puck drop (minus a buffer)."""
    c = cons[cons.game_id == game_id]
    if c.empty or pd.isna(start_utc):
        return None
    t = pd.to_datetime(start_utc, utc=True) - pd.Timedelta(minutes=min_before_minutes)
    c = c[pd.to_datetime(c.captured_at, utc=True) <= t].sort_values("captured_at")
    return float(c.home_prob_novig.iloc[-1]) if len(c) else None


def bet_clv(side: str, bet_decimal: float, close_home_prob: float) -> dict:
    """CLV of a bet on ``side`` at ``bet_decimal`` versus the closing no-vig probability."""
    p_close = close_home_prob if side == "home" else 1 - close_home_prob
    return {"close_prob": p_close, "bet_implied": 1 / bet_decimal, "clv_prob_pts": p_close - 1 / bet_decimal,
            "clv_ev": clv_ev(bet_decimal, p_close)}


def summarize_clv(clvs: pd.Series, B: int = 4000, seed: int = 0) -> dict:
    """Mean CLV (in EV per $) with a bootstrap CI and the share of bets that beat the close."""
    x = clvs.dropna().to_numpy(float)
    if len(x) < 5:
        return {"n": len(x), "mean": float(x.mean()) if len(x) else float("nan"), "ci": (float("nan"), float("nan")), "beat_close": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return {"n": len(x), "mean": float(x.mean()), "ci": tuple(np.percentile(means, [2.5, 97.5])), "beat_close": float((x > 0).mean())}
