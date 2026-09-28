"""Bankroll history, drawdown statistics and bet-log summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(bets: pd.DataFrame, start_bankroll: float) -> pd.DataFrame:
    """``bets``: resolved bets with ``date`` and ``profit`` (net of stake; a loss is -stake). Returns per-bet
    bankroll, running peak and drawdown (fraction below the peak)."""
    b = bets.sort_values("date").reset_index(drop=True)
    bank = start_bankroll + b["profit"].cumsum()
    peak = np.maximum.accumulate(np.r_[start_bankroll, bank.to_numpy()])[1:]
    return pd.DataFrame({"date": b["date"], "profit": b["profit"], "bankroll": bank, "peak": peak, "drawdown": (peak - bank) / peak})


def drawdown_stats(curve: pd.DataFrame) -> dict:
    if curve.empty:
        return {"max_drawdown": 0.0, "current_drawdown": 0.0, "longest_underwater_bets": 0}
    under = (curve.drawdown > 0).astype(int)
    run = longest = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return {"max_drawdown": float(curve.drawdown.max()), "current_drawdown": float(curve.drawdown.iloc[-1]), "longest_underwater_bets": int(longest)}


def summarize_bets(bets: pd.DataFrame) -> dict:
    """Staked, profit, ROI, win rate and average price of resolved bets (``stake``, ``profit``, ``decimal``)."""
    r = bets[bets.stake > 0]
    if r.empty:
        return {"n": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"), "win_rate": float("nan"), "avg_decimal": float("nan")}
    return {"n": int(len(r)), "staked": float(r.stake.sum()), "profit": float(r.profit.sum()), "roi": float(r.profit.sum() / r.stake.sum()),
            "win_rate": float((r.profit > 0).mean()), "avg_decimal": float(r.decimal.mean())}
