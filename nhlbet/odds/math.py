"""Odds conversion, vig removal, best price, edge and expected value. Pure functions, fully unit-tested.

Conventions: ``decimal`` odds include the stake (2.00 = even money). ``implied`` = 1 / decimal (includes vig).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


# ---- conversions -------------------------------------------------------------------------------
def american_to_decimal(a: float) -> float:
    a = float(a)
    if abs(a) < 100:
        raise ValueError(f"invalid American odds: {a}")
    return 1 + (a / 100 if a > 0 else 100 / -a)


def decimal_to_american(d: float) -> float:
    d = float(d)
    if d <= 1:
        raise ValueError(f"decimal odds must be > 1, got {d}")
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def implied_prob(decimal: float) -> float:
    if decimal <= 1:
        raise ValueError(f"decimal odds must be > 1, got {decimal}")
    return 1.0 / decimal


def overround(decimals: Sequence[float]) -> float:
    """Bookmaker margin: sum of implied probabilities minus 1 (e.g. -110/-110 -> 0.0476)."""
    return float(sum(1.0 / d for d in decimals) - 1.0)


# ---- vig removal -------------------------------------------------------------------------------
def devig_proportional(decimals: Sequence[float]) -> np.ndarray:
    """Divide each implied probability by the booksum (multiplicative / basic normalisation)."""
    q = 1.0 / np.asarray(decimals, float)
    return q / q.sum()


def devig_shin(decimals: Sequence[float], tol: float = 1e-12) -> np.ndarray:
    """Shin (1993) method: assumes a share ``z`` of money is from insiders; corrects favourite-longshot bias.

    Solves for z in [0, 1) such that the implied 'true' probabilities sum to 1. Falls back to the
    proportional method when there is no margin (booksum <= 1).
    """
    q = 1.0 / np.asarray(decimals, float)
    s = q.sum()
    if s <= 1.0 + 1e-12:
        return q / s

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z))

    lo, hi = 0.0, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if probs(mid).sum() > 1:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    p = probs((lo + hi) / 2)
    return p / p.sum()


def devig(decimals: Sequence[float], method: str = "shin") -> np.ndarray:
    if method == "proportional":
        return devig_proportional(decimals)
    if method == "shin":
        return devig_shin(decimals)
    raise ValueError(f"unknown devig method {method!r}")


# ---- best price / consensus --------------------------------------------------------------------
def best_price(prices_by_book: Mapping[str, float]) -> tuple[str, float]:
    """(book, decimal) with the highest payout."""
    if not prices_by_book:
        raise ValueError("no prices")
    book = max(prices_by_book, key=lambda b: prices_by_book[b])
    return book, float(prices_by_book[book])


def consensus_prob(two_way: Mapping[str, tuple[float, float]], method: str = "shin") -> float | None:
    """Mean no-vig probability of side A across books. ``two_way``: book -> (decimal_A, decimal_B)."""
    ps = [devig(pair, method)[0] for pair in two_way.values() if pair[0] > 1 and pair[1] > 1]
    return float(np.mean(ps)) if ps else None


# ---- edge / EV ---------------------------------------------------------------------------------
def expected_value(p: float, decimal: float) -> float:
    """Expected profit per $1 staked if the true win probability is ``p`` at ``decimal`` odds."""
    return p * (decimal - 1.0) - (1.0 - p)


def edge(p_model: float, p_market_novig: float) -> float:
    """Model probability minus the no-vig market probability (in probability points)."""
    return p_model - p_market_novig


def breakeven_prob(decimal: float) -> float:
    return 1.0 / decimal


def clv(bet_decimal: float, close_prob_novig: float) -> float:
    """Closing-line value as EV of the bet price against the no-vig closing probability.

    Positive = you beat the closing line (the market later agreed the price was too generous).
    Equivalent to (bet_decimal * p_close) - 1.
    """
    return bet_decimal * close_prob_novig - 1.0
