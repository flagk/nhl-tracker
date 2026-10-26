"""Kelly staking."""
from __future__ import annotations


def kelly_fraction(p: float, decimal: float) -> float:
    """Full-Kelly fraction of bankroll for win probability ``p`` at decimal odds: f* = (b p - q) / b, floored at 0."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1, got {decimal}")
    b = decimal - 1.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def kelly_fraction_push(p_win: float, p_lose: float, decimal: float) -> float:
    """Full-Kelly fraction when the bet can also PUSH (whole-number total / puck line / prop line; the stake is returned on a push).

    Maximising p_win*ln(1+f*b) + p_lose*ln(1-f) gives f* = (b*p_win - p_lose) / (b*(p_win + p_lose)): a push is neutral, so only the conditional
    win / lose split matters for the fraction, but using the unconditional win probability with the plain formula would oversize the bet.
    """
    if p_win < 0 or p_lose < 0 or p_win + p_lose > 1.0 + 1e-9:
        raise ValueError(f"probabilities out of range: win {p_win}, lose {p_lose}")
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1, got {decimal}")
    if p_win + p_lose <= 0:
        return 0.0
    b = decimal - 1.0
    return max(0.0, (b * p_win - p_lose) / (b * (p_win + p_lose)))


def stake(bankroll: float, p: float, decimal: float, fraction: float = 0.25, max_pct: float = 0.02) -> float:
    """Fractional-Kelly stake in currency, capped at ``max_pct`` of bankroll. Never negative."""
    if bankroll < 0:
        raise ValueError("bankroll must be >= 0")
    if not 0 < fraction <= 1:
        raise ValueError("kelly fraction must be in (0, 1]")
    f = min(fraction * kelly_fraction(p, decimal), max_pct)
    return bankroll * f


def stake_push(bankroll: float, p_win: float, p_lose: float, decimal: float, fraction: float = 0.25, max_pct: float = 0.02) -> float:
    """Fractional-Kelly stake in currency for a bet that can push (whole-number or quarter lines): fraction * f*, capped at ``max_pct`` of bankroll, never negative."""
    if bankroll < 0:
        raise ValueError("bankroll must be >= 0")
    if not 0 < fraction <= 1:
        raise ValueError("kelly fraction must be in (0, 1]")
    return bankroll * min(fraction * kelly_fraction_push(p_win, p_lose, decimal), max_pct)


def ev_per_unit(p_win: float, p_lose: float, decimal: float) -> float:
    """Expected profit per unit staked when a push refunds the stake: b * p_win - p_lose with b = decimal - 1."""
    return (decimal - 1.0) * p_win - p_lose
