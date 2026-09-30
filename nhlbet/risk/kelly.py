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


def stake(bankroll: float, p: float, decimal: float, fraction: float = 0.25, max_pct: float = 0.02) -> float:
    """Fractional-Kelly stake in currency, capped at ``max_pct`` of bankroll. Never negative."""
    if bankroll < 0:
        raise ValueError("bankroll must be >= 0")
    if not 0 < fraction <= 1:
        raise ValueError("kelly fraction must be in (0, 1]")
    f = min(fraction * kelly_fraction(p, decimal), max_pct)
    return bankroll * f
