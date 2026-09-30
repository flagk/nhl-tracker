"""Combine a model probability with market prices: best price per side, no-vig market probability, edge and EV."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nhlbet.odds.math import best_price, breakeven_prob, devig, expected_value


@dataclass
class SideQuote:
    side: str                 # 'home' | 'away'
    team: str
    model_prob: float
    market_prob: float        # consensus no-vig probability for this side
    best_book: str
    best_decimal: float
    breakeven: float          # implied probability of the best price (includes vig)
    edge: float               # model_prob - market_prob
    ev: float                 # expected profit per $1 at the best price using model_prob
    n_books: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_game(p_home: float, home: str, away: str, prices: pd.DataFrame, method: str = "shin") -> dict | None:
    """``prices``: per-book rows with ``book``, ``home``, ``away`` decimal odds (see ``latest_book_prices``).

    Returns both sides' quotes, or None when there are no usable prices (-> 'no bet: no odds')."""
    prices = prices.dropna(subset=["home", "away"])
    prices = prices[(prices.home > 1) & (prices.away > 1)]
    if prices.empty:
        return None
    p_mkt_home = float(np.mean([devig([h, a], method)[0] for h, a in zip(prices.home, prices.away)]))
    out = {}
    for side, team, p, pm, col in (("home", home, p_home, p_mkt_home, "home"), ("away", away, 1 - p_home, 1 - p_mkt_home, "away")):
        book, dec = best_price(dict(zip(prices.book, prices[col])))
        out[side] = SideQuote(side, team, float(p), pm, book, dec, breakeven_prob(dec), float(p - pm),
                              expected_value(float(p), dec), int(len(prices)))
    return out
