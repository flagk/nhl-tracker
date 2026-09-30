"""Parlay maths (reference implementation; the website's JavaScript mirrors it and is parity-tested).

Why parlays need guardrails: a parlay multiplies every leg's bookmaker margin, so unless each leg has a real edge it is a
worse bet than the same picks placed separately. Legs must also be independent (different games, never same-game).
"""
from __future__ import annotations

from math import prod
from typing import Mapping, Sequence


def parlay(legs: Sequence[Mapping], max_legs: int = 4) -> dict:
    """``legs``: dicts with ``game_id``, ``p`` (working win prob, already shrunk toward the market), ``best_decimal``
    and ``books`` (mapping book -> decimal for this leg's side).

    Returns the combined probability, fair/offered odds (offered = the best SINGLE book carrying every leg, since a parlay
    is placed at one book), the parlay's EV per $1, and the EV of betting the same legs separately at their best prices.
    """
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")
    if len(legs) > max_legs:
        raise ValueError(f"at most {max_legs} legs")
    ids = [l["game_id"] for l in legs]
    if len(set(ids)) != len(ids):
        raise ValueError("legs must come from different games (no same-game parlays: legs are correlated)")
    p = prod(l["p"] for l in legs)
    common = set.intersection(*(set(l["books"]) for l in legs))
    offered, book = 0.0, None
    for b in sorted(common):
        o = prod(l["books"][b] for l in legs)
        if o > offered:
            offered, book = o, b
    if book is None:
        offered = prod(l["best_decimal"] for l in legs)        # no single book has every leg: reference price only
    singles_ev = sum(l["p"] * l["best_decimal"] - 1 for l in legs) / len(legs)
    ev = p * offered - 1
    return {"legs": len(legs), "p": p, "fair_decimal": 1 / p, "offered_decimal": offered, "book": book,
            "single_book_available": book is not None, "ev": ev, "singles_ev": singles_ev, "better_as_singles": singles_ev >= ev}
