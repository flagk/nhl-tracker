"""Settlement rules, per book, from config/policy.yaml. One place decides how a shootout counts toward a total, so the model's goal distribution, the settlement of
paper bets and the exports can never disagree about it.

Sources and what is verified: docs/SETTLEMENT.md. A book whose rules page could not be read is applied with the industry convention and reported as ``verified: false``.
"""
from __future__ import annotations

import numpy as np

from nhlbet.policy import policy_get


VOID_STATES = frozenset({"PPD", "CNCL", "CANCELLED", "POSTPONED"})      # game states in which no bet can be graded: stakes are refunded


def is_void_game(status, planned_date, actual_date) -> bool:
    """A bet is void when its game is postponed / cancelled, or was moved to a different date than the one the pick was made for (books void or re-price those)."""
    if str(status or "").upper() in VOID_STATES:
        return True
    pd_, ad = str(planned_date or "")[:10], str(actual_date or "")[:10]
    iso = lambda x: len(x) == 10 and x[4] == "-" and x[7] == "-"
    return bool(iso(pd_) and iso(ad) and pd_ != ad)                    # an unreadable date proves nothing, so it never voids a bet


def rules_for(book: str | None = None) -> dict:
    """The rule set (a dict of booleans) that applies to ``book``; unknown or blank books get the default."""
    books = policy_get("settlement", "books", default={}) or {}
    entry = books.get((book or "").lower()) or books.get("default") or {}
    name = entry.get("ruleset") or policy_get("settlement", "default_ruleset", default="so_adds_goal")
    sets = policy_get("settlement", "rulesets", default={}) or {}
    if name not in sets:
        raise KeyError(f"settlement ruleset {name!r} is not defined in config/policy.yaml")
    return {**sets[name], "ruleset": name, "verified": bool(entry.get("verified", False))}


def shootout_in_total(book: str | None = None) -> bool:
    return bool(rules_for(book)["shootout_in_total"])


def game_total(home_score, away_score, last_period, book: str | None = None):
    """Goals that count toward the game total (scalars or arrays). Official scores already include the shootout winner's extra goal, so under ``so_excluded``
    a shootout game subtracts it."""
    so = (np.asarray(last_period) == "SO").astype(float)
    out = np.asarray(home_score, float) + np.asarray(away_score, float) - (0.0 if shootout_in_total(book) else so)
    return float(out) if out.ndim == 0 else out
