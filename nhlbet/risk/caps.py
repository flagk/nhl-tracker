"""Exposure caps: per bet, per game (correlated exposure), per day, and a limit on the number of bets.

Every market on one game is correlated (the moneyline, puck line, total and the players' props all depend on the same game), so their stakes are summed and capped together.
Stakes are only ever scaled DOWN and rounded DOWN to the cent, never up, so no cap can be exceeded by rounding.
"""
from __future__ import annotations

import math


def _down(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def apply_caps(bets: list[dict], bankroll: float, max_bet_pct: float, max_game_pct: float, max_daily_pct: float, max_bets: int, min_stake: float) -> list[dict]:
    """``bets``: dicts with ``id``, ``game_id``, ``stake`` and ``ev``. Returns one dict per input (same order) with ``stake`` (after caps), ``dropped`` (bool), ``reason`` and ``notes``."""
    out = [{**b, "stake": float(b["stake"]), "dropped": False, "reason": "", "notes": []} for b in bets]
    live = [o for o in out]
    for o in live:                                                                           # per-bet cap
        cap = bankroll * max_bet_pct
        if o["stake"] > cap:
            o["stake"], o["notes"] = _down(cap), o["notes"] + [f"Stake capped at the per-bet limit ({max_bet_pct:.1%} of bankroll)."]
    ranked = sorted(live, key=lambda o: -o["ev"])                                              # at most max_bets, best EV first
    for o in ranked[max_bets:]:
        o["dropped"], o["reason"] = True, f"more than {max_bets} qualifying bets today; kept only the highest-EV ones"
    live = [o for o in live if not o["dropped"]]
    games: dict = {}
    for o in live:
        games.setdefault(o["game_id"], []).append(o)
    gcap = bankroll * max_game_pct
    for g in games.values():                                                                   # per-game cap: correlated markets share one limit
        tot = sum(o["stake"] for o in g)
        if tot > gcap > 0:
            for o in g:
                o["stake"] = _down(o["stake"] * gcap / tot)
                o["notes"].append(f"Scaled by {gcap / tot:.2f} to respect the {max_game_pct:.1%} per-game exposure cap (all markets on the game together).")
    tot, dcap = sum(o["stake"] for o in live), bankroll * max_daily_pct
    if tot > dcap > 0:
        scale = dcap / tot
        for o in live:
            o["stake"] = _down(o["stake"] * scale)
            o["notes"].append(f"Scaled by {scale:.2f} to respect the {max_daily_pct:.0%} daily exposure cap.")
    for o in live:
        if o["stake"] < min_stake:
            o["dropped"], o["reason"], o["stake"] = True, "stake fell below the minimum after the exposure caps", 0.0
    for o in out:
        if o["dropped"]:
            o["stake"] = 0.0
    return out
