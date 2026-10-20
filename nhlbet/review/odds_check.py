"""Odds movement, computed in code before the agent runs, and the deterministic veto that does not depend on what the agent says.

Fresh prices come from the same source the pipeline already uses (The Odds API through ``OddsClient``): one bulk call for moneylines, totals and puck
lines (3 credits). Player prices are not re-fetched (a call per game and market would burn the monthly credits), so those picks carry
``status: not_refreshed`` and can only be judged on the price they had at the last capture. The edge is recomputed exactly as the pipeline does it:
the model's probability (unchanged) minus the Shin no-vig consensus probability across books at the pick's own line.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from nhlbet.odds.math import best_price, consensus_prob
from nhlbet.odds.snapshots import parse_events

log = logging.getLogger(__name__)
REFRESHABLE = {"h2h", "totals", "spreads"}
MIN_CREDITS_TO_FETCH = 60            # do not spend the month's last credits on a review: they keep the closing-line snapshots alive
SEVERITY = {"keep": 0, "flag": 1, "veto": 2}


def fetch_current(min_credits: int = MIN_CREDITS_TO_FETCH):
    """(events, captured_at, remaining, note). ``events`` is None when no fresh prices could be bought, and ``note`` says why."""
    if not os.environ.get("ODDS_API_KEY"):
        return None, None, None, "ODDS_API_KEY is not set: prices not refreshed"
    from nhlbet.odds.client import OddsAPIError, OddsClient
    try:
        client = OddsClient()
        rem = client.remaining_credits()
        if rem is not None and rem < min_credits:
            return None, None, rem, f"only {rem} odds credits left (< {min_credits}): prices not refreshed"
        f = client.fetch_odds(("h2h", "spreads", "totals"))
    except OddsAPIError as e:
        return None, None, None, f"odds fetch failed ({str(e)[:100]}): prices not refreshed"
    if f.stale:
        return None, None, f.remaining, "the odds feed only returned stale data: prices not refreshed"
    return f.events, f.captured_at, f.remaining, "refreshed"


def _two_way(rows, game, market, side, point):
    """book -> (decimal of the pick, decimal of the other side) at the pick's own line, from parsed odds rows."""
    home, away = game["home"], game["away"]
    mine = {}
    other = {}
    for r in rows:
        if r["home"] != home or r["away"] != away or r["market"] != market:
            continue
        if market == "h2h":
            key = r["outcome"]
            tgt = home if side == "home" else away
            (mine if key == tgt else other)[r["book"]] = r["price"]
        elif market == "totals":
            if abs(r["point"] - point) > 1e-9:
                continue
            key = str(r["outcome"]).lower()
            (mine if key == side else other)[r["book"]] = r["price"]
        elif market == "spreads":
            team = home if side == "home" else away
            if r["outcome"] == team and abs(r["point"] - point) < 1e-9:
                mine[r["book"]] = r["price"]
            elif r["outcome"] != team and abs(r["point"] + point) < 1e-9:
                other[r["book"]] = r["price"]
    return {b: (mine[b], other[b]) for b in mine if b in other}


def odds_facts(picks: list[dict], games: dict, events, captured_at: str | None, min_edge: float, note: str = "") -> dict[str, dict]:
    """pick_id -> deterministic odds fact. ``veto`` is True when the refreshed edge is below ``min_edge``; ``floor`` is the lowest decision this fact forces."""
    rows = parse_events(events, captured_at or "") if events else []
    out = {}
    for p in picks:
        base = {"price_then": p["odds"], "edge_then": p["edge"], "p_market_then": p["p_market"], "min_edge": min_edge, "captured_at": captured_at}
        if p["market"] not in REFRESHABLE or not events:
            why = note or "player prices are not re-fetched (credit budget)" if p["market"] not in REFRESHABLE else (note or "prices not refreshed")
            out[p["pick_id"]] = {**base, "status": "not_refreshed", "veto": False, "floor": "keep",
                                 "summary": f"unconfirmed: current price not refreshed ({why}); the candidate's own price {p['odds']:.2f} and edge {p['edge'] * 100:+.1f} pts stand"}
            continue
        g = games[str(p["game_id"])]
        pairs = _two_way(rows, g, p["market"], p["side"], p["line"])
        if not pairs:
            out[p["pick_id"]] = {**base, "status": "line_gone", "veto": False, "floor": "flag",
                                 "summary": f"the line/price this pick was made at is no longer offered at any book ({captured_at}); it cannot be placed as logged"}
            continue
        pm = consensus_prob(pairs)
        _, best = best_price({b: v[0] for b, v in pairs.items()})
        edge_now = p["p_model"] - pm
        veto = edge_now < min_edge
        out[p["pick_id"]] = {**base, "status": "refreshed", "price_now": float(best), "p_market_now": float(pm), "edge_now": float(edge_now), "n_books": len(pairs),
                             "edge_change_pts": float((edge_now - p["edge"]) * 100), "veto": bool(veto), "floor": "veto" if veto else "keep",
                             "summary": (f"price {p['odds']:.2f} -> {best:.2f} (best of {len(pairs)} books); no-vig market probability {p['p_market'] * 100:.1f}% -> {pm * 100:.1f}%; "
                                         f"edge {p['edge'] * 100:+.1f} -> {edge_now * 100:+.1f} pts (minimum {min_edge * 100:.0f}) at {captured_at}")}
    return out


def combine(agent: str | None, fact: dict | None) -> tuple[str, str]:
    """(decision, decided_by): the more severe of the agent's decision and the deterministic floor. The agent can never soften a code veto."""
    floor = (fact or {}).get("floor", "keep")
    a = agent or "keep"
    if SEVERITY[floor] > SEVERITY[a]:
        return floor, "code"
    return a, "agent" if agent else "code"
