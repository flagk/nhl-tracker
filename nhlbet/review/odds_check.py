"""Odds movement, computed in code before the agent runs, and the deterministic veto that does not depend on what the agent says.

Fresh prices come from the same source the pipeline already uses (The Odds API through ``OddsClient``): one bulk call for moneylines, totals and puck
lines (3 credits). Player prices are not re-fetched (a call per game and market would burn the monthly credits), so those picks carry
``status: not_refreshed`` and can only be judged on the price they had at the last capture. The edge is recomputed exactly as the pipeline does it:
the model's probability (unchanged) minus the Shin no-vig consensus probability across books at the pick's own line.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nhlbet.odds.math import best_price, consensus_prob
from nhlbet.odds.snapshots import parse_events
from nhlbet.policy import policy_get

log = logging.getLogger(__name__)
REFRESHABLE = {"h2h", "totals", "spreads"}
SEVERITY = {"keep": 0, "flag": 1, "veto": 2}


def _used_series(root: str | Path) -> list[tuple[str, int]]:
    """(captured_at, API 'used' counter) of every live call logged under data/logs/odds_fetch_log, oldest first."""
    rows = []
    for f in sorted((Path(root) / "data" / "logs" / "odds_fetch_log").glob("*.csv")):
        try:
            for ln in f.read_text().splitlines()[1:]:
                c = ln.split(",")
                if len(c) >= 5 and c[2] == "live" and c[4]:
                    rows.append((c[0], int(float(c[4]))))
        except (OSError, ValueError):
            continue
    return sorted(rows)


def credits_spent_today(root: str | Path, now: datetime) -> int:
    """Credits used since 00:00 UTC today, from the committed fetch logs (the API's own counter), so review waves and the daily jobs share one tally."""
    day = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    rows = _used_series(root)
    if not rows:
        return 0
    cur = max(u for _, u in rows)
    before = [u for c, u in rows if c < day]
    base = max(before) if before else (min(u for c, u in rows if c >= day) - 1)
    return int(max(0, cur - base))


def daily_budget_left(root: str | Path, remaining: int | None, now: datetime) -> int | None:
    """Credits the day can still spend beyond the closing-line snapshots, by the same formula the daily job uses for player prices (spread what is left over the days left)."""
    if remaining is None:
        return None
    res, core = int(policy_get("odds", "reserve_credits", default=25)), int(policy_get("odds", "core_after_late", default=4))
    nxt = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
    days_left = max(1, (nxt.date() - now.date()).days)
    return int(max(0, remaining - res) / days_left - credits_spent_today(root, now) - core)


def record_credit_spend(root: str | Path, captured_at: str, remaining, used, note: str) -> Path:
    """Append the call to the committed odds-fetch log (same format the daily job writes) so the next run's budget counts it."""
    d = Path(root) / "data" / "logs" / "odds_fetch_log"
    d.mkdir(parents=True, exist_ok=True)
    p = d / (captured_at.replace(":", "-").replace("+", "-") + ".csv")
    p.write_text("captured_at,ok,source,remaining,used,events,note\n" + f"{captured_at},1,live,{'' if remaining is None else remaining},{'' if used is None else used},,{note}\n")
    return p


def fetch_current(root: str | Path = ".", now: datetime | None = None):
    """(events, captured_at, remaining, note). ``events`` is None when no fresh prices could be bought, and ``note`` says why. A live call is recorded in the credit log."""
    now = now or datetime.now(timezone.utc)
    if not os.environ.get("ODDS_API_KEY"):
        return None, None, None, "ODDS_API_KEY is not set: prices not refreshed"
    from nhlbet.odds.client import OddsAPIError, OddsClient
    need = int(policy_get("review", "credits_per_wave", default=3))
    floor = int(policy_get("review", "min_credits_to_fetch", default=60))
    try:
        client = OddsClient()
        rem = client.remaining_credits()
        if rem is not None and rem < floor:
            return None, None, rem, f"only {rem} odds credits left (< {floor}): prices not refreshed"
        left = daily_budget_left(root, rem, now)
        if left is not None and left < need:
            return None, None, rem, f"today's odds-credit budget has {max(left, 0)} credits left (< {need}): prices not refreshed"
        f = client.fetch_odds(("h2h", "spreads", "totals"))
    except OddsAPIError as e:
        return None, None, None, f"odds fetch failed ({str(e)[:100]}): prices not refreshed"
    if f.stale:
        return None, None, f.remaining, "the odds feed only returned stale data: prices not refreshed"
    if f.source == "live":
        record_credit_spend(root, f.captured_at, f.remaining, f.used, "review:h2h,spreads,totals")
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
