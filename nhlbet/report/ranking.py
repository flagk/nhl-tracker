"""One ranked list of every pick the model can price today, across moneylines, totals, puck lines and player props.

This is a RANKING, not a recommendation. Only the moneyline policy can recommend a stake; everything else here is experimental and
shown so the day's picks can be compared in one order.

Score = confidence-weighted edge, in probability points: ``weight * (model probability - market probability)``. The weight is how far that
kind of model is trusted (the moneyline model is the only one with a track record, and its probability is already shrunk toward the market by
the live policy; the experimental models get much less), so a huge edge from a new, unvalidated model cannot jump the queue on its own.
It deliberately does NOT rank by expected value per dollar: EV scales with the price, so long shots (anytime scorers at 20/1) and one
bookmaker's outlier price would crowd out everything else. EV at the best price is shown as its own column.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from nhlbet.odds.props import PROP_MAX_AGE_MIN, STATS  # noqa: F401  (STATS documents the stat names used below)
from nhlbet.risk.policy import RiskConfig, assess_sides

# How far each kind of model is trusted (1 = take its probability at face value). Experimental markets have no betting track record yet.
# Player shots have a walk-forward backtest (calibrated against each player's own history) but no proof against the market; points, assists and
# goals have no backtest at all yet, and their first live prices disagreed with the market mostly in one direction (too many Unders), so they get the least trust.
WEIGHTS = {"moneyline": 1.0, "totals": 0.5, "spreads": 0.5, "player_sog": 0.3, "player_points": 0.2, "player_assists": 0.2, "player_goals": 0.15}
TYPE_NAMES = {"moneyline": "Moneyline", "totals": "Total", "spreads": "Puck line", "player_sog": "Player shots", "player_points": "Player points",
              "player_assists": "Player assists", "player_goals": "Anytime goalscorer"}
PROP_MIN_HISTORY = 10          # a player prop needs this many earlier games to be ranked at all


def _ev(p_win: float, p_push: float, dec: float) -> float:
    return p_win * (dec - 1.0) - (1.0 - p_win - p_push)


def _row(kind: str, s, label: str, p_model: float, p_market: float, dec: float, p_push: float, ev_raw: float, w: float, leg: dict | None = None, **extra) -> dict:
    score = w * (p_model - p_market)
    return {"kind": kind, "type": TYPE_NAMES[kind], "game_id": s.game_id, "game": f"{s.away} @ {s.home}", "start_utc": s.start_utc, "pick": label,
            "p_model": p_model, "p_market": p_market, "price": dec, "ev_raw": ev_raw, "score": score, "weight": w,
            "experimental": kind != "moneyline", "leg": {"game_id": s.game_id, "decimal": dec, "p_model": p_model, "p_market": p_market, **(leg or {})}, **extra}


def rank_picks(slate: Sequence, cfg: RiskConfig, limit: int = 80, now=None) -> list[dict]:
    """Every priced pick (the better side of each market per game, the better side per player prop), best score first, with rank 1..n."""
    rows: list[dict] = []
    for s in slate:
        if s.odds_stale or s.ctx.get("model_status") == "ALERT":
            continue                                                       # stale prices or a failing model: nothing to rank
        if now is not None and s.start_utc is not None and s.start_utc == s.start_utc and pd.Timestamp(s.start_utc) <= pd.Timestamp(now):
            continue                                                       # already started: that bet can no longer be placed
        if s.quotes:
            best = None
            for q, p_adj, ev, fails in assess_sides(s.quotes, cfg, s.ctx):
                cand = _row("moneyline", s, f"{q.team} moneyline", p_adj, q.market_prob, q.best_decimal, 0.0, ev, WEIGHTS["moneyline"],
                            recommended=bool(s.rec.action == "BET" and s.rec.side == q.side), note=("; ".join(fails) if fails else "clears the policy checks"),
                            leg={"market": "h2h", "side": q.side, "point": None, "player_id": 0})
                if best is None or cand["score"] > best["score"]:
                    best = cand
            if best:
                rows.append(best)
        for market in ("totals", "spreads"):
            qs = [q for q in (s.alt or []) if q.market == market]
            cands = [_row(market, s, q.label, q.model_prob, q.market_prob, q.best_decimal, q.p_push, q.ev, WEIGHTS[market], recommended=False,
                          note="experimental goals model, paper-traded only", leg={"market": market, "side": q.side, "point": q.point, "player_id": 0}) for q in qs]
            if cands:
                rows.append(max(cands, key=lambda r: r["score"]))
        for q in s.props or []:
            if q.n_prev < PROP_MIN_HISTORY:
                continue
            kind = STATS[q.stat]["store"]
            cands = []
            for side in ("over", "under"):
                if not q.has_side(side):
                    continue
                over = side == "over"
                p, m = (q.p_over, q.p_over_market) if over else (1 - q.p_over, 1 - q.p_over_market)
                dec = q.over_price if over else q.under_price
                label = f"{q.name} to score (anytime)" if q.one_sided else f"{q.name} {'Over' if over else 'Under'} {q.point:g} {STATS[q.stat]['noun']}"
                cands.append(_row(kind, s, label, p, m, dec, 0.0, q.ev_over if over else q.ev_under, WEIGHTS[kind], recommended=False,
                                  note="experimental player model, paper-traded only", stat=q.stat, player_id=q.player_id,
                                  leg={"market": kind, "side": side, "point": q.point, "player_id": int(q.player_id)}))
            if cands:
                rows.append(max(cands, key=lambda r: r["score"]))
    rows.sort(key=lambda r: (-r["score"], r["game_id"], r["pick"]))
    rows = rows[:limit]
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["has_edge"] = r["score"] > 0
    return rows
