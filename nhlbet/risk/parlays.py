"""AI-built parlays (fake money only): the model combines picks from different games into multi-leg tickets, and they are settled like any paper bet.

Legs always come from DIFFERENT games, so they are close to independent and the combined probability is just the product (same-game parlays are
correlated and priced differently by books, so they are not built). Parlays multiply both the payout and the bookmaker's margin, so the
no-skill control shows what that costs. Evaluate on ROI, never on one lucky ticket.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from nhlbet.risk.shadow import CONTROL_STAKE, paper_stake

MAX_PARLAYS_PER_STRATEGY = 1          # one ticket per strategy per run (the latest run of the day replaces earlier ones)


@dataclass(frozen=True)
class ParlayStrategy:
    name: str
    description: str
    legs: int
    source: str                       # 'top' = best-ranked picks | 'players' = best-ranked player props | 'favorites' = market favourites (control)


PARLAY_STRATEGIES: tuple[ParlayStrategy, ...] = (
    ParlayStrategy("parlay_top2", "The model's 2 best-ranked picks from different games combined into one ticket ($5-$30 by conviction)", 2, "top"),
    ParlayStrategy("parlay_top3", "The model's 3 best-ranked picks from different games combined into one ticket ($5-$30 by conviction)", 3, "top"),
    ParlayStrategy("parlay_players3", "The 3 best-ranked PLAYER props (shots, points, assists, anytime goals) from different games in one ticket ($5-$30 by conviction)", 3, "players"),
    ParlayStrategy("parlay_favorites_control", "CONTROL: flat $10 on the market favourites of 3 different games (no model; shows what parlay margin costs)", 3, "favorites"),
)


def _pick_legs(ranked: Sequence[dict], n: int, only_players: bool) -> list[dict]:
    """Best-ranked picks with a positive score, at most one per game, best first. Needs at least 2 to make a parlay."""
    out, seen = [], set()
    for r in ranked:
        if r["score"] <= 0 or r["game_id"] in seen or (only_players and not r["kind"].startswith("player_")):
            continue
        out.append(r)
        seen.add(r["game_id"])
        if len(out) == n:
            break
    return out


def _favorite_legs(ranked: Sequence[dict], games: Sequence, n: int) -> list[dict]:
    """The market's favourite side of up to n games (needs a priced moneyline), soonest game first. Built from the raw game quotes, not the ranking."""
    out = []
    for g in sorted((g for g in games if g.quotes and not g.odds_stale), key=lambda g: (str(g.start_utc), g.game_id)):
        q = max(g.quotes.values(), key=lambda x: x.market_prob)
        out.append({"label": f"{q.team} moneyline", "game": f"{g.away} @ {g.home}", "game_id": g.game_id, "score": 0.0,
                    "leg": {"game_id": g.game_id, "market": "h2h", "side": q.side, "point": None, "player_id": 0, "decimal": q.best_decimal,
                            "p_model": q.model_prob, "p_market": q.market_prob}, "pick": f"{q.team} moneyline"})
        if len(out) == n:
            break
    return out


def build_parlays(ranked: Sequence[dict], games: Sequence, run_id: str, run_at: str, date: str, strategies: Sequence[ParlayStrategy] = PARLAY_STRATEGIES,
                  now=None) -> list[dict]:
    """Rows for the ``parlay_bets`` table. ``ranked`` = ``nhlbet.report.ranking.rank_picks`` output computed with ``now`` so started games are excluded."""
    rows = []
    for st in strategies:
        legs = (_favorite_legs(ranked, [g for g in games if now is None or g.start_utc is None or g.start_utc != g.start_utc or g.start_utc > now], st.legs)
                if st.source == "favorites" else _pick_legs(ranked, st.legs, st.source == "players"))
        if len(legs) < 2:
            continue                                             # not enough independent games/picks for a parlay today
        dec, p_model, p_market = 1.0, 1.0, 1.0
        for r in legs:
            dec *= r["leg"]["decimal"]; p_model *= r["leg"]["p_model"]; p_market *= r["leg"]["p_market"]
        mean_score = sum(r["score"] for r in legs) / len(legs)
        stake = CONTROL_STAKE if st.source == "favorites" else paper_stake(mean_score / 0.06)          # a 6-point average weighted edge is the $30 maximum
        rows.append({"run_id": run_id, "run_at": run_at, "game_date": date, "strategy": st.name, "idx": 0, "n_legs": len(legs),
                     "label": " + ".join(r["pick"] if "pick" in r else r["label"] for r in legs),
                     "legs": json.dumps([{**r["leg"], "label": r["pick"], "game": r["game"]} for r in legs]),
                     "decimal": dec, "p_model": p_model, "p_market": p_market, "stake": stake, "ev": p_model * (dec - 1) - (1 - p_model)})
    return rows
