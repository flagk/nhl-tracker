"""Shared fixtures for the review-agent tests."""
from __future__ import annotations

from datetime import datetime, timezone

from nhlbet.review.contract import build_candidates

NOW = datetime(2026, 10, 14, 22, 0, tzinfo=timezone.utc)


def row(rank, gid, label, kind="moneyline", p_model=0.60, p_market=0.52, price=2.0, side="home", point=None, player_id=0, market="h2h"):
    return {"kind": kind, "type": kind, "game_id": gid, "game": "AAA @ BBB", "start_utc": "2026-10-14T23:30:00+00:00", "pick": label, "p_model": p_model, "p_market": p_market,
            "price": price, "ev_raw": p_model * price - 1, "score": p_model - p_market, "weight": 1.0, "experimental": kind != "moneyline", "p_push": 0.0, "captured_at": "2026-10-14T21:30:00+00:00",
            "leg": {"game_id": gid, "decimal": price, "p_model": p_model, "p_market": p_market, "market": market, "side": side, "point": point, "player_id": player_id},
            "rank": rank, "has_edge": True, "line": point}


def payload(extra_rows=()):
    games = [{"game_id": 1, "home": "BBB", "away": "AAA", "start_utc": "2026-10-14T23:30:00+00:00", "home_goalie": "H. Goalie", "away_goalie": "A. Goalie", "goalie_status": "probable",
              "sides": {"home": {"best_book": "bk1"}}, "alt": []},
             {"game_id": 2, "home": "DDD", "away": "CCC", "start_utc": "2026-10-15T02:00:00+00:00", "home_goalie": "X", "away_goalie": "Y", "goalie_status": "confirmed", "sides": {}, "alt": []}]
    ranking = [row(1, 1, "BBB moneyline"), row(2, 1, "Over 6.5", kind="totals", side="over", point=6.5, market="totals", p_model=0.58, p_market=0.50),
               row(3, 2, "CCC moneyline", side="away", p_model=0.55, p_market=0.50), row(4, 2, "Tiny edge", p_model=0.51, p_market=0.50), *extra_rows]
    return {"date": "2026-10-14", "run_type": "late", "generated_at": "2026-10-14T21:45:00+00:00", "model": {"version": "v1"}, "policy": {"min_edge": 0.03}, "games": games, "ranking": ranking}


def candidate():
    return build_candidates(payload())


def good_review(inp, decision="keep", **over):
    """A valid answer for ``inp`` (a review_input batch): every pick kept with one source and confirmed checks."""
    src = {"url": "https://example.com/news", "retrieved_at": "2026-10-14T21:50:00Z"}
    chk = {"found": "starter confirmed in morning skate report", "source_urls": [src["url"]]}
    reviews = []
    for p in inp["picks"]:
        r = {"pick_id": p["pick_id"], "decision": decision, "reason": "Starter and lineup confirmed as assumed.", "confidence": "high", "sources": [src],
             "checks": {"confirmed_goalie": chk, "lineup_changes": chk, "injuries": chk, "odds_movement": "unconfirmed"}}
        r.update(over)
        reviews.append(r)
    return {"candidate_id": inp["candidate_id"], "reviewed_at": "2026-10-14T21:55:00Z", "reviews": reviews}
