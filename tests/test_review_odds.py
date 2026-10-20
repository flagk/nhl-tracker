import json

import pytest

from nhlbet.review.odds_check import combine, fetch_current, odds_facts
from tests.review_helpers import candidate


def ev(home_h, away_h, over=None, under=None, spread=None, book="bk"):
    """A minimal Odds API event for BBB (home) v AAA (away)."""
    mk = [{"key": "h2h", "outcomes": [{"name": "Boston Bruins", "price": home_h}, {"name": "Toronto Maple Leafs", "price": away_h}]}]
    if over:
        mk.append({"key": "totals", "outcomes": [{"name": "Over", "price": over, "point": 6.5}, {"name": "Under", "price": under, "point": 6.5}]})
    return {"id": "e1", "home_team": "Boston Bruins", "away_team": "Toronto Maple Leafs", "commence_time": "2026-10-14T23:30:00Z", "bookmakers": [{"key": book, "markets": mk}]}


def cand_for_teams():
    c = candidate()
    c["games"]["1"]["home"], c["games"]["1"]["away"] = "BOS", "TOR"
    return c


def test_unchanged_prices_keep_the_pick():
    c = cand_for_teams()
    f = odds_facts(c["picks"], c["games"], [ev(1.92, 1.92)], "2026-10-14T22:00:00+00:00", 0.03)
    ml = f[c["picks"][0]["pick_id"]]
    assert ml["status"] == "refreshed" and ml["veto"] is False and ml["floor"] == "keep"
    assert abs(ml["p_market_now"] - 0.5) < 1e-6 and abs(ml["edge_now"] - 0.10) < 1e-6        # model 0.60 minus 0.50


def test_a_collapsed_edge_is_vetoed_in_code():
    c = cand_for_teams()
    f = odds_facts(c["picks"], c["games"], [ev(1.55, 2.60)], "2026-10-14T22:00:00+00:00", 0.03)      # market now has BOS ~62%: model 0.60 has no edge
    ml = f[c["picks"][0]["pick_id"]]
    assert ml["veto"] is True and ml["floor"] == "veto" and ml["edge_now"] < 0.03


def test_totals_use_the_picks_own_line_and_missing_lines_are_flagged():
    c = cand_for_teams()
    f = odds_facts(c["picks"], c["games"], [ev(1.92, 1.92, over=1.95, under=1.87)], "t", 0.03)
    tot = f[c["picks"][1]["pick_id"]]
    assert tot["status"] == "refreshed" and tot["p_market_now"] > 0.5 - 1e-9 - 0.05
    moved = ev(1.92, 1.92)
    moved["bookmakers"][0]["markets"].append({"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 7.0}, {"name": "Under", "price": 1.9, "point": 7.0}]})
    f = odds_facts(c["picks"], c["games"], [moved], "t", 0.03)
    assert f[c["picks"][1]["pick_id"]]["status"] == "line_gone" and f[c["picks"][1]["pick_id"]]["floor"] == "flag"


def test_without_fresh_prices_nothing_is_vetoed_and_it_says_unconfirmed():
    c = cand_for_teams()
    f = odds_facts(c["picks"], c["games"], None, None, 0.03, "ODDS_API_KEY is not set")
    assert all(v["status"] == "not_refreshed" and not v["veto"] and v["summary"].startswith("unconfirmed") for v in f.values())


def test_player_picks_are_never_refreshed():
    c = cand_for_teams()
    c["picks"][0]["market"] = "player_sog"
    f = odds_facts(c["picks"], c["games"], [ev(1.92, 1.92)], "t", 0.03)
    assert f[c["picks"][0]["pick_id"]]["status"] == "not_refreshed"


def test_the_agent_can_never_soften_a_code_veto():
    assert combine("keep", {"floor": "veto"}) == ("veto", "code")
    assert combine("flag", {"floor": "keep"}) == ("flag", "agent")
    assert combine("veto", {"floor": "flag"}) == ("veto", "agent")
    assert combine("keep", {"floor": "flag"}) == ("flag", "code")
    assert combine(None, None) == ("keep", "code")


def test_fetch_is_skipped_without_a_key_or_with_few_credits(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    ev_, at, rem, note = fetch_current()
    assert ev_ is None and "not set" in note
