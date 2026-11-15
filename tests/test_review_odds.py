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
    ev_, at, rem, note = fetch_current(".")
    assert ev_ is None and "not set" in note


def test_review_waves_share_one_credit_tally_with_the_daily_jobs(tmp_path):
    from datetime import datetime, timezone
    from nhlbet.review.odds_check import credits_spent_today, daily_budget_left, record_credit_spend
    d = tmp_path / "data/logs/odds_fetch_log"
    d.mkdir(parents=True)
    (d / "a.csv").write_text("captured_at,ok,source,remaining,used,events,note\n2026-10-19T23:00:00+00:00,1,live,200,300,11,h2h\n")
    (d / "b.csv").write_text("captured_at,ok,source,remaining,used,events,note\n2026-10-20T12:00:00+00:00,1,live,196,304,11,h2h\n")
    now = datetime(2026, 10, 20, 21, 0, tzinfo=timezone.utc)
    assert credits_spent_today(tmp_path, now) == 4
    p = record_credit_spend(tmp_path, "2026-10-20T21:00:00+00:00", 193, 307, "review:h2h,spreads,totals")      # a review wave's call is logged in the same format
    assert p.exists() and credits_spent_today(tmp_path, now) == 7
    left = daily_budget_left(tmp_path, 193, now)                                                                 # (193 - 25) / 12 days left - 7 spent - 4 held for the closes
    assert left == int((193 - 25) / 12 - 7 - 4)
    assert daily_budget_left(tmp_path, None, now) is None


def test_the_review_refuses_to_buy_prices_when_the_daily_budget_is_spent(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from nhlbet.review import odds_check
    monkeypatch.setenv("ODDS_API_KEY", "x")
    class FakeClient:
        def remaining_credits(self): return 150
        def fetch_odds(self, *a, **k): raise AssertionError("must not call the API")
    import nhlbet.odds.client as c
    monkeypatch.setattr(c, "OddsClient", lambda: FakeClient())
    d = tmp_path / "data/logs/odds_fetch_log"
    d.mkdir(parents=True)
    (d / "a.csv").write_text("captured_at,ok,source,remaining,used,events,note\n2026-10-19T23:00:00+00:00,1,live,160,340,11,h2h\n2026-10-20T20:00:00+00:00,1,live,150,350,11,h2h\n")
    ev_, at, rem, note = odds_check.fetch_current(tmp_path, datetime(2026, 10, 20, 21, 0, tzinfo=timezone.utc))
    assert ev_ is None and "budget" in note and rem == 150
