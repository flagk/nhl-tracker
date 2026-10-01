"""Regression: a run after puck drop saw in-play odds (a 2% home favourite) and replaced the earlier pre-game decision with them."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.hygiene import purge_inplay
from nhlbet.odds.client import OddsFetch
from nhlbet.odds.consensus import consensus_snapshots, latest_book_prices
from nhlbet.odds.markets import latest_alt_prices
from nhlbet.odds.snapshots import record_fetch
from nhlbet.pipeline import run_daily
from nhlbet.report.betlog import final_recommendations
from tests import fakes as F
from tests.test_alt_markets import event
from tests.test_report_pipeline import _e2e_setup


def store_with_game():
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=1, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="BOS", away="TOR", home_score=None,
                             away_score=None, status="LIVE", last_period=None, home_win=None, source="nhl_api", updated_at=None)], ["game_id"])
    return st


def ev_at(prob_home_price):
    e = event()
    e["commence_time"] = "2026-10-08T23:00:00Z"
    for bk in e["bookmakers"]:
        for m in bk["markets"]:
            if m["key"] == "h2h":
                m["outcomes"] = [{"name": "Boston Bruins", "price": prob_home_price[0]}, {"name": "Toronto Maple Leafs", "price": prob_home_price[1]}]
    return e


def test_in_play_quotes_are_ignored_for_prices_consensus_and_alt_lines():
    st = store_with_game()
    record_fetch(st, OddsFetch([ev_at((1.60, 2.40))], "2026-10-08T22:00:00+00:00", 400, 100, "live"))      # pre-game
    record_fetch(st, OddsFetch([ev_at((40.0, 1.02))], "2026-10-09T00:30:00+00:00", 399, 101, "live"))       # 90 minutes after puck drop: home is a 2% underdog
    cons = consensus_snapshots(st)
    assert list(cons.captured_at) == ["2026-10-08T22:00:00+00:00"] and cons.home_prob_novig.iloc[0] > 0.5
    assert latest_book_prices(st, 1).attrs["captured_at"] == "2026-10-08T22:00:00+00:00"
    assert latest_alt_prices(st, 1)["totals"].attrs["captured_at"] == "2026-10-08T22:00:00+00:00"


def test_purge_removes_rows_logged_after_puck_drop_and_keeps_the_rest():
    st = store_with_game()
    rec = lambda run, at, pm: dict(run_id=run, run_at=at, run_type="late", game_id=1, game_date="2026-10-08", home="BOS", away="TOR", home_goalie="", away_goalie="",
                                   goalie_status="probable", model_version="v", p_model=0.5, p_adj=None, p_market=pm, p_stack_raw=0.5, action="NO_BET", side=None, team=None,
                                   book=None, decimal=None, stake=0.0, edge=None, ev=None, reasons="", odds_captured_at=None, odds_stale=0, model_status="OK")
    st.upsert("recommendations", [rec("pre", "2026-10-08T21:00:00+00:00", 0.55), rec("post", "2026-10-09T00:32:00+00:00", 0.02)], ["run_id", "game_id"])
    st.upsert("odds_consensus", [dict(game_id=1, captured_at="2026-10-08T22:00:00+00:00", home_prob_novig=0.55, n_books=9),
                                 dict(game_id=1, captured_at="2026-10-09T00:32:00+00:00", home_prob_novig=0.02, n_books=1)], ["game_id", "captured_at"])
    assert purge_inplay(st) == {"recommendations": 1, "odds_consensus": 1}
    assert final_recommendations(st).p_market.iloc[0] == 0.55                      # the pre-game decision is final again
    assert purge_inplay(st) == {}                                                   # idempotent


def test_a_run_after_puck_drop_skips_the_game_and_leaves_the_pregame_record(league, tmp_path, monkeypatch):
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    books = [("bookA", 2.30, 1.65, f"{day}T15:00:00Z"), ("bookB", 2.25, 1.68, f"{day}T15:00:00Z")]
    record_fetch(st, type("Fx", (), {"events": [F.odds_event("e1", commence=f"{day}T23:00:00Z", books=books)], "captured_at": f"{day}T16:00:00+00:00", "remaining": 300, "used": 200, "source": "live"})())
    kw = dict(db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models")
    t0 = datetime.fromisoformat(f"{day}T17:00:00+00:00")
    run_daily(day, "morning", now=t0, **kw)
    assert len(Store(str(tmp_path / "t.db")).df("SELECT * FROM recommendations")) == 1
    r2 = run_daily(day, "late", now=t0 + timedelta(hours=7, minutes=30), **kw)      # 00:30 UTC: the 23:00 game is under way
    assert r2["games"] == 0
    s = Store(str(tmp_path / "t.db"))
    assert len(s.df("SELECT * FROM recommendations")) == 1 and final_recommendations(s).run_id.iloc[0].endswith("morning-1700")
