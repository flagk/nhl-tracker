import gzip
import json
import logging
import math

import numpy as np
import pandas as pd
import pytest
import requests

from nhlbet.data.store import Store
from nhlbet.features.market import attach_market_features
from nhlbet.odds import math as om
from nhlbet.odds.client import OddsAPIError, OddsClient, OddsConfigError, QuotaExhausted
from nhlbet.odds.clv import bet_clv, closing_consensus, summarize_clv
from nhlbet.odds.consensus import consensus_snapshots, latest_book_prices
from nhlbet.odds.edge import evaluate_game
from nhlbet.odds.snapshots import link_games, parse_events, record_fetch
from tests import fakes as F
from tests.fakes import FakeResp, FakeSession

KEY = "sekrit-key-123"


# ---------------------------------------------------------------- odds math
def test_conversions_roundtrip():
    assert om.american_to_decimal(-110) == pytest.approx(1.90909, abs=1e-4) and om.american_to_decimal(150) == 2.5
    assert om.american_to_decimal(100) == 2.0 and om.american_to_decimal(-200) == 1.5
    for a in (-250, -110, 100, 135, 400):
        assert om.decimal_to_american(om.american_to_decimal(a)) == a
    for bad in (0, 50, -50):
        with pytest.raises(ValueError):
            om.american_to_decimal(bad)
    with pytest.raises(ValueError):
        om.implied_prob(1.0)


def test_overround_and_devig():
    d = [om.american_to_decimal(-110)] * 2
    assert om.overround(d) == pytest.approx(0.047619, abs=1e-5)
    for m in ("proportional", "shin"):
        assert om.devig(d, m) == pytest.approx([0.5, 0.5])
    fav = [om.american_to_decimal(-180), om.american_to_decimal(150)]
    prop, shin = om.devig(fav, "proportional"), om.devig(fav, "shin")
    assert prop.sum() == pytest.approx(1) and shin.sum() == pytest.approx(1)
    assert prop[0] == pytest.approx(0.61644, abs=1e-4)
    assert shin[0] > prop[0]                                       # Shin: favourite-longshot bias -> favourite a bit higher
    three = om.devig([2.5, 3.4, 3.1], "shin")
    assert three.sum() == pytest.approx(1) and (three > 0).all()
    assert om.devig([2.0, 2.0], "shin") == pytest.approx([0.5, 0.5])          # no margin: unchanged
    with pytest.raises(ValueError):
        om.devig(d, "magic")


def test_best_price_consensus_ev_clv():
    assert om.best_price({"a": 1.8, "b": 1.95, "c": 1.9}) == ("b", 1.95)
    with pytest.raises(ValueError):
        om.best_price({})
    assert om.consensus_prob({"a": (1.9, 1.9), "b": (1.8, 2.1)}, "proportional") == pytest.approx((0.5 + 2.1 / 3.9 * 0 + (1 / 1.8) / (1 / 1.8 + 1 / 2.1)) / 2)
    assert om.expected_value(0.55, 1.9091) == pytest.approx(0.55 * 0.9091 - 0.45)
    assert om.expected_value(1 / 1.9091, 1.9091) == pytest.approx(0)          # fair bet
    assert om.edge(0.56, 0.52) == pytest.approx(0.04) and om.breakeven_prob(2.0) == 0.5
    assert om.clv(2.10, 0.50) == pytest.approx(0.05) and om.clv(1.90, 0.50) == pytest.approx(-0.05)


# ---------------------------------------------------------------- client
def client(tmp_path, routes, **kw):
    sess = FakeSession(routes)
    c = OddsClient(api_key=KEY, cache_dir=tmp_path / "c", session=sess, sleep=lambda s: None, **kw)
    return c, sess


def ok(events, rem="480", used="20"):
    return FakeResp(200, events, {"x-requests-remaining": rem, "x-requests-used": used})


def test_missing_key_gives_setup_instructions(monkeypatch, tmp_path):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsConfigError, match="the-odds-api.com"):
        OddsClient(cache_dir=tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", "fromenv")
    assert OddsClient(cache_dir=tmp_path)._key == "fromenv"


def test_fetch_caches_and_tracks_quota(tmp_path):
    c, s = client(tmp_path, {"/odds": [ok([F.odds_event()])]})
    f1 = c.fetch_odds()
    assert f1.source == "live" and f1.remaining == 480 and len(f1.events) == 1 and c.remaining_credits() == 480
    f2 = c.fetch_odds()
    assert f2.source == "cache" and len(s.calls) == 1              # within TTL: no second credit spent
    assert s.params["apiKey"] == KEY and s.params["markets"] == "h2h" and s.params["oddsFormat"] == "decimal"


def test_ttl_expiry_triggers_new_call(tmp_path):
    now = [1000.0]
    c, s = client(tmp_path, {"/odds": [ok([F.odds_event()]), ok([F.odds_event("e2")])]}, ttl=60)
    c._now = lambda: now[0]
    assert c.fetch_odds().events[0]["id"] == "e1"
    now[0] += 120
    assert c.fetch_odds().events[0]["id"] == "e2" and len(s.calls) == 2


def test_retries_then_succeeds(tmp_path):
    c, s = client(tmp_path, {"/odds": [FakeResp(500), requests.ConnectionError(f"boom {KEY}"), ok([F.odds_event()])]})
    assert c.fetch_odds().source == "live" and len(s.calls) == 3


def test_bad_key_raises_without_leaking_it(tmp_path):
    c, _ = client(tmp_path, {"/odds": [FakeResp(401, {"message": "bad"})]})
    with pytest.raises(OddsAPIError) as e:
        c.fetch_odds()
    assert KEY not in str(e.value) and "ODDS_API_KEY" in str(e.value)


def test_stale_fallback_when_api_down(tmp_path):
    now = [1000.0]
    c, s = client(tmp_path, {"/odds": [ok([F.odds_event()]), FakeResp(503)]}, ttl=60, retries=2)
    c._now = lambda: now[0]
    c.fetch_odds()
    now[0] += 3600                                                  # cache expired, API now failing
    f = c.fetch_odds()
    assert f.stale and f.source == "stale_cache" and len(f.events) == 1
    now[0] += 10 * 3600                                             # too old to be useful
    with pytest.raises(OddsAPIError):
        c.fetch_odds()


def test_quota_guard_blocks_call_and_uses_stale(tmp_path):
    now = [1000.0]
    c, s = client(tmp_path, {"/odds": [ok([F.odds_event()], rem="10")]}, ttl=60, reserve=25)
    c._now = lambda: now[0]
    c.fetch_odds()
    now[0] += 300
    calls = len(s.calls)
    f = c.fetch_odds()
    assert f.stale and len(s.calls) == calls                        # 10 credits left < reserve: did NOT hit the API
    c2, s2 = client(tmp_path / "x", {"/odds": [ok([])]})
    c2._save_quota(0, 500)                                          # quota gone and nothing cached to fall back on
    with pytest.raises(QuotaExhausted):
        c2.fetch_odds()
    assert not s2.calls


def test_429_with_zero_remaining_stops_immediately(tmp_path):
    c, s = client(tmp_path, {"/odds": [FakeResp(429, {}, {"x-requests-remaining": "0", "Retry-After": "1"})]})
    with pytest.raises(QuotaExhausted):
        c.fetch_odds()
    assert len(s.calls) == 1


def test_api_key_never_touches_disk_or_logs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    c, _ = client(tmp_path, {"/odds": [requests.ConnectionError(f"failed for apiKey={KEY}"), ok([F.odds_event()])]})
    c.fetch_odds()
    for p in (tmp_path / "c").rglob("*"):
        if p.is_file():
            raw = p.read_bytes()
            raw = gzip.decompress(raw) if p.suffix == ".gz" else raw
            assert KEY.encode() not in raw, p
    assert KEY not in caplog.text


# ---------------------------------------------------------------- snapshots / matching / consensus
def games_store():
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=1, season=20232024, game_type=2, game_date="2023-10-11", start_utc="2023-10-11T23:00:00Z", home="BOS",
                             away="TOR", home_score=None, away_score=None, status="FUT", last_period=None, home_win=None, source="nhl_api", updated_at=None)],
              ["game_id"])
    return st


def test_parse_events_markets_and_unknown_team():
    ev = [F.odds_event(extra_markets=True), F.odds_event("e2", home="Quebec Nordiques")]
    rows = parse_events(ev, "2023-10-11T16:00:00+00:00")
    assert {r["market"] for r in rows} == {"h2h", "spreads", "totals"} and all(r["event_id"] == "e1" for r in rows)
    h2h = [r for r in rows if r["market"] == "h2h" and r["book"] == "bookA"]
    assert {r["outcome"] for r in h2h} == {"BOS", "TOR"}
    assert {r["outcome"] for r in rows if r["market"] == "totals"} == {"Over", "Under"}
    assert any(r["point"] == -1.5 and r["outcome"] == "BOS" for r in rows if r["market"] == "spreads")


def test_snapshots_are_appended_idempotent_and_linked():
    st = games_store()
    f = lambda t, h, a: type("Fx", (), {"events": [F.odds_event(books=[("bookA", h, a, "2023-10-11T15:00:00Z")])], "captured_at": t,
                                        "remaining": 400, "used": 100, "source": "live"})()
    record_fetch(st, f("2023-10-11T14:00:00+00:00", 1.80, 2.10))
    record_fetch(st, f("2023-10-11T14:00:00+00:00", 1.80, 2.10))                 # same snapshot again -> no duplicates
    assert len(st.df("SELECT * FROM odds_snapshots")) == 2
    record_fetch(st, f("2023-10-11T22:30:00+00:00", 1.70, 2.25))                 # later snapshot appended, not overwritten
    snaps = st.df("SELECT * FROM odds_snapshots")
    assert len(snaps) == 4 and (snaps.game_id == 1).all()
    assert len(st.df("SELECT * FROM odds_fetch_log")) == 2
    cons = consensus_snapshots(st)
    assert list(cons.captured_at) == ["2023-10-11T14:00:00+00:00", "2023-10-11T22:30:00+00:00"]
    assert cons.home_prob_novig.iloc[1] > cons.home_prob_novig.iloc[0]           # market moved toward the home team


def test_link_games_ignores_wrong_teams_or_dates():
    st = games_store()
    rows = parse_events([F.odds_event(commence="2023-12-25T23:00:00Z")], "2023-12-25T12:00:00+00:00")
    st.upsert("odds_snapshots", rows, ["captured_at", "event_id", "book", "market", "outcome", "point"])
    assert link_games(st) == 0


def test_consensus_and_stale_book_filter_and_edge():
    st = games_store()
    books = [("bookA", 1.80, 2.10, "2023-10-11T15:59:00Z"), ("bookB", 1.87, 2.00, "2023-10-11T15:58:00Z"),
             ("staleBook", 2.20, 1.70, "2023-10-11T09:00:00Z")]                  # stale line: would be a fake 'best price' for home
    fx = type("Fx", (), {"events": [F.odds_event(books=books)], "captured_at": "2023-10-11T16:00:00+00:00", "remaining": 1, "used": 1, "source": "live"})()
    record_fetch(st, fx)
    cur = latest_book_prices(st, 1, max_book_age_min=90)
    assert set(cur.book) == {"bookA", "bookB"}
    q = evaluate_game(0.60, "BOS", "TOR", cur, "proportional")
    p_a = (1 / 1.80) / (1 / 1.80 + 1 / 2.10); p_b = (1 / 1.87) / (1 / 1.87 + 1 / 2.00)
    mkt_home = (p_a + p_b) / 2
    h, a = q["home"], q["away"]
    assert h.best_book == "bookB" and h.best_decimal == 1.87 and a.best_book == "bookA" and a.best_decimal == 2.10
    assert h.market_prob == pytest.approx(mkt_home) and h.edge == pytest.approx(0.60 - mkt_home)
    assert h.ev == pytest.approx(0.60 * 0.87 - 0.40) and a.ev == pytest.approx(0.40 * 1.10 - 0.60)
    assert h.market_prob + a.market_prob == pytest.approx(1) and h.n_books == 2
    assert evaluate_game(0.5, "BOS", "TOR", cur.iloc[0:0]) is None


def test_clv_flow_uses_last_snapshot_before_puck_drop():
    cons = pd.DataFrame({"game_id": [1, 1, 1], "captured_at": ["2023-10-11T14:00:00+00:00", "2023-10-11T22:30:00+00:00", "2023-10-11T23:20:00+00:00"],
                         "home_prob_novig": [0.52, 0.56, 0.99]})               # last is after start (in-play) and must be ignored
    close = closing_consensus(cons, 1, "2023-10-11T23:00:00Z")
    assert close == 0.56
    home = bet_clv("home", 1.95, close)                                        # bet home at 1.95 when close says 56%: +9.2% EV
    assert home["clv_ev"] == pytest.approx(1.95 * 0.56 - 1) and home["clv_prob_pts"] == pytest.approx(0.56 - 1 / 1.95)
    away = bet_clv("away", 2.10, close)                                        # bet away at 2.10 vs close 44%: -7.6%
    assert away["clv_ev"] == pytest.approx(2.10 * 0.44 - 1) and away["clv_ev"] < 0
    assert closing_consensus(cons, 2, "2023-10-11T23:00:00Z") is None
    s = summarize_clv(pd.Series(np.random.default_rng(0).normal(0.02, 0.03, 400)))
    assert s["n"] == 400 and s["ci"][0] > 0 and 0.6 < s["beat_close"] <= 1


def test_market_features_consume_consensus_without_closing_leak():
    st = games_store()
    for t, h in (("2023-10-11T09:00:00+00:00", 1.90), ("2023-10-11T21:00:00+00:00", 1.80), ("2023-10-11T23:30:00+00:00", 1.40)):
        record_fetch(st, type("Fx", (), {"events": [F.odds_event(books=[("bookA", h, 2.0, t)])], "captured_at": t, "remaining": 1, "used": 1, "source": "live"})())
    cons = consensus_snapshots(st)
    feats = pd.DataFrame({"start_utc": [pd.Timestamp("2023-10-11 23:00", tz="UTC")]}, index=pd.Index([1], name="game_id"))
    out = attach_market_features(feats, cons, lead_minutes=90)          # decision time 21:30 -> 23:30 snapshot excluded
    assert out.mkt_open_p[1] == pytest.approx(cons.home_prob_novig.iloc[0])
    assert out.mkt_move[1] == pytest.approx(cons.home_prob_novig.iloc[1] - cons.home_prob_novig.iloc[0])
