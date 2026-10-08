import numpy as np
import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.models.goals import ScoreDistribution
from nhlbet.odds.markets import alt_quotes, latest_alt_prices
from nhlbet.odds.math import devig
from nhlbet.odds.snapshots import parse_events, record_fetch
from nhlbet.odds.client import OddsFetch


def event(total_pts=(6.5, 6.5, 6.0), spread=-1.5, updated="2026-10-08T14:59:00Z"):
    books = []
    for i, tp in enumerate(total_pts):
        books.append({"key": f"bk{i}", "last_update": updated, "markets": [
            {"key": "totals", "last_update": updated, "outcomes": [{"name": "Over", "price": 1.91, "point": tp}, {"name": "Under", "price": 1.91 + 0.04 * i, "point": tp}]},
            {"key": "spreads", "last_update": updated, "outcomes": [{"name": "Boston Bruins", "price": 2.25 - 0.05 * i, "point": spread},
                                                                     {"name": "Toronto Maple Leafs", "price": 1.65, "point": -spread}]},
            {"key": "h2h", "last_update": updated, "outcomes": [{"name": "Boston Bruins", "price": 1.6}, {"name": "Toronto Maple Leafs", "price": 2.4}]}]})
    return {"id": "e1", "commence_time": "2026-10-08T23:00:00Z", "home_team": "Boston Bruins", "away_team": "Toronto Maple Leafs", "bookmakers": books}


def store_with(ev):
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=1, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="BOS", away="TOR", home_score=None,
                             away_score=None, status="FUT", last_period=None, home_win=None, source="nhl_api", updated_at=None)], ["game_id"])
    record_fetch(st, OddsFetch([ev], "2026-10-08T15:00:00+00:00", 400, 100, "live"))
    return st


def test_parse_keeps_points_and_team_outcomes():
    rows = parse_events([event()], "t")
    tot = [r for r in rows if r["market"] == "totals"]
    assert {r["outcome"] for r in tot} == {"Over", "Under"} and {r["point"] for r in tot} == {6.5, 6.0}
    sp = [r for r in rows if r["market"] == "spreads" and r["outcome"] == "BOS"]
    assert {r["point"] for r in sp} == {-1.5}


def test_main_line_is_modal_and_quotes_match_hand_math():
    st = store_with(event())
    prices = latest_alt_prices(st, 1)
    assert len(prices["totals"]) == 3 and len(prices["spreads"]) == 3
    d = ScoreDistribution(3.1, 2.9, 0.03, 0.57, 0.6, 0.235)
    q = {(x.market, x.side): x for x in alt_quotes(prices, d, "BOS", "TOR")}
    over = q[("totals", "over")]
    assert over.point == 6.5 and over.n_books == 2 and over.label == "Over 6.5"            # the 6.0 book is ignored
    mkt = np.mean([devig([1.91, 1.91 + 0.04 * i])[0] for i in (0, 1)])
    assert over.market_prob == pytest.approx(mkt) and q[("totals", "under")].market_prob == pytest.approx(1 - mkt)
    o, u, p = d.total_probs(6.5)
    assert over.model_prob == pytest.approx(o) and over.edge == pytest.approx(o - mkt) and over.ev == pytest.approx(o * 0.91 - u)
    assert over.best_decimal == 1.91 and q[("totals", "under")].best_decimal == pytest.approx(1.95)   # best price per side across books
    home, away = q[("spreads", "home")], q[("spreads", "away")]
    assert home.label == "BOS -1.5" and away.label == "TOR +1.5" and home.point == -1.5 and away.point == 1.5
    assert home.best_decimal == 2.25 and home.model_prob + away.model_prob == pytest.approx(1)


def test_whole_number_line_uses_push_in_ev_but_not_in_probability():
    st = store_with(event(total_pts=(6.0, 6.0)))
    d = ScoreDistribution(3.1, 2.9, 0.03, 0.57, 0.55, 0.235)
    q = {x.side: x for x in alt_quotes(latest_alt_prices(st, 1), d, "BOS", "TOR") if x.market == "totals"}
    o, u, p = d.total_probs(6.0)
    assert p > 0.08 and q["over"].p_push == pytest.approx(p)
    assert q["over"].model_prob == pytest.approx(o / (o + u)) and q["over"].ev == pytest.approx(o * 0.91 - u)


def test_stale_books_and_missing_markets_are_skipped():
    st = store_with(event(updated="2026-10-08T10:00:00Z"))                                     # book quotes 5 hours before capture
    prices = latest_alt_prices(st, 1)
    assert prices["totals"].empty and prices["spreads"].empty
    assert alt_quotes(prices, ScoreDistribution(3, 3, 0.03, 0.57), "BOS", "TOR") == []


# ---------------------------------------------------------------- paper strategies and settlement
from nhlbet.odds.markets import MarketQuote
from nhlbet.report.betlog import shadow_performance, shadow_resolved
from nhlbet.risk.policy import RiskConfig
from nhlbet.risk.shadow import STRATEGIES, shadow_bets


class G:
    def __init__(self, gid, alt, ctx=None):
        self.game_id, self.home, self.away, self.quotes, self.alt = gid, f"H{gid}", f"A{gid}", None, alt
        self.ctx = ctx if ctx is not None else {"model_status": "OK", "odds_stale": False}


def mq(market, side, point, pm, pk, dec=1.91, label=None):
    return MarketQuote(market, side, label or f"{side} {point}", point, pm, pk, 0.0, "bk", dec, pm - pk, pm * (dec - 1) - (1 - pm), 4)


def test_alt_strategies_follow_their_rules():
    cfg = RiskConfig(bankroll=1000)
    tot = lambda po: [mq("totals", "over", 6.5, po, 0.47), mq("totals", "under", 6.5, 1 - po, 0.53)]
    sp = [mq("spreads", "home", -1.5, 0.36, 0.34), mq("spreads", "away", 1.5, 0.64, 0.66)]
    games = [G(1, tot(0.52) + sp), G(2, tot(0.47)), G(3, []), G(4, tot(0.60), {"model_status": "ALERT", "odds_stale": False})]
    by = {}
    for r in shadow_bets(games, cfg, "r", "t", "d"):
        by.setdefault(r["strategy"], {})[r["game_id"]] = r
    assert len(by) == len(STRATEGIES)
    te = by["totals_edge"]
    assert te[1]["action"] == "BET" and te[1]["side"] == "over" and te[1]["stake"] == 18.0 and te[1]["market"] == "totals" and te[1]["point"] == 6.5     # +5% edge
    assert te[2]["action"] == "NO_BET" and te[3]["action"] == "NO_BET" and te[4]["action"] == "NO_BET"         # 0% edge / no odds / model ALERT
    ev = by["every_total"]
    assert ev[1]["side"] == "over" and ev[2]["side"] == "under" and ev[2]["action"] == "BET" and ev[3]["action"] == "NO_BET"   # model's side, edge ignored
    assert ev[1]["stake"] == 5.0 and ev[2]["stake"] == 8.0                        # trust 0.3: model 52% vs market 47% blends to 48.5% -> no conviction -> $5; the under (53% / 53%) -> 0.12 -> $8
    assert by["always_over"][2]["side"] == "over" and by["always_over"][2]["label"] == "over 6.5" and by["always_over"][2]["stake"] == 10.0     # control: always $10
    assert by["puckline_edge"][1]["action"] == "NO_BET"                                                          # +2% < 3% threshold
    assert by["every_puckline"][1]["side"] == "away" and by["every_puckline"][1]["market"] == "spreads"


def test_settlement_by_hand_for_totals_and_puckline():
    st = Store(":memory:")
    games = [(1, 4, 3, "SO"), (2, 5, 2, "REG"), (3, 3, 2, "OT")]     # 1: SO win (total 6 excl. shootout goal), 2: home by 3, 3: OT win (total 5)
    st.upsert("games", [dict(game_id=g, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="H", away="A", home_score=h, away_score=a,
                             status="FINAL", last_period=lp, home_win=int(h > a), source="nhl_api", updated_at=None) for g, h, a, lp in games], ["game_id"])
    def bet(gid, strat, side, market, point, dec=2.0, stake=10.0):
        return dict(run_id="r", run_at="t", game_id=gid, strategy=strat, game_date="2026-10-08", action="BET", side=side, team="x", book=None, decimal=dec, stake=stake,
                    p_model=0.5, p_adj=0.5, p_market=0.5, edge=0.0, ev=0.0, market=market, point=point, label="x")
    rows = [bet(1, "s_over55", "over", "totals", 5.5), bet(1, "s_under6", "under", "totals", 6.0),       # total 6: over 5.5 wins; under 6.0 PUSHES
            bet(2, "s_over65", "over", "totals", 6.5), bet(3, "s_under55", "under", "totals", 5.5),      # total 7 beats 6.5; total 5 is under 5.5
            bet(1, "s_home15", "home", "spreads", -1.5), bet(1, "s_away15", "away", "spreads", 1.5),     # SO winner wins by one: -1.5 loses, +1.5 covers
            bet(2, "s_home15b", "home", "spreads", -1.5), bet(3, "s_home05", "home", "spreads", -0.5)]   # 5-2 covers; OT win covers -0.5
    st.upsert("shadow_bets", rows, ["run_id", "game_id", "strategy"])
    d = shadow_resolved(st).set_index("strategy")
    assert d.loc["s_over55", "profit"] == 10 and d.loc["s_under6", "push"] and not d.loc["s_under6", "is_bet"] and d.loc["s_under6", "profit"] == 0
    assert d.loc["s_over65", "profit"] == 10 and d.loc["s_under55", "profit"] == 10
    assert d.loc["s_home15", "profit"] == -10 and d.loc["s_away15", "profit"] == 10 and d.loc["s_home15b", "profit"] == 10 and d.loc["s_home05", "profit"] == 10
    perf = shadow_performance(st)
    assert "s_under6" in perf.index and perf.loc["s_under6", "bets"] == 0                                         # pushes are not bets


# ---------------------------------------------------------------- end to end: odds -> goals model -> report, site, logs, paper bets
def test_end_to_end_alt_markets(league, tmp_path, monkeypatch):
    from nhlbet.pipeline import run_daily
    from tests.test_report_pipeline import _e2e_setup
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    from datetime import datetime, timedelta, timezone
    now = datetime.fromisoformat(f"{day}T17:00:00+00:00")                # pinned clock: the odds are 30 minutes old and the game starts at 23:00 UTC
    upd = (now - timedelta(minutes=35)).isoformat(timespec="seconds")
    ev = event(updated=upd)
    ev["commence_time"] = f"{day}T23:00:00Z"
    ev["bookmakers"][0]["key"], ev["bookmakers"][1]["key"], ev["bookmakers"][2]["key"] = "bookA", "bookB", "bookC"
    record_fetch(st, OddsFetch([ev], (now - timedelta(minutes=30)).isoformat(timespec="seconds"), 300, 200, "live"))
    r = run_daily(day, "morning", db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models", now=now)
    assert r["games"] == 1
    db = Store(str(tmp_path / "t.db"))
    aq = db.df("SELECT * FROM alt_quotes")
    assert set(zip(aq.market, aq.side)) == {("totals", "over"), ("totals", "under"), ("spreads", "home"), ("spreads", "away")}
    assert (aq.p_model.between(0.01, 0.99)).all() and aq.lam_home.notna().all() and aq.exp_total.between(3, 9).all()
    sb = db.df("SELECT * FROM shadow_bets WHERE strategy IN ('every_total','every_puckline','always_over')")
    assert (sb.action == "BET").all() and set(sb.market) == {"totals", "spreads"} and sb[sb.strategy == "always_over"].side.iloc[0] == "over"
    rep = (tmp_path / "reports/latest.md").read_text()
    assert "Other markets: totals and puck line" in rep and "Over 6.5" in rep and "BOS -1.5" in rep and "No real stakes" in rep
    assert "Puck line" in rep.split("fake bets on every game")[1]
    html = (tmp_path / "site/index.html").read_text()
    assert "Totals &amp; puck line" in html or "Totals & puck line" in html
    bets = (tmp_path / "site/bets.html").read_text()
    assert '"type": "Total"' in bets and '"type": "Puck line"' in bets
    published = {str(p): p.read_text(errors="ignore") for d in ("reports", "site", "data/logs") for p in (tmp_path / d).rglob("*") if p.is_file() and p.suffix in (".md", ".html", ".json", ".csv")}
    assert not [k for k, v in published.items() if "bookA" in v or "bookB" in v or "bookC" in v]
    assert any((tmp_path / "data/logs/alt_quotes").glob("*.csv"))


def test_alt_predictions_export_settles_outcomes_by_hand():
    from nhlbet.report.export import alt_market_predictions
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=g, season=1, game_type=2, game_date="2026-10-08", start_utc=None, home="H", away="A", home_score=h, away_score=a, status="FINAL" if h is not None else "FUT",
                             last_period=lp, home_win=None if h is None else int(h > a), source="nhl_api", updated_at=None) for g, h, a, lp in ((1, 4, 3, "SO"), (2, None, None, None))], ["game_id"])
    row = lambda run, at, gid, market, side, point: dict(run_id=run, run_at=at, game_id=gid, game_date="2026-10-08", market=market, side=side, label=f"{side} {point}", point=point,
                                                         p_model=0.5, p_market=0.5, p_push=0.0, book="b", decimal=1.9, edge=0.0, ev=0.0, n_books=3, lam_home=3.0, lam_away=3.0,
                                                         exp_total=6.0, odds_captured_at="t")
    st.upsert("alt_quotes", [row("r1", "t1", 1, "totals", "over", 7.5), row("r2", "t2", 1, "totals", "over", 5.5), row("r2", "t2", 1, "totals", "under", 6.0),
                             row("r2", "t2", 1, "spreads", "home", -1.5), row("r2", "t2", 2, "totals", "over", 5.5)], ["run_id", "game_id", "market", "side"])
    d = alt_market_predictions(st).set_index(["game_id", "market", "side"])
    assert len(d) == 5 - 1                                            # the earlier run's quote for the same game/market/side is superseded
    assert d.loc[(1, "totals", "over"), "outcome"] == "won" and d.loc[(1, "totals", "over"), "point"] == 5.5     # SO game: 4+3-1 = 6 goals
    assert d.loc[(1, "totals", "under"), "outcome"] == "push" and d.loc[(1, "spreads", "home"), "outcome"] == "lost"
    assert d.loc[(2, "totals", "over"), "outcome"] is None or pd.isna(d.loc[(2, "totals", "over"), "outcome"])


def test_role_strategies_pick_their_own_bet_type():
    cfg = RiskConfig(bankroll=1000)
    tot = [mq("totals", "over", 6.5, 0.46, 0.50), mq("totals", "under", 6.5, 0.54, 0.50)]                 # under +4%
    sp = [mq("spreads", "home", -1.5, 0.36, 0.40), mq("spreads", "away", 1.5, 0.64, 0.60)]               # dog +4%, fav -4%
    by = {r["strategy"]: r for r in shadow_bets([G(1, tot + sp)], cfg, "r", "t", "d")}
    assert by["under_edge"]["side"] == "under" and by["under_edge"]["action"] == "BET" and by["under_edge"]["stake"] == 15.0      # 4% edge -> conviction 0.4
    assert by["over_edge"]["action"] == "NO_BET"                                                          # over is -4%
    assert by["puckline_dog_edge"]["side"] == "away" and by["puckline_dog_edge"]["point"] == 1.5 and by["puckline_dog_edge"]["action"] == "BET"
    assert by["puckline_fav_edge"]["action"] == "NO_BET"
    assert by["always_under"]["side"] == "under" and by["always_under"]["stake"] == 10.0
    assert by["puckline_dog_control"]["side"] == "away" and by["puckline_dog_control"]["stake"] == 10.0


def test_old_totals_and_puckline_prices_are_not_used():
    st = store_with(event())                                                                   # captured 15:00 UTC
    fresh = latest_alt_prices(st, 1, now=pd.Timestamp("2026-10-08T16:30:00Z"))                 # 90 minutes later: still current
    assert len(fresh["totals"]) == 3 and len(fresh["spreads"]) == 3
    old = latest_alt_prices(st, 1, now=pd.Timestamp("2026-10-08T17:30:00Z"))                   # 2.5 hours later: too old to call current
    assert old["totals"].empty and old["spreads"].empty
