import numpy as np
import pandas as pd
import pytest

from nhlbet.data import ingest
from nhlbet.data.store import Store
from nhlbet.odds.edge import SideQuote
from nhlbet.odds.math import expected_value
from nhlbet.report.betlog import export_logs, restore_logs, shadow_performance, shadow_resolved
from nhlbet.report.markdown import render_report
from nhlbet.risk.policy import RiskConfig
from nhlbet.risk.shadow import STRATEGIES, shadow_bets


class G:                                     # minimal SlateGame stand-in
    def __init__(self, gid, quotes, ctx=None):
        self.game_id, self.home, self.away, self.quotes = gid, f"H{gid}", f"A{gid}", quotes
        self.ctx = ctx if ctx is not None else {"model_status": "OK", "odds_stale": False, "goalie_confirmed": True, "games_played_min": 40.0}


def q(ph, mh, dh, da):
    mk = lambda side, team, p, m, d, b: SideQuote(side, team, p, m, b, d, 1 / d, p - m, expected_value(p, d), 5)
    return {"home": mk("home", "H", ph, mh, dh, "bA"), "away": mk("away", "A", 1 - ph, 1 - mh, da, "bB")}


def by_strategy(rows):
    out = {}
    for r in rows:
        out.setdefault(r["strategy"], {})[r["game_id"]] = r
    return out


def test_one_row_per_game_and_strategy_and_flat_stakes():
    cfg = RiskConfig(bankroll=1000)
    games = [G(1, q(0.60, 0.54, 1.85, 2.10)), G(2, q(0.44, 0.47, 2.05, 1.88)), G(3, None)]
    rows = shadow_bets(games, cfg, "r1", "2026-10-08T15:00:00", "2026-10-08")
    assert len(rows) == 3 * len(STRATEGIES) and len({(r["game_id"], r["strategy"]) for r in rows}) == len(rows)
    by = by_strategy(rows)
    fav = by["market_favorite"]
    assert fav[1]["side"] == "home" and fav[2]["side"] == "away" and fav[1]["stake"] == fav[2]["stake"] == 10.0     # 1% flat on the favourite
    assert fav[3]["action"] == "NO_BET"                                                                              # no odds -> nothing to bet
    flat = by["flat_model_side"]
    assert flat[1]["action"] == "BET" and flat[1]["side"] == "home"                                                   # the larger edge
    assert flat[2]["side"] == "away" and flat[2]["action"] == "BET"                                                   # home is -3%, away +3%
    assert flat[3]["action"] == "NO_BET"


def test_flat_model_side_passes_when_no_positive_edge():
    rows = shadow_bets([G(1, q(0.50, 0.50, 1.95, 1.95))], RiskConfig(), "r", "t", "d")
    assert by_strategy(rows)["flat_model_side"][1]["action"] == "NO_BET"


def test_experiments_differ_from_the_live_policy_as_documented():
    cfg = RiskConfig(bankroll=1000)
    early = G(1, q(0.585, 0.535, 1.90, 2.12), {"model_status": "OK", "odds_stale": False, "goalie_confirmed": True, "games_played_min": 3.0})
    by = by_strategy(shadow_bets([early], cfg, "r", "t", "d"))
    assert by["no_guard"][1]["action"] == "BET"                    # the guard would have blocked this (only 3 games played)
    marginal = G(2, q(0.553, 0.53, 1.92, 2.05))                    # 2.3% edge: below the live 3%, above edge_1pct's 1%
    by2 = by_strategy(shadow_bets([marginal], cfg, "r", "t", "d"))
    assert by2["edge_1pct"][2]["action"] == "BET"
    big = G(3, q(0.58, 0.53, 1.90, 2.12))
    by3 = by_strategy(shadow_bets([big], cfg, "r", "t", "d"))
    assert by3["no_shrink"][3]["stake"] >= by3["no_guard"][3]["stake"] > 0     # trusting the raw model stakes at least as much


def test_flat_strategies_respect_stale_odds_and_model_alert():
    cfg = RiskConfig()
    stale = G(1, q(0.6, 0.54, 1.85, 2.1), {"model_status": "OK", "odds_stale": True, "games_played_min": 40.0})
    alert = G(2, q(0.6, 0.54, 1.85, 2.1), {"model_status": "ALERT", "odds_stale": False, "games_played_min": 40.0})
    rows = shadow_bets([stale, alert], cfg, "r", "t", "d")
    assert all(r["action"] == "NO_BET" for r in rows)


def test_control_loses_about_the_bookmaker_margin():
    """Property: betting the market favourite on a fair market with a 4.5% overround must lose ~4.3% per dollar. This is the
    'no skill' baseline every strategy is compared against."""
    rng = np.random.default_rng(0)
    st = Store(":memory:")
    games, bets = [], []
    for i in range(1, 6001):
        p = float(rng.uniform(0.35, 0.65))                          # true home win probability = the fair market
        dh, da = 1 / (p * 1.045), 1 / ((1 - p) * 1.045)
        home_win = int(rng.random() < p)
        fav_home = p >= 0.5
        games.append(dict(game_id=i, season=1, game_type=2, game_date="2026-10-08", start_utc=None, home="H", away="A", home_score=3 if home_win else 1,
                          away_score=1 if home_win else 3, status="FINAL", last_period="REG", home_win=home_win, source="nhl_api", updated_at=None))
        bets.append(dict(run_id="r", run_at="t", game_id=i, strategy="market_favorite", game_date="2026-10-08", action="BET",
                         side="home" if fav_home else "away", team="x", book="b", decimal=dh if fav_home else da, stake=10.0,
                         p_model=None, p_adj=None, p_market=None, edge=None, ev=None))
    st.upsert("games", games, ["game_id"]); st.upsert("shadow_bets", bets, ["run_id", "game_id", "strategy"])
    perf = shadow_performance(st)
    r = perf.loc["market_favorite"]
    assert r.bets == 6000 and -0.065 < r.roi < -0.02 and r.roi_lo < r.roi < r.roi_hi      # about -4.3%, CI brackets it


def test_settlement_by_hand_uses_latest_run_and_clv():
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=g, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="H", away="A", home_score=3, away_score=1,
                             status="FINAL", last_period="REG", home_win=1, source="nhl_api", updated_at=None) for g in (1, 2)], ["game_id"])
    mk = lambda run, at, gid, side, dec, stake, strat="flat_model_side": dict(run_id=run, run_at=at, game_id=gid, strategy=strat, game_date="2026-10-08", action="BET", side=side,
                                                                           team="t", book="b", decimal=dec, stake=stake, p_model=0.5, p_adj=0.5, p_market=0.5, edge=0.0, ev=0.0)
    st.upsert("shadow_bets", [mk("r1", "2026-10-08T10:00:00", 1, "away", 2.0, 99.0),       # superseded
                              mk("r2", "2026-10-08T20:00:00", 1, "home", 2.0, 10.0),       # home won: +10
                              mk("r2", "2026-10-08T20:00:00", 2, "away", 1.9, 10.0)],      # away lost: -10
              ["run_id", "game_id", "strategy"])
    for gid, hp in ((1, 0.58), (2, 0.50)):
        st.upsert("odds_snapshots", [dict(captured_at="2026-10-08T22:50:00+00:00", event_id=f"e{gid}", commence_time="2026-10-08T23:00:00Z", home="H", away="A", book="bk",
                                          market="h2h", outcome=o, point=0.0, price=pr, book_updated=None, game_id=gid)
                                     for o, pr in (("H", 1 / hp), ("A", 1 / (1 - hp)))], ["captured_at", "event_id", "book", "market", "outcome", "point"])
    r = shadow_performance(st).loc["flat_model_side"]
    assert r.bets == 2 and r.staked == 20.0 and r.profit == 0.0 and r.roi == 0.0 and r.win_rate == 0.5
    assert r.n_clv == 2 and r.avg_clv == pytest.approx((2.0 * 0.58 - 1 + 1.9 * 0.50 - 1) / 2)   # CLV from the last pre-start consensus


def test_shadow_export_restore_roundtrip(tmp_path):
    st = Store(":memory:")
    st.upsert("shadow_bets", [dict(run_id="r1", run_at="t", game_id=1, strategy=s.name, game_date="d", action="NO_BET", side=None, team=None, book=None, decimal=None,
                                   stake=0.0, p_model=None, p_adj=None, p_market=None, edge=None, ev=None) for s in STRATEGIES], ["run_id", "game_id", "strategy"])
    export_logs(st, tmp_path / "logs")
    assert (tmp_path / "logs/shadow_bets/r1.csv").exists()
    fresh = Store(":memory:")
    assert restore_logs(fresh, tmp_path / "logs")["shadow_bets"] == len(STRATEGIES)
    assert len(fresh.df("SELECT * FROM shadow_bets")) == len(STRATEGIES)


def test_report_explains_the_control_and_small_samples():
    perf = pd.DataFrame({"bets": [12, 40], "staked": [120.0, 400.0], "profit": [5.0, -17.0], "roi": [0.041, -0.0425], "roi_lo": [np.nan, -0.15], "roi_hi": [np.nan, 0.07],
                         "win_rate": [0.58, 0.49], "avg_clv": [0.012, -0.004], "n_clv": [10, 40]}, index=pd.Index(["flat_model_side", "market_favorite"], name="strategy"))
    txt = render_report("2026-10-08", "late", [], RiskConfig(), "v1", {"drift": {"status": "OK"}}, None, None, shadow=perf)
    assert "Paper trading" in txt and "no-skill control" in txt and "`market_favorite`" in txt and "+4.1%" in txt and "-4.2%" in txt
    assert "still small" in txt
    assert "No settled paper bets yet" in render_report("d", "m", [], RiskConfig(), "v1", {}, None, None, shadow=pd.DataFrame())


def test_old_seasons_do_not_request_seattle():
    assert "SEA" not in ingest.season_teams(20192020) and "SEA" in ingest.season_teams(20212022)
    assert len(ingest.season_teams(20192020)) == 31 and len(ingest.season_teams(20232024)) == 32


def test_every_game_bets_the_models_side_even_at_negative_edge_and_is_listed():
    cfg = RiskConfig(bankroll=1000)
    games = [G(1, q(0.60, 0.64, 1.55, 2.60)), G(2, q(0.45, 0.50, 1.95, 1.95)), G(3, None)]   # game 1: model likes home but the price is too short (edge -4%)
    eg = by_strategy(shadow_bets(games, cfg, "r", "t", "d"))["every_game"]
    assert eg[1]["action"] == "BET" and eg[1]["side"] == "home" and eg[1]["stake"] == 15.0 and eg[1]["edge"] < 0
    assert eg[2]["side"] == "away" and eg[2]["stake"] == 10.0 and eg[3]["action"] == "NO_BET"
    from nhlbet.report.markdown import _fake_bets_today
    md = "\n".join(_fake_bets_today(games, cfg, "d"))
    assert "NOT recommendations" in md and "A1 @ H1" in md and "A2 @ H2" in md and "A3" not in md and "Total pretend stake $25.00" in md


def test_paper_stakes_scale_with_conviction_between_5_and_30_and_controls_stay_flat():
    from nhlbet.risk.shadow import CONTROL_STAKE, edge_conviction, paper_stake, side_conviction
    assert [paper_stake(c) for c in (-1, 0, 0.5, 1, 7)] == [5.0, 5.0, 18.0, 30.0, 30.0]            # clipped, whole dollars (17.5 rounds to 18)
    assert paper_stake(edge_conviction(0.03)) == 12.0 and paper_stake(edge_conviction(0.10)) == 30.0 and paper_stake(edge_conviction(0.25)) == 30.0
    assert paper_stake(side_conviction(0.50)) == 5.0 and paper_stake(side_conviction(0.75)) == 30.0 and paper_stake(side_conviction(0.62)) == 17.0
    cfg = RiskConfig(bankroll=1000)
    games = [G(i, q(0.50 + 0.04 * i, 0.50, 1.95, 1.95)) for i in range(1, 7)]
    rows = shadow_bets(games, cfg, "r", "t", "d")
    stakes = {}
    for r in rows:
        if r["action"] == "BET":
            stakes.setdefault(r["strategy"], []).append(r["stake"])
    assert set(stakes["market_favorite"]) == {CONTROL_STAKE}                                         # controls: always $10, so their ROI is a clean baseline
    for name in ("every_game", "flat_model_side"):
        assert all(5.0 <= x <= 30.0 and x == int(x) for x in stakes[name]) and len(set(stakes[name])) > 1     # varies, whole dollars, in range
        assert stakes[name] == sorted(stakes[name])                                                  # a surer model never stakes less (games are in rising-probability order)


def test_moneyline_diversity_strategies():
    from nhlbet.risk.policy import RiskConfig
    cfg = RiskConfig(bankroll=1000)
    g = G(1, q(0.40, 0.45, 1.80, 2.10))                      # market favourite = home (55%); model likes the AWAY dog more than the market does (60% vs 55%... market 55% away? see below)
    by = by_strategy(shadow_bets([g], cfg, "r", "t", "d"))
    assert by["home_ml_control"][1]["side"] == "home" and by["home_ml_control"][1]["stake"] == 10.0
    assert by["underdog_ml_control"][1]["side"] == "home" and by["underdog_ml_control"][1]["stake"] == 10.0   # home market prob 0.45 < away 0.55
    assert by["underdog_ml"][1]["action"] == "NO_BET"                                                         # dog edge is only -5%
    g2 = G(2, q(0.55, 0.45, 2.20, 1.70))                    # model rates the home dog 10 points above the market
    r = by_strategy(shadow_bets([g2], cfg, "r", "t", "d"))["underdog_ml"][2]
    assert r["action"] == "BET" and r["side"] == "home" and r["stake"] == 30.0
