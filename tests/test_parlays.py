"""AI parlays: built from ranked picks across different games, priced as the product of the legs, settled by hand-checkable rules."""
import json

import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.report.betlog import export_logs, parlay_resolved, restore_logs, shadow_performance, shadow_resolved
from nhlbet.report.ranking import rank_picks
from nhlbet.risk.parlays import PARLAY_STRATEGIES, build_parlays
from nhlbet.risk.policy import RiskConfig
from tests.test_ranking import S, ml, mq, pq


def slate():
    return [S(1, ml(0.62, 0.52, 1.95, 1.95), [mq("totals", "over", 6.5, 0.50, 0.50), mq("totals", "under", 6.5, 0.50, 0.50)]),
            S(2, ml(0.58, 0.52, 1.95, 1.95), props=[pq(7, "Shooter", 0.70)]),
            S(3, ml(0.57, 0.53, 1.95, 1.95))]


def test_parlays_use_one_leg_per_game_and_multiply_prices():
    cfg = RiskConfig(bankroll=1000)
    g = slate()
    for x in g:
        x.start_utc = pd.Timestamp("2026-10-08T23:00:00Z")
    rows = build_parlays(rank_picks(g, cfg), g, "r", "t", "2026-10-08")
    by = {r["strategy"]: r for r in rows}
    assert set(by) == {"parlay_top2", "parlay_top3", "parlay_players3", "parlay_favorites_control"} - {"parlay_players3"}       # only one game has a player prop: no players parlay
    t3 = by["parlay_top3"]
    legs = json.loads(t3["legs"])
    assert t3["n_legs"] == 3 and len({l["game_id"] for l in legs}) == 3
    assert t3["decimal"] == pytest.approx(pd.Series([l["decimal"] for l in legs]).prod()) and t3["p_model"] == pytest.approx(pd.Series([l["p_model"] for l in legs]).prod())
    assert 5 <= t3["stake"] <= 30 and by["parlay_favorites_control"]["stake"] == 10.0
    assert len(json.loads(by["parlay_top2"]["legs"])) == 2 and json.loads(by["parlay_top2"]["legs"])[0]["label"] == json.loads(t3["legs"])[0]["label"]


def test_no_parlay_with_fewer_than_two_games_or_started_games():
    cfg = RiskConfig(bankroll=1000)
    one = [slate()[0]]
    assert build_parlays(rank_picks(one, cfg), one, "r", "t", "d") == []
    g = slate()
    g[0].start_utc = pd.Timestamp("2026-10-08T19:00:00Z")
    g[1].start_utc = g[2].start_utc = pd.Timestamp("2026-10-08T23:00:00Z")
    now = pd.Timestamp("2026-10-08T20:00:00Z")
    rows = build_parlays(rank_picks(g, cfg, now=now), g, "r", "t", "d", now=now)
    assert all(1 not in {l["game_id"] for l in json.loads(r["legs"])} for r in rows)            # the started game is never a leg


def seed_games(st):
    mk = lambda gid, hs, as_, per="REG": dict(game_id=gid, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home=f"H{gid}", away=f"A{gid}",
                                              home_score=hs, away_score=as_, status="FINAL" if hs is not None else "FUT", last_period=per if hs is not None else None,
                                              home_win=(1 if hs > as_ else 0) if hs is not None else None, source="nhl_api", updated_at=None)
    st.upsert("games", [mk(1, 4, 2), mk(2, 1, 3), mk(3, 5, 4, "OT"), mk(4, None, None)], ["game_id"])


def leg(gid, market, side, point, dec, pid=0):
    return {"game_id": gid, "market": market, "side": side, "point": point, "player_id": pid, "decimal": dec, "p_model": 0.5, "p_market": 0.5, "label": f"{market} {side}", "game": f"A{gid} @ H{gid}"}


def put(st, strat, legs, stake=10.0, dec=None, run="r1", at="t1", date="2026-10-08"):
    d = dec or float(pd.Series([l["decimal"] for l in legs]).prod())
    st.upsert("parlay_bets", [dict(run_id=run, run_at=at, game_date=date, strategy=strat, idx=0, n_legs=len(legs), label=" + ".join(l["label"] for l in legs), legs=json.dumps(legs),
                                   decimal=d, p_model=0.2, p_market=0.18, stake=stake, ev=0.0)], ["run_id", "strategy", "idx"])


def test_settlement_by_hand_won_lost_push_and_pending():
    st = Store(":memory:")
    seed_games(st)
    st.upsert("skater_game", [dict(game_id=1, team="H1", player_id=10, name="P", position="C", toi_sec=1000, goals=1, assists=0, points=1, sog=4)], ["game_id", "player_id"])
    put(st, "parlay_top2", [leg(1, "h2h", "home", None, 1.9), leg(3, "totals", "over", 8.5, 2.0)])                      # home won; total 9 (OT goals count) > 8.5: both hit
    put(st, "parlay_top3", [leg(1, "h2h", "home", None, 1.9), leg(2, "h2h", "home", None, 1.8), leg(4, "h2h", "away", None, 2.0)])      # leg 2 lost: lost even though game 4 is unfinished
    put(st, "parlay_players3", [leg(1, "player_sog", "over", 3.5, 1.9, 10), leg(4, "h2h", "home", None, 2.0)])           # shots 4 > 3.5 hit, but game 4 pending
    put(st, "parlay_favorites_control", [leg(1, "totals", "over", 6.0, 1.9), leg(3, "h2h", "home", None, 1.5)])           # total 6 pushes exactly: leg drops out, ticket pays the rest
    d = parlay_resolved(st).set_index("strategy")
    assert d.loc["parlay_top2", "result"] == "won" and d.loc["parlay_top2", "profit"] == pytest.approx(10 * (1.9 * 2.0 - 1))
    assert d.loc["parlay_top3", "result"] == "lost" and d.loc["parlay_top3", "profit"] == -10.0
    assert d.loc["parlay_players3", "result"] == "pending" and d.loc["parlay_players3", "profit"] == 0.0
    assert d.loc["parlay_favorites_control", "result"] == "won" and d.loc["parlay_favorites_control", "eff_decimal"] == pytest.approx(1.5)
    assert d.loc["parlay_favorites_control", "profit"] == pytest.approx(5.0)
    res = shadow_resolved(st)                                                                           # settled parlays flow into the paper-trading frame; pending ones do not
    assert set(res.strategy) == {"parlay_top2", "parlay_top3", "parlay_favorites_control"} and (res.market == "parlay").all()
    perf = shadow_performance(st)
    assert perf.loc["parlay_top2", "profit"] == pytest.approx(10 * 2.8) and perf.loc["parlay_top3", "roi"] == -1.0


def test_latest_run_per_day_wins_and_logs_roundtrip(tmp_path):
    st = Store(":memory:")
    seed_games(st)
    put(st, "parlay_top2", [leg(1, "h2h", "away", None, 2.0), leg(2, "h2h", "home", None, 2.0)], run="r1", at="t1")        # earlier ticket would lose
    put(st, "parlay_top2", [leg(1, "h2h", "home", None, 1.9), leg(3, "h2h", "home", None, 2.0)], run="r2", at="t2")        # the later run replaces it for the day
    d = parlay_resolved(st)
    assert len(d) == 1 and d.iloc[0].result == "won"
    export_logs(st, tmp_path, public_safe=True)
    fresh = Store(":memory:")
    assert restore_logs(fresh, tmp_path)["parlay_bets"] == 2
    assert len(fresh.df("SELECT * FROM parlay_bets")) == 2
