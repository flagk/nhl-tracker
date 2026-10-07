from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nhlbet.data.ingest import reparse_skaters
from nhlbet.data.parsers import parse_boxscore
from nhlbet.data.store import Store
from nhlbet.features.players import add_asof_features
from nhlbet.models.props import MIN_GAMES, ShotsModel, baseline_lam
from nhlbet.odds.client import OddsAPIError, OddsClient
from nhlbet.odds.props import SOG_MARKET, PropQuote, latest_prop_prices, match_players, norm_name, roster_candidates
from nhlbet.odds.snapshots import parse_events, record_fetch
from nhlbet.pipeline import fetch_player_props
from nhlbet.report.betlog import shadow_resolved
from nhlbet.report.props import PropEngine
from nhlbet.risk.policy import RiskConfig
from nhlbet.risk.shadow import CONTROL_STAKE, PROP_MAX_PER_DAY, prop_shadow_bets
from tests import fakes as F
from tests.fakes import FakeResp, FakeSession


# ---------------------------------------------------------------- data capture
def test_boxscore_parser_keeps_shots_on_goal():
    sk = {(r["team"], r["player_id"]): r["sog"] for r in parse_boxscore(F.boxscore())["skaters"]}
    assert sk[("BOS", 11)] == 4 and sk[("TOR", 21)] == 2 and ("BOS", 12) not in sk          # the 0:00 skater did not play


def test_reparse_fills_shots_for_games_ingested_before_they_were_captured():
    st = Store(":memory:")
    st.upsert("skater_game", [dict(game_id=2023020001, team="BOS", player_id=11, name="S11", position="C", toi_sec=1080, goals=1, assists=0, points=1, sog=None),
                              dict(game_id=2023020001, team="TOR", player_id=21, name="S21", position="C", toi_sec=1020, goals=0, assists=0, points=0, sog=None)], ["game_id", "player_id"])

    class Client:
        calls = 0
        def boxscore(self, gid):
            Client.calls += 1
            return F.boxscore()
    assert reparse_skaters(Client(), st) == 1 and Client.calls == 1
    got = st.df("SELECT player_id, sog FROM skater_game")
    assert dict(zip(got.player_id, got.sog)) == {11: 4, 21: 2, 13: 2}                    # sog = pid % 4 + 1 in the fixture; the defenceman (13) is added by the reparse
    assert reparse_skaters(Client(), st) == 0 and Client.calls == 1                      # nothing left to do: no further reads


# ---------------------------------------------------------------- as-of features
def synth(days=150, seed=0, n_players=40):
    rng = np.random.default_rng(seed)
    rates = {pid: (rng.gamma(6, 0.3), "D" if pid % 3 == 0 else "C") for pid in range(n_players)}
    rows = []
    for d in range(days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for pid, (rate, pos) in rates.items():
            if rng.random() < 0.45:
                continue
            home, opp = int(rng.random() < 0.5), int(rng.integers(0, 8))
            lam = rate * (0.7 if pos == "D" else 1.0) * (1.15 if opp < 2 else 1.0) * (1.06 if home else 1.0)
            rows.append(dict(game_id=d * 100 + pid, game_date=day, player_id=pid, name=f"P{pid}", team="T", position=pos, toi_sec=1000 if pos == "C" else 1300,
                             sog=int(rng.poisson(lam)), is_home=home, opp=f"O{opp}"))
    return pd.DataFrame(rows)


def test_player_features_use_only_earlier_games():
    pg = synth()
    f = add_asof_features(pg)
    p = f[f.player_id == 5].sort_values("game_date")
    sog = p.sog.to_numpy()
    for i in (3, 10, 25):
        assert p.n_prev.iloc[i] == i and p.mean_sog.iloc[i] == pytest.approx(sog[:i].mean())             # strictly earlier games
        assert p.l10_sog.iloc[i] == pytest.approx(sog[max(0, i - 10):i].mean())
    pg2 = pg.copy()
    cut = pd.Timestamp("2024-03-01")
    pg2.loc[pg2.game_date >= cut, "sog"] = 99                                                            # corrupt everything from the cut on
    f2 = add_asof_features(pg2)
    a, b = f.set_index(["player_id", "game_id"]), f2.set_index(["player_id", "game_id"])
    early = a.index[a.game_date <= cut]
    cols = ["n_prev", "mean_sog", "ewm_sog", "l10_sog", "toi_l10"]
    # rows ON the cut date use games before it only, so they are unchanged too
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])


def test_model_beats_the_players_own_rate_when_opponent_and_venue_matter():
    pg = synth(days=220, seed=2)
    allowance = pd.DataFrame({"team": [f"O{i}" for i in range(8)], "game_date": pd.Timestamp("2024-01-01"), "allow": [33, 33, 28, 28, 28, 28, 28, 28], "lg": 29.5})
    pg["opp_factor"] = pg.opp.map({f"O{i}": (33 if i < 2 else 28) / 29.5 for i in range(8)})
    f = add_asof_features(pg)
    f["opp_factor"] = pg.set_index(["player_id", "game_id"]).opp_factor.reindex(pd.MultiIndex.from_frame(f[["player_id", "game_id"]])).to_numpy()
    tr, te = f[f.game_date < "2024-06-01"], f[(f.game_date >= "2024-06-01") & (f.n_prev >= MIN_GAMES)]
    m = ShotsModel().fit(tr)
    lam, base = m.lam(te), baseline_lam(te, m.pos_mean_)
    from scipy.stats import poisson
    nll_m, nll_b = -poisson.logpmf(te.sog, lam).mean(), -poisson.logpmf(te.sog, base).mean()
    assert nll_m < nll_b                                                                                 # opponent + venue carry real signal here
    assert lam.mean() == pytest.approx(te.sog.mean(), rel=0.04)                                          # unbiased on average
    over = m.p_over(lam, 2.5)
    assert abs(over.mean() - (te.sog > 2.5).mean()) < 0.03                                              # and calibrated on the over rate


def test_p_over_uses_the_negative_binomial_tail_by_hand():
    m = ShotsModel(); m.dispersion_ = 0.0
    lam = 2.0                                                              # Poisson(2): P(X<=2) = e^-2 (1 + 2 + 2) = 0.6767
    assert float(m.p_over([lam], 2.5)[0]) == pytest.approx(1 - np.exp(-2) * 5, abs=1e-9)
    far = float(m.p_over([lam], 7.5)[0])
    m.dispersion_ = 0.2
    assert float(m.p_over([lam], 7.5)[0]) > far                            # overdispersion fattens the far tail (a big-shots night is likelier)


# ---------------------------------------------------------------- odds
def test_names_match_across_formats_and_ambiguity_is_refused():
    assert norm_name("Connor McDavid") == norm_name("C. McDavid") == ("c", "mcdavid") and norm_name("Tim Stützle") == norm_name("T. Stutzle")
    roster = pd.DataFrame({"player_id": [1, 2, 3], "name": ["C. McDavid", "J. Smith", "J. Smith"]})
    assert match_players(["Connor McDavid", "John Smith", "Zach Nobody"], roster) == {"Connor McDavid": 1}          # two J. Smiths on the two teams: skip, never guess


def prop_event(day, books=(("bk1", 1.90, 1.90), ("bk2", 1.95, 1.87)), player="Connor McDavid", point=3.5, updated=None):
    updated = updated or f"{day}T15:50:00Z"
    bks = [{"key": k, "last_update": updated, "markets": [{"key": SOG_MARKET, "last_update": updated, "outcomes": [
        {"name": "Over", "description": player, "price": o, "point": point}, {"name": "Under", "description": player, "price": u, "point": point}]}]} for k, o, u in books]
    return {"id": "e1", "commence_time": f"{day}T23:00:00Z", "home_team": "Edmonton Oilers", "away_team": "Vancouver Canucks", "bookmakers": bks}


def test_prop_quotes_parse_store_and_ignore_in_play_prices(tmp_path):
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=1, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="EDM", away="VAN", home_score=None, away_score=None,
                             status="FUT", last_period=None, home_win=None, source="nhl_api", updated_at=None)], ["game_id"])
    rows = parse_events([prop_event("2026-10-08")], "t")
    assert {r["outcome"] for r in rows} == {"Over|Connor McDavid", "Under|Connor McDavid"} and {r["point"] for r in rows} == {3.5}
    record_fetch(st, type("Fx", (), {"events": [prop_event("2026-10-08")], "captured_at": "2026-10-08T16:00:00+00:00", "remaining": 300, "used": 5, "source": "live"})())
    record_fetch(st, type("Fx", (), {"events": [prop_event("2026-10-08", books=(("bk1", 1.2, 4.5),))], "captured_at": "2026-10-09T00:30:00+00:00", "remaining": 299, "used": 6, "source": "live"})())
    p = latest_prop_prices(st, 1)
    assert len(p) == 2 and set(p.book) == {"bk1", "bk2"} and p.attrs["captured_at"] == "2026-10-08T16:00:00+00:00"        # the post-puck-drop capture is never used


def test_fetching_props_is_bounded_cheap_and_never_raises(tmp_path):
    now = datetime(2026, 10, 8, 17, 0, tzinfo=timezone.utc)
    ev = lambda i, hours: {**prop_event("2026-10-08"), "id": f"e{i}", "commence_time": (now + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")}
    events = [ev(1, 5), ev(2, 1), ev(3, 2), ev(4, 3), ev(5, 12), ev(6, -1)]                  # soonest first; none already started; none more than 6 h away
    sess = FakeSession({"/events/": {**prop_event("2026-10-08")}})
    client = OddsClient(api_key="k", cache_dir=tmp_path / "c", session=sess, sleep=lambda s: None)
    st = Store(":memory:")
    r = fetch_player_props(st, client, events, max_games=3, now=now)
    assert r == {"games": 3} and [u.split("/events/")[1].split("/")[0] for u in sess.calls] == ["e2", "e3", "e4"]
    bad = OddsClient(api_key="k", cache_dir=tmp_path / "c2", session=FakeSession({"/events/": [FakeResp(422, {})]}), sleep=lambda s: None)
    out = fetch_player_props(st, bad, events, max_games=3, now=now)                          # a plan without prop access: reported, not raised
    assert out["games"] == 0 and "422" in out["error"]


# ---------------------------------------------------------------- engine, strategies, settlement
def engine_store():
    st = Store(":memory:")
    rng = np.random.default_rng(1)
    plist = {t: [(1000 * (i + 1) + k, f"{'ABCDEFGHIJ'[k]}. Pl{t}{k}", "D" if k % 3 == 0 else "C", rng.gamma(6, 0.3)) for k in range(9)] for i, t in enumerate(["EDM", "VAN"])}
    games, tg, sk = [], [], []
    base = pd.Timestamp("2024-01-01")
    for d in range(200):
        day = base + pd.Timedelta(days=d)
        h, a = ("EDM", "VAN") if d % 2 == 0 else ("VAN", "EDM")
        games.append(dict(game_id=d + 1, season=20232024, game_type=2, game_date=str(day.date()), start_utc=f"{day.date()}T23:00:00Z", home=h, away=a, home_score=3, away_score=2,
                          status="FINAL", last_period="REG", home_win=1, source="nhl_api", updated_at=None))
        for t in ("EDM", "VAN"):
            tg.append(dict(game_id=d + 1, team=t, opp=a if t == h else h, is_home=int(t == h), goals=3, goals_against=2, sog_for=30, sog_against=28 + int(rng.integers(0, 5))))
            for pid, nm, pos, rate in plist[t]:
                if rng.random() < 0.1:
                    continue
                sk.append(dict(game_id=d + 1, team=t, player_id=pid, name=nm, position=pos, toi_sec=1000, goals=0, assists=0, points=0, sog=int(rng.poisson(rate * (0.7 if pos == "D" else 1)))))
    st.upsert("games", games, ["game_id"]); st.upsert("team_game", tg, ["game_id", "team"]); st.upsert("skater_game", sk, ["game_id", "player_id"])
    day = base + pd.Timedelta(days=200)
    st.upsert("games", [dict(game_id=500, season=20232024, game_type=2, game_date=str(day.date()), start_utc=f"{day.date()}T23:00:00Z", home="EDM", away="VAN", home_score=None,
                             away_score=None, status="FUT", last_period=None, home_win=None, source="nhl_api", updated_at=None)], ["game_id"])
    return st, plist, str(day.date())


def test_engine_prices_matched_players_with_history_and_skips_unknowns():
    st, plist, day = engine_store()
    names = {pid: nm.replace(nm.split(".")[0] + ".", {"A": "Alex", "B": "Ben", "C": "Carl", "D": "Dan"}[nm[0]]) for pid, nm, *_ in plist["EDM"][:4]}
    ev = prop_event(day)
    ev["bookmakers"] = []
    for k, (o, u) in (("bk1", (1.9, 1.9)), ("bk2", (1.95, 1.87))):
        outs = []
        for pid, full in names.items():
            outs += [{"name": "Over", "description": full, "price": o, "point": 2.5}, {"name": "Under", "description": full, "price": u, "point": 2.5}]
        outs += [{"name": "Over", "description": "Nobody Here", "price": 1.9, "point": 2.5}, {"name": "Under", "description": "Nobody Here", "price": 1.9, "point": 2.5}]
        ev["bookmakers"].append({"key": k, "last_update": f"{day}T15:50:00Z", "markets": [{"key": SOG_MARKET, "last_update": f"{day}T15:50:00Z", "outcomes": outs}]})
    ev["commence_time"] = f"{day}T23:00:00Z"
    record_fetch(st, type("Fx", (), {"events": [ev], "captured_at": f"{day}T16:00:00+00:00", "remaining": 300, "used": 5, "source": "live"})())
    eng = PropEngine.create(st, as_of=pd.Timestamp(day))
    assert eng is not None
    qs = eng.quotes_for_game(500, pd.Timestamp(day))
    assert len(qs) == 4 and {q.player_id for q in qs} == set(names)                                       # four matched; "Nobody Here" is dropped
    for q in qs:
        assert 0.0 < q.p_over < 1.0 and 0.3 < q.p_over_market < 0.7 and q.n_books == 2 and len(q.history) == 10 and q.n_prev >= MIN_GAMES
        assert q.over_price == 1.95 and q.under_price == 1.9 and q.ev_over == pytest.approx(q.p_over * 0.95 - (1 - q.p_over))
        assert q.hit_l10 == pytest.approx(np.mean([h["sog"] > 2.5 for h in q.history]))
        assert q.avg_season is None or q.avg_season > 0                                       # shown only with 5+ games this season


def pq(pid, name, p_over, pm=0.5, point=2.5, over=1.91, under=1.91, n_prev=30):
    return PropQuote(1, pid, name, "EDM", "VAN", point, 2.8, p_over, pm, over, under, p_over - pm, p_over * (over - 1) - (1 - p_over), (1 - p_over) * (under - 1) - p_over, 3, n_prev)


class Gm:
    def __init__(self, gid, props, ctx=None):
        self.game_id, self.props = gid, props
        self.ctx = ctx if ctx is not None else {"model_status": "OK", "odds_stale": False}


def test_prop_strategies_follow_their_rules():
    cfg = RiskConfig(bankroll=1000)
    big = [pq(i, f"P{i}", 0.62 + 0.001 * i) for i in range(1, 13)]                               # 12 players with a 12+ point edge on the over
    small = [pq(100, "Small", 0.52), pq(101, "Rookie", 0.80, n_prev=4), pq(102, "Under", 0.40, pm=0.50)]
    rows = prop_shadow_bets([Gm(1, big + small), Gm(2, [], None), Gm(3, [pq(200, "Stale", 0.7)], {"model_status": "ALERT", "odds_stale": False})], cfg, "r", "t", "d")
    edge = [r for r in rows if r["strategy"] == "sog_edge"]
    assert len(edge) == PROP_MAX_PER_DAY and all(r["edge"] >= cfg.min_edge and r["side"] == "over" for r in edge)
    assert all(r["player_id"] not in (100, 101, 200) for r in edge) and all(5 <= r["stake"] <= 30 and r["stake"] == int(r["stake"]) for r in edge)
    assert {r["stake"] for r in edge} == {30.0}                                                  # 12-point+ edges all hit the maximum
    ctrl = [r for r in rows if r["strategy"] == "sog_over_control"]
    assert len(ctrl) == PROP_MAX_PER_DAY and {r["stake"] for r in ctrl} == {CONTROL_STAKE} and {r["side"] for r in ctrl} == {"over"}
    u = prop_shadow_bets([Gm(1, [pq(9, "Dog", 0.40, pm=0.5, under=1.95)])], cfg, "r", "t", "d")
    assert u[0]["side"] == "under" and u[0]["label"] == "Dog Under 2.5" and u[0]["p_model"] == pytest.approx(0.60) and u[0]["decimal"] == 1.95


def test_prop_settlement_by_hand_including_void_and_unsettled():
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=g, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="EDM", away="VAN", home_score=3, away_score=2 if g < 3 else None,
                             status="FINAL", last_period="REG", home_win=1 if g < 3 else None, source="nhl_api", updated_at=None) for g in (1, 2, 3)], ["game_id"])
    sk = lambda g, pid, sog: dict(game_id=g, team="EDM", player_id=pid, name=f"P{pid}", position="C", toi_sec=1000, goals=0, assists=0, points=0, sog=sog)
    st.upsert("skater_game", [sk(1, 10, 4), sk(1, 11, 2), sk(1, 99, 1), sk(2, 10, None)], ["game_id", "player_id"])      # game 1 has shot data; game 2's boxscores not reparsed yet
    bet = lambda run, at, gid, pid, side, pt, dec=2.0, stake=10.0, strat="sog_edge": dict(run_id=run, run_at=at, game_id=gid, strategy=strat, game_date="2026-10-08", player_id=pid, name=f"P{pid}",
                                                                                         side=side, label=f"P{pid} {side} {pt}", point=pt, book=None, decimal=dec, stake=stake, p_model=0.55,
                                                                                         p_market=0.5, edge=0.05, ev=0.1, lam=3.0)
    st.upsert("prop_bets", [bet("r1", "t1", 1, 10, "under", 2.5, stake=99.0),                    # superseded by the later run below
                            bet("r2", "t2", 1, 10, "over", 3.5), bet("r2", "t2", 1, 11, "over", 2.5), bet("r2", "t2", 1, 12, "over", 1.5),      # 4>3.5 won; 2<2.5 lost; player 12 never dressed -> void
                            bet("r2", "t2", 2, 10, "over", 1.5), bet("r2", "t2", 3, 10, "over", 1.5)], ["run_id", "game_id", "strategy", "player_id", "side"])      # unsettled: no shots yet / game not final
    d = shadow_resolved(st).set_index(["game_id", "player_id"])
    assert set(d.index) == {(1, 10), (1, 11), (1, 12)}                                           # games 2 and 3 are not settleable yet
    assert d.loc[(1, 10), "profit"] == 10.0 and d.loc[(1, 11), "profit"] == -10.0
    assert bool(d.loc[(1, 12), "push"]) and not d.loc[(1, 12), "is_bet"] and d.loc[(1, 12), "profit"] == 0.0
    assert d.loc[(1, 10), "market"] == "player_sog"


# ---------------------------------------------------------------- end to end
def test_end_to_end_props_reach_the_report_site_logs_and_exports(league, tmp_path, monkeypatch):
    import re
    from nhlbet.pipeline import run_daily
    from nhlbet.report.export import prop_predictions
    from tests.test_report_pipeline import _e2e_setup, _noon
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    rng = np.random.default_rng(3)
    letters = lambda n: "".join(chr(97 + (n // 26 ** i) % 26) for i in range(4))
    sk = league["skater_game"].copy()
    sk["name"] = [f"X. Sk{letters(int(p))}" for p in sk.player_id]
    sk["sog"] = rng.poisson(2.2, len(sk))
    st.upsert("skater_game", sk[["game_id", "team", "player_id", "name", "position", "toi_sec", "goals", "assists", "points", "sog"]].to_dict("records"), ["game_id", "player_id"])
    bos = sk[sk.team == "BOS"].drop_duplicates("player_id").head(3)
    ev = {"id": "e1", "commence_time": f"{day}T23:00:00Z", "home_team": "Boston Bruins", "away_team": "Toronto Maple Leafs", "bookmakers": []}
    for bk, o, u in (("bookA", 1.9, 1.9), ("bookB", 1.95, 1.87)):
        outs = []
        for nm in bos.name:
            full = nm.replace("X.", "Xavier")
            outs += [{"name": "Over", "description": full, "price": o, "point": 2.5}, {"name": "Under", "description": full, "price": u, "point": 2.5}]
        ev["bookmakers"].append({"key": bk, "last_update": f"{day}T16:25:00Z", "markets": [{"key": SOG_MARKET, "last_update": f"{day}T16:25:00Z", "outcomes": outs}]})
    record_fetch(st, type("Fx", (), {"events": [ev], "captured_at": f"{day}T16:30:00+00:00", "remaining": 300, "used": 5, "source": "live"})())
    run_daily(day, "morning", db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models", now=_noon(day))
    db = Store(str(tmp_path / "t.db"))
    aq = db.df("SELECT * FROM alt_quotes WHERE market = 'player_sog'")
    assert len(aq) == 6 and set(aq.player_id) == set(bos.player_id) and aq.exp_total.between(0.3, 8).all()          # three players x over/under
    rep = (tmp_path / "reports/latest.md").read_text()
    assert "Player props: shots on goal" in rep and "Experimental, no real stakes" in rep
    html = (tmp_path / "site/index.html").read_text()
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    import json
    payload = json.loads(m.group(1).replace("<\\/", "</"))
    props = payload["games"][0]["props"]
    assert len(props) == 3 and all(len(p["history"]) == 10 and 0 < p["p_over"] < 1 for p in props) and "playerProps" in html
    assert any((tmp_path / "data/logs/alt_quotes").glob("*.csv"))
    published = {str(p): p.read_text(errors="ignore") for d in ("reports", "site", "data/logs") for p in (tmp_path / d).rglob("*") if p.is_file() and p.suffix in (".md", ".html", ".json", ".csv")}
    assert not [k for k, v in published.items() if "bookA" in v or "bookB" in v]
    assert len(prop_predictions(db)) == 6 and prop_predictions(db).outcome.isna().all()                              # game not played yet: no outcome


def test_backtest_reports_calibration_and_never_trains_on_the_future():
    from nhlbet.models.props_eval import calibration_table, evaluate, walk_forward_props
    f = add_asof_features(synth(days=240, seed=5))
    P = walk_forward_props(f, "2024-05-01", 28, min_train=2000)
    assert len(P) > 1000 and P.game_date.min() >= pd.Timestamp("2024-05-01")
    ev = evaluate(P)
    assert ev.what.iloc[0].startswith("shots per player-game") and {"over 1.5 (log loss)", "over 2.5 (log loss)"} <= set(ev.what)
    assert abs(P.lam.mean() - P.sog.mean()) < 0.08                                              # unbiased out of sample
    cal = calibration_table(P)
    assert (cal.predicted - cal.actual).abs().max() < 0.07 and cal.predicted.is_monotonic_increasing
    f2 = f.copy()
    f2.loc[f2.game_date >= "2024-07-01", "sog"] = 40                                              # corrupt the future: earlier predictions must not move
    P2 = walk_forward_props(f2, "2024-05-01", 28, min_train=2000)
    early = P.game_date < "2024-06-01"
    pd.testing.assert_series_equal(P.loc[early, "lam"].reset_index(drop=True), P2.loc[P2.game_date < "2024-06-01", "lam"].reset_index(drop=True))


def test_prop_over_and_under_only_strategies():
    cfg = RiskConfig(bankroll=1000)
    props = [pq(1, "Over", 0.60), pq(2, "Under", 0.40), pq(3, "Meh", 0.51), pq(4, "Rookie", 0.70, n_prev=3)]
    rows = prop_shadow_bets([Gm(1, props)], cfg, "r", "t", "d")
    o = [r for r in rows if r["strategy"] == "sog_over_edge"]
    u = [r for r in rows if r["strategy"] == "sog_under_edge"]
    assert [(r["player_id"], r["side"]) for r in o] == [(1, "over")] and [(r["player_id"], r["side"]) for r in u] == [(2, "under")]
