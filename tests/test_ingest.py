import requests

import pytest

from nhlbet.data import ingest
from nhlbet.data.client import NHLAPIError, NHLClient
from nhlbet.data.store import Store
from tests import fakes as F
from tests.fakes import FakeResp, FakeSession


def mk_client(tmp_path, routes, retries=3):
    sess = FakeSession(routes)
    return NHLClient(cache_dir=tmp_path / "raw", min_interval=0, retries=retries, session=sess, sleep=lambda s: None), sess


def test_client_caches_and_retries(tmp_path):
    routes = {"/v1/x": [FakeResp(500), requests.ConnectionError("boom"), FakeResp(200, {"ok": 1})]}
    c, s = mk_client(tmp_path, routes)
    assert c.get("/v1/x") == {"ok": 1}
    n = len(s.calls)
    assert n == 3
    assert c.get("/v1/x") == {"ok": 1} and len(s.calls) == n          # served from disk cache


def test_client_404_raises_and_gives_up(tmp_path):
    c, _ = mk_client(tmp_path, {})
    with pytest.raises(NHLAPIError):
        c.get("/v1/missing")
    c2, _ = mk_client(tmp_path / "b", {"/v1/y": [FakeResp(503)]}, retries=2)
    with pytest.raises(NHLAPIError):
        c2.get("/v1/y")


def test_client_falls_back_to_stale_cache(tmp_path):
    routes = {"/v1/z": [FakeResp(200, {"v": 1})]}
    c, s = mk_client(tmp_path, routes)
    c.get("/v1/z", ttl=1)
    import os, time
    p = c._path("https://api-web.nhle.com/v1/z"); os.utime(p, (time.time() - 100, time.time() - 100))
    s.routes["/v1/z"] = [FakeResp(500)]
    assert c.get("/v1/z", ttl=1) == {"v": 1}                          # expired but API down -> stale beats nothing


def test_client_honours_ttl(tmp_path):
    c, s = mk_client(tmp_path, {"/v1/t": [FakeResp(200, {"n": 1}), FakeResp(200, {"n": 2})]})
    assert c.get("/v1/t", ttl=3600) == {"n": 1}
    assert c.get("/v1/t", ttl=3600) == {"n": 1}
    assert len(s.calls) == 1


def _routes():
    pbp = F.pbp([F.play(1, "shot-on-goal", F.HOME_ID, x=80, y=5, shotType="wrist", goalieInNetId=41, shootingPlayerId=1)])
    sched = {"games": [F.schedule_game(2023020001, hs=3, as_=2), F.schedule_game(2023020002, home="NYR", away="PIT", date="2023-10-11", state="FUT")]}
    return {"club-schedule-season": sched, "/v1/schedule/": {"gameWeek": []}, "boxscore": F.boxscore(), "play-by-play": pbp}


def test_full_ingest_is_idempotent(tmp_path):
    client, sess = mk_client(tmp_path, _routes())
    store = Store(":memory:")
    ingest.ingest_schedule(client, store, 20232024)
    ok, bad = ingest.ingest_details(client, store)
    assert (ok, bad) == (1, 0)                                        # only the FINAL game gets detail
    def dump(t):
        return store.df(f"SELECT * FROM {t}").drop(columns=["updated_at"], errors="ignore")

    snap = {t: dump(t) for t in ("games", "team_game", "goalie_game", "skater_game", "shots")}
    assert len(snap["games"]) == 2 and len(snap["team_game"]) == 2 and len(snap["shots"]) == 1
    # second run: same DB state, zero new network calls, zero new rows
    calls = len(sess.calls)
    ingest.ingest_schedule(client, store, 20232024)
    assert ingest.ingest_details(client, store) == (0, 0)
    assert len(sess.calls) == calls
    for t, frame in snap.items():
        assert dump(t).equals(frame), t
    assert store.df("SELECT xg_faced FROM goalie_game WHERE player_id=41").xg_faced[0] > 0


def test_detail_failure_is_logged_and_retried_next_run(tmp_path):
    routes = _routes(); routes["boxscore"] = [FakeResp(500)]
    client, _ = mk_client(tmp_path, routes, retries=1)
    store = Store(":memory:")
    ingest.ingest_schedule(client, store, 20232024)
    assert ingest.ingest_details(client, store) == (0, 1)
    assert store.df("SELECT ok FROM ingest_log").ok[0] == 0
    client2, _ = mk_client(tmp_path / "again", _routes())
    assert ingest.ingest_details(client2, store) == (1, 0)             # failed games are retried, not skipped


def test_legacy_import_and_dedupe(tmp_path):
    from nhlbet.data.loaders import load_games
    csv = tmp_path / "h.csv"
    csv.write_text("Date,Home,Away,HomeScore,AwayScore,Winner,Points,GoalDiff,GoalsFor,GoalsAgainst\n"
                   "2023-10-10,BOS,TOR,3,2,BOS,2,1,3,2\n2023-10-12,NYR,PIT,1,2,PIT,0,-1,1,2\n")
    store = Store(":memory:")
    assert ingest.import_legacy_csv(store, csv) == 2
    assert ingest.import_legacy_csv(store, csv) == 2 and len(store.df("SELECT * FROM games")) == 2   # idempotent
    client, _ = mk_client(tmp_path, _routes())
    ingest.ingest_schedule(client, store, 20232024)
    g = load_games(store)
    assert len(g[(g.home == "BOS")]) == 1 and g[g.home == "BOS"].source.iloc[0] == "nhl_api"      # API replaces legacy
    assert (g.source == "legacy_csv").sum() == 1


def test_season_teams_handles_relocation():
    assert "ARI" in ingest.season_teams(20232024) and "UTA" not in ingest.season_teams(20232024)
    assert "UTA" in ingest.season_teams(20242025) and "ARI" not in ingest.season_teams(20242025)
    assert len(ingest.season_teams(20242025)) == 32


def test_circuit_breaker_stops_hammering_a_dead_api(tmp_path):
    sess = FakeSession({"/v1/": [FakeResp(503)]})
    c = NHLClient(cache_dir=tmp_path / "raw", min_interval=0, retries=2, session=sess, sleep=lambda s: None, max_consecutive_failures=2)
    for i in range(2):
        with pytest.raises(NHLAPIError):
            c.get(f"/v1/a{i}")
    calls = len(sess.calls)
    with pytest.raises(NHLAPIError, match="circuit open"):
        c.get("/v1/a3")
    assert len(sess.calls) == calls                                   # no further network attempts


def test_season_comes_from_the_api_id_not_the_calendar():
    """The 2019-20 bubble playoffs were played Aug-Sep 2020; a date rule files them under 2020-21. The API's id is authoritative,
    and legacy rows (no season id) still use the date rule."""
    from nhlbet.data.loaders import load_games
    st = Store(":memory:")
    base = dict(game_type=2, start_utc=None, home_score=3, away_score=1, status="FINAL", last_period="REG", home_win=1, updated_at=None)
    st.upsert("games", [
        dict(game_id=1, season=20192020, game_date="2019-10-03", home="BOS", away="TOR", source="nhl_api", **base),
        dict(game_id=2, season=20192020, game_date="2020-08-15", home="BOS", away="TBL", source="nhl_api", **{**base, "game_type": 3}),   # bubble playoff
        dict(game_id=3, season=20202021, game_date="2021-01-14", home="TOR", away="MTL", source="nhl_api", **base),
        dict(game_id=-1, season=None, game_date="2023-10-10", home="NYR", away="PIT", source="legacy_csv", **base),
        dict(game_id=-2, season=None, game_date="2024-03-10", home="NYR", away="BOS", source="legacy_csv", **base)], ["game_id"])
    g = load_games(st).set_index("game_id").season
    assert g[1] == 2019 and g[2] == 2019 and g[3] == 2020          # bubble playoff stays in 2019-20
    assert g[-1] == 2023 and g[-2] == 2023                          # legacy fallback: Aug-1 boundary
    assert str(g.dtype).startswith("int")
