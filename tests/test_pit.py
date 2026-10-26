"""Point-in-time machinery: immutable snapshots, knowledge-time as-of selection, and equivalence of the as-of builder with the plain builder when nothing was revised."""
from __future__ import annotations

import copy
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import FeatureBuilder
from nhlbet.pit.asof import make_adv_asof, revision_report, team_observations
from nhlbet.pit.snapshots import SnapshotStore
from tests import fakes as F


def test_snapshots_are_immutable_versioned_and_verifiable(tmp_path):
    s = SnapshotStore(tmp_path)
    v1 = s.put("pbp", 1, {"a": 1}, "2026-01-01T10:00:00+00:00", "u")
    v2 = s.put("pbp", 1, {"a": 2}, "2026-01-02T10:00:00+00:00", "u")
    v3 = s.put("pbp", 1, {"a": 2}, "2026-01-03T10:00:00+00:00", "u")           # unchanged refetch: new observation, same object
    assert v2.sha256 == v3.sha256 and v1.sha256 != v2.sha256 and len(s.versions("pbp", 1)) == 3
    assert s.as_of("pbp", 1, "2026-01-01T23:59:59+00:00") == v1 and s.as_of("pbp", 1, "2026-01-02T10:00:00+00:00") == v2
    assert s.as_of("pbp", 1, "2025-12-31T00:00:00+00:00") is None                    # nothing was known yet
    assert s.load(v1) == {"a": 1} and s.verify() == []
    obj = tmp_path / "objects" / v1.sha256[:2] / f"{v1.sha256}.json.gz"
    obj.write_bytes(gzip.compress(b'{"a":99}'))
    assert any("modified" in p for p in s.verify())
    with pytest.raises(RuntimeError):
        s.load(v1)
    obj.unlink()
    assert any("missing" in p for p in s.verify())


def test_snapshot_bytes_are_reproducible(tmp_path):
    a, b = SnapshotStore(tmp_path / "a"), SnapshotStore(tmp_path / "b")
    va, vb = a.put("x", 1, {"b": 2, "a": 1}, "2026-01-01T00:00:00Z", "u"), b.put("x", 1, {"a": 1, "b": 2}, "2026-05-05T00:00:00Z", "u")
    assert va.sha256 == vb.sha256
    pa = tmp_path / "a" / "objects" / va.sha256[:2] / f"{va.sha256}.json.gz"
    pb = tmp_path / "b" / "objects" / vb.sha256[:2] / f"{vb.sha256}.json.gz"
    assert pa.read_bytes() == pb.read_bytes()


def _league_tables(n_days=40, seed=0):
    from tests.conftest import make_league
    return make_league(n_days=n_days, seed=seed)


def _fake_pbp_for(game, scale=1.0):
    plays = [F.play(1, "shot-on-goal", 1, "05:00", 1, "1551", x=70.0, y=5.0, owner=1 if game["home"] == game["home"] else 2) for _ in range(1)] if hasattr(F, "play") else []
    return F.pbp(plays)


def test_asof_resolver_hides_later_versions_and_unknown_games():
    obs = [{"game_id": 1, "team": "BOS", "att_for": 50.0, "att_against": 40.0, "knowledge_time": "2026-01-01T20:00:00+00:00", "snapshot": "a"},
           {"game_id": 1, "team": "BOS", "att_for": 80.0, "att_against": 40.0, "knowledge_time": "2026-01-02T20:00:00+00:00", "snapshot": "b"}]
    r = make_adv_asof(obs, 13.0)
    d = lambda y, m, dd: pd.Timestamp(year=y, month=m, day=dd).toordinal()
    assert r(1, "BOS", d(2026, 1, 1)) == {}                                      # fetched at 20:00, decision at 13:00: not yet known
    assert r(1, "BOS", d(2026, 1, 2))["att_for"] == 50.0                         # first version
    assert r(1, "BOS", d(2026, 1, 3))["att_for"] == 80.0                         # the correction becomes visible only after it was fetched
    assert r(2, "BOS", d(2026, 1, 3)) == {}


def test_revision_report_counts_changes():
    obs = [{"game_id": 1, "team": "A", "att_for": 50.0, "knowledge_time": "t1", "snapshot": "a"}, {"game_id": 1, "team": "A", "att_for": 53.0, "knowledge_time": "t2", "snapshot": "b"},
           {"game_id": 2, "team": "A", "att_for": 10.0, "knowledge_time": "t1", "snapshot": "c"}]
    r = revision_report(obs)
    assert r["team_games"] == 2 and r["team_games_with_revisions"] == 1 and r["columns_changed"]["att_for"]["mean_abs_change"] == 3.0


def _tables_with_adv(seed=0):
    lg = _league_tables(seed=seed)
    return lg


def test_asof_builder_equals_plain_builder_when_nothing_was_revised():
    lg = _league_tables()
    tables = {k: v for k, v in lg.items() if k in ("games", "team_game", "goalie_game", "skater_game")}
    plain = FeatureBuilder().build(tables)
    adv_cols = [c for c in tables["team_game"].columns if c not in ("game_id", "team", "opp", "is_home", "goals", "goals_against")]
    # every game's stats were fetched the night of the game, long before the next decision date
    gd = tables["games"].set_index("game_id").game_date
    obs = [{"game_id": r.game_id, "team": r.team, **{c: getattr(r, c) for c in adv_cols}, "knowledge_time": (gd[r.game_id] + pd.Timedelta(hours=30)).tz_localize("UTC").isoformat(), "snapshot": "s"}
           for r in tables["team_game"].itertuples(index=False)]
    pit = FeatureBuilder().build(tables, adv_asof=make_adv_asof(obs, 13.0))
    pd.testing.assert_frame_equal(plain, pit, check_exact=False, rtol=0, atol=1e-12)


def test_a_late_correction_changes_only_decisions_after_it_was_fetched():
    lg = _league_tables()
    tables = {k: v for k, v in lg.items() if k in ("games", "team_game", "goalie_game", "skater_game")}
    adv_cols = [c for c in tables["team_game"].columns if c not in ("game_id", "team", "opp", "is_home", "goals", "goals_against")]
    gd = tables["games"].set_index("game_id").game_date
    base = [{"game_id": r.game_id, "team": r.team, **{c: getattr(r, c) for c in adv_cols}, "knowledge_time": (gd[r.game_id] + pd.Timedelta(hours=30)).tz_localize("UTC").isoformat(), "snapshot": "s"}
            for r in tables["team_game"].itertuples(index=False)]
    target = tables["games"].sort_values("game_date").game_id.iloc[30]
    fix_day = gd[target] + pd.Timedelta(days=5)
    rev = []
    for o in base:
        if o["game_id"] == target:
            n = copy.deepcopy(o)
            n["att_for"] = (n["att_for"] or 50.0) + 40.0
            n["knowledge_time"] = (fix_day + pd.Timedelta(hours=14)).tz_localize("UTC").isoformat()
            n["snapshot"] = "fix"
            rev.append(n)
    a = FeatureBuilder().build(tables, adv_asof=make_adv_asof(base, 13.0))
    b = FeatureBuilder().build(tables, adv_asof=make_adv_asof(base + rev, 13.0))
    diff = (a.fillna(-999) != b.fillna(-999)).any(axis=1)
    assert diff.any()                                                                       # the correction does matter once known
    assert (a.index[diff].map(a.game_date) > fix_day).all()                                 # ...and never before the day after it was fetched


def test_fetch_game_stores_versions_with_fetch_time_and_never_stores_failures(tmp_path):
    from datetime import datetime, timezone
    from nhlbet.pit.fetch import fetch_game, probe
    store = SnapshotStore(tmp_path)
    pbp, box = F.pbp([]), F.boxscore()
    answers = {"play-by-play": (200, pbp), "boxscore": (200, box), "shiftcharts": (500, None)}
    get = lambda url: next(v for k, v in answers.items() if k in url)
    clock = iter([datetime(2026, 1, 1, 12, tzinfo=timezone.utc), datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, 12, 2, tzinfo=timezone.utc)])
    r = fetch_game(store, 7, get=get, now=lambda: next(clock), sleep=lambda s: None)
    assert r["pbp"].startswith("stored") and r["boxscore"].startswith("stored") and r["shifts"] == "http 500"
    assert [v.kind for v in store.manifest()] == ["pbp", "boxscore"] and store.manifest()[0].knowledge_time.startswith("2026-01-01T12:00")
    answers["play-by-play"] = (200, {"unexpected": 1})                                   # a payload without the keys the parsers need is rejected, not stored
    r2 = fetch_game(store, 7, get=get, kinds=("pbp",), sleep=lambda s: None)
    assert r2["pbp"].startswith("schema") and len(store.manifest()) == 2
    res = probe(get=lambda url: (200, {"gameWeek": []}) if "schedule" in url else (404, None), game_id=7)
    assert res["endpoints"]["pbp"]["status"] == 404 and res["endpoints"]["pbp"]["has_required_keys"] is False


def test_goalie_event_log_keeps_every_confirmation_and_answers_as_of(tmp_path):
    from nhlbet.data.store import Store
    from nhlbet.pit.goalie_log import append_events, confirmed_as_of
    st = Store(str(tmp_path / "t.db")); log = tmp_path / "g" / "log.jsonl"
    row = lambda pid, name, t: dict(game_date="2026-10-20", team="BOS", player_id=pid, name=name, source="manual", captured_at=t)
    st.upsert("goalie_confirmations", [row(1, "A", "2026-10-20T15:00:00+00:00")], ["game_date", "team"])
    assert append_events(st, log) == 1 and append_events(st, log) == 0                        # idempotent
    st.upsert("goalie_confirmations", [row(2, "B", "2026-10-20T22:30:00+00:00")], ["game_date", "team"])   # late change overwrites the table row...
    assert append_events(st, log) == 1                                                       # ...but the log keeps both
    assert confirmed_as_of(log, "2026-10-20", "BOS", "2026-10-20T20:00:00+00:00")["name"] == "A"
    assert confirmed_as_of(log, "2026-10-20", "BOS", "2026-10-20T23:00:00+00:00")["name"] == "B"
    assert confirmed_as_of(log, "2026-10-20", "BOS", "2026-10-20T10:00:00+00:00") is None


def test_odds_archive_is_off_by_default_spends_nothing_and_respects_the_credit_floor(tmp_path, monkeypatch, capsys):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("odds_archive", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "odds_archive.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    class C:
        calls = 0
        def remaining_credits(self): return 120
        def fetch_odds(self, *a, **k):
            C.calls += 1
            return type("F", (), {"source": "live", "captured_at": "2026-10-20T12:00:00+00:00", "events": [], "remaining": 119, "used": 381})()
    assert m.main(client=C(), store=SnapshotStore(tmp_path)) == 0 and C.calls == 0           # disabled in the policy: no call at all
    monkeypatch.setattr(m, "policy_get", lambda *k, default=None: {"enabled": True, "min_interval_min": 10, "min_credits_remaining": 150}.get(k[-1], default))
    assert m.main(client=C(), store=SnapshotStore(tmp_path)) == 0 and C.calls == 0           # 120 < floor 150
    class C2(C):
        def remaining_credits(self): return 400
    assert m.main(client=C2(), store=SnapshotStore(tmp_path)) == 0 and C.calls == 1
    assert [v.kind for v in SnapshotStore(tmp_path).manifest()] == ["odds"]
    assert m.main(client=C2(), store=SnapshotStore(tmp_path), now=__import__("datetime").datetime(2026, 10, 20, 12, 5, tzinfo=__import__("datetime").timezone.utc)) == 0 and C.calls == 1   # inside the minimum gap
