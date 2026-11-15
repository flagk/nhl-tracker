"""Phase 1.5: duplicates, postponed / moved games, team-name mapping, time zones, and the odds-validity facts on the pick log."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.gate import game_day
from nhlbet.hygiene import validate_inputs
from nhlbet.report.betlog import shadow_resolved
from nhlbet.settle import VOID_STATES, is_void_game
from nhlbet.teams import team_from_name


def games(rows):
    return [dict(game_id=g, season=1, game_type=2, game_date=d, start_utc=f"{d}T23:00:00Z", home="H", away="A", home_score=h, away_score=a, status=st,
                 last_period="REG", home_win=None if h is None else int(h > a), source="nhl_api", updated_at=None) for g, d, h, a, st in rows]


def bet(gid, day, strat="s1", side="home", market=None, point=None, book=None):
    return dict(run_id="r", run_at=f"{day}T20:00:00Z", game_id=gid, strategy=strat, game_date=day, action="BET", side=side, team="x", book=book, decimal=2.0, stake=10.0,
                p_model=0.5, p_adj=0.5, p_market=0.5, edge=0.0, ev=0.0, market=market, point=point, label="x")


def test_a_bet_on_a_game_moved_to_another_date_is_void_not_graded():
    st = Store(":memory:")
    st.upsert("games", games([(1, "2026-10-08", 4, 2, "FINAL"),          # played as planned
                              (2, "2026-10-12", 3, 1, "FINAL")]), ["game_id"])   # the pick was made for 10-08, the game was moved to 10-12
    st.upsert("shadow_bets", [bet(1, "2026-10-08", "a"), bet(2, "2026-10-08", "b")], ["run_id", "game_id", "strategy"])
    d = shadow_resolved(st).set_index("strategy")
    assert d.loc["a", "is_bet"] and d.loc["a", "profit"] == 10.0
    assert not d.loc["b", "is_bet"] and d.loc["b", "push"] and d.loc["b", "profit"] == 0.0 and d.loc["b", "void_game"]      # stake refunded


def test_void_states_and_dates():
    assert all(is_void_game(s, "2026-10-08", "2026-10-08") for s in VOID_STATES)
    assert not is_void_game("FINAL", "2026-10-08", "2026-10-08") and is_void_game("FINAL", "2026-10-08", "2026-10-09")
    assert not is_void_game("FINAL", None, "2026-10-09")


def test_settlement_rules_are_applied_per_book(monkeypatch):
    """The same shootout game settles differently for a book that excludes the shootout goal from totals."""
    import nhlbet.settle as s
    real = s.rules_for
    monkeypatch.setattr(s, "rules_for", lambda book=None: {**real(book), "shootout_in_total": book != "strictbook"})
    st = Store(":memory:")
    g = games([(1, "2026-10-08", 4, 3, "FINAL")])
    g[0]["last_period"] = "SO"
    st.upsert("games", g, ["game_id"])
    st.upsert("shadow_bets", [bet(1, "2026-10-08", "dk", "over", "totals", 6.5, book="draftkings"), bet(1, "2026-10-08", "strict", "over", "totals", 6.5, book="strictbook")], ["run_id", "game_id", "strategy"])
    d = shadow_resolved(st).set_index("strategy")
    assert d.loc["dk", "profit"] == 10.0 and d.loc["strict", "profit"] == -10.0          # 7 goals counted vs 6


def test_postponed_games_are_never_picked(monkeypatch):
    from nhlbet.report import slate
    src = open(slate.__file__).read()
    assert "VOID_STATES" in src and "postponed / cancelled games get no picks" in src


def test_input_validation_flags_duplicate_games_and_inconsistent_results():
    st = Store(":memory:")
    ok = games([(1, "2026-10-08", 4, 2, "FINAL")])
    st.upsert("games", ok, ["game_id"])
    assert validate_inputs(st) == []
    dup = games([(2, "2026-10-08", 3, 1, "FINAL")])
    dup[0].update(game_date="2026-10-08", home="H", away="A")
    st.upsert("games", dup, ["game_id"])                                  # same teams, same date, different id
    assert validate_inputs(st)
    bad = Store(":memory:")
    b = games([(5, "2026-10-09", 4, 2, "FINAL")])
    b[0]["home_win"] = 0                                                  # winner flag contradicts the score
    bad.upsert("games", b, ["game_id"])
    assert validate_inputs(bad)


@pytest.mark.parametrize("name,abbr", [("Montréal Canadiens", "MTL"), ("St Louis Blues", "STL"), ("St. Louis Blues", "STL"), ("Utah Mammoth", "UTA"),
                                       ("Vegas Golden Knights", "VGK"), ("Toronto Maple Leafs", "TOR"), ("New York Rangers", "NYR"), ("New York Islanders", "NYI")])
def test_team_name_mapping(name, abbr):
    assert team_from_name(name) == abbr


def test_team_mapping_is_a_bijection_over_32_clubs():
    from nhlbet import teams
    names = {n for n in getattr(teams, "_NAME_INDEX")}
    abbrs = {teams._NAME_INDEX[n] for n in names}
    assert len(abbrs) >= 32 and None not in abbrs


@pytest.mark.parametrize("iso,day", [("2026-10-01T03:00:00+00:00", "2026-09-30"), ("2026-10-01T23:59:00+00:00", "2026-10-01"), ("2026-11-01T04:30:00+00:00", "2026-11-01"),
                                     ("2026-11-02T04:30:00+00:00", "2026-11-01"), ("2026-03-08T06:30:00+00:00", "2026-03-08"), ("2026-03-08T04:30:00+00:00", "2026-03-07")])
def test_game_day_is_the_eastern_calendar_day_including_dst_changes(iso, day):
    assert game_day(datetime.fromisoformat(iso)) == day


def test_log_integrity_reports_decision_time_price_facts(tmp_path):
    """Every bet must name a book, carry an odds timestamp before the run, and the run must precede puck drop."""
    import json
    from nhlbet.report.validation import log_integrity
    (tmp_path / "archive").mkdir()
    payload = {"games": [{"game_id": 1, "start_utc": "2026-10-08T23:00:00+00:00"}, {"game_id": 2, "start_utc": "2026-10-08T23:00:00+00:00"}]}
    (tmp_path / "archive" / "2026-10-08.html").write_text(f'<script id="payload" type="application/json">{json.dumps(payload)}</script>')
    rec = pd.DataFrame([
        dict(run_id="a", run_at="2026-10-08T21:00:00+00:00", game_id=1, action="BET", book="draftkings", odds_captured_at="2026-10-08T20:58:00+00:00", model_version="v1"),
        dict(run_id="a", run_at="2026-10-08T21:00:00+00:00", game_id=2, action="BET", book=None, odds_captured_at="2026-10-08T23:30:00+00:00", model_version="v1"),   # no book, odds after puck drop
        dict(run_id="a", run_at="2026-10-08T21:00:00+00:00", game_id=3, action="NO_BET", book=None, odds_captured_at="2026-10-08T20:59:00+00:00", model_version="v1")])
    (tmp_path / "logs" / "recommendations").mkdir(parents=True)
    rec.to_csv(tmp_path / "logs" / "recommendations" / "a.csv", index=False)
    r = log_integrity(tmp_path / "logs", tmp_path / "archive")["recommendations"]
    assert r["bets_without_book"] == 1 and r["odds_captured_after_puck_drop"] == 1 and r["odds_captured_after_run"] == 1


def test_a_goalie_confirmation_captured_after_the_decision_time_is_ignored():
    from nhlbet.data.goalies import goalie_display, starter_overrides
    st = Store(":memory:")
    st.upsert("goalie_confirmations", [dict(game_date="2026-10-08", team="H", player_id=7, name="Late Goalie", source="manual", captured_at="2026-10-08T22:30:00+00:00")], ["game_date", "team"])
    g = pd.DataFrame({"game_id": [1], "game_date": ["2026-10-08"], "home": ["H"], "away": ["A"]})
    before, after = pd.Timestamp("2026-10-08T21:00:00Z"), pd.Timestamp("2026-10-08T23:00:00Z")
    assert starter_overrides(st, g, as_of=before) == {} and starter_overrides(st, g, as_of=after) == {(1, "H"): 7}
    assert goalie_display(st, "H", "2026-10-08", as_of=before)[1] != "confirmed" and goalie_display(st, "H", "2026-10-08", as_of=after) == ("Late Goalie", "confirmed")


def test_reloading_the_confirmation_file_keeps_the_first_seen_time(tmp_path):
    from nhlbet.data.goalies import load_confirmations
    st = Store(":memory:")
    st.upsert("goalie_game", [dict(game_id=1, team="H", player_id=7, name="Starter G", started=1, toi_sec=3600, shots_against=30, saves=28, goals_against=2, xg_faced=2.0)], ["game_id", "player_id"])
    f = tmp_path / "c.csv"
    f.write_text("date,team,name\n2026-10-08,H,Starter G\n")
    load_confirmations(st, f)
    first = st.df("SELECT captured_at FROM goalie_confirmations").captured_at.iloc[0]
    st.upsert("goalie_confirmations", [dict(game_date="2026-10-08", team="H", player_id=7, name="Starter G", source="manual", captured_at="2026-10-08T15:00:00+00:00")], ["game_date", "team"])
    load_confirmations(st, f)                                                        # a later run re-reads the same file
    assert st.df("SELECT captured_at FROM goalie_confirmations").captured_at.iloc[0] == "2026-10-08T15:00:00+00:00" != first
