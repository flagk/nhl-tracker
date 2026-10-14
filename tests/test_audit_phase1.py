"""Audit phase 1: live predictions feed the monitor, stale data never publishes picks, tuning / selection stay before the evaluation window."""
import json
import pytest
from pathlib import Path

import pandas as pd

from nhlbet.data.store import Store
from nhlbet.hygiene import data_freshness
from nhlbet.train import live_predictions, resolved_history

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- live predictions feed the monitor
def _store_with_live():
    st = Store(":memory:")
    games = [dict(game_id=i, season=1, game_type=2, game_date=f"2026-10-0{i}", start_utc=f"2026-10-0{i}T23:00:00Z", home="H", away="A", home_score=3, away_score=1 + (i % 2) * 4,
                  status="FINAL", last_period="REG", home_win=1 if i % 2 else 0, source="nhl_api", updated_at=None) for i in range(1, 5)]
    st.upsert("games", games, ["game_id"])
    rec = lambda gid, at, p: dict(run_id=f"r{gid}{at}", run_at=f"2026-10-0{gid}T{at}:00:00+00:00", run_type="late", game_id=gid, game_date=f"2026-10-0{gid}", home="H", away="A",
                                  p_model=p, p_stack_raw=p, action="NO_BET", model_version="v")
    st.upsert("recommendations", [rec(1, "10", 0.40), rec(1, "20", 0.60), rec(2, "20", 0.55), rec(3, "20", 0.70), rec(4, "20", None)], ["run_id", "game_id"])
    F = pd.DataFrame({"game_date": pd.to_datetime([g["game_date"] for g in games]), "home_win": [g["home_win"] for g in games], "elo_prob": [0.5] * 4},
                     index=pd.Index([g["game_id"] for g in games], name="game_id"))
    return st, F


def test_live_predictions_use_the_last_pregame_run_and_only_resolved_games():
    st, F = _store_with_live()
    d = live_predictions(st, F)
    assert list(d["stack"]) == [0.60, 0.55, 0.70] and list(d.y) == [1, 0, 1] and (d.elo == 0.5).all()          # game 1 uses the 20:00 run, game 4 had no probability
    assert len(live_predictions(st, F, after=pd.Timestamp("2026-10-01"))) == 2


def test_resolved_history_appends_live_games_after_the_backtest(tmp_path):
    wf = tmp_path / "wf.csv"
    pd.DataFrame({"game_date": ["2026-09-28", "2026-10-01"], "y": [1, 0], "stack": [0.5, 0.5], "elo": [0.5, 0.5]}).to_csv(wf)
    st, F = _store_with_live()
    h = resolved_history(wf, live=live_predictions(st, F))
    assert list(h.game_date.dt.strftime("%m-%d")) == ["09-28", "10-01", "10-02", "10-03"]                  # the 10-01 live game is already in the backtest file: not counted twice


# ---------------------------------------------------------------- stale data never publishes picks
def test_stale_data_withholds_picks_only_when_games_are_scheduled(tmp_path):
    st = Store(":memory:")
    g = lambda gid, d, score: dict(game_id=gid, season=1, game_type=2, game_date=d, start_utc=f"{d}T23:00:00Z", home="H", away="A", home_score=score, away_score=1 if score else None,
                                   status="FINAL" if score else "FUT", last_period="REG" if score else None, home_win=1 if score else None, source="nhl_api", updated_at=None)
    st.upsert("games", [g(1, "2026-10-01", 3), g(2, "2026-10-10", None)], ["game_id"])
    assert data_freshness(st, "2026-10-10", None) == (True, "")                                    # refresh worked
    ok, msg = data_freshness(st, "2026-10-10", "timeout")
    assert not ok and "9 days" in msg                                                              # refresh failed, newest result is 9 days old, games tonight
    assert data_freshness(st, "2026-10-03", "timeout")[0]                                           # refresh failed but cached data is only 2 days old: allowed (with a note)
    assert data_freshness(st, "2026-10-20", "timeout")[0]                                           # nothing scheduled that day: nothing to withhold


# ---------------------------------------------------------------- provenance: nothing was tuned or selected on the evaluation window
def test_feature_selection_and_tuning_ended_before_the_walk_forward_evaluation_began():
    wf_first = pd.read_csv(ROOT / "reports/walkforward_predictions.csv", usecols=["game_date"]).game_date.min()
    sel = json.loads((ROOT / "data/models/feature_selection.json").read_text())["meta"]["selection_end"]
    tune = json.loads((ROOT / "data/models/hyperparams.json").read_text())["_meta"]["tune_end"]
    assert pd.Timestamp(sel) < pd.Timestamp(wf_first) and pd.Timestamp(tune) < pd.Timestamp(wf_first)


def test_input_validation_catches_duplicates_and_inconsistent_results():
    from nhlbet.hygiene import validate_inputs
    st = Store(":memory:")
    g = lambda gid, d, hs, as_, win: dict(game_id=gid, season=1, game_type=2, game_date=d, start_utc=f"{d}T23:00:00Z", home="H", away="A", home_score=hs, away_score=as_, status="FINAL",
                                          last_period="REG", home_win=win, source="nhl_api", updated_at=None)
    st.upsert("games", [g(1, "2026-10-01", 3, 1, 1), g(2, "2026-10-02", 2, 4, 0)], ["game_id"])
    assert validate_inputs(st) == []
    st.upsert("games", [g(3, "2026-10-02", 2, 4, 0)], ["game_id"])                                  # same date, home and away as game 2
    assert any("duplicate (date, home, away)" in p for p in validate_inputs(st))
    st.upsert("games", [g(4, "2026-10-03", 5, 2, 0)], ["game_id"])                                  # home won 5-2 but home_win says 0
    assert any("disagrees with the score" in p for p in validate_inputs(st))


def test_margin_free_closing_line_value_is_centred_on_zero_for_a_no_skill_bet():
    from nhlbet.report.betlog import performance, resolved
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=1, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="H", away="A", home_score=3, away_score=1,
                             status="FINAL", last_period="REG", home_win=1, source="nhl_api", updated_at=None)], ["game_id"])
    rec = dict(run_id="r", run_at="2026-10-08T20:00:00+00:00", run_type="late", game_id=1, game_date="2026-10-08", home="H", away="A", p_model=0.6, p_adj=0.55, p_market=0.50, p_stack_raw=0.6,
               action="BET", side="home", team="H", decimal=1.95, stake=10.0, model_version="v")
    st.upsert("recommendations", [rec], ["run_id", "game_id"])
    cons = pd.DataFrame([{"game_id": 1, "captured_at": "2026-10-08T22:50:00+00:00", "home_prob_novig": 0.53, "n_books": 5}])
    d = resolved(st, cons)
    assert d.clv_move_pts.iloc[0] == pytest.approx(3.0)                                              # the no-vig line moved 3 points toward home after the pick
    assert d.clv_ev.iloc[0] == pytest.approx(1.95 * 0.53 - 1)                                       # the older measure also pays the margin of the price taken: -3.4% here
    cons2 = cons.assign(home_prob_novig=0.50)
    assert resolved(st, cons2).clv_move_pts.iloc[0] == pytest.approx(0.0) and resolved(st, cons2).clv_ev.iloc[0] < 0      # no movement: margin-free CLV is zero while CLV-with-margin is negative
