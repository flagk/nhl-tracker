"""The real-data check functions, exercised on the synthetic league so they cannot silently rot between CI runs."""
import pandas as pd
import pytest

from nhlbet.validity import realdata
from tests.conftest import make_league

FEATS = ["d_elo", "h_b2b", "a_b2b", "d_xg_share"]


@pytest.fixture(scope="module")
def tables():
    return make_league(n_days=300, seed=5)


def test_truncation_invariance_report_passes_on_a_clean_pipeline(tables):
    r = realdata.truncation_invariance(tables, FEATS, pd.Timestamp("2024-05-01"), "2024-02-01", step_days=14, min_train=120)
    assert r["pass"] and r["games_compared"] > 50 and r["max_abs_diff_models_stack_calibrators"] < 1e-9


def test_availability_report_catches_every_planted_sentinel(tables):
    r = realdata.availability_and_sentinel(tables, FEATS)
    assert r["pass"] and all(r["sentinel_caught"].values()) and r["clean"]


def test_shuffle_null_report_passes_on_a_clean_pipeline_and_shows_real_skill_in_the_real_world(tables):
    r = realdata.shuffle_null(tables, FEATS, "2024-01-15", step_days=14, min_train=120, seeds=(1, 2))
    assert r["pass"] and r["real"]["auc_elo"] > 0.52
    assert all(abs(s["auc_logistic"] - 0.5) < 0.1 for s in r["shuffled"])


def test_settlement_sensitivity_counts_totals_bets_that_flip_with_the_shootout_rule():
    from nhlbet.data.store import Store
    st = Store(":memory:")
    games = [dict(game_id=1, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home="H", away="A", home_score=4, away_score=3, status="FINAL",
                  last_period="SO", home_win=1, source="nhl_api", updated_at=None)]
    st.upsert("games", games, ["game_id"])
    st.upsert("shadow_bets", [dict(run_id="r", run_at="2026-10-08T20:00:00Z", game_id=1, strategy="s", game_date="2026-10-08", action="BET", side="over", team="x", book=None, decimal=2.0,
                                   stake=10.0, p_model=.5, p_adj=.5, p_market=.5, edge=0.0, ev=0.0, market="totals", point=6.5, label="x")], ["run_id", "game_id", "strategy"])
    r = realdata.settlement_sensitivity(st)
    assert r["bets_whose_result_changes"] == 1 and r["profit_so_adds_goal"] == 10.0 and r["profit_so_excluded"] == -10.0
