"""Ablations run under the same walk-forward protocol and compare variants with the full model on identical games."""
import pandas as pd
import pytest

from nhlbet.analysis.ablation import group_ablation, period_stability, regularisation_sensitivity
from nhlbet.features.builder import BuilderConfig, FeatureBuilder

FEATS = ["d_gd_season_shrunk", "d_rest_days", "d_gd_ewm", "d_elo", "d_b2b_placeholder"]


@pytest.fixture(scope="module")
def F():
    from tests.conftest import make_league
    f = FeatureBuilder(BuilderConfig()).build(make_league(n_days=420, seed=3))
    f = f[f.home_score.notna()].sort_values("game_date")
    f["d_b2b_placeholder"] = 0.0
    return f


def test_group_ablation_compares_each_removal_with_the_full_model(F):
    feats = [c for c in FEATS if c in F.columns and c != "d_b2b_placeholder"]
    first = str(F.game_date.iloc[int(len(F) * 0.6)].date())
    t = group_ablation(F, feats, first)
    assert t.variant.iloc[0] == "all selected features" and t.verdict.iloc[0] == "reference" and t.cost_vs_full.iloc[0] == 0.0
    assert {"elo only", "home_rate only"} <= set(t.variant)
    assert (t.ci_low <= t.cost_vs_full + 1e-12).all() and (t.cost_vs_full <= t.ci_high + 1e-12).all()
    assert t.loc[t.variant == "home_rate only", "cost_vs_full"].iloc[0] > -0.02                  # synthetic strengths exist, so the base rate is not better than the model


def test_regularisation_sensitivity_and_period_stability_run_on_identical_games(F):
    feats = [c for c in FEATS if c in F.columns and c != "d_b2b_placeholder"]
    first = str(F.game_date.iloc[int(len(F) * 0.6)].date())
    s = regularisation_sensitivity(F, feats, first, grid=(0.003, 0.3))
    assert list(s.C) == [0.003, 0.3] and s.log_loss.between(0.4, 0.8).all()
    p = period_stability(F, feats, first, min_n=50)
    assert len(p) >= 1 and (p.n >= 50).all()
