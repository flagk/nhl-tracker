import numpy as np
import pandas as pd

from nhlbet.analysis.importance import candidate_columns, prune_correlated, select_features
from nhlbet.features.builder import BuilderConfig, FeatureBuilder, feature_group_map
from nhlbet.splits import expanding_folds, walk_forward_windows


def _dates():
    return pd.Series(np.repeat(pd.date_range("2023-10-01", periods=200), 6))


def test_walk_forward_is_strictly_time_ordered_and_never_splits_a_day():
    d = _dates()
    wins = list(walk_forward_windows(d, "2024-01-01", step_days=14, min_train=100))
    assert len(wins) > 5
    for tr, te in wins:
        assert d.iloc[tr].max() < d.iloc[te].min()                      # train strictly before test
        assert set(d.iloc[tr]).isdisjoint(set(d.iloc[te]))              # no shared calendar date
    all_test = np.concatenate([te for _, te in wins])
    assert len(all_test) == len(set(all_test))                          # every game tested at most once


def test_expanding_folds_ordered():
    d = _dates()
    folds = expanding_folds(d, 5)
    assert len(folds) == 5
    for tr, te in folds:
        assert d.iloc[tr].max() < d.iloc[te].min()
    assert all(len(a[0]) < len(b[0]) for a, b in zip(folds, folds[1:]))  # training window expands


def test_prune_correlated_keeps_more_predictive_twin():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    F = pd.DataFrame({"a": a, "a_copy": a + rng.normal(scale=0.01, size=500), "b": rng.normal(size=500)})
    kept, dropped = prune_correlated(F, list(F.columns), pd.Series({"a": 0.10, "a_copy": 0.05, "b": 0.02}), 0.9)
    assert set(kept) == {"a", "b"} and "a_copy" in dropped


def test_selection_finds_signal_and_rejects_noise():
    rng = np.random.default_rng(1)
    n = 1500
    sig, noise = rng.normal(size=n), rng.normal(size=n)
    y = (sig * 1.2 + rng.normal(size=n) > 0).astype(int)
    F = pd.DataFrame({"game_date": pd.date_range("2023-01-01", periods=n, freq="8h"), "home_win": y,
                      "d_signal": sig, "d_noise": noise, "d_noise2": rng.normal(size=n)})
    res = select_features(F, use_shap=False, n_folds=4, n_repeats=4)
    assert "d_signal" in res.kept and "d_noise" not in res.kept and "d_noise2" not in res.kept


def test_candidate_columns_prefers_differentials(league):
    f = FeatureBuilder(BuilderConfig()).build(league)
    cols = candidate_columns(f, max_missing=0.9)
    assert "d_gd_season_shrunk" in cols and "h_gd_season_shrunk" not in cols
    assert "h_b2b" in cols                                              # no differential exists -> raw kept
    assert set(feature_group_map(cols)) == set(cols)
