import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import FeatureBuilder
from nhlbet.validity.placebo import adversarial_validation, shuffled_market_placebo, time_shift_placebo
from tests.conftest import make_league


@pytest.fixture(scope="module")
def F():
    return FeatureBuilder().build(make_league(n_days=300, seed=11))


def test_time_shift_placebo_positive_control_future_values_jump_stale_values_fade(F):
    r = time_shift_placebo(F).set_index("shift_games")
    assert r.loc[-1, "auc"] > r.loc[0, "auc"] + 0.03                  # one game into the future carries the game's own result: the placebo SEES a leak
    assert r.loc[0, "auc"] >= r.loc[10, "auc"] - 0.03                 # stale values never beat the current ones by more than noise
    assert r.loc[-1, "meaning"] == "FUTURE value" and r.loc[0, "meaning"].startswith("current")


def test_adversarial_validation_detects_a_planted_shift_and_not_a_clean_split():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"d_{i}" for i in range(6)])
    X["game_date"] = pd.date_range("2024-01-01", periods=n, freq="D")
    mid = X.game_date.iloc[n // 2]
    feats = [c for c in X.columns if c.startswith("d_")]
    clean = adversarial_validation(X, feats, mid)
    bad_X = X.copy(); bad_X.loc[bad_X.game_date >= mid, "d_0"] += 1.5
    bad = adversarial_validation(bad_X, feats, mid)
    assert clean["auc"] < 0.58 and bad["auc"] > 0.8 and bad["top_features"][0][0] == "d_0"


def test_shuffled_market_placebo_efficient_market_and_uninformative_market():
    rng = np.random.default_rng(1)
    n = 20000
    p_true = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    market = np.clip(p_true + rng.normal(0, 0.03, n), 0.05, 0.95)                 # an almost efficient market
    model = np.clip(p_true + rng.normal(0, 0.06, n), 0.05, 0.95)                  # a noisier model than the market
    eff = shuffled_market_placebo(model, market, y)
    assert eff["shuffled_roi_mean"] > 0.2 and eff["actual_roi"] < 0.0              # profit against prices that do not belong to the game; a loss against the right ones
    assert eff["information_already_in_the_price"] > 0.8
    junk = rng.uniform(0.3, 0.7, n)                                                # a market with no information about the game at all
    unin = shuffled_market_placebo(model, junk, y)
    assert unin["actual_roi"] == pytest.approx(unin["shuffled_roi_mean"], abs=0.05) and abs(unin["information_already_in_the_price"]) < 0.15
