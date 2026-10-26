import numpy as np
import pandas as pd
import pytest

from nhlbet.stats.overfit import deflated_sharpe, expected_max_sharpe, pbo, reality_check


def noise(T=400, N=60, seed=0):
    return pd.DataFrame(np.random.default_rng(seed).normal(0, 1, (T, N)), columns=[f"s{i}" for i in range(N)])


def test_pbo_is_about_half_for_pure_noise_and_low_when_one_variant_truly_wins():
    n = pbo(noise())
    assert 0.35 < n["pbo"] < 0.65
    X = noise(seed=1); X["real"] = np.random.default_rng(2).normal(0.3, 1, len(X))
    assert pbo(X)["pbo"] < 0.1


def test_deflated_sharpe_penalises_the_number_of_trials():
    r = np.random.default_rng(3).normal(0.06, 1, 600)
    few = deflated_sharpe(r, 2, 0.0004)["dsr"]
    many = deflated_sharpe(r, 500, 0.0004)["dsr"]
    assert many < few and expected_max_sharpe(500, 0.0004) > expected_max_sharpe(2, 0.0004)
    best_of_noise = noise(600, 100, seed=4).mean().idxmax()
    X = noise(600, 100, seed=4)
    d = deflated_sharpe(X[best_of_noise].to_numpy(), 100, float((X.mean() / X.std()).var()))
    assert d["dsr"] < 0.95                                                          # the best of 100 noise strategies is not "significant" once deflated
    assert d["psr_vs_zero"] > d["dsr"]


def test_reality_check_and_spa_do_not_reward_data_mining_but_find_a_real_edge():
    X = noise(500, 80, seed=5)
    rc = reality_check(X, B=400)
    assert rc["p_white_reality_check"] > 0.1 and rc["p_hansen_spa"] > 0.1
    X["real"] = np.random.default_rng(6).normal(0.25, 1, len(X))
    rc2 = reality_check(X, B=400)
    assert rc2["best_variant"] == "real" and rc2["p_white_reality_check"] < 0.02 and rc2["p_hansen_spa"] < 0.02
