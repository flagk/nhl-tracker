import numpy as np
import pandas as pd
import pytest

from nhlbet.risk.uncertainty import estimate_theta, simulate_uncertain


def template(n=200, seed=0):
    rng = np.random.default_rng(seed)
    mk = rng.uniform(0.4, 0.6, n)
    return pd.DataFrame({"decimal": 1 / mk * 0.96, "p_market": mk, "p_model": mk + rng.uniform(0.02, 0.06, n)})


def test_theta_estimate_recovers_skill_and_its_uncertainty_shrinks_with_data():
    rng = np.random.default_rng(1)
    for true_theta in (0.0, 0.6):
        n = 4000
        mk = rng.uniform(0.35, 0.65, n); pm = mk + rng.normal(0, 0.05, n)
        y = (rng.uniform(size=n) < np.clip(mk + true_theta * (pm - mk), 0.01, 0.99)).astype(int)
        e = estimate_theta(pm, mk, y, np.arange(n) // 3, B=400)
        assert abs(e["theta"] - true_theta) < 3 * e["se"] + 0.05
    small = estimate_theta(pm[:200], mk[:200], y[:200], np.arange(200) // 3, B=400)
    assert small["se"] > e["se"]


def test_no_skill_loses_money_and_skill_makes_it_and_a_price_haircut_hurts():
    t = template()
    none = simulate_uncertain(t, 0.0, 0.0, n_paths=1500, horizon=300)
    good = simulate_uncertain(t, 1.0, 0.0, n_paths=1500, horizon=300)
    cut = simulate_uncertain(t, 1.0, 0.0, n_paths=1500, horizon=300, price_haircut=0.04)
    assert none["median_final"] < 1000 < good["median_final"] and cut["median_final"] < good["median_final"]
    assert none["p_loss"] > 0.7 and good["p_loss"] < 0.35 and good["p_loss"] < none["p_loss"] - 0.3


def test_full_kelly_has_deeper_drawdowns_than_quarter_kelly_and_uncertainty_widens_outcomes():
    t = template()
    q = simulate_uncertain(t, 0.5, 0.0, kelly_fraction=0.25, max_bet_pct=1.0, n_paths=1500, horizon=300)
    f = simulate_uncertain(t, 0.5, 0.0, kelly_fraction=1.0, max_bet_pct=1.0, n_paths=1500, horizon=300)
    assert f["p95_max_drawdown"] > q["p95_max_drawdown"]
    sure = simulate_uncertain(t, 0.5, 0.0, n_paths=1500, horizon=300)
    unsure = simulate_uncertain(t, 0.5, 0.6, n_paths=1500, horizon=300)
    assert (unsure["p95_final"] - unsure["p05_final"]) > (sure["p95_final"] - sure["p05_final"])
