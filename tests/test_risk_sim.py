import numpy as np
import pandas as pd
import pytest

from nhlbet.policy import policy_get
from nhlbet.risk.montecarlo import SCENARIOS, default_template, simulate, template_from_picks
from nhlbet.risk.policy import RiskConfig


def test_policy_asks_for_at_least_ten_thousand_paths_and_defines_the_kill_switch():
    assert policy_get("risk", "simulation", "paths") >= 10000
    assert 0 < policy_get("risk", "kill_switch", "max_drawdown_pct") < 1 and list(SCENARIOS.values()) == [1.0, 0.5, 0.0]


def test_scenarios_are_ordered_and_the_kill_switch_probability_is_reported():
    t = default_template(0.03, 1.95, 0.01)
    r = simulate(t, 4000, 200, (1.0, 0.5, 0.0), 1000.0, 0.5, (0.1, 0.2), seed=1, kill_dd=0.2)
    assert r.median_final.is_monotonic_decreasing and r.p_loss.is_monotonic_increasing and r.p_kill_switch.is_monotonic_increasing
    assert (r.p99_maxdd >= r.p90_maxdd).all() and (r.p90_maxdd >= r.median_maxdd).all()
    assert r.loc[0.0, "median_final"] < 1000                              # with no skill the margin eats the bankroll
    assert r.loc[0.0, "p_kill_switch"] == pytest.approx(r.loc[0.0, "p_maxdd_ge_20"], abs=1e-12)


def test_model_true_uses_the_raw_model_probability_not_the_shrunk_one():
    t = pd.DataFrame({"stake_pct": 0.01, "decimal": 2.0, "p_market": 0.5, "p_model": 0.60, "p_adj": 0.52}, index=range(5))
    hi = simulate(t, 3000, 100, (1.0,), seed=2).median_final.iloc[0]
    lo = simulate(t.assign(p_model=0.52), 3000, 100, (1.0,), seed=2).median_final.iloc[0]
    assert hi > lo


def test_template_stakes_follow_the_live_policy():
    cfg = RiskConfig()
    d = pd.DataFrame({"p_model": [0.62, 0.55, 0.51, 0.70], "p_market": [0.55, 0.50, 0.50, 0.50], "decimal": [1.8, 2.0, 2.0, 2.0]})
    t = template_from_picks(d, cfg)
    assert len(t) == 2                                                    # 0.51 has no 3-point edge; 0.70 vs 0.50 is a 20-point gap, which the live policy rejects as a probable model error
    assert (t.stake_pct <= cfg.max_bet_pct + 1e-12).all() and (t.stake_pct > 0).all()
