import numpy as np
import pytest

from nhlbet.betting_gate import ELIGIBLE, RESEARCH, evaluate_gate

D = np.arange(400) // 3
EQ = [100, 110, 108]


def test_eligible_only_when_every_criterion_holds():
    x = np.random.default_rng(0).normal(1.0, 2, 400)
    g = evaluate_gate(x, D, EQ)
    assert g["status"] == ELIGIBLE and all(c["ok"] for c in g["criteria"])
    assert evaluate_gate(x[:50], D[:50], EQ)["status"] == RESEARCH                       # too few picks
    assert evaluate_gate(x, D, [100, 130, 100])["status"] == RESEARCH                    # drawdown beyond the limit
    assert evaluate_gate(x, D, EQ, system_status="HALT_SOFT")["status"] == RESEARCH      # any halt reverts it
    assert evaluate_gate(np.random.default_rng(1).normal(0, 2, 400), D, EQ)["status"] == RESEARCH   # interval includes zero


def test_no_data_is_research_only():
    g = evaluate_gate([], [], [])
    assert g["status"] == RESEARCH and not g["criteria"][0]["ok"]


def test_small_positive_mean_below_the_bar_is_not_enough():
    x = np.random.default_rng(2).normal(0.3, 0.5, 600)
    g = evaluate_gate(x, np.arange(600) // 3, EQ)
    assert g["status"] == RESEARCH and not next(c for c in g["criteria"] if c["name"] == "min_mean_clv_pts")["ok"]
