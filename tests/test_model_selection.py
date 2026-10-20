import numpy as np
import pandas as pd
import pytest

from nhlbet.report.evaluation import dm_test
from nhlbet.report.model_selection import compare_candidates, score_models, select_model


def frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    p_true = np.clip(rng.normal(0.55, 0.12, n), 0.1, 0.9)
    y = (rng.random(n) < p_true).astype(int)
    d = pd.DataFrame({"game_date": pd.date_range("2025-01-01", periods=n // 6 + 1).repeat(6)[:n], "y": y})
    d["good"] = p_true                                                                       # knows the truth
    d["ref"] = np.clip(p_true + rng.normal(0, 0.10, n), 0.05, 0.95)                          # noisy version of it
    d["same"] = np.clip(d.ref + rng.normal(0, 0.002, n), 0.05, 0.95)                         # indistinguishable from the reference
    d["worse"] = np.clip(0.5 + 0.2 * (d.ref - 0.5), 0.05, 0.95)                              # overshrunk toward 0.5
    return d


def test_the_dm_test_finds_a_real_improvement_and_not_a_null_one():
    d = frame()
    ll = lambda p: -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))
    assert dm_test(ll(d.ref), ll(d.good))["p"] < 0.05 and dm_test(ll(d.ref), ll(d.good))["mean_diff"] > 0
    assert dm_test(ll(d.ref), ll(d.same))["p"] > 0.05 or abs(dm_test(ll(d.ref), ll(d.same))["mean_diff"]) < 1e-4
    assert dm_test(ll(d.ref), ll(d.worse))["p"] > 0.5                                       # a worse candidate is never "significantly better"


def test_the_decision_rule_switches_only_for_a_candidate_both_tests_support():
    t = compare_candidates(frame(), "ref", ["good", "same", "worse"], B=500)
    assert list(t.model) == ["ref", "good", "same", "worse"] and t.set_index("model").loc["good", "qualifies"]
    assert not t.set_index("model").loc[["same", "worse"], "qualifies"].any()
    assert select_model(t) == "good"
    t2 = compare_candidates(frame(), "ref", ["same", "worse"], B=500)
    assert select_model(t2) == "ref"                                                          # nothing qualifies: the status quo stays


def test_holm_correction_is_applied_across_candidates():
    t = compare_candidates(frame(), "ref", ["good", "same", "worse"], B=500).set_index("model")
    c = t.loc[["good", "same", "worse"]]
    assert (c.p_boot_adj >= c.p_boot - 1e-12).all() and (c.p_dm_adj >= c.p_dm - 1e-12).all()


def test_a_candidate_must_pass_both_tests():
    t = compare_candidates(frame(), "ref", ["good"], B=500)
    t.loc[t.model == "good", "p_dm_adj"] = 0.5
    t["qualifies"] = (t.role == "candidate") & (t.p_boot_adj < 0.05) & (t.p_dm_adj < 0.05) & (t.gain > 0)
    assert select_model(t) == "ref"


def test_score_models_reports_gain_over_the_baseline():
    d = frame()
    d["home_rate"] = d.y.mean()
    s = score_models(d, ["good", "ref", "home_rate"], B=300).set_index("model")
    assert s.loc["good", "gain_over_home_rate"] > s.loc["ref", "gain_over_home_rate"] > 0 and s.loc["home_rate", "gain_over_home_rate"] == pytest.approx(0, abs=1e-12)


def test_blend_scores_weights_interpolate_between_market_and_model():
    from nhlbet.report.evaluation import blend_scores
    d = frame(400, 3)
    d["p_market"], d["p_model"], d["home_win"], d["date"] = d["ref"], d["good"], d["y"], d.game_date.dt.strftime("%Y-%m-%d")
    b = blend_scores(d, B=300).set_index("model_weight")
    assert b.loc[0.0, "gain_over_market"] == pytest.approx(0, abs=1e-12) and b.loc[1.0, "gain_over_market"] > 0
    assert b.loc[0.5, "log_loss"] < b.loc[0.0, "log_loss"]


def test_the_selection_logic_cannot_touch_the_holdout_lock_and_the_protocol_exists():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "reports" / "model_comparison_protocol.md").exists()
    src = (root / "nhlbet" / "report" / "model_selection.py").read_text()
    assert "load_lock" not in src and "holdout_lock" not in src and "validity.holdout" not in src
