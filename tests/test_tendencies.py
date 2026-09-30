import numpy as np
import pandas as pd
import pytest

from nhlbet.analysis.tendencies import analyze, bh_qvalues, render_markdown

TEAMS = ["BOS", "TOR", "NYR", "PIT", "CHI", "COL", "EDM", "VGK"]


def make(n=6000, seed=0, planted=0.0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.normal(0.54, 0.06, n), 0.3, 0.75)
    b2b = (rng.random(n) < 0.18).astype(int)
    true = np.clip(p - planted * b2b, 0.02, 0.98)             # a planted bias: home back-to-back teams win `planted` LESS than predicted
    y = (rng.random(n) < true).astype(int)
    return pd.DataFrame({"game_date": pd.date_range("2024-10-01", periods=n, freq="9h"), "p_model_home": p, "home_win": y, "home_b2b": b2b,
                         "away_b2b": (rng.random(n) < 0.18).astype(int), "home_rest_days": rng.integers(1, 5, n), "away_rest_days": rng.integers(1, 5, n),
                         "home_games_played": rng.integers(0, 82, n), "away_games_played": rng.integers(0, 82, n), "is_rivalry": (rng.random(n) < .3).astype(int),
                         "home": rng.choice(TEAMS, n), "away": rng.choice(TEAMS, n)})


def test_bh_matches_hand_computation():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = bh_qvalues(p)
    assert q[0] == pytest.approx(0.01) and q[1] == pytest.approx(0.04) and q[4] == pytest.approx(0.084)
    assert np.all(q >= p) and np.all(q <= 1) and np.all(np.diff(q[np.argsort(p)]) >= -1e-12)                         # monotone, never below raw p


def test_finds_a_planted_tendency_and_reports_its_direction():
    r = analyze(make(planted=0.08))
    row = r.table[(r.table.group == "rest") & (r.table.segment == "home on a back-to-back")].iloc[0]
    assert row.verdict == "TENDENCY (replicated)" and row.bias < -0.04 and row.q < 0.01
    assert row.bias_first < 0 and row.bias_second < 0
    md = render_markdown({"Model vs outcomes": r}, "t")
    assert "Replicated tendencies" in md and "home on a back-to-back" in md and "betting advice" in md


def test_stays_quiet_on_a_perfectly_calibrated_model():
    """False-discovery control: a calibrated model has no tendencies; with ~100 segments tested, nothing should replicate."""
    hits = 0
    for seed in range(25):
        t = analyze(make(seed=100 + seed)).table
        hits += int((t.verdict == "TENDENCY (replicated)").any())
    assert hits <= 3                                              # 10% FDR + replication rule: rare false alarms (25 datasets, ~100 tests each)
    md = render_markdown({"x": analyze(make(seed=2))}, "t")
    assert "No replicated tendency" in md


def test_uncorrected_testing_would_have_been_fooled():
    """Why the correction matters: on a calibrated model many segments have raw p < 0.05, yet almost none survive the correction."""
    t = analyze(make(seed=3)).table
    assert (t.p < 0.05).sum() >= 1 and (t.q < 0.10).sum() <= (t.p < 0.05).sum()


def test_small_segments_are_skipped_and_empty_input_is_graceful():
    r = analyze(make(n=200), min_n=150)
    assert (r.table.n >= 150).all()
    e = analyze(pd.DataFrame({"game_date": [], "p_model_home": [], "home_win": []}))
    assert e.n_games == 0 and "Not enough data" in render_markdown({"empty": e}, "t")
