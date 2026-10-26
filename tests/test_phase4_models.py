import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import FeatureBuilder
from nhlbet.models.base import HomeRate
from nhlbet.models.calibration import Calibrator, calibration_slope_intercept, log_loss_
from nhlbet.models.hier import HierBT
from nhlbet.splits import walk_forward_windows
from tests.conftest import make_league


@pytest.fixture(scope="module")
def F():
    F = FeatureBuilder().build(make_league(n_days=300, seed=21))
    return F[(F.game_type == 2) & F.home_win.notna()]


def test_embargo_removes_the_days_before_each_block_and_zero_is_the_old_behaviour():
    d = pd.Series(pd.date_range("2024-01-01", periods=400, freq="D"))
    old = list(walk_forward_windows(d, "2024-09-01", 14, 50))
    same = list(walk_forward_windows(d, "2024-09-01", 14, 50, embargo_days=0))
    emb = list(walk_forward_windows(d, "2024-09-01", 14, 50, embargo_days=3))
    assert all((a[0] == b[0]).all() and (a[1] == b[1]).all() for a, b in zip(old, same))
    for (tr0, te), (tr3, te3) in zip(old, emb):
        assert (te == te3).all() and len(tr3) == len(tr0) - 3
        assert d[tr3].max() <= d[te].min() - pd.Timedelta(days=4)


def test_hierarchical_model_learns_strength_and_never_uses_future_rows(F):
    cut = F.game_date.sort_values().iloc[int(len(F) * 0.7)]
    tr, te = F[F.game_date < cut], F[F.game_date >= cut]
    m = HierBT().fit(tr, tr.home_win.astype(int))
    p = m.predict(te)
    base = HomeRate().fit(tr, tr.home_win.astype(int)).predict(te)
    y = te.home_win.astype(int).to_numpy()
    assert ((p > 0) & (p < 1)).all() and log_loss_(p, y) < log_loss_(base, y) + 0.005
    te2 = te.copy(); te2["home_win"] = 1 - te2.home_win
    assert np.allclose(m.predict(te2), p)                                          # predictions do not read the labels of the rows being predicted
    unseen = te.copy(); unseen["season"] = unseen.season + 5                      # a new season falls back to long-run strength
    q = m.predict(unseen)
    assert ((q > 0) & (q < 1)).all() and np.corrcoef(p, q)[0, 1] > 0.8


def test_beta_calibrator_fixes_a_distorted_probability_and_isotonic_runs():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.2, 0.8, 6000)
    y = (rng.uniform(size=6000) < p_true).astype(int)
    over = 1 / (1 + np.exp(-2.2 * np.log(p_true / (1 - p_true))))               # overconfident
    for method in ("platt", "beta", "isotonic"):
        c = Calibrator(method).fit(over[:3000], y[:3000])
        out = c.predict(over[3000:])
        assert log_loss_(out, y[3000:]) < log_loss_(over[3000:], y[3000:])
        if method != "isotonic":
            assert abs(calibration_slope_intercept(out, y[3000:])[0] - 1) < 0.15
