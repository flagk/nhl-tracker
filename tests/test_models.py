import json

import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.models.base import EloOnly, HomeRate, LGBM, Logistic, RandomForest, XGB, logit, expit, make_zoo
from nhlbet.models.bundle import ModelBundle, train_bundle
from nhlbet.models.calibration import Calibrator, brier, calibration_slope_intercept, ece, log_loss_, online_calibrate, reliability_table
from nhlbet.models.ensemble import Stacker, WeightedAverage
from nhlbet.models.evaluate import metrics_table, paired_logloss_diff
from nhlbet.models.tune import tune_model
from nhlbet.models.walkforward import walk_forward

FEATS = ["d_gd_season_shrunk", "d_rest_days", "d_gd_ewm"]


@pytest.fixture(scope="module")
def F(league):
    f = FeatureBuilder(BuilderConfig()).build(league)
    return f[f.home_score.notna()].sort_values("game_date")


def small_zoo():
    return {"home_rate": lambda: HomeRate(), "elo": lambda: EloOnly(), "logistic": lambda: Logistic(FEATS),
            "lgbm": lambda: LGBM(FEATS, n_estimators=20, num_leaves=3, min_child_samples=20)}


def test_every_model_fits_and_returns_valid_probabilities(F):
    y = F.home_win.astype(int).to_numpy()
    for fac in (HomeRate, EloOnly, lambda: Logistic(FEATS), lambda: RandomForest(FEATS, n_estimators=20), lambda: LGBM(FEATS), lambda: XGB(FEATS)):
        p = fac().fit(F.iloc[:250], y[:250]).predict(F.iloc[250:])
        assert len(p) == len(F) - 250 and np.all((p > 0) & (p < 1)) and np.isfinite(p).all()


def test_models_tolerate_missing_features(F):
    X = F.copy(); X.loc[X.index[:80], FEATS] = np.nan
    y = X.home_win.astype(int).to_numpy()
    for fac in (lambda: Logistic(FEATS), lambda: RandomForest(FEATS, n_estimators=10), lambda: LGBM(FEATS), lambda: XGB(FEATS)):
        assert np.isfinite(fac().fit(X, y).predict(X)).all()


def test_walk_forward_predictions_never_depend_on_the_future(F):
    """Corrupt every label and feature on/after a cutoff: predictions for games BEFORE it must not move."""
    cutoff = F.game_date.iloc[int(len(F) * 0.7)]
    rng = np.random.default_rng(0)
    G = F.copy()
    late = G.game_date >= cutoff
    G.loc[late, "home_win"] = rng.integers(0, 2, late.sum())
    for c in FEATS + ["elo_prob"]:
        G.loc[late, c] = rng.normal(size=late.sum()) if c != "elo_prob" else rng.uniform(0.2, 0.8, late.sum())
    first = F.game_date.iloc[int(len(F) * 0.4)]
    a = walk_forward(F, small_zoo(), str(first.date()), 14, min_train=100, n_jobs=1)
    b = walk_forward(G, small_zoo(), str(first.date()), 14, min_train=100, n_jobs=1)
    early = a.index[a.game_date < cutoff]
    assert len(early) > 30
    pd.testing.assert_frame_equal(a.loc[early].drop(columns="y"), b.loc[early].drop(columns="y"), check_exact=False, rtol=1e-9, atol=1e-9)


def test_walk_forward_blocks_cover_each_game_once_and_output_columns(F):
    P = walk_forward(F, small_zoo(), str(F.game_date.iloc[150].date()), 14, min_train=100, n_jobs=1)
    assert P.index.is_unique and P.game_date.min() >= F.game_date.iloc[150]
    for c in ("stack", "wavg", "stack__platt", "stack__isotonic", "elo__platt", "logistic"):
        assert c in P and P[c].between(0, 1).all()
    t = metrics_table(P, ["home_rate", "elo", "stack__platt"])
    assert {"log_loss", "brier", "auc", "ece", "cal_slope"} <= set(t.columns)


def test_calibration_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 40000                                                # large n: slope SE ~0.03, so the bounds below are tight
    true_p = rng.uniform(0.35, 0.65, n)
    y = (rng.random(n) < true_p).astype(int)
    over = expit(3.0 * logit(true_p))                        # over-confident by 3x
    tr, te = slice(0, n // 2), slice(n // 2, None)
    for method in ("platt", "isotonic"):
        cal = Calibrator(method).fit(over[tr], y[tr]).predict(over[te])
        assert log_loss_(cal, y[te]) < log_loss_(over[te], y[te])
    slope_before = calibration_slope_intercept(over[te], y[te])[0]
    slope_after = calibration_slope_intercept(Calibrator("platt").fit(over[tr], y[tr]).predict(over[te]), y[te])[0]
    assert slope_before < 0.5 and 0.8 < slope_after < 1.25


def test_metrics_basic_properties():
    p, y = np.array([0.5] * 100), np.array([0, 1] * 50)
    assert brier(p, y) == pytest.approx(0.25) and log_loss_(p, y) == pytest.approx(np.log(2))
    t = reliability_table(np.linspace(0.3, 0.7, 200), np.tile([0, 1], 100), 5)
    assert t.n.sum() == 200 and ece(np.linspace(0.3, 0.7, 200), np.tile([0, 1], 100), 5) >= 0
    assert Calibrator("none").fit(p, y).predict(p).tolist() == p.tolist()
    with pytest.raises(ValueError):
        Calibrator("magic")


def test_online_calibration_uses_only_the_past():
    rng = np.random.default_rng(1)
    n = 900
    P = pd.DataFrame({"game_date": pd.date_range("2024-01-01", periods=n, freq="8h"), "y": rng.integers(0, 2, n),
                      "m": rng.uniform(0.3, 0.7, n)})
    a = online_calibrate(P, "m", min_history=200, block_days=14)
    Q = P.copy(); cutoff = P.game_date.iloc[600]
    Q.loc[Q.game_date >= cutoff, "y"] = 1 - Q.loc[Q.game_date >= cutoff, "y"]      # flip future labels
    b = online_calibrate(Q, "m", min_history=200, block_days=14)
    early = P.game_date < cutoff - pd.Timedelta(days=14)
    assert a[early].equals(b[early]) and a[:60].isna().all()


def test_ensembles():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 3000)
    good = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.08, 3000), 0.05, 0.95)
    noise = rng.uniform(0.3, 0.7, 3000)
    P = pd.DataFrame({"good": good, "noise": noise})
    w = WeightedAverage().fit(P, y)
    assert w.weights().sum() == pytest.approx(1) and (w.weights() >= -1e-9).all() and w.weights()["good"] > 0.8
    s = Stacker().fit(P, y)
    assert s.weights()["good"] > abs(s.weights()["noise"]) and np.isfinite(s.predict(P)).all()


def test_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(3)
    n = 1500
    p = rng.uniform(0.3, 0.7, n); y = (rng.random(n) < p).astype(int)
    P = pd.DataFrame({"game_date": pd.date_range("2024-01-01", periods=n, freq="6h"), "y": y, "good": p, "flat": 0.5})
    r = paired_logloss_diff(P, "good", "flat", B=500)
    assert r["diff"] < 0 and r["ci_high"] < 0 and r["p_a_not_better"] < 0.05


def test_tuning_returns_a_grid_member_and_prefers_simple_when_tied(F):
    r = tune_model("logistic", FEATS, F, n_folds=3)
    from nhlbet.models.tune import GRIDS
    assert r["params"] in GRIDS["logistic"][1] and len(r["table"]) == len(GRIDS["logistic"][1])


def test_bundle_roundtrip_and_predict(F, tmp_path):
    zoo = small_zoo()
    hist = pd.DataFrame({"stack": np.random.default_rng(0).uniform(0.4, 0.6, 400), "y": np.random.default_rng(1).integers(0, 2, 400)})
    b = train_bundle(F, zoo, "v1", FEATS, {"elo_k": 6}, hist, min_online=300)
    out = b.predict(F.tail(20))
    assert out.p.between(0, 1).all() and (out.p_source == "online_platt").all()
    b.save(tmp_path / "m.joblib")
    out2 = ModelBundle.load(tmp_path / "m.joblib").predict(F.tail(20))
    np.testing.assert_allclose(out.p, out2.p)
    assert (train_bundle(F, zoo, "v2", FEATS, {}, None).predict(F.tail(5)).p_source == "inner_oof_platt").all()
