import math

import numpy as np
import pandas as pd
import pytest

from nhlbet.models.bundle import train_bundle
from nhlbet.report.stats import feature_label, game_stats, model_inputs
from tests.test_models import F, FEATS, small_zoo  # noqa: F401  (fixtures)


def test_game_stats_picks_the_better_side_by_direction_and_skips_missing():
    row = {"h_elo": 1540.0, "a_elo": 1500.0,                       # higher is better -> home
           "h_xga_pg": 2.4, "a_xga_pg": 2.9,                       # lower is better -> home
           "h_pp_pct": 0.18, "a_pp_pct": 0.24,                     # higher is better -> away
           "h_sh_pct": 0.12, "a_sh_pct": 0.09,                     # luck check: no 'better'
           "h_b2b": 1.0, "a_b2b": 0.0,                             # back-to-back: lower is better -> away
           "h_cf_pct": float("nan"), "a_cf_pct": float("nan"),     # missing on both sides: row omitted
           "h_travel_km": 0.0, "a_travel_km": 2620.7}
    s = {r["key"]: r for r in game_stats(row)}
    assert s["elo"]["better"] == "home" and s["elo"]["home_s"] == "1540" and s["xga_pg"]["better"] == "home"
    assert s["pp_pct"]["better"] == "away" and s["pp_pct"]["home_s"] == "18.0%" and s["sh_pct"]["better"] is None
    assert s["b2b"]["better"] == "away" and s["b2b"]["home_s"] == "yes" and s["b2b"]["away_s"] == "no"
    assert "cf_pct" not in s and s["travel_km"]["away_s"] == "2,621" and s["travel_km"]["better"] == "home"


def test_labels_are_plain_language():
    assert feature_label("d_elo") == "Elo rating (home minus away)" and feature_label("h_b2b") == "Second game in two nights (home team)"
    assert feature_label("d_xg_share").startswith("Expected-goals share") and feature_label("d_unknown_thing") == "unknown thing (home minus away)"


def test_model_inputs_use_importance_shares(tmp_path):
    p = tmp_path / "imp.csv"
    p.write_text("feature,perm_mean,perm_std,folds_positive,univariate_auc_edge,group,shap_mean_abs,shap_direction\n"
                 "d_elo,0.02,0.005,1,0.13,strength,0.30,1\nd_cf_pct,0.001,0.002,0.6,0.1,advanced,0.10,1\nh_b2b,0.0007,0.0007,0.8,0.01,rest_schedule,0.10,-1\n")
    r = model_inputs(["h_b2b", "d_elo", "d_cf_pct"], p)
    assert r[0]["feature"] == "d_elo" and len(r) == 3
    assert sum(x["share"] for x in r) == pytest.approx(1.0) and r[0]["share"] == pytest.approx(0.6) and r[0]["group"] == "strength"
    assert model_inputs(["d_elo"], tmp_path / "missing.csv")[0]["share"] is None                                # no csv: still lists the inputs, no fake ranking


def test_driver_contributions_reproduce_the_logistic_models_own_logit(F):  # noqa: F811
    b = train_bundle(F, small_zoo(), "v1", FEATS, {}, None)
    rows = F.tail(12)
    d = b.drivers(rows, top=len(FEATS))
    lg = b.fitted["logistic"]
    p_lr = lg.predict(rows)
    for gid, p in zip(rows.index, p_lr):
        total = sum(x["value"] for x in d[gid]) + float(lg.m_[-1].intercept_[0])
        assert total == pytest.approx(math.log(p / (1 - p)), abs=1e-9)             # contributions + intercept = the model's own logit, nothing invented
    top = b.drivers(rows, top=2)[rows.index[0]]
    assert len(top) == 2 and abs(top[0]["value"]) >= abs(top[1]["value"])
