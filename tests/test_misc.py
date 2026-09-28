import math

import numpy as np
import pandas as pd
import pytest

from nhlbet.data.xg import XGModel, geometry, is_high_danger
from nhlbet.features.elo import Elo
from nhlbet.features.market import attach_market_features
from nhlbet.teams import TEAMS, canon, haversine_km, is_rivalry, tz_shift


def test_teams_and_distance():
    assert len(TEAMS) == 33 and canon("TB") == "TBL" and canon("la") == "LAK"
    assert haversine_km("BOS", "BOS") == 0
    assert 3750 < haversine_km("BOS", "VGK") < 3900 and haversine_km("NYR", "NYI") < 40
    assert tz_shift("BOS", "VGK") == -3 and tz_shift("VGK", "BOS") == 3 and tz_shift("BOS", "NYR") == 0
    assert is_rivalry("BOS", "MTL") and is_rivalry("BOS", "BUF") and not is_rivalry("BOS", "SJS")


def test_xg_monotonic_and_reasonable():
    m = XGModel()
    near, far = m.xg(85, 0, "wrist", "sog"), m.xg(40, 0, "wrist", "sog")
    assert near > far > 0 and 0.1 < near < 0.4 and far < 0.05
    assert m.xg(80, 20, "wrist", "sog") < m.xg(80, 0, "wrist", "sog")           # wider angle is worse
    assert m.xg(80, 0, "wrist", "miss") < m.xg(80, 0, "wrist", "sog")
    assert m.xg(None, 3, "wrist", "sog") is None and m.xg(float("nan"), 3, None, "sog") is None
    assert is_high_danger(80, 3) and not is_high_danger(40, 0) and not is_high_danger(85, 40)
    assert geometry(-89, 0) == (0.0, 0.0)


def test_xg_save_load_roundtrip(tmp_path):
    m = XGModel(intercept=-1.0); m.save(tmp_path / "m.json")
    assert XGModel.load(tmp_path / "m.json").intercept == -1.0 and XGModel.load(tmp_path / "none.json").intercept == XGModel().intercept


def test_elo_zero_sum_and_direction():
    e = Elo()
    assert e.expected_home("A", "B") > 0.5                                      # home-ice edge for equals
    e.update("A", "B", 5, 1)
    assert e.rating("A") > 1500 > e.rating("B") and math.isclose(e.rating("A") + e.rating("B"), 3000)
    big = Elo(); big.update("A", "B", 6, 0)
    small = Elo(); small.update("A", "B", 2, 1)
    assert big.rating("A") > small.rating("A")                                   # margin matters
    up = Elo(); up.ratings = {"A": 1400.0, "B": 1600.0}; up.update("A", "B", 3, 2)
    fav = Elo(); fav.ratings = {"A": 1600.0, "B": 1400.0}; fav.update("A", "B", 3, 2)
    assert up.rating("A") - 1400 > fav.rating("A") - 1600                       # upset moves ratings more
    e.new_season(); assert abs(e.rating("A") - 1500) < abs(2 * 0 + 1500 + (e.rating("A") - 1500)) + 1


def test_market_features_ignore_snapshots_after_decision_time():
    feats = pd.DataFrame({"start_utc": [pd.Timestamp("2024-01-01 00:00", tz="UTC")]}, index=pd.Index([1], name="game_id"))
    snaps = pd.DataFrame({"game_id": [1, 1, 1, 1],
                          "captured_at": ["2023-12-31 12:00", "2023-12-31 20:00", "2023-12-31 22:29", "2023-12-31 23:55"],
                          "home_prob_novig": [0.50, 0.52, 0.55, 0.70]})                # last one = near-closing line
    out = attach_market_features(feats, snaps, lead_minutes=90)
    assert out.mkt_open_p[1] == 0.50 and math.isclose(out.mkt_move[1], 0.05)           # 23:55 snapshot ignored
    with pytest.raises(ValueError):
        attach_market_features(feats, snaps.drop(columns="home_prob_novig"))
