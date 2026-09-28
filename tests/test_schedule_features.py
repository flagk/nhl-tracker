"""Hand-computed expectations for rest/travel/schedule features."""
import math

import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.teams import haversine_km


def tables(spec):
    games, tg = [], []
    for i, (d, h, a, hs, as_) in enumerate(spec, 1):
        games.append(dict(game_id=i, game_date=pd.Timestamp(d), season=2023, game_type=2, home=h, away=a, home_score=hs,
                          away_score=as_, home_win=int(hs > as_), status="FINAL", last_period="REG", start_utc=pd.NaT, source="x"))
        tg += [dict(game_id=i, team=h, opp=a, is_home=1, goals=hs, goals_against=as_),
               dict(game_id=i, team=a, opp=h, is_home=0, goals=as_, goals_against=hs)]
    return {"games": pd.DataFrame(games), "team_game": pd.DataFrame(tg), "goalie_game": pd.DataFrame(), "skater_game": pd.DataFrame()}


SPEC = [("2023-10-10", "TOR", "BOS", 2, 3),   # id1  BOS away
        ("2023-10-11", "VGK", "BOS", 4, 1),   # id2  BOS away again, back-to-back
        ("2023-10-13", "BOS", "TOR", 3, 1)]   # id3  BOS home, 3rd game in 4 nights; TOR rested


@pytest.fixture(scope="module")
def F():
    return FeatureBuilder(BuilderConfig()).build(tables(SPEC))


def test_first_game_defaults(F):
    r = F.loc[1]
    assert r.h_rest_days == 7 and r.a_rest_days == 7 and r.h_gp_season == 0 and r.h_travel_km == 0
    assert r.a_travel_km == pytest.approx(haversine_km("BOS", "TOR"))    # away team travels from its own arena
    assert r.h_road_trip_n == 0 and r.a_road_trip_n == 1 and r.h_homestand_n == 1


def test_back_to_back_road_trip_and_travel(F):
    r = F.loc[2]                                     # BOS at VGK, one day after playing at TOR
    assert r.a_rest_days == 1 and r.a_b2b == 1 and r.h_b2b == 0
    assert r.a_travel_km == pytest.approx(haversine_km("TOR", "VGK"))
    assert r.a_tz_shift == -3 and r.h_tz_shift == 0
    assert r.a_road_trip_n == 2 and r.a_g3in4 == 0
    assert r.h_rest_days == 7                        # VGK's first game
    assert r.d_rest_days == 7 - 1


def test_third_game_in_four_nights_and_return_home(F):
    r = F.loc[3]
    assert r.h_rest_days == 2 and r.h_b2b == 0
    assert r.h_g3in4 == 1                            # Oct 10, 11, 13 = 3 games in 4 nights
    assert r.h_travel_km == pytest.approx(haversine_km("VGK", "BOS"))
    assert r.h_tz_shift == 3 and r.h_road_trip_n == 0 and r.h_homestand_n == 1
    assert r.a_rest_days == 3 and r.a_g3in4 == 0
    assert r.a_travel_km == pytest.approx(haversine_km("TOR", "BOS"))
    assert r.h_games_last_7d == 2


def test_form_features_are_as_of(F):
    r = F.loc[3]
    # BOS before game 3: won 3-2 at TOR, lost 1-4 at VGK -> goals diff +1, -3
    assert r.h_gd_l10 == pytest.approx((1 - 3) / 2)
    assert r.h_win_pct_l10 == 0.5
    assert r.a_gd_l10 == pytest.approx(-1)           # TOR lost 2-3
    assert F.loc[1, "h_gd_l10"] != F.loc[1, "h_gd_l10"]      # NaN with no history (never a fake zero)
