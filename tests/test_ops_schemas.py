import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.ops.schemas import GAMES, ODDS, schema_validation_check, validate_frame


def games(**over):
    d = pd.DataFrame([dict(game_id=1, game_date="2026-10-01", home="BOS", away="TOR", game_type=2, home_score=3.0, away_score=2.0, home_win=1.0),
                      dict(game_id=2, game_date="2026-10-02", home="NYR", away="PIT", game_type=2, home_score=None, away_score=None, home_win=None)])
    for k, v in over.items():
        d.loc[0, k] = v
    return d


def test_valid_games_pass_and_each_kind_of_corruption_is_named():
    assert validate_frame(GAMES, games()) == []
    for bad, why in ((dict(home_score=-1.0), "home_score"), (dict(home_win=2.0), "home_win"), (dict(away_score=3.0), "equal scores"), (dict(away="BOS"), "plays itself"), (dict(game_type=9), "game_type")):
        errs = validate_frame(GAMES, games(**bad))
        assert errs and why in " ".join(errs), (bad, errs)
    d = games(); d.loc[1, "game_id"] = 1
    assert any("game_id" in e for e in validate_frame(GAMES, d))                         # duplicate id


def test_odds_prices_must_be_decimal_odds():
    ok = pd.DataFrame({"captured_at": ["t"], "event_id": ["e"], "book": ["b"], "market": ["h2h"], "price": [1.9]})
    assert validate_frame(ODDS, ok) == []
    assert validate_frame(ODDS, ok.assign(price=-110.0)) and validate_frame(ODDS, ok.assign(price=0.0)) and validate_frame(ODDS, ok.assign(market="moneyline"))


def test_store_check_halts_softly_on_a_corrupt_row_and_passes_on_clean_data(tmp_path):
    st = Store(str(tmp_path / "t.db"))
    base = dict(season=1, start_utc=None, status="FINAL", last_period="REG", source="nhl_api", updated_at=None)
    st.upsert("games", [dict(game_id=1, game_type=2, game_date="2026-10-01", home="BOS", away="TOR", home_score=3, away_score=2, home_win=1, **base)], ["game_id"])
    assert schema_validation_check(st).ok
    st.upsert("games", [dict(game_id=2, game_type=2, game_date="2026-10-01", home="NYR", away="PIT", home_score=-4, away_score=2, home_win=0, **base)], ["game_id"])
    r = schema_validation_check(st)
    assert not r.ok and r.level == "soft" and "home_score" in r.detail
