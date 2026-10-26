"""Phase 3 of the audit: leak detectors must catch deliberately planted leaks, and must not cry wolf on the clean pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.validity.canary import MODES, LeakyBuilder, detect
from tests.conftest import make_league


@pytest.fixture(scope="module")
def lg():
    return make_league(n_days=240, seed=5)


@pytest.fixture(scope="module")
def results(lg):
    cut = lg["games"].game_date.sort_values().iloc[len(lg["games"]) // 2]
    return {m: detect(lg, m, cut) for m in (None,) + MODES}


def test_clean_pipeline_raises_no_alarm(results):
    r = results[None]
    assert not r["caught"], r


@pytest.mark.parametrize("mode", MODES)
def test_every_canary_is_caught_by_at_least_one_detector(results, mode):
    r = results[mode]
    assert r["caught"], f"canary {mode} slipped through every detector: {r}"


def test_no_single_detector_is_sufficient(results):
    """Documented weakness of any one check: each detector misses at least one canary."""
    for det in ("guard", "truncation", "corruption", "same_day", "shuffle_null", "too_good"):
        assert any(not results[m]["flags"][det] for m in MODES), f"detector {det} caught every canary: the matrix claim in audit/PHASE3 needs revisiting"


def test_expected_detector_assignments(results):
    f = {m: results[m]["flags"] for m in MODES}
    assert f["own_result"]["guard"] is False or f["own_result"]["too_good"]                       # a copy of the margin is caught by some check
    assert f["next_game"]["truncation"] and f["next_game"]["corruption"]                         # future-dependent: caught by the structural checks
    assert f["same_day_other"]["same_day"] and not f["same_day_other"]["truncation"]            # sideways-in-time leak: only the same-date perturbation test sees it
    assert not f["next_game"]["shuffle_null"]                                                    # ...and invisible to the shuffled-world test (it is unrelated to this game's label)
    assert f["own_result"]["shuffle_null"] or f["noisy_own_result"]["shuffle_null"]              # peeking at the own result survives the shuffle (no real signal left)


def test_default_configuration_is_morning_safe():
    c = BuilderConfig()
    assert c.goalie_mode == "probable" and c.lineup_mode == "off"
    from nhlbet.config import load_builder_config
    b = load_builder_config()
    assert b.goalie_mode == "probable" and b.lineup_mode == "off"


def test_changing_the_games_own_starter_does_not_move_features_in_default_mode_but_does_in_actual_mode(lg):
    gg = lg["goalie_game"]
    victim = lg["games"].sort_values("game_date").game_id.iloc[200]
    t = {k: v.copy() for k, v in lg.items()}
    m = t["goalie_game"].game_id == victim
    assert m.sum() >= 2
    t["goalie_game"].loc[m, "started"] = 1 - t["goalie_game"].loc[m, "started"]
    cols = lambda F: [c for c in F.select_dtypes("number").columns if c not in ("home_score", "away_score", "home_win")]
    for mode, must_move in (("probable", False), ("actual", True)):
        a = FeatureBuilder(BuilderConfig(goalie_mode=mode)).build(lg); b = FeatureBuilder(BuilderConfig(goalie_mode=mode)).build(t)
        moved = not np.allclose(a.loc[victim, cols(a)].astype(float).fillna(-9).to_numpy(), b.loc[victim, cols(b)].astype(float).fillna(-9).to_numpy())
        assert moved == must_move, f"goalie_mode={mode}"


def test_market_features_ignore_snapshots_after_the_decision_time():
    from nhlbet.features.market import attach_market_features
    feats = pd.DataFrame({"start_utc": ["2026-01-01T23:00:00Z"]}, index=pd.Index([1], name="game_id"))
    snaps = pd.DataFrame({"game_id": [1, 1, 1], "captured_at": ["2026-01-01T12:00:00Z", "2026-01-01T21:00:00Z", "2026-01-01T22:55:00Z"], "home_prob_novig": [0.50, 0.52, 0.99]})
    a = attach_market_features(feats, snaps.iloc[:2])
    b = attach_market_features(feats, snaps)                                              # a closing-line snapshot (0.99) after the decision time
    assert a.loc[1, "mkt_move"] == b.loc[1, "mkt_move"] == pytest.approx(0.02)


def test_labels_inside_a_walk_forward_test_block_never_change_that_blocks_predictions(lg):
    from nhlbet.models.base import make_zoo
    from nhlbet.models.walkforward import walk_forward
    from nhlbet.validity.realdata import _zoo
    F = FeatureBuilder().build(lg); F = F[F.game_type == 2]
    feats = [c for c in F.columns if c[:2] == "d_"][:6]
    first = F.game_date.sort_values().iloc[len(F) // 2]
    base = walk_forward(F, _zoo(feats), first, 28, min_train=100, inner_folds=3, n_jobs=1)
    flipped = F.copy()
    blk = (flipped.game_date >= first) & (flipped.game_date < first + pd.Timedelta(days=28))
    flipped.loc[blk, "home_win"] = 1 - flipped.loc[blk, "home_win"]
    alt = walk_forward(flipped, _zoo(feats), first, 28, min_train=100, inner_folds=3, n_jobs=1)
    ids = base.index[(base.game_date >= first) & (base.game_date < first + pd.Timedelta(days=28))]
    pd.testing.assert_frame_equal(base.loc[ids].drop(columns=["game_date", "y"]), alt.loc[ids].drop(columns=["game_date", "y"]), check_exact=False, rtol=0, atol=1e-9)
