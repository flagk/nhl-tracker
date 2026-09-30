"""The core guarantee: a game's features depend only on strictly earlier dates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import BuilderConfig, FeatureBuilder, GROUPS_FLAT, feature_group_map


def _build(tables, **cfg):
    return FeatureBuilder(BuilderConfig(**cfg)).build(tables)


def _scramble_from(tables, cutoff, rng):
    """Corrupt *everything* result-like for games on/after ``cutoff`` (scores, stats, goalies, skaters)."""
    t = {k: v.copy() for k, v in tables.items()}
    g = t["games"]
    late_ids = set(g.loc[g.game_date >= cutoff, "game_id"])
    m = g.game_id.isin(late_ids)
    g.loc[m, "home_score"] = rng.integers(0, 9, m.sum()).astype(float)
    g.loc[m, "away_score"] = rng.integers(0, 9, m.sum()).astype(float)
    g.loc[m, "home_score"] = np.where(g.loc[m, "home_score"] == g.loc[m, "away_score"], g.loc[m, "home_score"] + 1, g.loc[m, "home_score"])
    g.loc[m, "home_win"] = (g.loc[m, "home_score"] > g.loc[m, "away_score"]).astype(int)
    g.loc[m, "last_period"] = "SO"
    for name in ("team_game", "goalie_game", "skater_game"):
        df = t[name]
        mm = df.game_id.isin(late_ids)
        for c in df.columns:
            if c in ("game_id", "team", "opp", "is_home", "name", "position", "player_id", "started"):
                continue
            df[c] = df[c].astype(float)
            df.loc[mm, c] = rng.uniform(0, 50, mm.sum())
    return t


@pytest.mark.parametrize("cutoff", ["2023-11-15", "2024-01-20", "2024-02-20"])
def test_future_corruption_never_changes_earlier_or_same_day_features(league, cutoff):
    cutoff = pd.Timestamp(cutoff)
    base = _build(league)
    corrupted = _build(_scramble_from(league, cutoff, np.random.default_rng(1)))
    ids = base.index[base.game_date <= cutoff]  # includes games ON the cutoff date
    assert len(ids) > 20
    cols = [c for c in base.columns if c not in ("home_score", "away_score", "home_win")]
    a, b = base.loc[ids, cols], corrupted.loc[ids, cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_dropping_future_games_gives_identical_history(league):
    cutoff = pd.Timestamp("2024-01-10")
    trunc = {k: v[v.game_id.isin(league["games"].loc[league["games"].game_date <= cutoff, "game_id"])] for k, v in league.items()}
    full, cut = _build(league), _build(trunc)
    ids = cut.index
    cols = [c for c in full.columns if c not in ("home_score", "away_score", "home_win")]
    pd.testing.assert_frame_equal(full.loc[ids, cols], cut[cols], check_exact=False, rtol=1e-12, atol=1e-12)


def test_same_day_results_do_not_leak(league):
    """Changing the outcome of game X must not change features of other games on the same date."""
    g = league["games"]
    day = g.game_date.iloc[len(g) // 2]
    same = g[g.game_date == day]
    assert len(same) >= 2
    victim = same.game_id.iloc[0]
    t = {k: v.copy() for k, v in league.items()}
    t["games"].loc[t["games"].game_id == victim, ["home_score", "away_score"]] = [9.0, 0.0]
    t["team_game"].loc[t["team_game"].game_id == victim, ["sog_for", "xg_for"]] = 99.0
    base, mod = _build(league), _build(t)
    cols = [c for c in base.columns if c not in ("home_score", "away_score", "home_win")]
    on_day = base.index[base.game_date == day]
    pd.testing.assert_frame_equal(base.loc[on_day, cols], mod.loc[on_day, cols], check_exact=False, rtol=1e-12, atol=1e-12)


def test_row_order_does_not_matter(league):
    shuffled = {k: v.sample(frac=1.0, random_state=3).reset_index(drop=True) for k, v in league.items()}
    a, b = _build(league), _build(shuffled)
    cols = [c for c in a.columns if c not in ("home_score", "away_score", "home_win")]
    pd.testing.assert_frame_equal(a[cols], b.loc[a.index, cols], check_exact=False, rtol=1e-12, atol=1e-12)


def test_unplayed_games_get_features_from_history_only(league):
    """A scheduled-but-unplayed game (NaN score) is featurised from history; a future game never alters anything."""
    g = league["games"]
    last = g.game_date.max()
    fut = pd.DataFrame([dict(game_id=99999, game_date=last + pd.Timedelta(days=1), season=int(g.season.iloc[-1]),
                             game_type=2, home="BOS", away="TOR", home_score=np.nan, away_score=np.nan, home_win=np.nan,
                             status="FUT", last_period=None, start_utc=pd.NaT, source="nhl_api")])
    t = dict(league); t["games"] = pd.concat([g, fut], ignore_index=True)
    with_fut, without = _build(t), _build(league)
    assert 99999 in with_fut.index
    cols = [c for c in without.columns if c not in ("home_score", "away_score", "home_win")]
    pd.testing.assert_frame_equal(with_fut.loc[without.index, cols], without[cols], check_exact=False, rtol=1e-12, atol=1e-12)
    assert with_fut.loc[99999, "h_gp_season"] > 0


def test_actual_goalie_mode_uses_identity_not_tonights_stats(league):
    """In late-info mode, only WHO started is used. Tonight's saves/xG faced must not matter."""
    g = league["games"]
    victim = g.game_id.iloc[len(g) // 2]
    t = {k: v.copy() for k, v in league.items()}
    m = t["goalie_game"].game_id == victim
    t["goalie_game"].loc[m, ["saves", "goals_against", "xg_faced", "shots_against"]] = [0, 20, 9.9, 20]
    a = _build(league, goalie_mode="actual", lineup_mode="actual")
    b = _build(t, goalie_mode="actual", lineup_mode="actual")
    cols = [c for c in a.columns if c not in ("home_score", "away_score", "home_win")]
    pd.testing.assert_frame_equal(a.loc[[victim], cols], b.loc[[victim], cols], check_exact=False, rtol=1e-12, atol=1e-12)


def test_default_mode_ignores_tonights_starter_identity(league):
    g = league["games"]
    victim = g.game_id.iloc[len(g) // 2]
    t = {k: v.copy() for k, v in league.items()}
    m = t["goalie_game"].game_id == victim
    t["goalie_game"].loc[m, "player_id"] = 424242
    a, b = _build(league), _build(t)
    cols = [c for c in a.columns if c not in ("home_score", "away_score", "home_win")]
    pd.testing.assert_frame_equal(a.loc[[victim], cols], b.loc[[victim], cols], check_exact=False, rtol=1e-12, atol=1e-12)


def test_every_feature_column_is_grouped(league):
    f = _build(league)
    meta = {"game_date", "season", "game_type", "home", "away", "home_score", "away_score", "home_win", "start_utc"}
    ungrouped = [c for c in f.columns if c not in meta and c not in feature_group_map(list(f.columns))]
    assert not ungrouped, ungrouped


def test_no_forbidden_columns(league):
    from nhlbet.features.market import assert_no_closing_columns
    assert_no_closing_columns(_build(league).columns)
    with pytest.raises(ValueError):
        assert_no_closing_columns(["h_elo", "closing_prob"])
