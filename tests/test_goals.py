import numpy as np
import pandas as pd
import pytest

from nhlbet.models.goals import GoalsModel, ScoreDistribution, add_goal_targets, regulation_goals


def test_regulation_goals_by_hand():
    g = pd.DataFrame({"home_score": [4, 3, 2, 5, None], "away_score": [2, 2, 3, 1, None], "last_period": ["REG", "OT", "SO", "REG", None]})
    r = regulation_goals(g)
    assert r.hr.tolist()[:4] == [4, 2, 2, 5] and r.ar.tolist()[:4] == [2, 2, 2, 1]        # OT/SO winners lose their extra goal -> tied after 60
    assert r.ot.tolist()[:4] == [False, True, False, False] and r.so.tolist()[:4] == [False, False, True, False] and np.isnan(r.hr.iloc[4])


def dist(lh=3.1, la=2.9, ph=0.54, **kw):
    return ScoreDistribution(lh, la, kw.get("k", 0.03), kw.get("p_ot", 0.57), ph, kw.get("tie", 0.235))


def test_probabilities_are_coherent():
    d = dist()
    for line in (4.5, 5.5, 6.0, 6.5):
        o, u, p = d.total_probs(line)
        assert o + u + p == pytest.approx(1) and o >= 0 and u >= 0 and p >= 0
    assert d.total_probs(5.5)[2] == 0 and d.total_probs(6.0)[2] > 0.08                    # half lines never push; whole lines do
    assert d.total_probs(4.5)[0] > d.total_probs(5.5)[0] > d.total_probs(6.5)[0]          # over gets harder as the line rises
    c, a, p = d.spread_probs(-1.5)
    assert c + a + p == pytest.approx(1) and p == 0 and d.spread_probs(1.5)[0] > d.p_home > c
    assert d.p_tie == pytest.approx(0.235) and d.p_home == pytest.approx(0.54)             # tie rate and moneyline both honoured
    assert d.spread_probs(-0.5)[0] == pytest.approx(0.54)                                  # -0.5 is just the moneyline


def test_mirror_symmetry_and_favourite_covers_more():
    a, b = dist(3.4, 2.6, 0.62), dist(2.6, 3.4, 0.38)
    assert a.spread_probs(-1.5)[0] == pytest.approx(b.spread_probs(1.5)[1], abs=1e-9)
    assert a.total_probs(5.5)[0] == pytest.approx(b.total_probs(5.5)[0], abs=1e-9)
    assert a.spread_probs(-1.5)[0] > dist(3.0, 3.0, 0.5).spread_probs(-1.5)[0]
    assert dist(3.6, 3.2).total_probs(6.5)[0] > dist(2.6, 2.4).total_probs(6.5)[0]


def test_total_probs_match_a_direct_simulation():
    """Independent check of the tie/overtime/shootout logic by sampling games from the same grid."""
    d = dist(3.0, 3.0, 0.5)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d.w), size=400_000, p=d.w)
    m, t = d.reg_margin[idx], d.reg_total[idx].astype(float)
    tied = m == 0
    t = t + (tied & (rng.random(len(idx)) < d.p_ot))                                       # OT goal only for tied games decided in OT
    assert np.mean(t > 5.5) == pytest.approx(d.total_probs(5.5)[0], abs=0.004)
    assert np.mean(t == 6.0) == pytest.approx(d.total_probs(6.0)[2], abs=0.004)
    home = np.where(tied, rng.random(len(idx)) < d.s_home_tie, m > 0)
    assert np.mean(home) == pytest.approx(0.5, abs=0.004)
    fm = np.where(tied, np.where(home, 1, -1), m)
    assert np.mean(fm - 1.5 > 0) == pytest.approx(d.spread_probs(-1.5)[0], abs=0.004)


def synth(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    lh, la = np.exp(1.1 + 0.15 * x[:, 0] + 0.05 * x[:, 2]), np.exp(1.05 - 0.10 * x[:, 0] + 0.08 * x[:, 1])
    hr, ar = rng.poisson(lh), rng.poisson(la)
    X = pd.DataFrame(x, columns=["a", "b", "c"], index=np.arange(n))
    return X, hr, ar, lh, la


def test_model_recovers_rates_and_is_calibrated_out_of_sample():
    X, hr, ar, lh, la = synth()
    tr, te = slice(0, 2400), slice(2400, None)
    m = GoalsModel(["a", "b", "c"], alpha=1.0).fit(X[tr], hr[tr], ar[tr])
    lam = m.predict_lambdas(X[te])
    assert np.corrcoef(lam.lam_home, lh[te])[0, 1] > 0.9 and np.corrcoef(lam.lam_away, la[te])[0, 1] > 0.8
    assert lam.lam_home.mean() == pytest.approx(hr[te].mean(), abs=0.15)
    d = m.distributions(X[te])
    p_over = np.mean([x.total_probs(5.5)[0] for x in d])
    assert p_over == pytest.approx(np.mean(hr[te] + ar[te] > 5.5), abs=0.05)
    assert m.dispersion_ <= 0.03                                                          # the data were Poisson: no spurious overdispersion


def test_add_goal_targets_joins_on_game_id():
    F = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index([10, 11], name="game_id"))
    games = pd.DataFrame({"game_id": [10, 11, 12], "home_score": [3, 2, 1], "away_score": [1, 3, 0], "last_period": ["REG", "OT", "REG"]})
    out = add_goal_targets(F, games)
    assert out.loc[11, "hr"] == 2 and out.loc[11, "ar"] == 2 and bool(out.loc[11, "ot"]) and 12 not in out.index


def _league_frame(days=260):
    from nhlbet.models.goals import goal_feature_columns
    from nhlbet.data.store import Store
    from nhlbet.features.builder import BuilderConfig, build_features
    from tests.conftest import make_league
    st = Store(":memory:")
    for t, df in make_league(days).items():
        d = df.copy()
        for c in d.columns:
            if str(d[c].dtype).startswith("datetime"):
                d[c] = d[c].astype(str)
        d.to_sql(t, st.conn, if_exists="append", index=False)
    F = build_features(st, BuilderConfig())
    F = add_goal_targets(F[(F.game_type == 2) & F.home_score.notna()], st.df("select game_id, home_score, away_score, last_period from games"))
    return F, goal_feature_columns(F)


def test_walk_forward_never_uses_the_future():
    from nhlbet.models.goals_eval import evaluate, walk_forward_goals
    F, feats = _league_frame()
    start = "2024-01-01"
    P = walk_forward_goals(F, feats, start, 14, min_train=200)
    assert len(P) > 200 and set(P.index) <= set(F.index)
    cut = pd.Timestamp("2024-02-15")
    G = F.copy()
    late = G.game_date >= cut
    G.loc[late, ["hr", "ar", "tot", "mar"]] = 99.0                                  # corrupt every result from the cut onward
    P2 = walk_forward_goals(G, feats, start, 14, min_train=200)
    early = P.index[P.game_date < cut]
    pd.testing.assert_frame_equal(P.loc[early, ["lam_home", "lam_away", "p_over_5.5", "p_cover_-1.5"]], P2.loc[early, ["lam_home", "lam_away", "p_over_5.5", "p_cover_-1.5"]])
    ev = evaluate(P)
    assert (ev["diff"] < 0.01).all() and set(ev.market) >= {"total over 5.5", "home -1.5 cover"}       # no leak: any gain is small (goals are hard to predict)


def test_outcome_columns_can_never_be_features():
    """Regression: the first backtest accidentally included hr/ar/tot/mar as predictors and reported an impossible +0.3 nat/game improvement."""
    with pytest.raises(ValueError, match="leakage"):
        GoalsModel(["a", "hr"])
    F, feats = _league_frame(120)
    assert not set(feats) & {"hr", "ar", "ot", "so", "tot", "mar", "home_score", "away_score", "home_win"}


def test_walk_forward_gain_is_small_and_realistic_on_pure_noise():
    """With features that carry no information about goals the model must not beat the base rate by anything like a leak would."""
    from nhlbet.models.goals_eval import evaluate, goal_rate_table, walk_forward_goals
    F, _ = _league_frame()
    rng = np.random.default_rng(0)
    F = F.copy()
    for c in ("n1", "n2", "n3"):
        F[c] = rng.normal(size=len(F))
    P = walk_forward_goals(F, ["n1", "n2", "n3"], "2024-01-01", 14, min_train=200)
    assert abs(goal_rate_table(P)["diff"]).max() < 0.03 and evaluate(P)["diff"].abs().max() < 0.03
