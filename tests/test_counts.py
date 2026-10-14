"""Count modelling: one joint score grid prices every market, so the prices must agree; whole-number and quarter lines settle correctly."""
import numpy as np
import pandas as pd
import pytest

from nhlbet.models.goals import ScoreDistribution, coherence_issues


def dist(lh=3.1, la=2.9, k=0.03, p_ot=0.57, p_home=0.55, tie=0.235):
    return ScoreDistribution(lh, la, k, p_ot, p_home, tie)


@pytest.mark.parametrize("args", [dict(), dict(lh=2.2, la=3.8, p_home=0.3), dict(lh=4.0, la=3.5, k=0.0, p_home=0.5), dict(lh=2.5, la=2.5, p_home=None)])
def test_all_markets_from_one_grid_are_coherent(args):
    assert coherence_issues(dist(**args)) == []


def test_moneyline_equals_puck_line_minus_half_and_probabilities_sum_to_one():
    d = dist()
    assert d.spread_probs(-0.5)[0] == pytest.approx(d.p_home, abs=1e-9)
    for line in (4.5, 5.0, 5.25, 6.0, 6.25, 6.5, 6.75):
        assert sum(d.total_probs(line)) == pytest.approx(1.0)
    assert d.total_probs(6.5)[2] == 0.0 and d.total_probs(6.0)[2] > 0.1                      # half-point lines never push; whole-number lines do


def test_half_point_over_is_one_minus_cdf_and_whole_number_has_three_outcomes():
    d = dist()
    over5, under5, push5 = d.total_probs(5.0)                                                 # P(X >= 6), P(X <= 4), P(X = 5)
    o55, u55, _ = d.total_probs(5.5)
    o45, u45, _ = d.total_probs(4.5)
    assert over5 == pytest.approx(o55) and under5 == pytest.approx(u45) and push5 == pytest.approx(u55 - u45)
    # EV counts the push as a refund: win p*(d-1), lose u, push 0
    dec = 1.91
    ev_push_aware = over5 * (dec - 1) - under5
    ev_if_push_were_a_loss = over5 * (dec - 1) - (under5 + push5)
    assert ev_push_aware > ev_if_push_were_a_loss                                              # folding pushes into losses understates the bet's value


def test_quarter_lines_are_half_stakes_on_the_adjacent_lines():
    d = dist()
    q = d.total_probs(6.25)                                                                    # half on 6.0 (can push), half on 6.5
    a, b = d.total_probs(6.0), d.total_probs(6.5)
    assert q == pytest.approx(tuple((x + y) / 2 for x, y in zip(a, b)))
    assert d.total_probs(6.75) == pytest.approx(tuple((x + y) / 2 for x, y in zip(d.total_probs(6.5), d.total_probs(7.0))))
    t = d.team_total_probs("home", 2.75)
    assert sum(t) == pytest.approx(1.0) and t[2] < d.team_total_probs("home", 3.0)[2]


def test_team_totals_and_both_teams_to_score_follow_from_the_same_grid():
    d = dist()
    assert d.team_total_probs("home", 2.5)[0] > d.team_total_probs("home", 3.5)[0] > d.team_total_probs("home", 4.5)[0]
    assert d.team_total_probs("home", 2.5)[2] == 0.0 and d.team_total_probs("away", 3.0)[2] > 0
    stronger = dist(lh=3.8, la=3.4)
    assert stronger.btts_prob() > d.btts_prob() > 0.7
    # a shootout goal never counts: a 0-0 game decided in a shootout is not "both teams scored"
    scoreless = dist(lh=0.6, la=0.6, p_ot=0.0)
    assert scoreless.btts_prob() < 0.25


def test_the_coherence_check_flags_a_broken_distribution():
    d = dist()
    d.p_ot = 0.9                                                                               # change a parameter without rebuilding the grid views: expected_total no longer matches the cells
    assert coherence_issues(d) == []                                                           # p_ot is read by both sides, so still coherent
    d.s_home_tie = 0.1                                                                         # now the moneyline (p_home) no longer equals the -0.5 puck line the grid implies
    d.p_home_reg = d.p_home_reg + 0.05
    assert any("moneyline" in m for m in coherence_issues(d))


def test_player_prop_whole_number_line_is_priced_with_a_push():
    import pandas as pd
    from nhlbet.models.props import over_push_under
    from nhlbet.odds.snapshots import record_fetch
    from nhlbet.report.props import PropEngine
    from tests.test_props import engine_store, prop_event

    ov, pu, un = over_push_under(0.3, np.array([2.4]), 3.0)                                     # NB(mean 2.4): P(X>3), P(X=3), P(X<3)
    assert float((ov + pu + un)[0]) == pytest.approx(1.0) and pu[0] > 0.1
    half = over_push_under(0.3, np.array([2.4]), 3.5)
    assert float(half[1][0]) == 0.0 and float(half[0][0]) == pytest.approx(float(ov[0]), rel=1e-9)  # P(X>3.5) = P(X>=4) = P(X>3)

    st, plist, day = engine_store()
    names = {pid: nm.replace(nm.split(".")[0] + ".", {"A": "Alex", "B": "Ben", "C": "Carl", "D": "Dan"}[nm[0]]) for pid, nm, *_ in plist["EDM"][:3]}
    ev = prop_event(day)
    outs = [x for full in names.values() for x in ({"name": "Over", "description": full, "price": 2.1, "point": 2.0}, {"name": "Under", "description": full, "price": 1.8, "point": 2.0})]
    ev["bookmakers"] = [{"key": "bk1", "last_update": f"{day}T15:50:00Z", "markets": [{"key": "player_shots_on_goal", "last_update": f"{day}T15:50:00Z", "outcomes": outs}]}]
    ev["commence_time"] = f"{day}T23:00:00Z"
    record_fetch(st, type("Fx", (), {"events": [ev], "captured_at": f"{day}T16:00:00+00:00", "remaining": 300, "used": 5, "source": "live"})())
    qs = PropEngine.create(st, as_of=pd.Timestamp(day)).quotes_for_game(500, pd.Timestamp(day))
    assert len(qs) == 3 and all(q.point == 2.0 and q.p_push > 0.05 for q in qs)                # a whole-number line is no longer skipped
    for q in qs:
        pu_, po_u = q.p_push, None
        un_u = (1 - q.p_over) * (1 - pu_)                                                      # p_over is conditional on no push
        ov_u = q.p_over * (1 - pu_)
        assert q.ev_over == pytest.approx(ov_u * 1.1 - un_u) and q.ev_under == pytest.approx(un_u * 0.8 - ov_u)


def test_line_aware_calibration_recovers_overconfidence_that_a_pooled_map_hides():
    from nhlbet.models.goals_eval import fit_platt, fit_platt_line
    from nhlbet.odds.markets import recalibrate
    rng = np.random.default_rng(0)
    n = 6000
    lines = rng.choice([5.5, 6.5], n)
    base = np.where(lines == 5.5, 0.57, 0.44)
    lg = lambda p: np.log(p / (1 - p))
    true = np.clip(base + rng.normal(0, 0.04, n), 0.05, 0.95)
    y = (rng.random(n) < true).astype(int)
    p = 1 / (1 + np.exp(-(lg(base) + 1.6 * (lg(true) - lg(base)))))                    # a model that exaggerates every line's deviation by 60%
    pooled, per_line = fit_platt(p, y), fit_platt_line(p, y, lines)
    assert pooled["b"] > 0.8 > per_line["b"] > 0.55                                    # the pooled slope looks nearly calibrated; the per-line one shows the overconfidence
    cal = {"a": per_line["a"], "b": per_line["b"], "c": per_line["c"]}
    assert recalibrate(0.6, cal, 5.5) != recalibrate(0.6, cal, 6.5)                    # the intercept moves with the line
    assert recalibrate(0.6, {"a": 0.0, "b": 1.0}, 6.5) == pytest.approx(0.6)           # old maps without the line term still work


def test_ranked_probability_score_and_pmf_scoring_by_hand():
    from nhlbet.models.props_eval import KMAX, _pmf_matrix, evaluate, rps
    pmf = np.zeros((2, KMAX + 1))
    pmf[0, 2] = 1.0                                                                            # certain it is 2
    pmf[1, 0] = 1.0                                                                            # certain it is 0
    assert rps(pmf, np.array([2, 0])).tolist() == [0.0, 0.0]
    assert rps(pmf, np.array([3, 2]))[0] == pytest.approx(1 / KMAX)                            # one step off costs one CDF bucket
    assert rps(pmf, np.array([0, 3]))[1] == pytest.approx(3 / KMAX)                            # three steps off costs three, so distance matters (a hit rate would not see it)
    m = _pmf_matrix(np.array([2.0]), np.array([0.1]))
    assert m.shape == (1, KMAX + 1) and m.sum() == pytest.approx(1.0) and float((m[0, :KMAX] * np.arange(KMAX)).sum()) == pytest.approx(2.0, abs=0.02)
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.2, 1.5, 3000)
    k = np.full(3000, 0.05)
    y = rng.poisson(lam)
    P = pd.DataFrame({"game_date": pd.date_range("2024-01-01", periods=3000, freq="2h"), "y": y, "lam": lam, "base": np.full(3000, lam.mean()), "k": k})
    ev = evaluate(P, "goals")
    nll, score = ev.iloc[0], ev.iloc[1]
    assert nll["diff"] < 0 and score["diff"] < 0 and set(ev.what) >= {"over 0.5 (log loss)"}        # the true rate beats a constant rate on both count scores
    assert ev["verdict"].iloc[0] in ("skill vs own rate", "insufficient evidence") and np.isfinite(ev.n_to_confirm.iloc[0])


def test_kelly_with_a_push_is_neutral_and_matches_a_numeric_optimum():
    from nhlbet.risk.kelly import kelly_fraction, kelly_fraction_push
    pw, pl, pp, dec = 0.42, 0.40, 0.18, 1.95
    f = kelly_fraction_push(pw, pl, dec)
    grid = np.linspace(0, 0.9, 90001)
    growth = pw * np.log1p(grid * (dec - 1)) + pl * np.log1p(-grid)                    # a push leaves the bankroll unchanged
    assert f == pytest.approx(grid[np.argmax(growth)], abs=2e-5)
    assert kelly_fraction_push(0.5, 0.5, dec) == pytest.approx(kelly_fraction(0.5, dec))   # no push mass: identical to the plain formula
    assert kelly_fraction_push(0.30, 0.40, dec) == 0.0                                  # negative edge: no bet


def _saves_store(days=260, seed=0):
    from nhlbet.data.store import Store
    rng = np.random.default_rng(seed)
    st = Store(":memory:")
    teams = ["A", "B", "C", "D", "E", "F"]
    rate = dict(zip(teams, rng.uniform(26, 33, len(teams))))
    goalie = {t: 100 + i for i, t in enumerate(teams)}
    sv = {t: rng.uniform(0.895, 0.92) for t in teams}
    games, tg, gg = [], [], []
    gid = 0
    base = pd.Timestamp("2024-10-01")
    for d in range(days):
        order = list(rng.permutation(teams))
        for i in range(0, 6, 2):
            h, a = order[i], order[i + 1]
            gid += 1
            sa_h = rng.poisson(rate[h] * 0.5 + rate[a] * 0.5 - 1)          # shots against the home team
            sa_a = rng.poisson(rate[a] * 0.5 + rate[h] * 0.5 + 1)
            games.append(dict(game_id=gid, season=1, game_type=2, game_date=str((base + pd.Timedelta(days=d)).date()), start_utc=None, home=h, away=a, home_score=3, away_score=2,
                              status="FINAL", last_period="REG", home_win=1, source="nhl_api", updated_at=None))
            for t, o, ih, sa, sf in ((h, a, 1, sa_h, sa_a), (a, h, 0, sa_a, sa_h)):
                tg.append(dict(game_id=gid, team=t, opp=o, is_home=ih, goals=3, goals_against=2, sog_for=sf, sog_against=sa))
                saves = rng.binomial(sa, sv[t])
                gg.append(dict(game_id=gid, team=t, player_id=goalie[t], name=f"G{t}", started=1, toi_sec=3600, shots_against=sa, saves=int(saves), goals_against=int(sa - saves), xg_faced=None))
    st.upsert("games", games, ["game_id"]); st.upsert("team_game", tg, ["game_id", "team"]); st.upsert("goalie_game", gg, ["game_id", "player_id"])
    return st


def test_goalie_saves_model_is_an_integer_pmf_unbiased_and_never_sees_the_future():
    from nhlbet.models.props_eval import evaluate
    from nhlbet.models.saves import SavesModel, add_asof_features, goalie_starts, walk_forward_saves
    d = goalie_starts(_saves_store())
    assert len(d) == 260 * 6 and set(d.columns) >= {"saves", "team_sa", "opp_sf", "is_home"}
    P = walk_forward_saves(d, "2025-03-01", min_train=600)
    assert len(P) > 300 and abs(P.lam.mean() - P.y.mean()) < 0.6                                # unbiased out of sample
    ev = evaluate(P, "saves")
    assert ev.what.iloc[0].startswith("saves per player-game") and np.isfinite(ev["diff"].iloc[:2]).all()
    assert any(w.startswith("over 26.5") or w.startswith("over 24.5") for w in ev.what)
    # a binomially thinned NB is NB: P(over) from the model sums with push and under to one on a whole-number line
    f = add_asof_features(d)
    m = SavesModel().fit(f.iloc[:1200])
    sub = f.iloc[1300:1310]
    o, p, u = m.over_push_under(sub, 26.0)
    assert np.allclose(o + p + u, 1.0) and (p > 0.02).all()
    # corrupt every result after a cutoff: predictions before it must not move
    cut = pd.Timestamp("2025-04-15")
    d2 = d.copy()
    late = d2.game_date >= cut
    d2.loc[late, "saves"] = 5; d2.loc[late, "shots_against"] = 8
    P2 = walk_forward_saves(d2, "2025-03-01", min_train=600)
    a, b = P[P.game_date < cut].sort_values(["game_id", "player_id"]), P2[P2.game_date < cut].sort_values(["game_id", "player_id"])
    assert len(a) > 100 and np.allclose(a.lam.to_numpy(), b.lam.to_numpy())
