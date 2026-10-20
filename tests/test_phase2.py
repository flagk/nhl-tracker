"""Phase 2: every count PMF sums to one, every line's three outcomes sum to one, all game markets agree with a direct summation of one joint distribution,
and a quarter line is exactly half the stake on each adjacent line."""
import itertools

import numpy as np
import pytest

from nhlbet.models.goals import ScoreDistribution
from nhlbet.models.props import over_push_under
from nhlbet.models.props_eval import _pmf_matrix
from nhlbet.odds.markets import _ev

PARAMS = [(3.1, 2.9, 0.03, 0.57, 0.54, 0.235), (2.2, 3.8, 0.0, 0.5, 0.35, 0.2), (4.0, 4.0, 0.12, 0.6, None, None), (1.8, 2.0, 0.05, 0.4, 0.5, 0.3)]
LINES = [x * 0.25 for x in range(2, 49)]            # 0.5 to 12 in quarter steps


def f1(x) -> float:
    return float(np.ravel(x)[0])


def dists():
    return [ScoreDistribution(*p) for p in PARAMS]


@pytest.mark.parametrize("so", [True, False])
def test_the_total_pmf_sums_to_one_and_agrees_with_every_line(so):
    for d in dists():
        pmf = d.total_pmf(30, so_in_total=so)
        assert abs(pmf.sum() - 1.0) < 1e-9 and (pmf >= 0).all()
        for line in (4.5, 5.5, 6.5, 7.5):
            assert d.total_probs(line, so)[0] == pytest.approx(pmf[int(line) + 1:].sum(), abs=1e-12)


def test_every_line_has_three_outcomes_that_sum_to_one_for_every_game_market():
    for d in dists():
        for so in (True, False):
            for line in LINES:
                o, u, p = d.total_probs(line, so)
                assert abs(o + u + p - 1) < 1e-9 and min(o, u, p) >= -1e-12
        for pt in [x * 0.25 for x in range(-14, 15)]:
            c, a, p = d.spread_probs(pt)
            assert abs(c + a + p - 1) < 1e-9 and min(c, a, p) >= -1e-12
        for side in ("home", "away"):
            for line in (0.5, 1.5, 2.5, 1.0, 2.0, 1.25, 2.75):
                o, u, p = d.team_total_probs(side, line)
                assert abs(o + u + p - 1) < 1e-9


@pytest.mark.parametrize("lam,k", [(0.3, 0.0), (2.0, 0.1), (8.0, 0.3), (24.0, 0.05), (0.02, 0.2)])
def test_player_count_pmf_sums_to_one_and_three_outcomes_sum_to_one(lam, k):
    from nhlbet.models.props_eval import KMAX
    pm = _pmf_matrix(np.array([lam]), np.array([k]), 60)
    assert abs(pm.sum() - 1.0) < 1e-9 and (pm >= 0).all()
    for line in [x * 0.25 for x in range(1, 140)]:
        o, p, u = over_push_under(k, np.array([lam]), line)
        assert abs(f1(o + p + u) - 1) < 1e-9 and min(f1(o), f1(p), f1(u)) >= -1e-12
        if line % 1 == 0.5 and line < 55:
            assert f1(o) == pytest.approx(pm[0, int(np.floor(line)) + 1:].sum(), abs=1e-9)


def _direct_cells(d, so_adds):
    """An independent enumeration of the official result: for each regulation score (i, j) with its probability, a tied game is decided in overtime with probability
    p_ot (home wins it with s_home_tie) and otherwise in a shootout (same winner probability); returns (weight, home_official, away_official, so_game)."""
    n = d.grid.shape[0]
    out = []
    for i, j in itertools.product(range(n), range(n)):
        w = float(d.grid[i, j])
        if i != j:
            out.append((w, i, j, False))
        else:
            for decided_ot, pr in ((True, d.p_ot), (False, 1 - d.p_ot)):
                out.append((w * pr * d.s_home_tie, i + 1, j, not decided_ot))
                out.append((w * pr * (1 - d.s_home_tie), i, j + 1, not decided_ot))
    return out


def test_every_market_equals_a_direct_summation_of_the_same_joint_distribution():
    for d in dists():
        for so_adds in (True, False):
            cells = _direct_cells(d, so_adds)
            assert abs(sum(c[0] for c in cells) - 1) < 1e-9
            assert d.p_home == pytest.approx(sum(w for w, h, a, _ in cells if h > a), abs=1e-9)                                # moneyline
            for pt in (-1.5, -0.5, 0.5, 1.5, 2.5):                                                                              # puck line: official margin
                cover = sum(w for w, h, a, _ in cells if h - a + pt > 0)
                push = sum(w for w, h, a, _ in cells if h - a + pt == 0)
                assert d.spread_probs(pt)[0] == pytest.approx(cover, abs=1e-9) and d.spread_probs(pt)[2] == pytest.approx(push, abs=1e-9)
            for line in (4.5, 5.0, 5.5, 6.0, 6.5, 7.5):                                                                         # game total: official score, minus the shootout goal if excluded
                tot = lambda h, a, so: h + a - (0 if so_adds else (1 if so else 0))
                over = sum(w for w, h, a, so in cells if tot(h, a, so) > line)
                push = sum(w for w, h, a, so in cells if tot(h, a, so) == line)
                o, u, p = d.total_probs(line, so_adds)
                assert o == pytest.approx(over, abs=1e-9) and p == pytest.approx(push, abs=1e-9)
            for line in (1.5, 2.5, 3.0):                                                                                         # team totals never count the shootout goal
                tt = lambda h, so: h - (0 if not so else 0)
                over = sum(w for w, h, a, so in cells if (h - (1 if (so and h > a) else 0)) > line)
                o, u, p = d.team_total_probs("home", line)
                assert o == pytest.approx(over, abs=1e-9)


def _ev_of(o, p, u, dec):
    return o * (dec - 1) - u            # a push returns the stake: it adds nothing and loses nothing


@pytest.mark.parametrize("dec", [1.8, 2.05, 3.2])
def test_a_quarter_line_is_half_the_stake_on_each_adjacent_line(dec):
    for d in dists():
        for line in (5.25, 5.75, 6.25, 6.75):
            quarter = _ev_of(*(d.total_probs(line)[i] for i in (0, 2, 1)), dec)                    # (over, push, under) -> EV via the averaged probabilities
            legs = 0.5 * _ev_of(*(d.total_probs(line - 0.25)[i] for i in (0, 2, 1)), dec) + 0.5 * _ev_of(*(d.total_probs(line + 0.25)[i] for i in (0, 2, 1)), dec)
            assert quarter == pytest.approx(legs, abs=1e-12)
            assert _ev(d.total_probs(line)[0], d.total_probs(line)[2], dec) == pytest.approx(
                0.5 * _ev(d.total_probs(line - 0.25)[0], d.total_probs(line - 0.25)[2], dec) + 0.5 * _ev(d.total_probs(line + 0.25)[0], d.total_probs(line + 0.25)[2], dec), abs=1e-12)
        for pt in (-1.25, -1.75, 1.25):
            c, a, p = d.spread_probs(pt)
            c1, a1, p1 = d.spread_probs(pt - 0.25)
            c2, a2, p2 = d.spread_probs(pt + 0.25)
            assert (c, p) == pytest.approx(((c1 + c2) / 2, (p1 + p2) / 2), abs=1e-12)
    # a player-prop quarter line, derived independently from the outcome of each half-stake bet
    k, lam = 0.1, np.array([1.8])
    o, p, u = over_push_under(k, lam, 1.25)
    o1, p1, u1 = over_push_under(k, lam, 1.0)
    o2, p2, u2 = over_push_under(k, lam, 1.5)
    assert _ev(f1(o), f1(p), dec) == pytest.approx(0.5 * _ev(f1(o1), f1(p1), dec) + 0.5 * _ev(f1(o2), f1(p2), dec), abs=1e-12)


def test_quarter_line_by_hand_for_a_player_prop():
    """X = 1: the 1.0 leg pushes (refund) and the 1.5 leg loses: half a stake lost overall; X >= 2 wins both; X = 0 loses both."""
    k, lam = 0.0, np.array([1.0])
    from scipy.stats import poisson
    p0, p1, pge2 = poisson.pmf(0, 1.0), poisson.pmf(1, 1.0), 1 - poisson.cdf(1, 1.0)
    o, p, u = over_push_under(k, lam, 1.25)
    assert f1(o) == pytest.approx(pge2) and f1(p) == pytest.approx(p1 / 2) and f1(u) == pytest.approx(p0 + p1 / 2)


def test_a_shrinkage_value_on_the_edge_of_its_grid_is_not_used(tmp_path):
    import json
    from nhlbet.models.props import load_shrink
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"shrink_k": {"sog": 25.0, "points": 400.0}, "grid_audit": {"sog": {"pinned_to_grid_edge": False}, "points": {"pinned_to_grid_edge": True}}}))
    assert load_shrink(str(p)) == {"sog": 25.0}                    # points was pinned to the top of its grid: back to the default, not a published parameter
    p.write_text(json.dumps({"shrink_k": {"sog": 25.0}}))
    assert load_shrink(str(p)) == {"sog": 25.0}                    # files written before the audit existed still load


def test_the_goal_calibration_is_only_applied_under_the_rule_it_was_fitted_for():
    src = open("nhlbet/train.py").read()
    assert 'gcal.get("_rule") != rule' in src and "UNCALIBRATED" in src and 'last.get("goals_rule")' in src


def test_saves_are_never_priced_live():
    from nhlbet.odds.props import STATS
    assert "saves" not in STATS and not any("save" in v["odds"] for v in STATS.values())
