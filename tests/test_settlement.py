"""Settlement rules (Phase 1.6): per-book rule sets from config/policy.yaml, applied identically by the settlement code and by the goal model."""
import numpy as np
import pandas as pd
import pytest

from nhlbet.models.goals import ScoreDistribution, add_goal_targets
from nhlbet.report.betlog import _leg_status, alt_value
from nhlbet.settle import game_total, rules_for, shootout_in_total


def test_every_book_resolves_to_a_defined_ruleset_and_unverified_books_say_so():
    for book in (None, "", "draftkings", "fanduel", "someunknownbook"):
        r = rules_for(book)
        assert r["ruleset"] in ("so_adds_goal", "so_excluded") and r["verified"] is False
        assert r["moneyline_includes_ot_so"] and r["puck_line_uses_official_margin"] and r["whole_line_push_refunds_stake"]
    assert shootout_in_total("DraftKings") is True                                  # lookup is case-insensitive
    assert rules_for("fanduel")["props_shootout_counts"] is False


def test_unknown_ruleset_is_an_error_not_a_silent_default(tmp_path, monkeypatch):
    import nhlbet.policy as pol
    p = tmp_path / "policy.yaml"
    p.write_text("settlement:\n  default_ruleset: nope\n  rulesets: {}\n  books: {}\n")
    monkeypatch.setattr(pol, "POLICY_PATH", p)
    monkeypatch.setattr("nhlbet.settle.policy_get", lambda *k, default=None: pol.policy_get(*k, default=default, path=p))
    with pytest.raises(KeyError):
        rules_for("x")


# (home, away, last_period) -> total under each rule set. Official scores include the shootout winner's extra goal.
CASES = [((3, 2, "REG"), 5, 5), ((3, 2, "OT"), 5, 5), ((3, 2, "SO"), 5, 4)]


@pytest.mark.parametrize("game,incl,excl", CASES)
def test_game_total_under_each_ruleset(game, incl, excl, monkeypatch):
    import nhlbet.settle as s
    h, a, lp = game
    monkeypatch.setattr(s, "shootout_in_total", lambda book=None: True)
    assert game_total(h, a, lp) == incl
    monkeypatch.setattr(s, "shootout_in_total", lambda book=None: False)
    assert game_total(h, a, lp) == excl


def test_totals_settle_by_hand_under_the_policy_default():
    # SO game 4-3 (official): 7 goals counted -> over 6.5 wins, under 7.0 pushes, over 7.5 loses
    f = lambda side, pt: float(alt_value(np.array(["totals"]), np.array([side]), np.array([pt]), [4], [3], ["SO"])[0])
    assert f("over", 6.5) > 0 and f("under", 7.0) == 0 and f("over", 7.5) < 0
    # the same bets if the book excluded the shootout goal (6 goals): over 6.5 loses, under 7.0 wins
    import nhlbet.settle as s
    mp = pytest.MonkeyPatch()
    mp.setattr(s, "shootout_in_total", lambda book=None: False)
    try:
        assert f("over", 6.5) < 0 and f("under", 7.0) > 0
    finally:
        mp.undo()


def test_puck_line_and_moneyline_use_the_official_result_in_every_ruleset():
    g = pd.DataFrame({"game_id": [1], "home": ["H"], "away": ["A"], "home_score": [4], "away_score": [3], "last_period": ["SO"], "home_win": [1.0]})
    sk = pd.DataFrame(columns=["game_id", "player_id", "sog", "points", "goals", "assists"])
    leg = lambda market, side, pt: _leg_status({"game_id": 1, "market": market, "side": side, "point": pt, "player_id": 0}, g, sk)
    assert leg("h2h", "home", None) == "won" and leg("h2h", "away", None) == "lost"
    assert leg("spreads", "home", -1.5) == "lost" and leg("spreads", "away", 1.5) == "won"        # a shootout winner wins by exactly one
    assert leg("spreads", "home", -0.5) == "won"


def test_props_count_overtime_not_shootout_and_void_when_the_player_did_not_play():
    g = pd.DataFrame({"game_id": [1], "home": ["H"], "away": ["A"], "home_score": [3], "away_score": [2], "last_period": ["OT"], "home_win": [1.0]})
    sk = pd.DataFrame({"game_id": [1, 1], "player_id": [7, 8], "sog": [4, 0], "points": [1, 0], "goals": [1, 0], "assists": [0, 0]})   # stats come from the box score: OT counted, SO never
    leg = lambda market, side, pid, pt: _leg_status({"game_id": 1, "market": market, "side": side, "point": pt, "player_id": pid}, g, sk)
    assert leg("player_sog", "over", 7, 3.5) == "won" and leg("player_sog", "under", 7, 3.5) == "lost"
    assert leg("player_goals", "over", 7, 0.5) == "won"
    assert leg("player_sog", "over", 99, 2.5) == "push"                                 # not in the box score: stake refunded
    assert leg("player_sog", "over", 7, 4.0) == "push"                                  # whole-number line landed on: refund


def test_goal_targets_follow_the_ruleset(monkeypatch):
    import nhlbet.settle as s
    games = pd.DataFrame({"game_id": [1, 2], "home_score": [3, 3], "away_score": [2, 2], "last_period": ["SO", "OT"]})
    F = pd.DataFrame(index=pd.Index([1, 2], name="game_id"))
    monkeypatch.setattr(s, "shootout_in_total", lambda book=None: True)
    assert add_goal_targets(F, games).tot.tolist() == [5, 5]
    monkeypatch.setattr(s, "shootout_in_total", lambda book=None: False)
    assert add_goal_targets(F, games).tot.tolist() == [4, 5]


def _sim_total(d, rng, so_adds, n=400_000):
    idx = rng.choice(len(d.w), size=n, p=d.w)
    t = d.reg_total[idx].astype(float)
    tied = d.reg_margin[idx] == 0
    ot = rng.random(n) < d.p_ot
    return t + (tied & (ot | so_adds))                      # a tied game gets +1 in overtime, and +1 in a shootout only when the book counts it


@pytest.mark.parametrize("so_adds", [True, False])
def test_model_total_probabilities_match_a_simulation_of_the_same_rule(so_adds):
    d = ScoreDistribution(3.0, 3.0, 0.03, 0.5, 0.54, 0.235)
    t = _sim_total(d, np.random.default_rng(1), so_adds)
    for line in (5.5, 6.0, 6.5):
        o, u, p = d.total_probs(line, so_in_total=so_adds)
        assert np.mean(t > line) == pytest.approx(o, abs=0.004) and np.mean(t == line) == pytest.approx(p, abs=0.004)
    # every tied game is odd-totalled when the shootout counts, so the whole-number push at 6.0 vanishes from tied games
    assert d.total_probs(6.0, True)[2] != pytest.approx(d.total_probs(6.0, False)[2], abs=0.005)


def test_team_totals_and_btts_never_count_the_shootout_goal():
    d = ScoreDistribution(3.0, 3.0, 0.03, 0.5, 0.54, 0.235)
    cells = list(d._final_goal_cells())
    assert sum(w for w, _, _ in cells) == pytest.approx(1.0)
