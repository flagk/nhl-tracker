"""Phase 5: push-aware Kelly and EV derived from first principles, the policy ceiling on the Kelly fraction, and the exposure caps (per bet, per game, per day)."""
import numpy as np
import pytest

from nhlbet.risk.caps import apply_caps
from nhlbet.risk.kelly import ev_per_unit, kelly_fraction, kelly_fraction_push, stake_push
from nhlbet.risk.policy import RiskConfig


@pytest.mark.parametrize("pw,pl,dec", [(0.45, 0.40, 2.1), (0.30, 0.25, 3.5), (0.55, 0.35, 1.95), (0.60, 0.10, 1.4), (0.2, 0.5, 4.0)])
def test_push_aware_kelly_maximises_the_expected_log_growth(pw, pl, dec):
    b = dec - 1
    f = np.linspace(0, 0.999, 200001)
    g = pw * np.log1p(b * f) + pl * np.log1p(-f)                       # a push (probability 1 - pw - pl) leaves the bankroll unchanged
    grid_best = f[np.argmax(g)] if g.max() > 0 else 0.0
    assert kelly_fraction_push(pw, pl, dec) == pytest.approx(grid_best, abs=2e-5)
    assert kelly_fraction_push(pw, pl, dec) == pytest.approx(max(0.0, (b * pw - pl) / (b * (pw + pl))))


def test_with_no_push_probability_it_is_the_ordinary_kelly_formula():
    for p, dec in ((0.55, 1.95), (0.4, 2.8), (0.7, 1.5)):
        assert kelly_fraction_push(p, 1 - p, dec) == pytest.approx(kelly_fraction(p, dec), abs=1e-12)          # the half-point case: pp = 0


def test_a_push_is_neutral_so_only_the_conditional_win_probability_matters():
    pw, pl, dec = 0.40, 0.40, 2.3                                           # 20% push
    assert kelly_fraction_push(pw, pl, dec) == pytest.approx(kelly_fraction(pw / (pw + pl), dec), abs=1e-12)
    assert kelly_fraction_push(pw, pl, dec) == pytest.approx((1.3 * 0.4 - 0.4) / (1.3 * 0.8))
    assert kelly_fraction(pw, dec) == 0.0                                    # the plain formula fed the unconditional win probability treats the push as a loss: it would refuse a bet with positive growth


def test_ev_per_unit_by_hand():
    assert ev_per_unit(0.45, 0.40, 2.1) == pytest.approx(1.1 * 0.45 - 0.40)
    assert ev_per_unit(0.5, 0.5, 2.0) == pytest.approx(0.0)
    assert ev_per_unit(0.3, 0.3, 2.0) == pytest.approx(0.0)                 # even-money bet with equal win and lose probability and a 40% push: zero


def test_stake_push_is_fractional_and_capped():
    assert stake_push(1000, 0.55, 0.45, 2.0, fraction=0.25, max_pct=1.0) == pytest.approx(1000 * 0.25 * 0.10)
    assert stake_push(1000, 0.80, 0.20, 2.0, fraction=0.25, max_pct=0.02) == pytest.approx(20.0)             # capped at 2% of bankroll
    assert stake_push(1000, 0.40, 0.60, 2.0) == 0.0                                                          # negative edge: no stake
    with pytest.raises(ValueError):
        stake_push(1000, 0.5, 0.5, 2.0, fraction=1.5)


def test_defaults_come_from_the_policy_and_the_kelly_ceiling_is_enforced(tmp_path, monkeypatch):
    from nhlbet.policy import policy_get
    c = RiskConfig()
    assert c.kelly_fraction == policy_get("staking", "kelly_fraction") <= policy_get("staking", "kelly_fraction_max") == 0.25
    assert c.max_bet_pct == policy_get("staking", "max_bet_pct") and c.max_game_exposure_pct == policy_get("staking", "max_game_exposure_pct")
    import nhlbet.risk.policy as rp
    monkeypatch.setattr(rp, "_pol", lambda key, default: 0.5 if key == "kelly_fraction" else (0.25 if key == "kelly_fraction_max" else default))
    with pytest.raises(ValueError):
        RiskConfig()                                                           # a policy file asking for half Kelly is rejected at load time


B = 1000.0


def caps(bets, **kw):
    a = dict(bankroll=B, max_bet_pct=0.02, max_game_pct=0.03, max_daily_pct=0.05, max_bets=5, min_stake=1.0)
    a.update(kw)
    return apply_caps(bets, **a)


def bet(i, game, stake, ev=0.05):
    return {"id": i, "game_id": game, "stake": stake, "ev": ev}


def test_per_bet_cap():
    o = caps([bet(1, 10, 50.0)])
    assert o[0]["stake"] == 20.0 and any("per-bet" in n for n in o[0]["notes"])


def test_correlated_markets_on_one_game_share_one_cap():
    o = caps([bet(1, 7, 20.0), bet(2, 7, 20.0), bet(3, 7, 20.0)])           # moneyline + total + puck line on the same game: 60 asked, cap is 30
    assert sum(x["stake"] for x in o) <= 30.0 and all(x["stake"] == 10.0 for x in o)
    other = caps([bet(1, 7, 20.0), bet(2, 8, 20.0), bet(3, 9, 20.0)])
    assert [x["stake"] for x in other] == [20.0, 20.0, 10.0] or sum(x["stake"] for x in other) <= 50.0     # different games are not tied together; the daily cap (50) still binds


def test_daily_cap_scales_down_never_up_and_rounds_down():
    o = caps([bet(i, i, 20.0) for i in range(5)] , max_daily_pct=0.05)
    assert sum(x["stake"] for x in o) <= 50.0 + 1e-9 and all(x["stake"] <= 20.0 for x in o)
    o2 = caps([bet(1, 1, 3.337), bet(2, 2, 3.337)], max_daily_pct=0.0061)
    assert sum(x["stake"] for x in o2) <= 6.1 + 1e-9 and all(abs(x["stake"] * 100 - round(x["stake"] * 100)) < 1e-6 for x in o2)


def test_only_the_best_ev_bets_survive_the_count_limit():
    o = caps([bet(i, i, 10.0, ev=0.01 * i) for i in range(1, 8)], max_bets=3, max_daily_pct=1.0)
    kept = [x["id"] for x in o if not x["dropped"]]
    assert sorted(kept) == [5, 6, 7] and all(x["reason"] for x in o if x["dropped"])


def test_a_stake_pushed_below_the_minimum_is_dropped():
    o = caps([bet(1, 1, 0.6)], min_stake=1.0)
    assert o[0]["dropped"] and o[0]["stake"] == 0.0


def test_caps_are_idempotent_and_total_exposure_never_exceeds_any_cap():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        bets = [bet(i, int(rng.integers(0, 4)), float(rng.uniform(0.5, 60)), float(rng.uniform(-0.02, 0.2))) for i in range(n)]
        o = caps(bets)
        by_game = {}
        for x in o:
            by_game[x["game_id"]] = by_game.get(x["game_id"], 0) + x["stake"]
            assert x["stake"] <= B * 0.02 + 1e-9
        assert all(v <= B * 0.03 + 1e-9 for v in by_game.values()) and sum(x["stake"] for x in o) <= B * 0.05 + 1e-9
        assert sum(1 for x in o if not x["dropped"]) <= 5
        again = caps([{**x, "dropped": False} for x in o if not x["dropped"]])
        assert [a["stake"] for a in again] == [x["stake"] for x in o if not x["dropped"]]


def test_the_recommendation_pipeline_uses_the_same_caps():
    from nhlbet.risk.policy import Recommendation, apply_portfolio_caps
    cfg = RiskConfig(bankroll=1000, max_bets_per_day=5, max_game_exposure_pct=0.01, max_daily_exposure_pct=0.05, max_bet_pct=0.02)
    recs = [Recommendation(1, "H", "A", action="BET", stake=20.0, ev=0.1), Recommendation(1, "H", "A", action="BET", stake=20.0, ev=0.09)]
    out = apply_portfolio_caps(recs, cfg)
    assert sum(r.stake for r in out) <= 10.0 + 1e-9
