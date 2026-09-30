import math

import numpy as np
import pandas as pd
import pytest

from nhlbet.odds.edge import SideQuote
from nhlbet.odds.math import expected_value
from nhlbet.risk.bankroll import drawdown_stats, equity_curve, summarize_bets
from nhlbet.risk.kelly import kelly_fraction, stake
from nhlbet.risk.montecarlo import default_template, simulate
from nhlbet.risk.policy import RiskConfig, apply_portfolio_caps, recommend_game, recommend_slate
from nhlbet.risk.shrink import estimate_trust, shrink_to_market, trust_weight


def q(side, team, p, m, dec, book="bookB"):
    return SideQuote(side, team, p, m, book, dec, 1 / dec, p - m, expected_value(p, dec), 5)


def quotes(ph, mh, dh=1.90, da=2.15):
    return {"home": q("home", "BOS", ph, mh, dh), "away": q("away", "TOR", 1 - ph, 1 - mh, da)}


# ------------------------------------------------------------------ Kelly
def test_kelly_known_values():
    d = 1.0 + 100 / 110
    assert kelly_fraction(0.55, d) == pytest.approx((0.9091 * 0.55 - 0.45) / 0.9091, abs=1e-4)      # ~5.5%
    assert kelly_fraction(0.5, 2.0) == 0.0 and kelly_fraction(0.4, 2.0) == 0.0                     # no edge / negative -> 0
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)                                          # classic: 60% at even money
    assert kelly_fraction(1.0, 2.0) == 1.0
    for bad in ((1.2, 2.0), (-0.1, 2.0), (0.5, 1.0), (0.5, 0.9)):
        with pytest.raises(ValueError):
            kelly_fraction(*bad)


def test_stake_fractional_and_capped():
    d = 1.90909
    full = 1000 * kelly_fraction(0.55, d)
    assert stake(1000, 0.55, d, fraction=0.25, max_pct=1.0) == pytest.approx(full / 4)
    assert stake(1000, 0.75, d, fraction=0.25, max_pct=0.02) == pytest.approx(20.0)                # cap binds
    assert stake(1000, 0.45, d) == 0.0
    assert stake(2000, 0.55, d) == pytest.approx(2 * stake(1000, 0.55, d))                         # scales with bankroll
    with pytest.raises(ValueError):
        stake(1000, 0.55, d, fraction=0)
    with pytest.raises(ValueError):
        stake(-1, 0.55, d)


def test_kelly_growth_is_maximised_at_full_kelly():
    p, d = 0.55, 2.0
    g = lambda f: p * math.log(1 + f * (d - 1)) + (1 - p) * math.log(1 - f)
    fk = kelly_fraction(p, d)
    assert g(fk) > g(fk / 2) > g(0) and g(fk) > g(fk * 1.5) and g(2 * fk) < g(fk)


# ------------------------------------------------------------------ shrinkage
def test_shrink_properties():
    assert shrink_to_market(0.53, 0.53) == pytest.approx(0.53)
    small, big = shrink_to_market(0.55, 0.53), shrink_to_market(0.70, 0.53)
    assert 0.53 < small < 0.55 and 0.53 < big < 0.70
    assert (0.70 - 0.53) > 2 * (big - 0.53)                                     # big disagreements are discounted much more
    assert trust_weight(0.02) > trust_weight(0.06) > trust_weight(0.15)          # trust falls with disagreement
    assert (big - 0.53) < 0.03                                                   # even a 17-pt gap yields a small adjusted edge
    assert shrink_to_market(0.40, 0.55) > 0.40 and shrink_to_market(0.9, 0.5) < 0.9   # symmetric, never overshoots
    assert shrink_to_market(0.5, 0.5, w0=0.0) == 0.5


def test_adjusted_edge_peaks_near_d0_then_falls():
    adj = lambda d: shrink_to_market(0.53 + d, 0.53) - 0.53
    assert adj(0.03) < adj(0.06)                                                 # rises while disagreement is plausible...
    assert adj(0.06) > adj(0.10) > adj(0.16)                                     # ...then falls: huge gaps are distrusted


def test_estimate_trust_recovers_true_blend():
    rng = np.random.default_rng(0)
    n = 20000
    pm = rng.uniform(0.35, 0.65, n)
    noise = rng.normal(0, 0.05, n)
    model = np.clip(pm + noise, 0.05, 0.95)
    # truth: market + 0.3 * (model-market)  => optimal trust ~0.3
    z = np.log(pm / (1 - pm)) + 0.3 * (np.log(model / (1 - model)) - np.log(pm / (1 - pm)))
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    assert 0.15 < estimate_trust(model, pm, y) < 0.5
    y0 = (rng.random(n) < pm).astype(int)                                        # model is pure noise
    assert estimate_trust(model, pm, y0) < 0.2


# ------------------------------------------------------------------ policy
def test_bets_a_clear_edge_within_caps():
    cfg = RiskConfig(bankroll=1000)
    r = recommend_game(1, "BOS", "TOR", quotes(0.58, 0.53), cfg)
    assert r.action == "BET" and r.team == "BOS" and r.book == "bookB" and 0 < r.stake <= 1000 * cfg.max_bet_pct
    assert r.market_prob == 0.53 and r.model_prob == 0.58 and r.adj_prob < r.model_prob and r.ev > 0
    assert "BOS" in r.explain() and "edge" in r.explain()


def test_no_bet_is_the_default_outcome():
    cfg = RiskConfig()
    assert recommend_game(1, "BOS", "TOR", quotes(0.54, 0.53), cfg).action == "NO_BET"       # 1% edge < 3%
    r = recommend_game(1, "BOS", "TOR", quotes(0.50, 0.50), cfg)
    assert r.action == "NO_BET" and r.stake == 0 and r.reasons and r.explain().startswith("No bet")
    assert recommend_game(1, "BOS", "TOR", None, cfg).reasons == ["no usable odds available"]


def test_wild_disagreement_is_treated_as_model_error():
    r = recommend_game(1, "BOS", "TOR", quotes(0.72, 0.53), RiskConfig())
    assert r.action == "NO_BET" and "model error" in r.explain()


def test_confidence_and_longshot_filters():
    cfg = RiskConfig()
    dog = {"home": q("home", "BOS", 0.62, 0.66, 1.5), "away": q("away", "TOR", 0.38, 0.34, 3.0)}   # away: edge 4% but only 38% likely
    assert recommend_game(1, "BOS", "TOR", dog, cfg).action == "NO_BET"
    assert recommend_game(1, "BOS", "TOR", dog, RiskConfig(min_prob=0.30)).action == "BET"
    far = {"home": q("home", "BOS", 0.80, 0.83, 1.15), "away": q("away", "TOR", 0.20, 0.17, 6.0)}
    assert recommend_game(1, "BOS", "TOR", far, cfg).action == "NO_BET"


def test_health_stale_odds_and_unconfirmed_goalie():
    cfg = RiskConfig()
    good = quotes(0.58, 0.53)
    assert recommend_game(1, "BOS", "TOR", good, cfg, {"model_status": "ALERT"}).action == "NO_BET"
    assert recommend_game(1, "BOS", "TOR", good, cfg, {"model_status": "WARN"}).action == "BET"
    assert "stale" in recommend_game(1, "BOS", "TOR", good, cfg, {"odds_stale": True}).explain()
    assert recommend_game(1, "BOS", "TOR", good, RiskConfig(allow_stale_odds=True), {"odds_stale": True}).action == "BET"
    marginal = quotes(0.565, 0.53)                                                  # 3.5% edge: OK if goalie confirmed, not otherwise
    assert recommend_game(1, "BOS", "TOR", marginal, cfg, {"goalie_confirmed": True}).action == "BET"
    r = recommend_game(1, "BOS", "TOR", marginal, cfg, {"goalie_confirmed": False})
    assert r.action == "NO_BET" and "goalie" in r.explain()


def test_picks_the_better_side_and_never_both():
    cfg = RiskConfig()
    away_edge = {"home": q("home", "BOS", 0.47, 0.53, 1.90), "away": q("away", "TOR", 0.53, 0.47, 2.15)}
    r = recommend_game(1, "BOS", "TOR", away_edge, cfg)
    assert r.action == "BET" and r.side == "away" and r.team == "TOR"


def test_stake_scales_with_bankroll_and_kelly_fraction():
    s = lambda **k: recommend_game(1, "BOS", "TOR", quotes(0.575, 0.53, 2.05, 1.80), RiskConfig(max_bet_pct=1.0, **k)).stake
    assert s(bankroll=2000) == pytest.approx(2 * s(bankroll=1000), abs=0.02)
    assert s(bankroll=1000, kelly_fraction=0.5) == pytest.approx(2 * s(bankroll=1000, kelly_fraction=0.25), abs=0.02)


def test_portfolio_caps_enforced():
    cfg = RiskConfig(bankroll=1000, max_bets_per_day=3, max_daily_exposure_pct=0.03, max_bet_pct=0.02)
    slate = [(i, "BOS", "TOR", quotes(0.56 + i * 0.004, 0.53), {}) for i in range(8)]   # edges 3.0%..5.8%, below d0 so EV rises with edge
    recs = recommend_slate(slate, cfg)
    bets = [r for r in recs if r.action == "BET"]
    assert len(bets) == 3 and len(recs) == 8                                       # one output per game
    assert sum(r.stake for r in bets) <= 1000 * 0.03 + 1e-9                        # exposure cap
    assert all(r.stake <= 1000 * 0.02 + 1e-9 for r in bets)                        # per-bet cap
    assert {r.game_id for r in bets} == {7, 6, 5}                                  # highest EV kept
    assert all("more than 3" in r.explain() for r in recs if r.game_id in (0, 1, 2, 3, 4) and r.action == "NO_BET" and r.reasons)


def test_min_stake_and_cap_note():
    tiny = recommend_game(1, "BOS", "TOR", quotes(0.565, 0.53), RiskConfig(bankroll=50))
    assert tiny.action == "NO_BET" and "minimum" in tiny.explain()
    capped = recommend_game(1, "BOS", "TOR", quotes(0.62, 0.53, 2.2, 1.7), RiskConfig(max_bet_pct=0.005))
    assert capped.action == "BET" and capped.stake_pct <= 0.005 + 1e-9 and "capped" in capped.explain()


def test_policy_invariants_on_random_slates():
    rng = np.random.default_rng(0)
    cfg = RiskConfig(bankroll=1000)
    for _ in range(200):
        slate = []
        for i in range(int(rng.integers(1, 12))):
            m = rng.uniform(0.3, 0.7); p = float(np.clip(m + rng.normal(0, 0.06), 0.05, 0.95))
            dh, da = 1 / m / 1.04, 1 / (1 - m) / 1.04
            slate.append((i, "BOS", "TOR", quotes(p, m, dh, da), {}))
        recs = recommend_slate(slate, cfg)
        bets = [r for r in recs if r.action == "BET"]
        assert sum(r.stake for r in bets) <= cfg.bankroll * cfg.max_daily_exposure_pct + 1e-9
        assert len(bets) <= cfg.max_bets_per_day
        for r in bets:
            assert 0 < r.stake <= cfg.bankroll * cfg.max_bet_pct + 1e-9 and r.edge >= cfg.min_edge - 1e-12 and r.ev > 0 and abs(r.edge) <= cfg.max_disagreement
        assert all(r.stake == 0 for r in recs if r.action == "NO_BET")


# ------------------------------------------------------------------ bankroll + Monte Carlo
def test_equity_curve_and_drawdown():
    b = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6), "profit": [10, -20, -10, 5, 30, -5], "stake": 10, "decimal": 1.9})
    c = equity_curve(b, 100)
    assert list(c.bankroll) == [110, 90, 80, 85, 115, 110]
    ds = drawdown_stats(c)
    assert ds["max_drawdown"] == pytest.approx((110 - 80) / 110) and ds["current_drawdown"] == pytest.approx((115 - 110) / 115)
    assert ds["longest_underwater_bets"] == 3
    s = summarize_bets(b)
    assert s["n"] == 6 and s["profit"] == 10 and s["roi"] == pytest.approx(10 / 60) and s["win_rate"] == pytest.approx(3 / 6)
    assert drawdown_stats(equity_curve(b.iloc[0:0], 100))["max_drawdown"] == 0


def test_monte_carlo_behaves():
    t = default_template(edge=0.03, stake_pct=0.01)
    r = simulate(t, n_paths=3000, horizon=300, seed=1)
    assert r.p_loss[0.0] > r.p_loss[0.5] > r.p_loss[1.0]                            # more real edge -> less chance of loss
    assert r.median_final[0.0] < 1000 < r.median_final[1.0]                          # no edge loses to the vig; full edge wins
    assert r.p_maxdd_ge_20[0.0] > r.p_maxdd_ge_20[1.0]
    assert simulate(t, 500, 100, seed=7).equals(simulate(t, 500, 100, seed=7))       # reproducible
    big = simulate(default_template(0.03, stake_pct=0.06), 3000, 300, skills=(0.0,), seed=1)
    small = simulate(default_template(0.03, stake_pct=0.01), 3000, 300, skills=(0.0,), seed=1)
    assert big.p_ruin_50pct[0.0] > small.p_ruin_50pct[0.0] and big.median_maxdd[0.0] > small.median_maxdd[0.0]   # bigger stakes -> more ruin
    assert ((r.filter(like="p_") >= 0) & (r.filter(like="p_") <= 1)).all().all()
