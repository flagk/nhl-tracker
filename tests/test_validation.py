"""Honest-evaluation helpers (audit phase 2): hand-checkable statistics, pick tables, random-pick baseline, model vs market scoring."""
import numpy as np
import pandas as pd
import pytest

from nhlbet.report.validation import (cluster_bootstrap, longest_losing_streak, max_drawdown, pick_table, random_pick_baseline, required_n, score_vs_market,
                                      split_table, summarize_picks)


# ---------------------------------------------------------------- hand-checkable statistics
def test_streak_drawdown_and_required_n_by_hand():
    assert longest_losing_streak([1, 0, 0, 0, 1, 0, 0]) == 3 and longest_losing_streak([1, 1]) == 0
    assert max_drawdown([1, -1, -1, -1, 2, -4]) == pytest.approx(5.0)                 # cumulative 1,0,-1,-2,0,-4: the peak of 1 to the trough of -4 is 5
    assert required_n(0.05, 1.0) == int(np.ceil(((1.6448536 + 0.8416212) / 0.05) ** 2))
    assert required_n(0.0, 1.0) == 0


def test_cluster_bootstrap_interval_contains_the_mean_and_is_wider_with_clustering():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 10)
    v = rng.normal(0, 1, 200) + np.repeat(rng.normal(0, 1, 20), 10)                  # strong day effect
    lo, hi = cluster_bootstrap(v, day, np.mean, 2000)
    lo2, hi2 = cluster_bootstrap(v, np.arange(200), np.mean, 2000)                    # pretending every bet is independent
    assert lo < v.mean() < hi and (hi - lo) > (hi2 - lo2)


def _games(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.35, 0.65, n)
    hw = (rng.random(n) < ph).astype(int)
    return pd.DataFrame({"game_id": np.arange(n), "game_date": [f"2026-10-{1 + i // 4:02d}" for i in range(n)], "p_model_home": ph, "p_market_home": ph,
                         "p_close_home": ph, "elo_prob_home": ph, "home_win": hw})


def _paper(gp, strategy="every_game", price=1.9):
    side = np.where(gp.p_market_home >= 0.5, "home", "away")
    won = np.where(side == "home", gp.home_win == 1, gp.home_win == 0).astype(float)
    return pd.DataFrame({"date": gp.game_date, "strategy": strategy, "game_id": gp.game_id, "market": "moneyline", "side": side, "price": price, "stake": 10.0,
                         "won": won, "profit": np.where(won == 1, 10 * (price - 1), -10.0), "clv_ev": -0.02, "team": "x"})


def test_pick_table_side_probabilities_edge_and_units():
    gp = _games()
    gp.loc[0, ["p_model_home", "p_market_home", "p_close_home"]] = [0.30, 0.40, 0.45]       # an away pick: model 70%, market 60%, close 55%
    p = _paper(gp)
    p.loc[0, "side"] = "away"
    d = pick_table(gp, p, "every_game")
    r = d[d.game_id == 0].iloc[0]
    assert r.p_model == pytest.approx(0.70) and r.p_market == pytest.approx(0.60) and r.p_close == pytest.approx(0.55)
    assert r.edge == pytest.approx(0.10) and r.clv_prob == pytest.approx(-0.05) and r.units == pytest.approx(r.profit / 10)
    s = summarize_picks(d, B=300)
    assert s["n"] == 40 and s["units"] == pytest.approx(d.units.sum()) and 0 <= s["win_rate"] <= 1 and s["roi_ci"][0] <= s["roi"] <= s["roi_ci"][1]
    assert set(split_table(d, "role").role) <= {"favorite", "underdog"} and split_table(d, "month").n.sum() == 40


def test_random_baseline_matches_a_fair_coin_when_nothing_distinguishes_the_sides():
    gp = _games(400, seed=3)
    gp["p_market_home"] = 0.5
    p = _paper(gp, price=2.0)                                                           # fair even-money prices, no margin
    d = pick_table(gp, p, "every_game")
    b = random_pick_baseline(d, gp, B=2000)
    assert abs(b["random_mean_roi"]) < 0.02 and 0.0 <= b["p_random_at_least_as_good"] <= 1.0
    assert b["random_roi_95"][0] < b["random_mean_roi"] < b["random_roi_95"][1]


def test_score_vs_market_ranks_a_perfect_market_above_a_noisy_model():
    gp = _games(300, seed=5)
    rng = np.random.default_rng(1)
    gp["p_model_home"] = np.clip(gp.p_market_home + rng.normal(0, 0.08, len(gp)), 0.05, 0.95)         # the market is exactly right, the model adds noise
    out = score_vs_market(gp, B=500)
    assert out["models"]["market_close"]["log_loss"] < out["models"]["model"]["log_loss"]
    assert out["model_minus_close_log_loss"]["diff"] > 0 and out["model_minus_close_log_loss"]["p_value_model_better"] > 0.5




def test_market_summary_separates_model_picks_from_controls_and_counts_distinct_picks():
    from nhlbet.report.validation import control_names, market_summary
    assert {"always_over", "market_favorite", "parlay_favorites_control"} <= control_names() and "every_game" not in control_names()
    rows = []
    for i in range(40):
        for strat in ("totals_edge", "every_total"):                                   # two strategies make the same pick: it must count once
            rows.append({"date": f"2026-10-{1 + i // 4:02d}", "strategy": strat, "game_id": i, "market": "totals", "team": "Over 6.5", "stake": 10.0, "won": float(i % 3 == 0), "profit": 9.0 if i % 3 == 0 else -10.0})
        rows.append({"date": f"2026-10-{1 + i // 4:02d}", "strategy": "always_over", "game_id": i, "market": "totals", "team": "Over 6.5", "stake": 10.0, "won": 0.0, "profit": -10.0})
    paper = pd.DataFrame(rows)
    m = market_summary(paper)["totals"]
    assert m["picks"] == 40 and m["roi"] == pytest.approx((14 * 9 - 26 * 10) / 400) and m["roi_ci"][0] <= m["roi"] <= m["roi_ci"][1]
    assert m["verdict"] in ("insufficient evidence (interval spans 0)", "credibly LOSING")
    c = market_summary(paper, controls=True)["totals"]
    assert c["picks"] == 40 and c["roi"] == pytest.approx(-1.0)
    assert market_summary(paper.head(10))["totals"]["verdict"].startswith("too few picks")


def test_log_integrity_flags_rows_logged_after_puck_drop_and_duplicates(tmp_path):
    import json
    from nhlbet.report.validation import log_integrity
    arch = tmp_path / "archive"; arch.mkdir()
    payload = json.dumps({"games": [{"game_id": 1, "start_utc": "2026-10-08T23:00:00+00:00"}, {"game_id": 2, "start_utc": "2026-10-08T23:00:00+00:00"}]})
    (arch / "2026-10-08.html").write_text(f'<script type="application/json" id="payload">{payload}</script>')
    d = tmp_path / "logs" / "recommendations"; d.mkdir(parents=True)
    pd.DataFrame([{"run_id": "r1", "game_id": 1, "run_at": "2026-10-08T20:00:00+00:00", "odds_captured_at": "2026-10-08T19:59:00+00:00", "model_version": "v1"},
                  {"run_id": "r2", "game_id": 2, "run_at": "2026-10-08T23:30:00+00:00", "odds_captured_at": "2026-10-08T23:29:00+00:00", "model_version": "v1"},    # in-play
                  {"run_id": "r2", "game_id": 2, "run_at": "2026-10-08T23:30:00+00:00", "odds_captured_at": "2026-10-08T23:29:00+00:00", "model_version": None}]).to_csv(d / "x.csv", index=False)
    r = log_integrity(tmp_path / "logs", arch)["recommendations"]
    assert r["rows"] == 3 and r["logged_at_or_after_puck_drop"] == 2 and r["duplicate_keys"] == 1 and r["rows_without_model_version"] == 1
