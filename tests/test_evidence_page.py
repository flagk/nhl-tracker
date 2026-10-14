"""The evidence page and the no-JavaScript table: static HTML, escaped, with verdicts computed by rule from the numbers."""
import re

from nhlbet.site.build import render_html, static_games_table
from nhlbet.site.validation_page import render_validation_html, verdicts
from tests.test_site import payload_for, random_slate
from nhlbet.risk.policy import RiskConfig
import numpy as np


def res(diff=0.015, lo=-0.002, hi=0.03, roi_lo=-0.2, roi_hi=0.15):
    return {"data_through": "2026-10-14", "games_with_results": 100,
            "game_level": {"n": 100, "models": {"model": {"log_loss": 0.658, "brier": 0.233, "n": 100}, "market_close": {"log_loss": 0.643, "brier": 0.226, "n": 100}},
                           "model_minus_close_log_loss": {"diff": diff, "ci": [lo, hi], "p_value_model_better": 0.9}, "reliability": []},
            "picks": {"every_game": {"n": 100, "roi": -0.02, "roi_ci": [roi_lo, roi_hi], "win_rate": 0.58, "break_even_win_rate": 0.583, "units": -2.0, "max_drawdown_units": 7.8,
                                     "longest_losing_streak": 7, "mean_edge_vs_no_vig": 0.009, "clv": {"mean_prob_points": 0.08}}},
            "random_baseline": {"every_game": {"p_random_at_least_as_good": 0.38}}, "required_bets": {"roi": {"1%": 45000, "3%": 5000, "5%": 1800}},
            "by_market": {"totals": {"picks": 97, "win_rate": 0.485, "roi": -0.08, "roi_ci": [-0.35, 0.16], "units": -7.9, "p_roi_not_above_0": 0.7, "picks_needed_for_5pct_roi": 2255,
                                     "verdict": "insufficient evidence (interval spans 0)"}},
            "controls_by_market": {}, "splits": {}, "log_integrity": {}, "backtest_by_season": {}}


def test_verdicts_follow_the_numbers_not_the_wish():
    assert "WORSE" in verdicts(res(lo=0.005, hi=0.03))[0][1]                       # interval above zero: the model is worse than the market
    assert "BETTER" in verdicts(res(diff=-0.02, lo=-0.03, hi=-0.01))[0][1]
    mid = verdicts(res())[0][1]
    assert "No demonstrable difference" in mid and "favours the market" in mid
    assert any("5,028" not in t and "5,000" in t for _, t in verdicts(res()))      # the sample-size statement is built from required_bets


def test_evidence_page_is_static_escaped_and_carries_the_disclaimer():
    r = res()
    r["by_market"]["<script>alert(1)</script>"] = r["by_market"]["totals"]
    page = render_validation_html(r, "body{}")
    assert "<script" not in page.lower().replace("&lt;script", "")                  # no script element anywhere, injected names are escaped
    assert "&lt;script&gt;" in page and "Is there an edge?" in page and "not betting advice" in page.lower()
    assert "Backtest (not live)" in page and "Known limitations" in page and "insufficient evidence" in page


def test_basic_table_works_without_javascript_and_escapes_team_names():
    slate, _ = random_slate(np.random.default_rng(0), 3, RiskConfig())
    p = payload_for(slate, RiskConfig())
    p["games"][0]["away"] = "<img src=x onerror=alert(1)>"
    t = static_games_table(p)
    assert "<img" not in t and "&lt;img" in t and t.count("<tr>") >= 4 and "Model P(home)" in t and "Prices as of" in t
    page = render_html(p)
    ns = re.search(r"<noscript>(.*?)</noscript>", page, re.S).group(1)
    assert "<table" in ns and "evidence page" in ns
