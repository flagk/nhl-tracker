"""Phase 9: status banners and per-pick fields exist without JavaScript, and the page offers nothing above the policy staking ceiling."""
import re
from pathlib import Path

from nhlbet.policy import policy_get
from nhlbet.site import build

SITE = Path(build.__file__).parent


def payload(**kw):
    g = {"game_id": 1, "start_utc": "2026-10-20T23:00:00Z", "home": "BOS", "away": "TOR", "p_home": 0.55, "blocked": None, "odds_captured_at": "2026-10-20T16:00:00+00:00",
         "sides": {"home": {"team": "BOS", "p_model": 0.55, "p_market": 0.5, "p_adj": 0.54, "edge": 0.05, "ev": 0.04, "best_decimal": 2.0, "qualifies": True, "fails": []}},
         "server_action": "BET", "server_stake": 12.5, "alt": [{"market": "totals", "side": "over", "label": "Over 6.0", "point": 6.0, "p_model": 0.45, "p_push": 0.1, "p_market": 0.44,
                                                              "edge": 0.01, "ev": 0.0, "best_decimal": 1.95, "best_book": None, "n_books": 3}]}
    return {"date": "2026-10-20", "generated_at": "2026-10-20T16:00:00+00:00", "model": {"version": "v", "status": "OK"}, "games": [g], "ranking": [], "disclaimer": "Research only.", **kw}


def test_static_page_shows_status_stake_push_probabilities_and_limits():
    html = build.render_html(payload(system={"status": "HALT_HARD", "halts": [{"check": "drawdown", "reason": "too deep", "next_step": "decide", "kind": "hard", "id": "H-1", "opened_at": "x"}]},
                                      betting={"status": "RESEARCH ONLY", "why": "40 live picks (need 300)"}))
    noscript = html.split("<noscript>")[1].split("</noscript>")[0]
    assert "HALT_HARD" in noscript and "picks are withheld" in noscript and "RESEARCH ONLY" in noscript and "40 live picks" in noscript
    assert "$12.50" in noscript and "P(push)" in noscript and "10.0%" in noscript and "45.0%" in noscript and "45.0%" in noscript and "Method and limits" in noscript
    assert "guarantee" not in noscript.lower() or "no guarantee" in noscript.lower() or "nothing guarantees" in noscript.lower()


def test_default_status_is_research_only_when_the_gate_is_absent():
    assert "Betting status: RESEARCH ONLY" in build.render_html(payload())


def test_page_offers_nothing_above_the_policy_staking_ceiling():
    core, ui = (SITE / "core.js").read_text(), (SITE / "ui.js").read_text()
    mult = float(re.search(r"function kellyMultiplier\(\)\s*\{\s*return ([0-9.]+);", core).group(1))
    assert mult <= float(policy_get("staking", "kelly_fraction_max"))
    assert "Half Kelly" not in ui and "half" not in re.findall(r'value: "(\w+)"', ui)
    assert "P.max_bet_pct" in ui and "P.max_daily_exposure_pct" in ui              # caps can be lowered, never raised past the policy
