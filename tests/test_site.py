"""The picks page: payload, HTML safety, and Python<->JavaScript parity of the stake and parlay maths."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhlbet.odds.edge import SideQuote
from nhlbet.odds.math import expected_value
from nhlbet.report.slate import SlateGame
from nhlbet.risk.parlay import parlay
from nhlbet.risk.policy import RiskConfig, recommend_slate
from nhlbet.site.build import build_payload, build_site, render_html

NODE = shutil.which("node")
CORE = Path(__file__).resolve().parent.parent / "nhlbet/site/core.js"


def quotes(ph, mh, dh, da):
    mk = lambda side, team, p, m, d, b: SideQuote(side, team, p, m, b, d, 1 / d, p - m, expected_value(p, d), 5)
    return {"home": mk("home", "H", ph, mh, dh, "bookB"), "away": mk("away", "A", 1 - ph, 1 - mh, da, "bookA")}


def random_slate(rng, n, cfg):
    inputs, games = [], []
    for i in range(n):
        m = rng.uniform(0.35, 0.65)
        p = float(np.clip(m + rng.normal(0.01, 0.05), 0.05, 0.95))
        dh, da = 1 / m / 1.04, 1 / (1 - m) / 1.04
        q = quotes(p, m, dh, da)
        for s in q.values():
            s.team = f"T{i}{s.side[0]}"
        ctx = {"model_status": "OK", "odds_stale": False, "goalie_confirmed": bool(rng.random() < 0.6), "games_played_min": 30.0}
        inputs.append((i, f"H{i}", f"A{i}", q, ctx))
        books = [{"book": "bookA", "home": dh - 0.03, "away": da}, {"book": "bookB", "home": dh, "away": da - 0.04}]
        games.append((i, q, ctx, books, p))
    recs = recommend_slate(inputs, cfg)
    slate = [SlateGame(i, pd.Timestamp("2026-10-08 23:00", tz="UTC"), f"H{i}", f"A{i}", "g", "g", "probable", p, p, q, r, None, False, [], books, ctx)
             for (i, q, ctx, books, p), r in zip(games, recs)]
    return slate, recs


def payload_for(slate, cfg, **kw):
    return build_payload(slate, cfg, kw.get("perf"), {"version": "v", "drift": {"status": "OK"}}, {"enabled": True}, "2026-10-08", "morning")


def node_eval(expr: str, data: dict):
    code = f"const C=require({json.dumps(str(CORE))});const D=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", code], input=json.dumps(data), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_payload_is_json_safe_and_complete():
    cfg = RiskConfig()
    slate, _ = random_slate(np.random.default_rng(0), 6, cfg)
    p = payload_for(slate, cfg)
    json.dumps(p, allow_nan=False)                                              # no NaN/Infinity anywhere
    assert {"games", "policy", "model", "odds", "track", "disclaimer", "generated_at"} <= set(p)
    g = p["games"][0]
    assert {"home", "away", "sides", "blocked", "p_home"} <= set(g) and set(g["sides"]["home"]) >= {"kelly_full", "qualifies", "books", "p_adj", "ev", "fails"}
    assert "educational" in p["disclaimer"] and "afford to lose" in p["disclaimer"]


def test_blocked_games_carry_the_reason_and_never_qualify():
    cfg = RiskConfig()
    slate, _ = random_slate(np.random.default_rng(1), 3, cfg)
    for s in slate:
        s.ctx = {**s.ctx, "model_status": "ALERT"}
    p = payload_for(slate, cfg)
    assert all(g["blocked"] and not any(sd["qualifies"] for sd in g["sides"].values()) for g in p["games"])
    empty = payload_for([], cfg)
    assert empty["games"] == []


def test_html_embeds_data_safely(tmp_path):
    cfg = RiskConfig()
    slate, _ = random_slate(np.random.default_rng(2), 2, cfg)
    slate[0].notes = ["</script><script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
    p = payload_for(slate, cfg)
    html = render_html(p)
    assert "</script><script>alert(1)" not in html                              # cannot break out of the data block
    assert html.count("</script>") == 3                                         # payload + core + ui only
    start = html.index('type="application/json">') + len('type="application/json">')
    back = json.loads(html[start:html.index("</script>", start)].replace("<\\/", "</"))
    assert back["games"][0]["notes"][0] == "</script><script>alert(1)</script>"   # round-trips intact
    assert "textContent" in (CORE.parent / "ui.js").read_text() and "innerHTML" not in (CORE.parent / "ui.js").read_text()
    out = build_site(p, tmp_path / "site")
    assert out.exists() and (tmp_path / "site/picks.json").exists() and html == out.read_text()
    assert html.count("educational") >= 1


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_syntax():
    for f in ("core.js", "ui.js"):
        r = subprocess.run([NODE, "--check", str(CORE.parent / f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("mode,mult", [("quarter", 0.25), ("half", 0.5)])
def test_js_stakes_match_python_policy(mode, mult):
    """For random slates the browser logic must pick the same bets with the same stakes as the Python recommender."""
    rng = np.random.default_rng(10)
    for trial in range(40):
        cfg = RiskConfig(bankroll=float(rng.choice([500, 1000, 2500])), kelly_fraction=mult, max_bets_per_day=int(rng.integers(2, 6)),
                         max_daily_exposure_pct=float(rng.choice([0.02, 0.03, 0.05])))
        slate, recs = random_slate(rng, int(rng.integers(3, 12)), cfg)
        payload = payload_for(slate, cfg)
        s = {"bankroll": cfg.bankroll, "unit": 10, "mode": mode, "maxPct": cfg.max_bet_pct, "dailyPct": cfg.max_daily_exposure_pct,
             "maxBets": cfg.max_bets_per_day, "minStake": cfg.min_stake}
        js = node_eval("C.allocate(D.games, D.s)", {"games": payload["games"], "s": s})
        py = {r.game_id: (r.side, r.stake) for r in recs if r.action == "BET"}
        assert {int(k) for k in js} == set(py), (trial, js, py)
        for k, v in js.items():
            assert v["side"] == py[int(k)][0] and abs(v["stake"] - py[int(k)][1]) <= 0.011, (trial, k, v, py[int(k)])


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_flat_mode_and_unit_sizes():
    cfg = RiskConfig(bankroll=1000)
    slate, recs = random_slate(np.random.default_rng(5), 10, cfg)
    payload = payload_for(slate, cfg)
    base = {"bankroll": 1000, "unit": 10, "mode": "flat", "maxPct": 0.02, "dailyPct": 0.05, "maxBets": 5, "minStake": 1}
    js = node_eval("C.allocate(D.games, D.s)", {"games": payload["games"], "s": base})
    assert all(v["stake"] <= 10.0 + 1e-9 for v in js.values()) and sum(v["stake"] for v in js.values()) <= 50.0 + 1e-9   # 5% daily cap
    big = node_eval("C.allocate(D.games, D.s)", {"games": payload["games"], "s": {**base, "unit": 1000}})
    assert all(v["stake"] <= 20.0 + 1e-9 for v in big.values())                  # a huge unit is still capped at 2% of bankroll


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_parlay_matches_python():
    rng = np.random.default_rng(3)
    for _ in range(40):
        n = int(rng.integers(2, 5))
        legs = []
        for i in range(n):
            books = {b: float(rng.uniform(1.6, 2.4)) for b in rng.choice(["a", "b", "c", "d"], size=int(rng.integers(2, 5)), replace=False)}
            legs.append({"game_id": i, "p": float(rng.uniform(0.4, 0.65)), "best_decimal": max(books.values()), "books": books})
        py, js = parlay(legs), node_eval("C.parlay(D.legs, 4)", {"legs": legs})
        for k in ("p", "fair_decimal", "offered_decimal", "ev", "singles_ev"):
            assert js[k] == pytest.approx(py[k], rel=1e-12), k
        assert js["book"] == py["book"] and js["better_as_singles"] == py["better_as_singles"]


def test_parlay_rules():
    leg = lambda i, p=0.55, d=1.9: {"game_id": i, "p": p, "best_decimal": d, "books": {"a": d, "b": d - 0.05}}
    r = parlay([leg(1), leg(2)])
    assert r["p"] == pytest.approx(0.3025) and r["offered_decimal"] == pytest.approx(3.61) and r["book"] == "a"
    assert r["ev"] == pytest.approx(0.3025 * 3.61 - 1) and r["fair_decimal"] == pytest.approx(1 / 0.3025)
    with pytest.raises(ValueError, match="at least 2"):
        parlay([leg(1)])
    with pytest.raises(ValueError, match="different games"):
        parlay([leg(1), leg(1)])
    with pytest.raises(ValueError, match="at most 4"):
        parlay([leg(i) for i in range(5)])
    no_common = parlay([{"game_id": 1, "p": 0.5, "best_decimal": 2.0, "books": {"a": 2.0}}, {"game_id": 2, "p": 0.5, "best_decimal": 2.0, "books": {"b": 2.0}}])
    assert no_common["book"] is None and not no_common["single_book_available"]
    # edge-less legs: the parlay multiplies the vig, so it loses more per dollar than the singles do
    fair = 1 / 0.5 / 1.05
    worse = parlay([{"game_id": i, "p": 0.5, "best_decimal": fair, "books": {"a": fair}} for i in range(3)])
    assert worse["ev"] < worse["singles_ev"] < 0 and worse["better_as_singles"]


def test_public_payload_has_no_bookmaker_names_or_per_book_prices():
    cfg = RiskConfig()
    slate, _ = random_slate(np.random.default_rng(4), 5, cfg)
    pub = build_payload(slate, cfg, None, {"version": "v", "drift": {"status": "OK"}}, {"enabled": True}, "2026-10-08", "morning", public_safe=True)
    txt = json.dumps(pub)
    assert "bookA" not in txt and "bookB" not in txt and pub["public_safe"] is True
    for g in pub["games"]:
        for sd in g["sides"].values():
            assert sd["best_book"] is None and sd["books"] == {} and sd["best_decimal"] > 1      # one aggregated best price, no book
    priv = build_payload(slate, cfg, None, {"version": "v", "drift": {"status": "OK"}}, {"enabled": True}, "2026-10-08", "morning", public_safe=False)
    assert "bookA" in json.dumps(priv) and priv["public_safe"] is False
    assert "bookA" not in render_html(pub)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_price_in_my_app_maths_by_hand():
    """The user's own price drives EV, break-even price and stake; nothing depends on other books."""
    r = node_eval("[C.toDecimal(-115,'american'), C.toDecimal(150,'american'), C.toDecimal(50,'american'), C.toDecimal(1.87,'decimal'), C.toDecimal(0.9,'decimal')]", {})
    assert r[0] == pytest.approx(1 + 100 / 115) and r[1] == pytest.approx(2.5) and r[2] is None and r[3] == 1.87 and r[4] is None
    # 55% to win at 1.95: EV = 0.55*0.95 - 0.45 = 0.0725; break-even price 1/0.55
    assert node_eval("C.evAt(0.55, 0, 1.95)", {}) == pytest.approx(0.0725) and node_eval("C.minDecimal(0.55, 0)", {}) == pytest.approx(1 / 0.55)
    # a whole-number total: 45% win, 12% push, 43% lose at 2.0 -> EV = 0.45*1 - 0.43 = 0.02; break-even (1-0.12)/0.45
    assert node_eval("C.evAt(0.45, 0.12, 2.0)", {}) == pytest.approx(0.02) and node_eval("C.minDecimal(0.45, 0.12)", {}) == pytest.approx(0.88 / 0.45)
    assert node_eval("C.evAt(0.45, 0.12, C.minDecimal(0.45, 0.12))", {}) == pytest.approx(0, abs=1e-12)            # EV is exactly zero at the break-even price
    # quarter Kelly at p=0.55, d=1.95: full Kelly = (0.95*0.55-0.45)/0.95 = 0.07632; 25% of it = 1.908% of $1000 = $19.08 (below the 2% cap); a 1% cap gives $10.00
    s = {"bankroll": 1000, "unit": 10, "mode": "quarter", "maxPct": 0.02, "dailyPct": 0.05, "maxBets": 5, "minStake": 1}
    assert node_eval(f"C.stakeAt(0.55, 1.95, {json.dumps(s)})", {}) == pytest.approx(19.08, abs=0.01)
    assert node_eval(f"C.stakeAt(0.55, 1.95, {json.dumps({**s, 'maxPct': 0.01})})", {}) == pytest.approx(10.0)
    assert node_eval(f"C.stakeAt(0.50, 1.90, {json.dumps(s)})", {}) == 0                                        # negative EV at this price -> no stake
