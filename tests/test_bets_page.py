import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from nhlbet.data.store import Store
from nhlbet.site.bets import build_bets_page, build_bets_payload, paper_bets, render_bets_html

NODE = shutil.which("node")
SITE = Path(__file__).resolve().parent.parent / "nhlbet/site"


def store_with_paper():
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=g, season=1, game_type=2, game_date="2026-10-08", start_utc="2026-10-08T23:00:00Z", home=f"H{g}", away=f"A{g}",
                             home_score=3 if w else 1, away_score=1 if w else 3, status="FINAL" if w is not None else "FUT", last_period="REG", home_win=w,
                             source="nhl_api", updated_at=None) for g, w in ((1, 1), (2, 0), (3, None))], ["game_id"])
    mk = lambda run, at, gid, side, dec, stake, strat="every_game", action="BET": dict(
        run_id=run, run_at=at, game_id=gid, strategy=strat, game_date="2026-10-08", action=action, side=side, team=f"T{gid}", book="secretbook", decimal=dec, stake=stake,
        p_model=0.5, p_adj=0.5, p_market=0.5, edge=0.0, ev=0.0)
    st.upsert("shadow_bets", [mk("r1", "t1", 1, "away", 9.0, 99.0),                 # superseded by the later run
                              mk("r2", "t2", 1, "home", 2.0, 10.0), mk("r2", "t2", 2, "home", 1.9, 10.0), mk("r2", "t2", 3, "away", 2.2, 10.0),
                              mk("r2", "t2", 3, None, None, 0.0, "market_favorite", "NO_BET")], ["run_id", "game_id", "strategy"])
    return st


def test_paper_bets_use_latest_run_settle_and_skip_no_bets():
    d = paper_bets(store_with_paper()).set_index("game_id")
    assert len(d) == 3 and set(d.strategy) == {"every_game"}
    assert d.loc[1, "result"] == "won" and d.loc[1, "stake"] == 10.0 and d.loc[2, "result"] == "lost" and d.loc[3, "result"] == "pending"
    assert d.loc[1, "game"] == "A1 @ H1" and d.loc[1, "pick"] == "T1 moneyline"


def test_page_is_self_contained_safe_and_has_no_bookmaker_name(tmp_path):
    st = store_with_paper()
    games = [{"away": "A3", "home": "H3", "model_side": "home", "sides": {"home": {"team": "H3", "best_decimal": 1.9}}}]
    out = build_bets_page(st, games, 1000.0, "2026-10-08T15:00:00", "2026-10-08", tmp_path)
    html = out.read_text()
    assert "secretbook" not in html and "__DATA__" not in html and "/*__" not in html
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    p = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(p["paper_bets"]) == 3 and p["bankroll"] == 1000.0 and "afford to lose" in p["disclaimer"]
    assert "innerHTML" not in html                                   # text only: bet names can never inject markup
    assert any(s["name"] == "every_game" for s in p["paper_strategies"])


def test_empty_store_still_renders():
    p = build_bets_payload(Store(":memory:"), [], 1000.0, "t", "d")
    assert p["paper_bets"] == [] and "<script" in render_bets_html(p)


def js(expr):
    code = f"const C=require({json.dumps(str(SITE / 'bets_core.js'))});console.log(JSON.stringify({expr}));"
    r = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_syntax_and_maths_by_hand():
    for f in ("bets_core.js", "bets_ui.js"):
        assert subprocess.run([NODE, "--check", str(SITE / f)], capture_output=True).returncode == 0
    assert js("C.toDecimal(-110,'american')") == pytest.approx(1 + 100 / 110)
    assert js("C.toDecimal(150,'american')") == pytest.approx(2.5)
    assert js("[C.toDecimal(50,'american'), C.toDecimal(1,'decimal'), C.toDecimal('x','decimal')]") == [None, None, None]
    # won $10 @2.5 (+15), lost $10 (-10), push $5 (0), pending $7
    s = js("C.summarize([{stake:10,decimal:2.5,result:'won'},{stake:10,decimal:2,result:'lost'},{stake:5,decimal:2,result:'push'},{stake:7,decimal:2,result:'pending'}])")
    assert s["profit"] == 5 and s["staked"] == 20 and s["roi"] == pytest.approx(0.25) and s["hit"] == 0.5 and s["pending"] == 1 and s["open_stake"] == 7
    c = js("C.cumulative([{date:'2026-01-02',stake:10,decimal:2,result:'lost'},{date:'2026-01-01',stake:10,decimal:2,result:'won'},{date:'2026-01-03',stake:1,decimal:2,result:'pending'}])")
    assert [x["cum"] for x in c] == [10, 0]
