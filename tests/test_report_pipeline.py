import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from nhlbet.data.goalies import goalie_display, load_confirmations, starter_overrides
from nhlbet.data.store import Store
from nhlbet.models.bundle import ModelBundle
from nhlbet.odds.edge import SideQuote
from nhlbet.odds.math import expected_value
from nhlbet.odds.snapshots import record_fetch
from nhlbet.pipeline import game_day, run_daily
from nhlbet.report.betlog import export_logs, final_recommendations, performance, resolved, restore_logs
from nhlbet.report.markdown import DISCLAIMER, render_report
from nhlbet.report.slate import SlateGame, build_slate, context_notes
from nhlbet.risk.policy import Recommendation, RiskConfig
from tests import fakes as F

ROOT = Path(__file__).resolve().parent.parent


def sq(side, team, p, m, dec, book="bookB"):
    return SideQuote(side, team, p, m, book, dec, 1 / dec, p - m, expected_value(p, dec), 4)


def mk_slate(bet=True):
    q = {"home": sq("home", "BOS", 0.58, 0.53, 1.90), "away": sq("away", "TOR", 0.42, 0.47, 2.15)}
    rec = Recommendation(1, "BOS", "TOR", "BET", "home", "BOS", "bookB", 1.90, 9.73, 0.0097, 0.58, 0.545, 0.53, 0.05, 0.035, 0.05, [])
    nobet = Recommendation(2, "NYR", "PIT", reasons=["edge +1.0% below the 3% minimum"])
    a = SlateGame(1, pd.Timestamp("2024-01-05 00:00", tz="UTC"), "BOS", "TOR", "Swayman", "Stolarz", "confirmed", 0.58, 0.57, q, rec if bet else nobet, "2024-01-04T20:00:00+00:00", False, ["TOR is on the second night of a back-to-back."])
    b = SlateGame(2, None, "NYR", "PIT", "unknown", "unknown", "unknown", 0.51, 0.51, None, nobet, None, False, [])
    return [a, b]


# ---------------------------------------------------------------- report rendering
def test_report_has_every_required_field_and_the_disclaimer_twice():
    txt = render_report("2024-01-04", "late", mk_slate(), RiskConfig(), "v1", {"p_source": "online_platt", "drift": {"status": "OK"}}, None,
                        {"captured_at": "2024-01-04T20:00:00+00:00", "stale": False, "remaining": 412})
    assert txt.count(DISCLAIMER) == 2 and "educational" in txt and "afford to lose" in txt
    for needle in ("TOR @ BOS", "19:00 ET", "Stolarz / Swayman (confirmed)", "58.0%", "53.0%", "BOS 1.90 @ bookB", "+5.0%", "+3.5%", "$9.73", "BET BOS",
                   "NYR", "no odds", "edge +1.0% below the 3% minimum", "second night of a back-to-back", "1 recommended bet", "credits left: 412"):
        assert needle in txt, needle
    assert "No resolved recommendations yet" in txt


def test_pass_rows_do_not_advertise_an_ev():
    """A game the policy passes on must not show an 'EV per $1' (it is computed from the raw model probability)."""
    txt = render_report("2024-01-04", "late", mk_slate(bet=False), RiskConfig(), "v1", {"drift": {"status": "OK"}})
    row = [l for l in txt.splitlines() if l.startswith("| TOR @ BOS")][0]
    assert "+3.5%" not in row and row.split("|")[7].strip() == "-"
    assert "+5.0%" in row                                                             # the edge is still shown


def test_report_no_bets_and_health_banners():
    txt = render_report("2024-01-04", "morning", mk_slate(bet=False), RiskConfig(), "v1",
                        {"drift": {"status": "ALERT", "performance": {"reasons": ["recent log loss worse than coin flip"]}}})
    assert "**No bets today.**" in txt and "Model health: ALERT" in txt and "suspended" in txt and "worse than coin flip" in txt
    stale = render_report("d", "m", mk_slate(), RiskConfig(), "v1", {"drift": {"status": "OK"}}, None, {"captured_at": "x", "stale": True})
    assert "(STALE)" in stale
    assert "No NHL games" in render_report("d", "m", [], RiskConfig(), "v1", {})


# ---------------------------------------------------------------- bet log: settlement, ROI, CLV, Brier
def seeded_store():
    st = Store(":memory:")
    games = [(1, "2024-01-05", "BOS", "TOR", 1), (2, "2024-01-05", "NYR", "PIT", 0), (3, "2024-01-06", "COL", "DAL", 1), (4, "2024-01-06", "EDM", "VAN", 0)]
    st.upsert("games", [dict(game_id=g, season=20232024, game_type=2, game_date=d, start_utc=f"{d}T23:00:00Z", home=h, away=a, home_score=3 if w else 1,
                             away_score=1 if w else 3, status="FINAL", last_period="REG", home_win=w, source="nhl_api", updated_at=None) for g, d, h, a, w in games], ["game_id"])
    def rec(gid, run, at, action, side, team, dec, stake, p, pm):
        return dict(run_id=run, run_at=at, run_type="late", game_id=gid, game_date=[g[1] for g in games if g[0] == gid][0], home="H", away="A", home_goalie="x",
                    away_goalie="y", goalie_status="probable", model_version="v1", p_model=p, p_adj=None, p_market=pm, p_stack_raw=p, action=action, side=side,
                    team=team, book="bookB", decimal=dec, stake=stake, edge=None, ev=None, reasons="", odds_captured_at=None, odds_stale=0, model_status="OK")
    st.upsert("recommendations", [
        rec(1, "r1", "2024-01-05T14:00:00", "BET", "away", "TOR", 2.2, 5.0, 0.60, 0.55),      # superseded by the later run below
        rec(1, "r2", "2024-01-05T21:00:00", "BET", "home", "BOS", 2.0, 10.0, 0.60, 0.55),      # final for game 1: home won -> +10
        rec(2, "r2", "2024-01-05T21:00:00", "BET", "home", "NYR", 1.9, 10.0, 0.58, 0.53),      # home lost -> -10
        rec(3, "r3", "2024-01-06T21:00:00", "NO_BET", None, None, None, 0.0, 0.55, 0.50),
        rec(4, "r3", "2024-01-06T21:00:00", "BET", "away", "VAN", 2.5, 4.0, 0.45, 0.40)],      # away won -> +6
        ["run_id", "game_id"])
    # closing consensus (last snapshot before puck drop) for CLV
    for gid, home_p in ((1, 0.58), (2, 0.50), (4, 0.62)):
        st.upsert("odds_snapshots", [dict(captured_at=f"2024-01-0{5 if gid < 3 else 6}T22:55:00+00:00", event_id=f"e{gid}", commence_time=f"2024-01-0{5 if gid < 3 else 6}T23:00:00Z",
                                          home="H", away="A", book="bk", market="h2h", outcome=o, point=0.0, price=price, book_updated=None, game_id=gid)
                                     for o, price in (("H", 1 / home_p), ("A", 1 / (1 - home_p)))], ["captured_at", "event_id", "book", "market", "outcome", "point"])
    return st


def test_settlement_uses_final_recommendation_and_computes_profit():
    st = seeded_store()
    fin = final_recommendations(st)
    assert len(fin) == 4 and fin[fin.game_id == 1].run_id.iloc[0] == "r2"          # later run supersedes
    d = resolved(st)
    prof = dict(zip(d.game_id, d.profit))
    assert prof == {1: 10.0, 2: -10.0, 3: 0.0, 4: 6.0}
    assert d[d.game_id == 3].is_bet.iloc[0] == False


def test_performance_metrics_by_hand():
    st = seeded_store()
    perf = performance(st, start_bankroll=1000)
    b = perf["bets"]
    assert b["n"] == 3 and b["staked"] == 24.0 and b["profit"] == 6.0 and b["roi"] == pytest.approx(6 / 24) and b["win_rate"] == pytest.approx(2 / 3)
    assert perf["bankroll"] == 1006.0 and perf["no_bet_rate"] == pytest.approx(0.25)
    y = np.array([1, 0, 1, 0]); p = np.array([0.60, 0.58, 0.55, 0.45])
    assert perf["model"]["brier"] == pytest.approx(np.mean((p - y) ** 2))
    # CLV: game1 bet home @2.0, close home 0.58 -> 2.0*0.58-1=+0.16; game2 bet home @1.9, close 0.50 -> -0.05; game4 away @2.5, close away 0.38 -> -0.05
    d = resolved(st).set_index("game_id")
    assert d.clv_ev[1] == pytest.approx(0.16) and d.clv_ev[2] == pytest.approx(-0.05) and d.clv_ev[4] == pytest.approx(2.5 * 0.38 - 1)
    assert perf["clv"]["n"] == 3 and perf["drawdown"]["max_drawdown"] == pytest.approx(10 / 1010)


def test_log_export_restore_roundtrip_is_idempotent(tmp_path):
    st = seeded_store()
    files = export_logs(st, tmp_path / "logs")
    names = {p.parent.name + "/" + p.name for p in files}
    assert "recommendations/r1.csv" in names and "recommendations/r3.csv" in names and "logs/bet_log.csv" in names
    assert sum(1 for p in files if p.parent.name == "odds") == 2                      # one file per odds capture
    fresh = Store(":memory:")
    n = restore_logs(fresh, tmp_path / "logs")
    assert n["recommendations"] == 5 and n["odds_snapshots"] == 6
    before = {str(p.relative_to(tmp_path)): p.read_text() for p in (tmp_path / "logs").rglob("*.csv")}
    restore_logs(fresh, tmp_path / "logs"); restore_logs(fresh, tmp_path / "logs")          # repeated restores add nothing
    assert len(fresh.df("SELECT * FROM recommendations")) == 5 and len(fresh.df("SELECT * FROM odds_snapshots")) == 6
    export_logs(fresh, tmp_path / "logs")
    after = {str(p.relative_to(tmp_path)): p.read_text() for p in (tmp_path / "logs").rglob("*.csv") if p.name != "bet_log.csv"}
    assert all(after[k] == before[k] for k in after) and set(after) == {k for k in before if not k.endswith("bet_log.csv")}   # byte-identical


def test_concurrent_jobs_write_disjoint_files(tmp_path):
    """Regression: the morning run, late run and closing-line job rewrote the SAME csv files, so a push after a concurrent job's
    commit hit a git conflict. Every capture/run now owns its files, so two jobs' outputs can never overlap."""
    base = seeded_store()
    a, b = Store(":memory:"), Store(":memory:")
    for st in (a, b):
        restore_logs(st, export_and_dir(base, tmp_path / "origin"))
    a.upsert("odds_snapshots", [dict(captured_at="2024-01-07T10:00:00+00:00", event_id="e9", commence_time="2024-01-07T23:00:00Z", home="H", away="A", book="bk",
                                     market="h2h", outcome="H", point=0.0, price=1.9, book_updated=None, game_id=None)], ["captured_at", "event_id", "book", "market", "outcome", "point"])
    a.upsert("odds_fetch_log", [dict(captured_at="2024-01-07T10:00:00+00:00", ok=1, source="live", remaining=9, used=1, events=1, note="h2h")], ["captured_at"])
    b.upsert("odds_snapshots", [dict(captured_at="2024-01-07T22:30:00+00:00", event_id="e9", commence_time="2024-01-07T23:00:00Z", home="H", away="A", book="bk",
                                     market="h2h", outcome="H", point=0.0, price=1.8, book_updated=None, game_id=None)], ["captured_at", "event_id", "book", "market", "outcome", "point"])
    fa = {str(p.relative_to(tmp_path / "a")) for p in export_logs(a, tmp_path / "a")}
    fb = {str(p.relative_to(tmp_path / "b")) for p in export_logs(b, tmp_path / "b")}
    new_a, new_b = fa - {str(p.relative_to(tmp_path / "origin")) for p in (tmp_path / "origin").rglob("*.csv")}, fb - {str(p.relative_to(tmp_path / "origin")) for p in (tmp_path / "origin").rglob("*.csv")}
    assert new_a and new_b and new_a.isdisjoint(new_b)                                # each job adds only its own new files
    merged = Store(":memory:")                                                        # git would merge both file sets cleanly
    for d in ("origin", "a", "b"):
        restore_logs(merged, tmp_path / d)
    assert len(merged.df("SELECT * FROM odds_snapshots WHERE event_id='e9'")) == 2


def export_and_dir(store, path):
    export_logs(store, path)
    return path


def test_restore_reads_the_legacy_single_file_layout(tmp_path):
    """History committed by the first live runs used recommendations.csv / odds_fetch_log.csv / odds/<YYYY-MM>.csv."""
    st = seeded_store()
    root = tmp_path / "logs"; (root / "odds").mkdir(parents=True)
    st.df("SELECT * FROM recommendations").to_csv(root / "recommendations.csv", index=False)
    st.df("SELECT * FROM odds_snapshots").to_csv(root / "odds" / "2024-01.csv", index=False)
    fresh = Store(":memory:")
    n = restore_logs(fresh, root)
    assert n["recommendations"] == 5 and n["odds_snapshots"] == 6


# ---------------------------------------------------------------- goalies
def test_goalie_confirmations(tmp_path):
    st = Store(":memory:")
    st.upsert("games", [dict(game_id=1, season=1, game_type=2, game_date="2024-01-01", start_utc=None, home="BOS", away="TOR", home_score=2, away_score=1,
                             status="FINAL", last_period="REG", home_win=1, source="nhl_api", updated_at=None),
                        dict(game_id=2, season=1, game_type=2, game_date="2024-01-05", start_utc=None, home="BOS", away="NYR", home_score=None, away_score=None,
                             status="FUT", last_period=None, home_win=None, source="nhl_api", updated_at=None)], ["game_id"])
    st.upsert("goalie_game", [dict(game_id=1, team="BOS", player_id=31, name="Swayman", started=1, toi_sec=3600, shots_against=20, saves=19, goals_against=1, xg_faced=2.0)], ["game_id", "player_id"])
    assert goalie_display(st, "BOS", "2024-01-05") == ("Swayman", "probable") and goalie_display(st, "NYR", "2024-01-05") == ("unknown", "unknown")
    csv = tmp_path / "c.csv"; csv.write_text("date,team,name\n2024-01-05,BOS,Swayman\n2024-01-05,NYR,Shesterkin\n")
    assert load_confirmations(st, csv) == 2
    assert goalie_display(st, "BOS", "2024-01-05") == ("Swayman", "confirmed")
    g = st.df("SELECT * FROM games WHERE game_id=2").assign(game_date=lambda d: pd.to_datetime(d.game_date))
    assert starter_overrides(st, g) == {(2, "BOS"): 31}                    # NYR goalie unknown to history: name only, no id override
    (tmp_path / "bad.csv").write_text("day,team\n")
    with pytest.raises(ValueError, match="columns"):
        load_confirmations(st, tmp_path / "bad.csv")


def test_context_notes():
    row = pd.Series({"d_gd_season_shrunk": 0.5, "h_b2b": 0, "a_b2b": 1, "a_travel_km": 3000, "a_tz_shift": -3, "is_rivalry": 1, "h_gp_season": 5, "a_gp_season": 6})
    n = context_notes(row, "BOS", "TOR")
    assert any("BOS have the clearly better goal differential" in x for x in n) and any("TOR" in x and "back-to-back" in x for x in n)
    assert len(n) <= 4 and context_notes(pd.Series({}), "A", "B") == []


# ---------------------------------------------------------------- end-to-end on a synthetic league
def _e2e_setup(tmp_path, league, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/models").mkdir(parents=True)
    (tmp_path / "data/models/feature_selection.json").write_text(json.dumps({"kept": ["d_gd_season_shrunk", "d_rest_days"]}))
    g = league["games"].copy(); g["game_date"] = g.game_date.dt.strftime("%Y-%m-%d"); g["start_utc"] = g.start_utc.astype(str)
    st = Store(str(tmp_path / "t.db"))
    st.upsert("games", g.assign(updated_at=None).to_dict("records"), ["game_id"])
    st.upsert("team_game", league["team_game"].to_dict("records"), ["game_id", "team"])
    st.upsert("goalie_game", league["goalie_game"].to_dict("records"), ["game_id", "player_id"])
    last = pd.Timestamp(league["games"].game_date.max())
    day = (last + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    st.upsert("games", [dict(game_id=99999, season=20232024, game_type=2, game_date=day, start_utc=f"{day}T23:00:00Z", home="BOS", away="TOR", home_score=None,
                             away_score=None, status="FUT", last_period=None, home_win=None, source="nhl_api", updated_at=None)], ["game_id"])
    return st, day


def test_end_to_end_daily_run(league, tmp_path, monkeypatch):
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    # a strongly mispriced market so the policy has something to say
    books = [("bookA", 2.30, 1.65, f"{day}T15:00:00Z"), ("bookB", 2.25, 1.68, f"{day}T15:00:00Z")]
    fx = type("Fx", (), {"events": [F.odds_event("e1", commence=f"{day}T23:00:00Z", books=books)], "captured_at": f"{day}T16:00:00+00:00", "remaining": 300, "used": 200, "source": "live"})()
    record_fetch(st, fx)
    assert (st.df("SELECT game_id FROM odds_snapshots").game_id == 99999).all()
    r = run_daily(day, "morning", db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models")
    assert r["games"] == 1
    rep = (tmp_path / "reports" / "latest.md").read_text()
    assert (tmp_path / "reports/daily" / f"{day}-morning.md").exists() and "TOR @ BOS" in rep and rep.count(DISCLAIMER) == 2
    recs = Store(str(tmp_path / "t.db")).df("SELECT * FROM recommendations")
    assert len(recs) == 1 and recs.action.iloc[0] in ("BET", "NO_BET") and recs.p_market.iloc[0] is not None
    assert any((tmp_path / "data/logs/recommendations").glob("*.csv")) and any((tmp_path / "data/logs/odds").glob("*.csv"))
    # late run: no retrain, adds a second recommendation row for the same game (history kept, final = latest)
    r2 = run_daily(day, "late", db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models")
    assert r2["model"] == r["model"]
    s2 = Store(str(tmp_path / "t.db"))
    assert len(s2.df("SELECT * FROM recommendations")) == 2 and len(final_recommendations(s2)) == 1


def test_stale_odds_produce_no_bet(league, tmp_path, monkeypatch):
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    books = [("bookA", 2.60, 1.55, f"{day}T15:00:00Z")]
    fx = type("Fx", (), {"events": [F.odds_event("e1", commence=f"{day}T23:00:00Z", books=books)], "captured_at": "2000-01-01T00:00:00+00:00", "remaining": 1, "used": 1, "source": "live"})()
    record_fetch(st, fx)
    run_daily(day, "morning", db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models")
    rec = Store(str(tmp_path / "t.db")).df("SELECT * FROM recommendations").iloc[0]
    assert rec.action == "NO_BET" and rec.odds_stale == 1 and "stale" in rec.reasons


def test_no_odds_means_no_bet_and_is_reported(league, tmp_path, monkeypatch):
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    run_daily(day, "morning", db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models")
    assert "no odds" in (tmp_path / "reports/latest.md").read_text()
    assert Store(str(tmp_path / "t.db")).df("SELECT action FROM recommendations").action.iloc[0] == "NO_BET"


def test_game_day_is_us_eastern():
    assert game_day(pd.Timestamp("2024-01-05 03:00", tz="UTC").to_pydatetime()) == "2024-01-04"      # late-night UTC still belongs to the ET game day


# ---------------------------------------------------------------- workflows & secrets hygiene
@pytest.fixture(scope="module")
def workflows():
    return {p.name: yaml.safe_load(p.read_text()) for p in (ROOT / ".github/workflows").glob("*.yml")}


def test_workflows_are_valid_and_scheduled(workflows):
    assert {"ci.yml", "daily.yml", "odds-close.yml", "backfill.yml"} <= set(workflows)
    daily = workflows["daily.yml"][True]
    crons = [s["cron"] for s in daily["schedule"]]
    assert len(crons) == 2 and all(len(c.split()) == 5 for c in crons)              # morning + later-in-the-day goalie-confirmed rerun
    assert workflows["daily.yml"]["permissions"]["contents"] == "write"
    for name in ("daily.yml", "odds-close.yml", "backfill.yml"):
        assert workflows[name]["concurrency"]["group"] == "nhl-data"                # writers never overlap


def test_no_hardcoded_secrets_anywhere():
    pat = re.compile(r"(api[_-]?key|apikey|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9]{16,}", re.I)
    offenders = []
    for p in list((ROOT / "nhlbet").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / ".github").rglob("*.yml")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{p.name}:{i}")
    assert not offenders, offenders
    daily = (ROOT / ".github/workflows/daily.yml").read_text()
    assert "secrets.ODDS_API_KEY" in daily and "ODDS_API_KEY:" in daily


def test_every_path_the_workflow_commits_is_not_gitignored():
    """Regression: an over-broad `logs/` pattern in .gitignore also ignored data/logs/ (odds snapshots, recommendations), so the
    daily job's commit step failed and nothing irreplaceable would ever have been saved."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git not available")
    must_track = ["data/logs/recommendations/2026-10-08-morning-1430.csv", "data/logs/odds/2026-10-08T15-24-51-00-00.csv", "data/logs/bet_log.csv",
                  "data/logs/odds_fetch_log/2026-10-08T15-24-51-00-00.csv", "data/logs/goalie_confirmations.csv",
                  "reports/latest.md", "reports/daily/2026-10-08-morning.md", "reports/performance.png", "site/index.html", "site/picks.json",
                  "data/models/registry.json", "data/models/model_20261008-abc123.joblib", "data/models/hyperparams.json"]
    r = subprocess.run(["git", "check-ignore", "--no-index", *must_track], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == "", f"these must NOT be ignored: {r.stdout}"
    r2 = subprocess.run(["git", "check-ignore", "--no-index", "logs/nhlbet.log", "data/nhl.db"], cwd=ROOT, capture_output=True, text=True)
    assert set(r2.stdout.split()) == {"logs/nhlbet.log", "data/nhl.db"}             # the things we DO want ignored stay ignored
