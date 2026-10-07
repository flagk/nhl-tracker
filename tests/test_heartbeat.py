"""File-based gate decisions and the heartbeat dispatcher (no network: fake ``gh``)."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nhlbet.gate import decide_files

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("heartbeat", ROOT / "scripts" / "heartbeat.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def repo(tmp_path, date="2026-10-03", starts=("2026-10-03T23:00:00Z",), done=(), capture=None):
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "picks.json").write_text(json.dumps({"date": date, "games": [{"start_utc": s} for s in starts]}))
    (tmp_path / "reports" / "daily").mkdir(parents=True)
    for k in done:
        (tmp_path / "reports" / "daily" / f"{date}-{k}.md").write_text("x")
    if capture:
        d = tmp_path / "data" / "logs" / "odds_fetch_log"
        d.mkdir(parents=True)
        (d / f"{capture}.csv").write_text("x")
    return tmp_path


def test_morning_due_until_done(tmp_path):
    r = repo(tmp_path)
    assert decide_files("morning", utc("2026-10-03T15:52:00"), r)[0]
    (r / "reports" / "daily" / "2026-10-03-morning.md").write_text("x")
    assert not decide_files("morning", utc("2026-10-03T15:52:00"), r)[0]


def test_morning_not_at_midnight(tmp_path):
    r = repo(tmp_path)
    assert not decide_files("morning", utc("2026-10-04T04:02:00"), r)[0]
    assert decide_files("morning", utc("2026-10-04T12:05:00"), r)[0]


def test_late_window(tmp_path):
    r = repo(tmp_path)                       # first game 23:00Z
    assert not decide_files("late", utc("2026-10-03T19:00:00"), r)[0]    # 4 h away
    assert decide_files("late", utc("2026-10-03T21:00:00"), r)[0]        # 2 h away
    assert not decide_files("late", utc("2026-10-03T22:50:00"), r)[0]    # too close
    assert not decide_files("late", utc("2026-10-03T23:30:00"), r)[0]    # started


def test_late_needs_todays_slate(tmp_path):
    r = repo(tmp_path, date="2026-10-02")
    assert not decide_files("late", utc("2026-10-03T21:00:00"), r)[0]


def test_close_window_and_gap(tmp_path):
    r = repo(tmp_path)
    assert not decide_files("close", utc("2026-10-03T21:00:00"), r)[0]
    assert decide_files("close", utc("2026-10-03T22:40:00"), r)[0]
    (r / "data" / "logs" / "odds_fetch_log").mkdir(parents=True)
    (r / "data" / "logs" / "odds_fetch_log" / "2026-10-03T22-35-00.csv").write_text("x")
    assert not decide_files("close", utc("2026-10-03T22:40:00"), r)[0]
    assert decide_files("close", utc("2026-10-03T22:58:00"), r)[0]


def test_tick_dispatches_respects_cooldown_and_busy(tmp_path):
    r = repo(tmp_path)
    calls = []
    state = {"busy": False}

    def run(args):
        calls.append(args)
        if args[:3] == ["gh", "run", "list"]:
            return json.dumps([{"databaseId": 1}]) if state["busy"] else "[]"
        return ""

    last = {}
    now = utc("2026-10-03T15:52:00")
    assert hb.tick(now, last, str(r), run)[0].startswith("morning")
    wf = [c for c in calls if c[:3] == ["gh", "workflow", "run"]]
    assert wf and wf[0][3] == "daily.yml" and "gated=true" in wf[0] and "run_type=morning" in wf[0]
    assert hb.tick(now, last, str(r), run) == []                 # cooldown
    last.clear()
    state["busy"] = True
    n = len(wf)
    assert hb.tick(now, last, str(r), run) == []                 # a run is already going
    assert len([c for c in calls if c[:3] == ["gh", "workflow", "run"]]) == n


def test_workflow_structure():
    w = yaml.safe_load((ROOT / ".github/workflows/heartbeat.yml").read_text())
    assert w["permissions"]["actions"] == "write"
    text = (ROOT / ".github/workflows/heartbeat.yml").read_text()
    assert "ran_minutes" in text and "gh workflow run heartbeat.yml" in text
    for f in ("daily.yml", "odds-close.yml"):
        on = yaml.safe_load((ROOT / ".github/workflows" / f).read_text())[True]
        assert "gated" in on["workflow_dispatch"]["inputs"]


def test_unstick_cancels_old_waiting_pages_runs_and_redeploys():
    calls = []

    def run(args):
        calls.append(args)
        if args[:2] == ["gh", "api"] and "status=waiting" in args[2]:
            return json.dumps({"workflow_runs": [{"id": 7, "created_at": "2026-10-06T19:36:51Z"}, {"id": 8, "created_at": "2026-10-07T21:30:00Z"}]})
        return json.dumps({"workflow_runs": []})

    out = hb.unstick_pages(utc("2026-10-07T21:40:00"), run)
    assert out == ["7"]                                                  # the 26-hour-old one only; the 10-minute-old one is left alone
    assert any(c[:3] == ["gh", "workflow", "run"] and c[3] == "pages.yml" for c in calls)
    calls.clear()
    assert hb.unstick_pages(utc("2026-10-07T21:40:00"), lambda a: json.dumps({"workflow_runs": []})) == []
