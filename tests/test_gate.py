import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from nhlbet.gate import decide, game_day

T = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def mkdb(tmp_path, starts, fetches=()):
    p = tmp_path / "n.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE games (game_id INTEGER PRIMARY KEY, game_date TEXT, start_utc TEXT, source TEXT, home_score INTEGER)")
    c.execute("CREATE TABLE odds_fetch_log (captured_at TEXT PRIMARY KEY, source TEXT)")
    for i, (d, s) in enumerate(starts):
        c.execute("INSERT INTO games VALUES (?,?,?,?,NULL)", (i, d, s, "nhl_api"))
    for f in fetches:
        c.execute("INSERT INTO odds_fetch_log VALUES (?,?)", (f, "live"))
    c.commit(); c.close()
    return str(p)


def test_game_day_is_eastern():
    assert game_day(T("2026-10-01T03:00:00")) == "2026-09-30"


def test_morning_runs_once_per_game_day(tmp_path):
    kw = dict(db=str(tmp_path / "none.db"), report_dir=str(tmp_path / "reports"))
    assert decide("morning", T("2026-10-01T14:30:00"), **kw)[0]
    (tmp_path / "reports/daily").mkdir(parents=True)
    (tmp_path / "reports/daily/2026-10-01-morning.md").write_text("x")
    run, why = decide("morning", T("2026-10-01T16:00:00"), **kw)
    assert not run and "already done" in why
    assert decide("morning", T("2026-10-02T14:30:00"), **kw)[0]                  # next game day


def test_late_run_window(tmp_path):
    db = mkdb(tmp_path, [("2026-10-01", "2026-10-01T23:00:00Z"), ("2026-10-01", "2026-10-02T02:00:00Z")])
    kw = dict(db=db, report_dir=str(tmp_path / "reports"))
    assert not decide("late", T("2026-10-01T18:00:00"), **kw)[0]                 # 5 h before the first game: goalies not confirmed
    assert decide("late", T("2026-10-01T20:30:00"), **kw)[0]                     # 2.5 h before
    assert not decide("late", T("2026-10-01T22:50:00"), **kw)[0]                 # 10 min before: too late to be useful
    run, why = decide("late", T("2026-10-02T00:32:00"), **kw)                    # a trigger delayed to 00:32 UTC: the 23:00 game is under way but the 02:00 one is still ahead
    assert run and "88 min" in why                                               # (the slate skips the started game itself)
    assert not decide("late", T("2026-10-02T01:50:00"), **kw)[0]                 # 10 min before the last game
    assert not decide("late", T("2026-10-02T02:30:00"), **kw)[0]                 # everything has started
    (tmp_path / "reports/daily").mkdir(parents=True)
    (tmp_path / "reports/daily/2026-10-01-late.md").write_text("x")
    assert decide("late", T("2026-10-01T20:30:00"), **kw) == (False, "late run for 2026-10-01 already done")


def test_late_run_skips_when_no_unstarted_games(tmp_path):
    db = mkdb(tmp_path, [])
    assert not decide("late", T("2026-10-01T20:30:00"), db=db, report_dir=str(tmp_path / "r"))[0]


def test_close_snapshot_only_just_before_a_start_and_not_twice(tmp_path):
    db = mkdb(tmp_path, [("2026-10-01", "2026-10-01T23:00:00Z")], fetches=["2026-10-01T22:20:00+00:00"])
    kw = dict(db=db, report_dir=str(tmp_path))
    assert not decide("close", T("2026-10-01T21:00:00"), **kw)[0]                # 2 h before: not the close
    run, why = decide("close", T("2026-10-01T22:35:00"), **kw)                   # 25 min before puck drop but a live capture was 15 min ago (< 20 min gap)
    assert not run and "15 min ago" in why
    assert decide("close", T("2026-10-01T22:45:00"), **kw)[0]                    # 25 min since the last capture: take the closing snapshot
    assert not decide("close", T("2026-10-01T23:05:00"), **kw)[0]                # puck has dropped: nothing starts soon, in-play prices are not wanted
