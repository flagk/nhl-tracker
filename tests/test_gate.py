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


def _close_db(tmp_path, name, starts, fetches=()):
    d = tmp_path / name
    d.mkdir()
    return dict(db=mkdb(d, [("2026-10-01", s) for s in starts], fetches=fetches), report_dir=str(tmp_path))


def test_close_snapshot_is_taken_once_per_start_wave(tmp_path):
    waves = ["2026-10-01T23:00:00Z", "2026-10-01T23:30:00Z", "2026-10-02T00:00:00Z"]
    a = _close_db(tmp_path, "a", waves, fetches=["2026-10-01T22:20:00+00:00"])
    assert not decide("close", T("2026-10-01T21:00:00"), **a)[0]                 # 2 h before: not the close
    assert decide("close", T("2026-10-01T22:40:00"), **a)[0]                     # 20 min before 23:00; the 22:20 capture is outside its final 25 min: take the closing snapshot
    b = _close_db(tmp_path, "b", waves, fetches=["2026-10-01T22:45:00+00:00"])
    run, why = decide("close", T("2026-10-01T22:55:00"), **b)                    # the 22:45 capture IS the 23:00 game's closing snapshot
    assert not run
    assert not decide("close", T("2026-10-01T23:03:00"), **b)[0]                 # 23:30 is 27 min away: not yet inside its own window
    assert decide("close", T("2026-10-01T23:07:00"), **b)[0]                     # 23 min before 23:30 and 22:45 is outside that game's window: capture the second wave
    c = _close_db(tmp_path, "c", waves, fetches=["2026-10-01T23:08:00+00:00"])
    assert not decide("close", T("2026-10-01T23:20:00"), **c)[0]                 # the 23:08 capture is inside the 23:30 game's final 25 minutes: already its closing snapshot
    d = _close_db(tmp_path, "d", ["2026-10-01T23:00:00Z"])
    assert not decide("close", T("2026-10-01T23:05:00"), **d)[0]                 # puck has dropped: in-play prices are not wanted


def test_morning_waits_until_8am_et(tmp_path):
    kw = dict(db=str(tmp_path / "x.db"), report_dir=str(tmp_path))
    assert not decide("morning", T("2026-10-05T04:02:00"), **kw)[0]      # midnight ET: last night's games not settled yet
    assert not decide("morning", T("2026-10-05T11:59:00"), **kw)[0]      # 7:59 am EDT
    assert decide("morning", T("2026-10-05T12:00:00"), **kw)[0]
