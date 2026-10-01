"""Decide whether a scheduled workflow trigger should do any work. Standard library only (it runs before dependencies are installed).

GitHub's cron is best-effort: on this repository triggers have arrived 3-6 hours late (a 21:15 UTC "late" run fired after puck drop, and the closing-line
snapshots meant for 22:45 / 23:20 UTC fired at 01:35 / 02:13). So instead of a few fixed times, the workflows are triggered often and each trigger asks
this gate whether *now* is a useful moment. Triggers that arrive too late simply do nothing, and the odds API credits are only spent in a useful window.

- ``morning``: once per game day (ET), never again after ``reports/daily/<date>-morning.md`` exists.
- ``late``: once per game day, only while the first unstarted game begins within the next ``LATE_WINDOW`` (so goalies are mostly confirmed and
  the run is still before puck drop).
- ``close``: only when some game starts within the next ``CLOSE_WINDOW_MIN`` minutes and no live odds capture happened in the last ``MIN_GAP_MIN``.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

LATE_MIN_MIN = 20          # a late run needs at least this long before puck drop to be worth doing
LATE_WINDOW_H = 3.0
CLOSE_WINDOW_MIN = 35
MIN_GAP_MIN = 20


def game_day(now: datetime) -> str:
    return now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _starts(db: str, date: str) -> list[datetime]:
    p = Path(db)
    if not p.exists():
        return []
    con = sqlite3.connect(str(p))
    try:
        rows = con.execute("SELECT start_utc FROM games WHERE game_date = ? AND source = 'nhl_api' AND home_score IS NULL AND start_utc IS NOT NULL", (date,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out = []
    for (s,) in rows:
        try:
            out.append(datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc))
        except ValueError:
            continue
    return sorted(out)


def _last_live_fetch(db: str) -> datetime | None:
    p = Path(db)
    if not p.exists():
        return None
    con = sqlite3.connect(str(p))
    try:
        row = con.execute("SELECT MAX(captured_at) FROM odds_fetch_log WHERE source = 'live'").fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not row or not row[0]:
        return None
    try:
        return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def decide(kind: str, now: datetime, db: str = "data/nhl.db", report_dir: str = "reports", force: bool = False) -> tuple[bool, str]:
    """(run, reason). ``force`` is for manual dispatches: always run."""
    if force:
        return True, "manual run"
    now = now.astimezone(timezone.utc)
    date = game_day(now)
    if kind in ("morning", "late"):
        if (Path(report_dir) / "daily" / f"{date}-{kind}.md").exists():
            return False, f"{kind} run for {date} already done"
        if kind == "morning":
            return True, "first morning run of the day"
        up = [s for s in _starts(db, date) if s > now]
        if not up:
            return False, "no unstarted games left today"
        wait = up[0] - now
        if wait < timedelta(minutes=LATE_MIN_MIN):
            return False, f"first game starts in {int(wait.total_seconds() // 60)} min: too late for a useful late run"
        if wait > timedelta(hours=LATE_WINDOW_H):
            return False, f"first game is {wait.total_seconds() / 3600:.1f} h away: too early (goalies not confirmed yet)"
        return True, f"first game in {int(wait.total_seconds() // 60)} min"
    if kind == "close":
        soon = [s for s in _starts(db, date) + _starts(db, game_day(now - timedelta(hours=24))) if now - timedelta(minutes=2) <= s <= now + timedelta(minutes=CLOSE_WINDOW_MIN)]
        if not soon:
            return False, "no game starts in the next %d min" % CLOSE_WINDOW_MIN
        last = _last_live_fetch(db)
        if last is not None and now - last < timedelta(minutes=MIN_GAP_MIN):
            return False, f"odds were captured {int((now - last).total_seconds() // 60)} min ago"
        return True, f"{len(soon)} game(s) start within {CLOSE_WINDOW_MIN} min"
    raise ValueError(f"unknown kind {kind!r}")
