"""Decide whether a scheduled workflow trigger should do any work. Standard library only (it runs before dependencies are installed).

GitHub's cron is best-effort: on this repository triggers have arrived 3-6 hours late (a 21:15 UTC "late" run fired after puck drop, and the closing-line
snapshots meant for 22:45 / 23:20 UTC fired at 01:35 / 02:13). So instead of a few fixed times, the workflows are triggered often and each trigger asks
this gate whether *now* is a useful moment. Triggers that arrive too late simply do nothing, and the odds API credits are only spent in a useful window.

- ``morning``: once per game day (ET), from 8 am ET on, never again after ``reports/daily/<date>-morning.md`` exists.
- ``late``: once per game day, only while the first unstarted game begins within the next ``LATE_WINDOW`` (so goalies are mostly confirmed and
  the run is still before puck drop).
- ``close``: only when a game starts within the next ``CLOSE_WINDOW_MIN`` minutes and has no capture in its final ``CLOSE_WINDOW_MIN`` minutes yet (one capture covers every game
  starting soon), and none happened in the last ``MIN_GAP_MIN``: about one capture per start wave instead of one every 20 minutes.
- ``review``: the pre-game review agent, once per start wave: ``REVIEW_LEAD_MIN`` minutes before the wave's first puck drop (taken from the committed
  schedule in ``site/picks.json``, never a fixed clock time), until ``REVIEW_LATEST_MIN`` before it. Games more than ``WAVE_GAP_MIN`` apart are different
  waves, so a late west-coast block gets its own pass. A wave whose review already ran (``data/review/picks_final.json``), even a failed one, is not re-run
  automatically: a failure should reach a person, not spend the subscription in a loop.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

LATE_MIN_MIN = 20          # a late run needs at least this long before puck drop to be worth doing
LATE_WINDOW_H = 3.0
CLOSE_WINDOW_MIN = 25      # a game's closing snapshot is one taken in the 25 minutes before it starts
MIN_GAP_MIN = 10           # never two captures closer than this
REVIEW_LEAD_MIN = 75       # start the review this long before a wave's first puck drop
REVIEW_LATEST_MIN = 5      # ...and not at all once it is this close (the answer could no longer be acted on)
WAVE_GAP_MIN = 120         # games starting this far apart or further belong to different waves (7:00 and 8:00 pm ET share a pass, 10:00 pm gets its own)
MORNING_FROM_ET_H = 8      # yesterday's late games are final (settlement) and the day's lines exist; a midnight run was too early


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


def _morning_ok(now: datetime) -> tuple[bool, str]:
    if now.astimezone(ZoneInfo("America/New_York")).hour < MORNING_FROM_ET_H:
        return False, "too early for the morning run (waits until 8 am ET so last night's games are settled)"
    return True, "first morning run of the day"


def _uncovered(starts: list[datetime], last: datetime | None, now: datetime) -> list[datetime]:
    """Games starting within the close window whose closing snapshot has not been taken yet.

    One capture returns every game's odds, so it covers all games starting in the next 25 minutes; a game is covered once any capture falls in its final
    25 minutes. This replaces 'one capture every 20 minutes while anything is about to start', which spent about 17 credits on a busy night.
    """
    win = timedelta(minutes=CLOSE_WINDOW_MIN)
    soon = [s for s in starts if now - timedelta(minutes=2) <= s <= now + win]
    return [s for s in soon if last is None or not (s - win <= last <= s)]


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
            return _morning_ok(now)
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
        starts = _starts(db, date) + _starts(db, game_day(now - timedelta(hours=24)))
        last = _last_live_fetch(db)
        if not [s for s in starts if now - timedelta(minutes=2) <= s <= now + timedelta(minutes=CLOSE_WINDOW_MIN)]:
            return False, "no game starts in the next %d min" % CLOSE_WINDOW_MIN
        if last is not None and now - last < timedelta(minutes=MIN_GAP_MIN):
            return False, f"odds were captured {int((now - last).total_seconds() // 60)} min ago"
        due = _uncovered(starts, last, now)
        if not due:
            return False, "every game starting soon already has its closing snapshot"
        return True, f"{len(due)} game(s) start within {CLOSE_WINDOW_MIN} min without a closing snapshot"
    raise ValueError(f"unknown kind {kind!r}")


# ---------------------------------------------------------------------------------------------------------------------------
# The same decisions from committed files only (no database), for the always-on heartbeat job, which has no cache to restore.
# ``site/picks.json`` lists the current slate with start times; ``reports/daily/<date>-<kind>.md`` marks completed runs; the file names under
# ``data/logs/odds_fetch_log/`` are the times of odds captures.
import json  # noqa: E402


def _slate(root: str | Path) -> tuple[str | None, list[datetime]]:
    p = Path(root) / "site" / "picks.json"
    if not p.exists():
        return None, []
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None, []
    starts = []
    for g in d.get("games", []):
        s = g.get("start_utc")
        if s:
            try:
                starts.append(datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc))
            except ValueError:
                continue
    return d.get("date"), sorted(starts)


def _last_capture_from_files(root: str | Path) -> datetime | None:
    names = sorted((Path(root) / "data" / "logs" / "odds_fetch_log").glob("*.csv"))
    for p in reversed(names):
        try:
            return datetime.strptime(p.stem[:19], "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def decide_files(kind: str, now: datetime, root: str | Path = ".") -> tuple[bool, str]:
    """(run, reason) for ``morning`` / ``late`` / ``close`` using only files in the repository checkout."""
    now = now.astimezone(timezone.utc)
    date = game_day(now)
    if kind in ("morning", "late"):
        if (Path(root) / "reports" / "daily" / f"{date}-{kind}.md").exists():
            return False, f"{kind} run for {date} already done"
        if kind == "morning":
            return _morning_ok(now)
    slate_date, starts = _slate(root)
    if kind == "late":
        if slate_date != date:
            return False, "today's slate is not built yet (the morning run comes first)"
        up = [s for s in starts if s > now]
        if not up:
            return False, "no unstarted games left today"
        wait = up[0] - now
        if wait < timedelta(minutes=LATE_MIN_MIN):
            return False, f"first game starts in {int(wait.total_seconds() // 60)} min: too late for a useful late run"
        if wait > timedelta(hours=LATE_WINDOW_H):
            return False, f"first game is {wait.total_seconds() / 3600:.1f} h away: too early (goalies not confirmed yet)"
        return True, f"first game in {int(wait.total_seconds() // 60)} min"
    if kind == "review":
        t = review_target(now, root)
        return (True, t["reason"]) if t.get("key") else (False, t["reason"])
    if kind == "close":
        if slate_date not in (date, game_day(now - timedelta(hours=24))):
            return False, "no current slate"
        last = _last_capture_from_files(root)
        if not [s for s in starts if now - timedelta(minutes=2) <= s <= now + timedelta(minutes=CLOSE_WINDOW_MIN)]:
            return False, "no game starts in the next %d min" % CLOSE_WINDOW_MIN
        if last is not None and now - last < timedelta(minutes=MIN_GAP_MIN):
            return False, f"odds were captured {int((now - last).total_seconds() // 60)} min ago"
        due = _uncovered(starts, last, now)
        if not due:
            return False, "every game starting soon already has its closing snapshot"
        return True, f"{len(due)} game(s) start within {CLOSE_WINDOW_MIN} min without a closing snapshot"
    raise ValueError(f"unknown kind {kind!r}")


def slate_waves(root: str | Path) -> tuple[str | None, list[dict]]:
    """(slate date, waves) from ``site/picks.json``: each wave is ``{"key", "start", "game_ids"}`` with games sorted by start time."""
    p = Path(root) / "site" / "picks.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None, []
    games = []
    for g in d.get("games", []):
        try:
            games.append((datetime.fromisoformat(str(g["start_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc), int(g["game_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    games.sort()
    waves: list[list[tuple[datetime, int]]] = []
    for st, gid in games:
        if waves and st - waves[-1][-1][0] < timedelta(minutes=WAVE_GAP_MIN):
            waves[-1].append((st, gid))
        else:
            waves.append([(st, gid)])
    date = d.get("date")
    return date, [{"key": f"rv-{date}-{w[0][0]:%H%M}", "start": w[0][0], "game_ids": [g for _, g in w]} for w in waves]


def review_target(now: datetime, root: str | Path = ".") -> dict:
    """The wave that should be reviewed now: ``{"key", "start", "game_ids", "reason"}``, or ``{"key": None, "reason": ...}`` when nothing is due."""
    now = now.astimezone(timezone.utc)
    date = game_day(now)
    slate_date, waves = slate_waves(root)
    if slate_date != date:
        return {"key": None, "reason": "today's slate is not built yet"}
    if not waves:
        return {"key": None, "reason": "no games today"}
    try:
        state = json.loads((Path(root) / "data" / "review" / "picks_final.json").read_text())
    except (OSError, ValueError):
        state = {}
    done = {x.get("run_id") for x in state.get("passes", [])} if state.get("date") == date else set()
    try:
        cand = json.loads((Path(root) / "data" / "review" / "picks_candidate.json").read_text())
    except (OSError, ValueError):
        cand = {}
    for w in waves:
        wait = w["start"] - now
        if wait > timedelta(minutes=REVIEW_LEAD_MIN):
            return {"key": None, "reason": f"next wave starts in {int(wait.total_seconds() // 60)} min: review opens {REVIEW_LEAD_MIN} min before"}
        if wait < timedelta(minutes=REVIEW_LATEST_MIN):
            continue                                                        # this wave is (almost) under way; look at the next one
        if w["key"] in done:
            continue
        if cand.get("date") != date or not any(p.get("game_id") in w["game_ids"] for p in cand.get("picks", [])):
            return {"key": None, "reason": f"wave {w['key']} has no candidate picks yet"}
        return {**w, "reason": f"wave {w['key']} starts in {int(wait.total_seconds() // 60)} min and has not been reviewed"}
    return {"key": None, "reason": "every remaining wave is reviewed or already under way"}
