"""Reliable scheduler: stays alive for hours, and every few minutes asks the gate whether a morning run, a late run or a closing-odds
snapshot is due; if so, dispatches that workflow. GitHub's own cron has been delivering triggers 3-6 hours late, so this does not depend on it.

    python3 scripts/heartbeat.py --minutes 340 --interval 180

Needs the ``gh`` CLI with ``GH_TOKEN`` (workflow permissions: actions write). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.gate import decide_files  # noqa: E402

TARGETS = {"morning": ("daily.yml", {"run_type": "morning", "gated": "true"}),
           "late": ("daily.yml", {"run_type": "late", "gated": "true"}),
           "close": ("odds-close.yml", {"gated": "true"}),
           "review": ("review.yml", {"gated": "true"})}
COOLDOWN_S = {"morning": 3600, "late": 3600, "close": 1200, "review": 1800}     # never dispatch the same job again sooner than this


def ping_liveness(url: str | None = None, timeout: float = 10.0) -> bool:
    """Tell the external dead-man's switch (Healthchecks.io, HC_PING_URL) this heartbeat is alive. Never raises: a missed ping is the alarm, not a crash.
    Docs: https://healthchecks.io/docs/http_api/ (a GET on the check's ping URL means "alive")."""
    import urllib.request
    url = url or os.environ.get("HC_PING_URL", "").strip()
    if not url:
        return False
    for _ in range(2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:                                    # noqa: BLE001
            time.sleep(2)
    return False


def sh(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout


def busy(workflow: str, run=sh) -> bool:
    """True when that workflow already has a queued or running run (so we do not pile up duplicates)."""
    for status in ("queued", "in_progress"):
        out = run(["gh", "run", "list", "--workflow", workflow, "--status", status, "--json", "databaseId", "--limit", "1"])
        try:
            if json.loads(out or "[]"):
                return True
        except ValueError:
            continue
    return False


def dispatch(workflow: str, inputs: dict, run=sh) -> None:
    args = ["gh", "workflow", "run", workflow, "--ref", "main"]
    for k, v in inputs.items():
        args += ["-f", f"{k}={v}"]
    run(args)


def tick(now: datetime, last: dict, root: str = ".", run=sh, decide=decide_files) -> list[str]:
    """One heartbeat: returns the jobs dispatched. ``last`` remembers when each was last dispatched."""
    fired = []
    for kind in ("morning", "late", "close", "review"):
        ok, why = decide(kind, now, root)
        if not ok:
            continue
        if (now.timestamp() - last.get(kind, 0)) < COOLDOWN_S[kind]:
            continue
        wf, inputs = TARGETS[kind]
        if busy(wf, run):
            continue
        dispatch(wf, inputs, run)
        last[kind] = now.timestamp()
        fired.append(f"{kind} ({why})")
    return fired


STUCK_AFTER_S = 20 * 60          # a Pages deploy normally takes ~1 minute


def unstick_pages(now: datetime, run=sh) -> list[str]:
    """Cancel Pages deploys stuck waiting/queued/pending for 20+ minutes (one such run blocked every later deploy for a day), then start a fresh one."""
    cancelled = []
    for status in ("waiting", "queued", "pending", "in_progress"):
        out = run(["gh", "api", f"repos/{os.environ.get('GITHUB_REPOSITORY', 'flagk/nhl-tracker')}/actions/workflows/pages.yml/runs?status={status}&per_page=20"])
        try:
            runs = json.loads(out or "{}").get("workflow_runs", [])
        except ValueError:
            continue
        for r in runs:
            age = now.timestamp() - datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).timestamp()
            if age > STUCK_AFTER_S:
                run(["gh", "api", "-X", "POST", f"repos/{os.environ.get('GITHUB_REPOSITORY', 'flagk/nhl-tracker')}/actions/runs/{r['id']}/cancel"])
                cancelled.append(str(r["id"]))
    if cancelled:
        dispatch("pages.yml", {}, run)
    return cancelled


def refresh(run=sh) -> None:
    run(["git", "fetch", "--depth=1", "-q", "origin", "main"])
    run(["git", "reset", "--hard", "-q", "origin/main"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=340)
    ap.add_argument("--interval", type=float, default=180)
    a = ap.parse_args()
    start, last, last_ping = time.time(), {}, 0.0
    ping_every = float(os.environ.get("HEARTBEAT_PING_MIN", "10")) * 60          # config/policy.yaml autonomy.heartbeat.ping_interval_min
    while time.time() - start < a.minutes * 60:
        if time.time() - last_ping >= ping_every:
            ping_liveness()
            last_ping = time.time()
        refresh()
        now = datetime.now(timezone.utc)
        fired = tick(now, last)
        stuck = unstick_pages(now)
        if stuck:
            fired.append("cancelled stuck Pages deploy " + ",".join(stuck))
        print(f"{now:%H:%M:%S}Z heartbeat: " + (", ".join(fired) if fired else "nothing due"), flush=True)
        time.sleep(a.interval)
    ran = (time.time() - start) / 60
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"ran_minutes={ran:.0f}\n")


if __name__ == "__main__":
    main()
