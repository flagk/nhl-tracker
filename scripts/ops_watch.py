"""Weekly watch and digest: credential expiry, odds credits, keepalive health, then the digest message. Alerts are keyed per week/month so each is sent once.

    python scripts/ops_watch.py            # needs GH_TOKEN + GITHUB_REPOSITORY for the keepalive lookups; NTFY_TOPIC to deliver
Warnings never halt the picks; they exist so that nothing expires silently.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nhlbet.ops import digest, notify, state as S, watch  # noqa: E402
from nhlbet.policy import policy_get  # noqa: E402

log = logging.getLogger("ops_watch")


def remaining_credits(root: Path = ROOT) -> int | None:
    fs = sorted((root / "data" / "logs" / "odds_fetch_log").glob("*.csv"))
    if not fs:
        return None
    try:
        d = pd.concat([pd.read_csv(f) for f in fs[-3:]]).sort_values("captured_at")
        v = d.remaining.dropna()
        return int(v.iloc[-1]) if len(v) else None
    except Exception:                                                      # noqa: BLE001
        return None


def commit_age_days(now: datetime) -> float | None:
    out = subprocess.run(["git", "log", "-1", "--format=%ct"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return (now.timestamp() - float(out)) / 86400 if out else None


def keepalive_success_age_days(now: datetime) -> float | None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return None
    out = subprocess.run(["gh", "api", f"repos/{repo}/actions/workflows/keepalive.yml/runs?status=success&per_page=1"], capture_output=True, text=True).stdout
    try:
        runs = json.loads(out or "{}").get("workflow_runs", [])
        return (now - datetime.fromisoformat(runs[0]["updated_at"].replace("Z", "+00:00"))).total_seconds() / 86400 if runs else None
    except (ValueError, KeyError, IndexError):
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    now = datetime.now(timezone.utc)
    credits = remaining_credits()
    events = watch.credential_events(now, credits) + watch.keepalive_events(commit_age_days(now), keepalive_success_age_days(now), now)
    for e in events:
        a = notify.event_to_alert(e)
        print(("sent " if notify.send(a["title"], a["message"], a["priority"], a["tags"], a["key"]) else "NOT sent ") + a["title"])
    state = S.load_state(ROOT / "state" / "system_state.json")
    ev = {}
    try:
        ev = json.loads((ROOT / "reports" / "evaluation.json").read_text())
    except (OSError, ValueError):
        pass
    exp = [e["message"].split(".")[0] for e in events if e["check"] == "claude_oauth_token"]
    title, body = digest.build_digest(state, ev, None, credits, exp, now)
    print(("sent " if notify.send(title, body, "default", ("bar_chart",), f"digest:{now:%G-W%V}") else "NOT sent ") + title)
    print(body)


if __name__ == "__main__":
    main()
