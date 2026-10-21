"""The halt state machine: OK, HALT_SOFT, HALT_HARD, kept in a committed state file that is separate from the policy file.

* A failed check opens a halt with a unique id, a reason, a timestamp and the exact next step. Re-running while it persists opens nothing new.
* SOFT halts clear after ``soft_resume_healthy_runs`` consecutive runs in which every check passed.
* HARD halts never clear by themselves: only when their id appears in ``autonomy.acknowledged_halts`` of the policy file, which only the owner edits. After an acknowledgement
  the same failing check is suppressed (no new halt, one reminder) until it passes once, so an accepted-but-persisting condition does not page every run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nhlbet.ops.checks import CheckResult

ROOT = Path(__file__).resolve().parent.parent.parent
STATE_PATH = ROOT / "state" / "system_state.json"


def empty_state() -> dict:
    return {"schema": 1, "status": "OK", "halts": [], "healthy_streak": 0, "last_run": None, "policy_sha256": None, "suppressed": [], "alerts_sent": []}


def load_state(path: Path | str = STATE_PATH) -> dict:
    try:
        s = json.loads(Path(path).read_text())
        return {**empty_state(), **s}
    except (OSError, ValueError):
        return empty_state()


def save_state(state: dict, path: Path | str = STATE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1, sort_keys=True))


def open_halts(state: dict) -> list[dict]:
    return [h for h in state["halts"] if h["cleared_at"] is None]


def status_of(state: dict) -> str:
    o = open_halts(state)
    return "HALT_HARD" if any(h["kind"] == "hard" for h in o) else "HALT_SOFT" if o else "OK"


def update(state: dict, results: list[CheckResult], now: datetime, soft_resume_runs: int, acknowledged: list[str], policy_sha: str | None = None) -> tuple[dict, list[dict]]:
    """Apply one run's check results. Returns (new state, events); events are ``halt_opened`` / ``halt_escalated`` / ``halt_cleared`` / ``acknowledged_but_persisting``."""
    st = json.loads(json.dumps(state))                                        # never mutate the caller's copy
    now_s = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    events: list[dict] = []
    ack = set(acknowledged or [])
    suppressed = set(st["suppressed"])
    by_name = {r.name: r for r in results}
    # 1. owner acknowledgements clear hard halts (and only those)
    for h in open_halts(st):
        if h["kind"] == "hard" and h["id"] in ack:
            h["cleared_at"], h["cleared_by"] = now_s, "owner acknowledgement in config/policy.yaml"
            suppressed.add(h["check"])
            events.append({"type": "halt_cleared", "id": h["id"], "check": h["check"], "by": h["cleared_by"]})
    # 2. a suppressed check that passes is no longer suppressed
    for name in list(suppressed):
        if name in by_name and by_name[name].ok:
            suppressed.discard(name)
    # 3. open / escalate halts for failures
    open_by_check = {h["check"]: h for h in open_halts(st)}
    for r in results:
        if r.ok:
            continue
        h = open_by_check.get(r.name)
        if r.name in suppressed and h is None:
            events.append({"type": "acknowledged_but_persisting", "check": r.name, "detail": r.detail, "id": None})
            continue
        if h is None:
            h = {"id": f"H-{now:%Y%m%d-%H%M%S}-{r.name}", "check": r.name, "kind": r.level, "reason": r.detail, "next_step": r.next_step, "opened_at": now_s, "last_seen": now_s,
                 "cleared_at": None, "cleared_by": None}
            st["halts"].append(h)
            open_by_check[r.name] = h
            events.append({"type": "halt_opened", "id": h["id"], "check": r.name, "kind": r.level, "reason": r.detail, "next_step": r.next_step})
        else:
            h["last_seen"], h["reason"] = now_s, r.detail
            if r.level == "hard" and h["kind"] == "soft":
                h["kind"], h["next_step"] = "hard", r.next_step
                events.append({"type": "halt_escalated", "id": h["id"], "check": r.name, "kind": "hard", "reason": r.detail, "next_step": r.next_step})
    # 4. soft halts clear after N consecutive fully healthy runs
    healthy = all(r.ok for r in results)
    st["healthy_streak"] = st["healthy_streak"] + 1 if healthy else 0
    if healthy and st["healthy_streak"] >= max(1, int(soft_resume_runs)):
        for h in open_halts(st):
            if h["kind"] == "soft":
                h["cleared_at"], h["cleared_by"] = now_s, f"{st['healthy_streak']} consecutive healthy runs"
                events.append({"type": "halt_cleared", "id": h["id"], "check": h["check"], "by": h["cleared_by"]})
    st["suppressed"] = sorted(suppressed)
    st["status"] = status_of(st)
    st["last_run"] = now_s
    if policy_sha is not None:
        st["policy_sha256"] = policy_sha
    return st, events


def can_publish(state: dict) -> bool:
    """Picks are published only in the OK state."""
    return state.get("status", "OK") == "OK"
