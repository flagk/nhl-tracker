"""Push alerts (ntfy) and the external dead-man's switch (Healthchecks.io). Neither ever raises into the pipeline: an alert that cannot be delivered is reported as False and logged.

ntfy (docs: https://docs.ntfy.sh/publish/ , source of the docs https://github.com/binwiederhier/ntfy/blob/main/docs/publish.md): POST the message body to https://ntfy.sh/<topic>; headers Title,
Priority (1-5 or min/low/default/high/urgent) and Tags; a message is at most 4,096 bytes and a title 1 KB; the public server allows 250 messages a day and the topic name is effectively
a password, so it is a secret (NTFY_TOPIC). An optional access token goes in ``Authorization: Bearer``.

Healthchecks.io (docs: https://healthchecks.io/docs/http_api/ , source https://github.com/healthchecks/healthchecks/blob/master/templates/docs/http_api.md): HEAD/GET/POST on the check's ping URL
means "alive"; append /fail to signal a failure; pinging one check more than 5 times a minute may be rate limited; the check's own period and grace time (set in its web settings, see
config/policy.yaml ``autonomy.heartbeat``) decide when a missing ping alerts the owner.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger(__name__)
PRIORITY = {"hard": "urgent", "soft": "high", "info": "default", "low": "low"}


def _topic_url() -> str | None:
    t = os.environ.get("NTFY_TOPIC", "").strip()
    if not t:
        return None
    return t if t.startswith("http") else f"https://ntfy.sh/{t}"


def send(title: str, message: str, priority: str = "default", tags: tuple[str, ...] = (), key: str | None = None, post: Callable | None = None, sleep: Callable = time.sleep) -> bool:
    """Deliver one push notification. ``key`` becomes the ntfy Sequence-ID so a re-sent alert updates the earlier one instead of stacking. Returns True when the server accepted it."""
    url = _topic_url()
    if url is None:
        log.warning("NTFY_TOPIC is not set: alert not delivered: %s", title)
        return False
    if post is None:
        import requests
        post = requests.post
    headers = {"Title": title.encode("ascii", "replace").decode()[:900], "Priority": priority, "Tags": ",".join(tags)[:480]}
    if key:
        headers["X-Sequence-ID"] = key[:200]
    if os.environ.get("NTFY_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["NTFY_TOKEN"]
    body = message.encode("utf-8")[:4000]
    for i in range(3):
        try:
            r = post(url, data=body, headers=headers, timeout=15)
            if 200 <= getattr(r, "status_code", 0) < 300:
                return True
            log.warning("ntfy answered %s (attempt %d)", getattr(r, "status_code", "?"), i + 1)
        except Exception as e:                                           # noqa: BLE001 - an alert must never crash the run
            log.warning("ntfy delivery failed (attempt %d): %s", i + 1, type(e).__name__)
        sleep(2 ** i)
    return False


def event_to_alert(e: dict) -> dict | None:
    """Turn a state-machine event into {title, message, priority, tags, key}; None for events nobody needs to hear about."""
    t = e["type"]
    if t in ("halt_opened", "halt_escalated"):
        kind = e.get("kind", "soft")
        return {"title": f"NHL model HALT ({kind}): {e['check']}", "priority": PRIORITY[kind], "tags": ("rotating_light",) if kind == "hard" else ("warning",), "key": f"{t}:{e['id']}",
                "message": f"Picks are withheld.\nWhy: {e['reason']}\nNext step: {e['next_step']}\nHalt id: {e['id']}" + ("\nThis halt does NOT clear by itself: add the id to autonomy.acknowledged_halts in config/policy.yaml." if kind == "hard" else "\nThis halt clears by itself after the policy's number of healthy runs.")}
    if t == "halt_cleared":
        return {"title": f"NHL model resumed: {e['check']}", "priority": PRIORITY["info"], "tags": ("white_check_mark",), "key": f"halt_cleared:{e['id']}", "message": f"Halt {e['id']} cleared ({e['by']}). Picks publish again once no other halt is open."}
    if t == "acknowledged_but_persisting":
        return {"title": f"NHL model: acknowledged condition persists: {e['check']}", "priority": PRIORITY["low"], "tags": ("information_source",), "key": f"persist:{e['check']}:{datetime.now(timezone.utc):%Y%m%d}",
                "message": f"You acknowledged a halt for {e['check']}, but the condition is still true: {e.get('detail', '')}. No new halt was opened."}
    if t == "warning":
        return {"title": f"NHL model warning: {e['check']}", "priority": PRIORITY["info"], "tags": ("warning",), "key": e["key"], "message": e["message"]}
    return None


def dispatch(events: list[dict], state: dict, post: Callable | None = None) -> dict:
    """Send each event's alert once: its key is remembered in ``state['alerts_sent']`` only after the server accepted it, so re-running a day never repeats an alert and a failed delivery is retried."""
    sent = list(state.get("alerts_sent", []))
    for e in events:
        a = event_to_alert(e)
        if a is None or a["key"] in sent:
            continue
        if send(a["title"], a["message"], a["priority"], a["tags"], a["key"], post=post):
            sent.append(a["key"])
    state = {**state, "alerts_sent": sent[-500:]}
    return state


def ping(status: str = "ok", url: str | None = None, get: Callable | None = None, sleep: Callable = time.sleep) -> bool:
    """Tell the external dead-man's switch the system is alive (``ok``) or failing (``fail`` appends /fail). Returns True on a 2xx answer; never raises."""
    url = url or os.environ.get("HC_PING_URL", "").strip()
    if not url:
        return False
    if status == "fail":
        url = url.rstrip("/") + "/fail"
    if get is None:
        import requests
        get = requests.get
    for i in range(3):
        try:
            r = get(url, timeout=10)
            if 200 <= getattr(r, "status_code", 0) < 300:
                return True
        except Exception as e:                                           # noqa: BLE001
            log.warning("heartbeat ping failed (attempt %d): %s", i + 1, type(e).__name__)
        sleep(2 ** i)
    return False


def should_ping(last_ping_ts: float | None, now_ts: float, interval_min: float) -> bool:
    return last_ping_ts is None or (now_ts - last_ping_ts) >= interval_min * 60
