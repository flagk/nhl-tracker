"""Send one push alert (ntfy) from a workflow step, e.g. when a job crashed. Never fails the step: an undeliverable alert is logged and exits 0.

    python scripts/notify.py --title "Daily job failed" --message "see the run" --priority high --key daily_failed:$GITHUB_RUN_ID
Needs NTFY_TOPIC in the environment (a secret: the topic name is the only protection on the public server).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.ops.notify import send  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--priority", default="high")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    print("delivered" if send(a.title, a.message, a.priority, ("rotating_light",), a.key) else "NOT delivered")


if __name__ == "__main__":
    main()
