"""Workflow gate: print and export ``run=true|false`` for a scheduled trigger. Standard library only.

    python3 scripts/gate.py --kind morning|late|close [--force] [--now 2026-10-01T22:30:00+00:00]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.gate import decide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True, choices=["morning", "late", "close"])
ap.add_argument("--force", action="store_true", help="manual dispatch: always run")
ap.add_argument("--now", help="ISO time (tests)")
ap.add_argument("--db", default="data/nhl.db")
a = ap.parse_args()
now = datetime.fromisoformat(a.now) if a.now else datetime.now(timezone.utc)
run, reason = decide(a.kind, now, a.db, force=a.force)
print(f"gate[{a.kind}] run={str(run).lower()}: {reason}")
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"run={str(run).lower()}\nreason={reason}\n")
