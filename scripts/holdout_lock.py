"""Write reports/holdout_lock.json from the committed walk-forward predictions (refuses to overwrite an existing lock).

    python scripts/holdout_lock.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.validity.holdout import LOCK_PATH, compute_lock  # noqa: E402

if LOCK_PATH.exists():
    sys.exit(f"{LOCK_PATH} already exists: a lock is written once. Delete it deliberately (and say why in the commit) to re-lock.")
lock = compute_lock()
LOCK_PATH.write_text(json.dumps(lock, indent=1))
print(json.dumps(lock, indent=1))
