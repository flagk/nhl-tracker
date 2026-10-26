"""Record the pre-registration: hash of audit/preregistration.md plus the hashes of the frozen code paths and the policy. Run once, in the same commit that adds the file."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ["nhlbet/features", "nhlbet/models", "nhlbet/odds", "nhlbet/risk", "nhlbet/train.py"]
POLICY_SECTIONS = ("staking", "evaluation", "settlement", "risk")        # the sections that change what is bet; operations and alert settings may evolve without voiding the test


def tree_hash(root: Path, paths=FROZEN) -> dict:
    out = {}
    for p in paths:
        base = root / p
        files = [base] if base.is_file() else sorted(f for f in base.rglob("*") if f.is_file() and f.suffix in (".py", ".yaml", ".json") and "__pycache__" not in f.parts)
        h = hashlib.sha256()
        for f in files:
            h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
        out[p] = h.hexdigest()
    import yaml
    pol = yaml.safe_load((root / "config/policy.yaml").read_text())
    out["config/policy.yaml#" + ",".join(POLICY_SECTIONS)] = hashlib.sha256(json.dumps({k: pol.get(k) for k in POLICY_SECTIONS}, sort_keys=True, default=str).encode()).hexdigest()
    return out


def main() -> int:
    lock = ROOT / "audit/prereg_lock.json"
    if lock.exists():
        sys.exit("already locked: a changed protocol needs a new pre-registration, not a rewrite")
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lock.write_text(json.dumps({"locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "preregistration_sha256": sha("audit/preregistration.md"), "parent_commit": head,
                                "frozen_paths_sha256": tree_hash(ROOT), "trigger": {"qualifying_games": 1312, "or_date": "2027-06-30"}}, indent=1))
    print(lock.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
