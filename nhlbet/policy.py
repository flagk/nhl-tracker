"""Read-only access to config/policy.yaml. Nothing in this package writes to the policy file (tests/test_policy_readonly.py enforces it)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import yaml

POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "policy.yaml"


def policy_hash(path: Path = POLICY_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    with open(path, "r") as f:                      # read-only
        return yaml.safe_load(f) or {}


def load_policy(path: Path | str = POLICY_PATH) -> dict:
    p = Path(path)
    return _load(str(p), p.stat().st_mtime)


def policy_get(*keys, default=None, path: Path | str = POLICY_PATH):
    cur = load_policy(path)
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
