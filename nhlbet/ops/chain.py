"""Hash-chained, append-only pick-log ledger (``data/logs/chain.jsonl``).

Every log file under ``data/logs/<table>/`` (per-run and per-capture CSVs, the review JSONL logs) gets ledger entries ``{seq, path, bytes, sha256, prev, hash}`` where ``sha256`` covers
the first ``bytes`` bytes of the file and ``hash = sha256(prev | seq | path | bytes | sha256)``, so each entry commits to the one before it. A file that only grows (a JSONL log) gets a new
entry each time it grows. Verification recomputes the chain and re-hashes the covered prefix of every file: edited, truncated or deleted history, or a broken link, is an integrity failure.
Derived aggregate files at the top of ``data/logs`` (``bet_log.csv`` and similar) are rewritten every run and are deliberately outside the chain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64
LEDGER = "chain.jsonl"
SUFFIXES = (".csv", ".jsonl")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _link(prev: str, seq: int, path: str, nbytes: int, sha: str) -> str:
    return _sha(f"{prev}|{seq}|{path}|{nbytes}|{sha}".encode())


def tracked_files(root: Path) -> list[Path]:
    """Files under data/logs/<subdir>/... (depth >= 2), sorted: the ledger covers these only."""
    base = Path(root) / "data" / "logs"
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in SUFFIXES and p.parent != base and p.name != LEDGER)


def read_ledger(root: Path) -> list[dict]:
    p = Path(root) / "data" / "logs" / LEDGER
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


def verify(root: Path) -> list[str]:
    """Every reason the ledger or the logs it covers cannot be trusted (empty = intact)."""
    root = Path(root)
    problems, prev = [], GENESIS
    for i, e in enumerate(read_ledger(root)):
        if e["seq"] != i or e["prev"] != prev or e["hash"] != _link(e["prev"], e["seq"], e["path"], e["bytes"], e["sha256"]):
            problems.append(f"ledger link {i} is broken")
            break
        prev = e["hash"]
        f = root / e["path"]
        if not f.exists():
            problems.append(f"{e['path']} was deleted")
            continue
        data = f.read_bytes()
        if len(data) < e["bytes"]:
            problems.append(f"{e['path']} was truncated")
        elif _sha(data[:e["bytes"]]) != e["sha256"]:
            problems.append(f"{e['path']} was modified")
    return sorted(set(problems))


def append_new(root: Path) -> int:
    """Add entries for new files and for files that have grown since their last entry. Refuses to touch a ledger that does not verify. Returns the entries added."""
    root = Path(root)
    if verify(root):
        raise RuntimeError("the ledger does not verify; not appending: " + "; ".join(verify(root)))
    led = read_ledger(root)
    last = {}
    for e in led:
        last[e["path"]] = e
    prev = led[-1]["hash"] if led else GENESIS
    new = []
    for f in tracked_files(root):
        rel = str(f.relative_to(root))
        data = f.read_bytes()
        e = last.get(rel)
        if e is not None and len(data) <= e["bytes"]:
            continue
        seq = len(led) + len(new)
        sha = _sha(data)
        h = _link(prev, seq, rel, len(data), sha)
        new.append({"seq": seq, "path": rel, "bytes": len(data), "sha256": sha, "prev": prev, "hash": h})
        prev = h
    if new:
        p = root / "data" / "logs" / LEDGER
        with p.open("a") as fh:
            for e in new:
                fh.write(json.dumps(e, sort_keys=True) + "\n")
    return len(new)
