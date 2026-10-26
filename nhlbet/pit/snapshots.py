"""Immutable, versioned raw-data snapshots with knowledge time.

Every fetch of a raw payload (play-by-play, boxscore, shift chart, odds, goalie confirmation) is stored once, content-addressed by sha256, and recorded in an append-only
manifest with the moment it was fetched (``knowledge_time``). The same URL fetched later is a NEW version, never an overwrite, so "what did we know at time t" can be answered
and revisions by the data provider become measurable. Objects live under ``root/objects`` (large, kept out of git and pushed with DVC to a private remote); the manifest is small
and is committed, so a changed or missing object is detectable from hashes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = "manifest.jsonl"


@dataclass(frozen=True)
class Version:
    kind: str                # 'pbp', 'boxscore', 'shifts', 'odds', 'goalie'
    key: str                 # e.g. the game id
    sha256: str
    knowledge_time: str      # ISO-8601 UTC: when this payload was fetched
    source: str              # URL or logical source
    status: int = 200
    nbytes: int = 0


def _canon(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _utc(t) -> str:
    if isinstance(t, datetime):
        return t.astimezone(timezone.utc).isoformat(timespec="seconds")
    return datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds")


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)

    # ---- writing -------------------------------------------------------------------------
    def put(self, kind: str, key: str, payload, fetched_at, source: str, status: int = 200) -> Version:
        """Store ``payload`` (JSON-serialisable) as a new version. Identical content fetched again is still recorded as a new observation (it proves the value was unchanged at that time)."""
        raw = _canon(payload)
        sha = hashlib.sha256(raw).hexdigest()
        obj = self.root / "objects" / sha[:2] / f"{sha}.json.gz"
        if obj.exists():
            if hashlib.sha256(gzip.decompress(obj.read_bytes())).hexdigest() != sha:
                raise RuntimeError(f"object {sha} is corrupt on disk; refusing to continue")
        else:
            obj.parent.mkdir(parents=True, exist_ok=True)
            tmp = obj.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(raw, mtime=0))            # mtime=0: the bytes are reproducible
            tmp.replace(obj)
        v = Version(kind, str(key), sha, _utc(fetched_at), source, int(status), len(raw))
        with (self.root / MANIFEST).open("a") as f:
            f.write(json.dumps(asdict(v), sort_keys=True) + "\n")
        return v

    # ---- reading -------------------------------------------------------------------------
    def manifest(self) -> list[Version]:
        p = self.root / MANIFEST
        if not p.exists():
            return []
        return [Version(**json.loads(ln)) for ln in p.read_text().splitlines() if ln.strip()]

    def versions(self, kind: str, key: str) -> list[Version]:
        return sorted((v for v in self.manifest() if v.kind == kind and v.key == str(key) and v.status == 200), key=lambda v: v.knowledge_time)

    def load(self, v: Version):
        raw = gzip.decompress((self.root / "objects" / v.sha256[:2] / f"{v.sha256}.json.gz").read_bytes())
        if hashlib.sha256(raw).hexdigest() != v.sha256:
            raise RuntimeError(f"object {v.sha256} does not match its hash")
        return json.loads(raw)

    def as_of(self, kind: str, key: str, t) -> Version | None:
        """The newest version whose knowledge time is at or before ``t``; None if nothing was known yet."""
        t = _utc(t)
        ok = [v for v in self.versions(kind, key) if v.knowledge_time <= t]
        return ok[-1] if ok else None

    def verify(self) -> list[str]:
        """Problems: missing or modified objects."""
        problems = []
        for i, v in enumerate(self.manifest()):
            p = self.root / "objects" / v.sha256[:2] / f"{v.sha256}.json.gz"
            if not p.exists():
                problems.append(f"line {i}: object {v.sha256[:10]} missing ({v.kind} {v.key})")
                continue
            if hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() != v.sha256:
                problems.append(f"line {i}: object {v.sha256[:10]} modified ({v.kind} {v.key})")
        return problems
