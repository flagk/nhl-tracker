"""Model registry: an append-only JSON list of every trained version with its metrics and lineage."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path("data/models/registry.json")


def data_fingerprint(game_ids, last_date) -> str:
    h = hashlib.sha1(",".join(map(str, sorted(game_ids))).encode()).hexdigest()[:10]
    return f"{h}@{pd.Timestamp(last_date).date()}"


class ModelRegistry:
    def __init__(self, path: str | Path = REGISTRY_PATH) -> None:
        self.path = Path(path)

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def history(self) -> list[dict]:
        return self._load()

    def latest(self) -> dict | None:
        h = self._load()
        return h[-1] if h else None

    def register(self, entry: dict) -> dict:
        entry = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
        h = self._load()
        if any(e["version"] == entry["version"] for e in h):
            raise ValueError(f"version {entry['version']} already registered")
        h.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(h, indent=1, default=str))
        return entry

    def prune_artifacts(self, keep: int = 3) -> list[str]:
        """Delete artifact files of all but the newest ``keep`` versions (registry rows stay for lineage)."""
        removed = []
        for e in self._load()[:-keep]:
            p = Path(e.get("artifact", ""))
            if p.is_file():
                p.unlink(); removed.append(str(p))
        return removed
