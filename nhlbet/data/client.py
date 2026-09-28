"""Thin, cached, retrying client for the free NHL web API (api-web.nhle.com)."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)
BASE = "https://api-web.nhle.com"


class NHLAPIError(RuntimeError):
    pass


class NHLClient:
    """GET JSON with an on-disk cache.

    ``ttl=None`` caches forever (use for completed games, which never change);
    a numeric ``ttl`` (seconds) is used for schedules and other mutable endpoints.
    """

    def __init__(self, cache_dir: str | Path = "data/cache/raw", min_interval: float = 0.25,
                 retries: int = 4, timeout: float = 20.0, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.retries, self.timeout = min_interval, retries, timeout
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last = 0.0
        self.network_calls = 0

    def _path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json.gz"

    def _read(self, p: Path, ttl: float | None) -> Any | None:
        if not p.exists():
            return None
        if ttl is not None and time.time() - p.stat().st_mtime > ttl:
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            log.warning("corrupt cache file %s ignored", p)
            return None

    def get(self, path: str, ttl: float | None = None) -> Any:
        url = path if path.startswith("http") else f"{BASE}{path}"
        p = self._path(url)
        cached = self._read(p, ttl)
        if cached is not None:
            return cached
        delay, last_err = 1.0, None
        for attempt in range(1, self.retries + 1):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.time()
            self.network_calls += 1
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:  # network error: retry
                last_err = e
            else:
                if r.status_code == 200:
                    data = r.json()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(".tmp")
                    with gzip.open(tmp, "wt", encoding="utf-8") as f:
                        json.dump(data, f)
                    tmp.replace(p)
                    return data
                if r.status_code == 404:
                    raise NHLAPIError(f"404 not found: {url}")
                last_err = NHLAPIError(f"HTTP {r.status_code} for {url}")
                if r.status_code == 429:
                    delay = max(delay, float(r.headers.get("Retry-After", 5)))
            log.warning("attempt %d/%d failed for %s: %s", attempt, self.retries, url, last_err)
            self._sleep(delay)
            delay *= 2
        # graceful fallback: a stale cache entry beats no data
        stale = self._read(p, None)
        if stale is not None:
            log.error("using STALE cache for %s after failures: %s", url, last_err)
            return stale
        raise NHLAPIError(f"giving up on {url}: {last_err}")

    # -- endpoints -----------------------------------------------------------------------------
    def club_schedule_season(self, team: str, season: int, ttl: float | None = 6 * 3600) -> Any:
        return self.get(f"/v1/club-schedule-season/{team}/{season}", ttl=ttl)

    def schedule(self, date: str, ttl: float | None = 900) -> Any:
        return self.get(f"/v1/schedule/{date}", ttl=ttl)

    def boxscore(self, game_id: int, ttl: float | None = None) -> Any:
        return self.get(f"/v1/gamecenter/{game_id}/boxscore", ttl=ttl)

    def play_by_play(self, game_id: int, ttl: float | None = None) -> Any:
        return self.get(f"/v1/gamecenter/{game_id}/play-by-play", ttl=ttl)

    def landing(self, game_id: int, ttl: float | None = 600) -> Any:
        return self.get(f"/v1/gamecenter/{game_id}/landing", ttl=ttl)

    def roster(self, team: str, ttl: float | None = 3600) -> Any:
        return self.get(f"/v1/roster/{team}/current", ttl=ttl)
