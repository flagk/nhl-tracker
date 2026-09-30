"""The Odds API (https://the-odds-api.com) client: quota-aware, cached, retrying, with stale fallback.

Setup: get a free key at https://the-odds-api.com, then ``export ODDS_API_KEY=...`` locally and add
``ODDS_API_KEY`` as a GitHub Actions secret. The key is read from the environment ONLY; it is never written
to disk (cache keys/log lines/exceptions all exclude it).

Quota: 1 credit per (region x market) per call. The free tier (500 credits/month) supports roughly 16 calls
of one market in one region per day - so the default is a single ``h2h`` (moneyline) call per fetch.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import requests

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"


class OddsAPIError(RuntimeError):
    pass


class OddsConfigError(OddsAPIError):
    pass


class QuotaExhausted(OddsAPIError):
    pass


@dataclass
class OddsFetch:
    events: list[dict]
    captured_at: str            # UTC ISO time the data was retrieved from the API
    remaining: int | None
    used: int | None
    source: str                 # 'live' | 'cache' | 'stale_cache'

    @property
    def stale(self) -> bool:
        return self.source == "stale_cache"


class OddsClient:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/cache/odds", ttl: float = 600.0,
                 max_stale: float = 6 * 3600.0, reserve: int = 25, retries: int = 3, timeout: float = 20.0,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.time) -> None:
        self._key = api_key if api_key is not None else os.environ.get("ODDS_API_KEY", "")
        if not self._key:
            raise OddsConfigError("ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com and run "
                                  "`export ODDS_API_KEY=<key>` (locally) or add it as a GitHub Actions secret.")
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl, self.max_stale, self.reserve, self.retries, self.timeout = ttl, max_stale, reserve, retries, timeout
        self.session, self._sleep, self._now = session or requests.Session(), sleep, now
        self._quota_path = self.cache_dir / "quota.json"

    # -- helpers ---------------------------------------------------------------------------------
    def _scrub(self, text: str) -> str:
        return str(text).replace(self._key, "***")

    def _cache_path(self, params: dict) -> Path:
        h = hashlib.sha1(json.dumps(sorted(params.items())).encode()).hexdigest()[:16]   # key excluded on purpose
        return self.cache_dir / f"odds_{h}.json.gz"

    def _read(self, p: Path):
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, p: Path, obj: dict) -> None:
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(obj, f)
        tmp.replace(p)

    def remaining_credits(self) -> int | None:
        try:
            return json.loads(self._quota_path.read_text()).get("remaining") if self._quota_path.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def _save_quota(self, remaining, used) -> None:
        self._quota_path.write_text(json.dumps({"remaining": remaining, "used": used, "at": self._now()}))

    # -- main call -------------------------------------------------------------------------------
    def fetch_odds(self, markets: Sequence[str] = ("h2h",), regions: str = "us", force: bool = False) -> OddsFetch:
        params = {"regions": regions, "markets": ",".join(markets), "oddsFormat": "decimal", "dateFormat": "iso"}
        path = self._cache_path(params)
        cached = self._read(path)
        age = (self._now() - cached["fetched_epoch"]) if cached else None
        if cached and not force and age is not None and age <= self.ttl:
            return OddsFetch(cached["events"], cached["captured_at"], cached.get("remaining"), cached.get("used"), "cache")

        cost = len(markets) * len(regions.split(","))
        remaining = self.remaining_credits()
        if remaining is not None and remaining < cost + self.reserve:
            log.error("odds quota low (%s credits left, reserve %s): not calling the API", remaining, self.reserve)
            return self._fallback(cached, age, QuotaExhausted(f"only {remaining} credits left (reserve {self.reserve})"))

        url = f"{BASE}/sports/{SPORT}/odds"
        delay, last_err = 1.0, None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params={**params, "apiKey": self._key}, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = OddsAPIError(f"network error: {self._scrub(e)}")
            else:
                if r.status_code == 200:
                    rem = _int(r.headers.get("x-requests-remaining")); used = _int(r.headers.get("x-requests-used"))
                    self._save_quota(rem, used)
                    if rem is not None and rem < 100:
                        log.warning("odds API credits running low: %s remaining", rem)
                    from datetime import datetime, timezone
                    captured = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    data = r.json()
                    self._write(path, {"events": data, "captured_at": captured, "fetched_epoch": self._now(),
                                       "remaining": rem, "used": used})
                    return OddsFetch(data, captured, rem, used, "live")
                if r.status_code in (401, 403):
                    raise OddsAPIError("Odds API rejected the key (HTTP %s). Check ODDS_API_KEY." % r.status_code)
                if r.status_code == 422:
                    raise OddsAPIError(f"Odds API rejected the request parameters (HTTP 422): {self._scrub(r.text)[:200]}")
                last_err = OddsAPIError(f"HTTP {r.status_code}")
                if r.status_code == 429:
                    delay = max(delay, float(r.headers.get("Retry-After", 5)))
                    if r.headers.get("x-requests-remaining") == "0":
                        self._save_quota(0, _int(r.headers.get("x-requests-used")))
                        last_err = QuotaExhausted("Odds API quota exhausted")
                        break
            log.warning("odds attempt %d/%d failed: %s", attempt, self.retries, last_err)
            self._sleep(delay); delay *= 2
        return self._fallback(cached, age, last_err)

    def _fallback(self, cached, age, err: Exception) -> OddsFetch:
        if cached and age is not None and age <= self.max_stale:
            log.error("using STALE odds (%.0f min old) because the API call failed: %s", age / 60, self._scrub(err))
            return OddsFetch(cached["events"], cached["captured_at"], cached.get("remaining"), cached.get("used"), "stale_cache")
        raise err if isinstance(err, OddsAPIError) else OddsAPIError(self._scrub(err))


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
