"""The gate ``run_daily`` calls: run every check, update the halt state, alert, and say whether picks may be published.

Prime directive (config/policy.yaml): when anything looks wrong the system stops publishing and alerts the owner. Nothing here changes a threshold, a rule or a model setting.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nhlbet.ops import chain, health, killswitch, notify, policyguard
from nhlbet.ops import state as S
from nhlbet.ops.checks import CheckResult, failed, passed
from nhlbet.policy import policy_get, policy_hash

log = logging.getLogger(__name__)


def _safe(name: str, level: str, fn) -> CheckResult:
    """A check that itself crashes is a failed check: silence must never read as health."""
    try:
        return fn()
    except Exception as e:                                                 # noqa: BLE001
        log.exception("check %s crashed", name)
        return failed(name, level, f"the check itself crashed: {type(e).__name__}: {e}", "Read the daily workflow log; fix the check or the data it reads.")


def pre_checks(store, date: str, refresh_error: str | None, odds_meta: dict | None, remaining_credits: int | None, scheduled_games: int, perf: dict | None, repo_root: Path,
               state: dict) -> tuple[list[CheckResult], str]:
    """All checks that do not need the built payload. Returns (results, policy hash to record)."""
    res = [_safe("schema", "soft", lambda: health.schema_check(store)),
           _safe("freshness", "soft", lambda: health.freshness_check(store, date, refresh_error)),
           _safe("row_counts", "soft", lambda: health.row_count_check(store)),
           _safe("input_validation", "soft", lambda: health.input_validation_check(store)),
           _safe("cross_source", "soft", lambda: health.cross_source_check(store, date)),
           _safe("odds_source", "soft", lambda: health.odds_source_check(odds_meta, scheduled_games)),
           _safe("odds_quota", "soft", lambda: health.quota_check(remaining_credits))]
    cur = policy_hash()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN")
    fetch = lambda: policyguard.last_policy_commit_github(repo, token) if repo else None
    pol, rec = policyguard.check_policy(state.get("policy_sha256"), cur, fetch)
    res.append(pol)
    perf = perf or {}
    curve = perf.get("curve")
    if curve is not None and len(curve):
        start = float(curve.bankroll.iloc[0] - curve.profit.iloc[0])
        res.append(_safe("drawdown", "hard", lambda: killswitch.drawdown_check(np.r_[start, curve.bankroll.to_numpy()])))
    else:
        res.append(passed("drawdown", "hard", "no settled official bets yet"))
    res.append(_safe("log_integrity", "hard", lambda: killswitch.integrity_check(chain.verify(repo_root))))
    return res, rec


def apply(state: dict, results: list[CheckResult], policy_sha: str | None = None, now: datetime | None = None, post=None) -> tuple[dict, list[dict]]:
    """Update the state machine with this run's results and send each new alert once."""
    now = now or datetime.now(timezone.utc)
    ack = policy_get("autonomy", "acknowledged_halts", default=[]) or []
    new, events = S.update(state, results, now, int(policy_get("autonomy", "soft_resume_healthy_runs", default=3)), ack, policy_sha)
    new = notify.dispatch(events, new, post=post)
    return new, events


def system_block(state: dict) -> dict:
    """What the page and the report show: status, open halts with the reason and next step."""
    return {"status": state.get("status", "OK"), "halts": [{k: h[k] for k in ("id", "check", "kind", "reason", "next_step", "opened_at")} for h in S.open_halts(state)],
            "last_run": state.get("last_run"), "policy_sha256": state.get("policy_sha256")}
