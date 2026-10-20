"""Kill switch: drawdown, live closing-line value and integrity failures open HARD halts. Limits come from config/policy.yaml."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.ops.checks import CheckResult, failed, passed
from nhlbet.policy import policy_get


def drawdown_check(equity: pd.Series | np.ndarray | list, limit: float | None = None) -> CheckResult:
    """``equity``: the official bankroll after each settled bet, in order. Fails when the current drawdown from the high-water mark reaches the limit."""
    limit = float(limit if limit is not None else policy_get("risk", "kill_switch", "max_drawdown_pct", default=0.20))
    e = np.asarray(equity, float)
    if len(e) == 0:
        return passed("drawdown", "hard", "no settled official bets yet")
    peak = np.maximum.accumulate(e)
    dd = float(((peak - e) / peak)[-1])
    if dd >= limit:
        return failed("drawdown", "hard", f"drawdown {dd:.1%} from the high-water mark reached the {limit:.0%} limit",
                      "Stop betting. Read reports/evaluation.md and reports/risk.md, decide whether to resume, then add this halt's id to autonomy.acknowledged_halts in config/policy.yaml.")
    return passed("drawdown", "hard", f"drawdown {dd:.1%} (limit {limit:.0%})")


def clv_check(clv_pts: pd.Series | np.ndarray | list, dates=None, seed: int = 0) -> CheckResult:
    """Fails when, with at least ``min_picks`` live moneyline picks, the mean margin-free CLV is below the policy limit and its 95% interval (resampling dates) lies entirely below zero."""
    mn = int(policy_get("autonomy", "kill_switch", "clv", "min_picks", default=100))
    lim = float(policy_get("autonomy", "kill_switch", "clv", "halt_below_mean_pts", default=-1.0))
    x = np.asarray(clv_pts, float)
    ok = ~np.isnan(x)
    x = x[ok]
    d = np.asarray(dates)[ok] if dates is not None else np.arange(len(x))
    if len(x) < mn:
        return passed("live_clv", "hard", f"{len(x)} live picks with a closing-line value (the limit applies from {mn})")
    from nhlbet.report.evaluation import cluster_dist
    dist = cluster_dist(x, d, 2000, seed)
    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
    mean = float(x.mean())
    if mean < lim and hi < 0:
        return failed("live_clv", "hard", f"mean live CLV {mean:+.2f} points (95% interval {lo:+.2f} to {hi:+.2f}) over {len(x)} picks is below the {lim:+.1f} limit",
                      "Stop betting: the live line moves against the picks. Read reports/evaluation.md, then add this halt's id to autonomy.acknowledged_halts in config/policy.yaml if you decide to resume.")
    return passed("live_clv", "hard", f"mean live CLV {mean:+.2f} points (95% interval {lo:+.2f} to {hi:+.2f}) over {len(x)} picks")


def integrity_check(problems: list[str]) -> CheckResult:
    """The hash-chained log ledger and the log files must verify."""
    if problems:
        return failed("log_integrity", "hard", "log integrity failure: " + "; ".join(problems[:5]),
                      "Do not edit the logs. Restore them from the log-backup branch (git checkout log-backup -- data/logs), find out what changed them, then add this halt's id to autonomy.acknowledged_halts.")
    return passed("log_integrity", "hard", "ledger verified")
