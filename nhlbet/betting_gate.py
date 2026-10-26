"""Betting eligibility: RESEARCH ONLY unless every pre-registered criterion in ``betting_gate`` (config/policy.yaml) holds.

Recomputed from the live log on every run, so a failing criterion reverts the status at once. Nothing here places a bet or touches a sportsbook account;
ELIGIBLE means only "the live evidence clears the pre-registered bar", never a promise of profit.
"""
from __future__ import annotations

import numpy as np

from nhlbet.policy import policy_get

RESEARCH, ELIGIBLE = "RESEARCH ONLY", "ELIGIBLE"


def _g(k, default=None):
    return policy_get("betting_gate", k, default=default)


def evaluate_gate(clv_pts, dates, equity, system_status: str = "OK", seed: int = 0) -> dict:
    """``clv_pts``: margin-free CLV in probability points of each live settled official moneyline pick with a closing price; ``dates``: their game dates;
    ``equity``: official bankroll after each settled bet (including the start). Returns {status, criteria, why}."""
    from nhlbet.report.evaluation import cluster_dist
    x, d = np.asarray(clv_pts, float), np.asarray(dates)
    ok = ~np.isnan(x)
    x, d = x[ok], d[ok]
    need = int(_g("min_live_clv_picks", 300))
    crit = [{"name": "min_live_clv_picks", "ok": len(x) >= need, "detail": f"{len(x)} live picks with a closing price (need {need})"}]
    if len(x) >= 2:
        dist = cluster_dist(x, d, 4000, seed)
        lvl = float(_g("clv_interval_level", 0.95))
        lo, hi = float(np.percentile(dist, 100 * (1 - lvl) / 2)), float(np.percentile(dist, 100 * (1 + lvl) / 2))
        mean = float(x.mean())
    else:
        lo = hi = mean = float("nan")
    mn = float(_g("min_mean_clv_pts", 0.5))
    crit.append({"name": "clv_interval_above_zero", "ok": bool(lo > 0), "detail": f"mean CLV {mean:+.2f} points, interval {lo:+.2f} to {hi:+.2f}" if len(x) >= 2 else "no CLV data"})
    crit.append({"name": "min_mean_clv_pts", "ok": bool(mean >= mn), "detail": f"mean CLV {mean:+.2f} points (need at least {mn:+.2f})" if len(x) >= 2 else "no CLV data"})
    e = np.asarray(equity, float)
    dd = float(((np.maximum.accumulate(e) - e) / np.maximum.accumulate(e))[-1]) if len(e) else 0.0
    lim = float(_g("max_drawdown_pct", 0.10))
    crit.append({"name": "max_drawdown_pct", "ok": dd <= lim, "detail": f"current drawdown {dd:.1%} (limit {lim:.0%})"})
    if _g("require_system_ok", True):
        crit.append({"name": "system_ok", "ok": system_status == "OK", "detail": f"system status {system_status}"})
    bad = [c for c in crit if not c["ok"]]
    status = ELIGIBLE if not bad else RESEARCH
    return {"status": status, "markets": list(_g("markets", ["h2h"])), "criteria": crit, "why": "all criteria met" if not bad else "; ".join(c["detail"] for c in bad)}
