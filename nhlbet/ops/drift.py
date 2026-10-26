"""Drift limits from the policy: feature shift (PSI), rolling calibration, and model-versus-market disagreement. Warnings alert; halt limits open SOFT halts (they clear after N healthy runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.monitor import psi
from nhlbet.ops.checks import CheckResult, failed, passed
from nhlbet.policy import policy_get


def _p(*k, default=None):
    return policy_get("autonomy", "drift", *k, default=default)


def feature_drift_check(ref: pd.DataFrame, new: pd.DataFrame, cols: list[str]) -> tuple[CheckResult, list[str]]:
    """(halt check, warnings). Warn above ``psi_warn``; halt above ``psi_halt`` for any feature."""
    warn_at, halt_at = float(_p("psi_warn", default=0.2)), float(_p("psi_halt", default=0.5))
    scores = {c: psi(ref[c], new[c]) for c in cols if c in ref and c in new}
    scores = {c: v for c, v in scores.items() if v == v}
    halted = {c: round(v, 2) for c, v in scores.items() if v > halt_at}
    warns = [f"feature {c} drifted (PSI {v:.2f} > {warn_at})" for c, v in sorted(scores.items()) if warn_at < v <= halt_at]
    if halted:
        return failed("feature_drift", "soft", f"feature drift above the halt limit {halt_at}: {halted}", "The inputs changed shape: check for a data source change; the halt clears after healthy runs, retrain if the shift is real."), warns
    return passed("feature_drift", "soft", f"{len(scores)} features compared"), warns


def calibration_check(p, y) -> tuple[CheckResult, list[str]]:
    """Rolling calibration slope of the last ``performance_window`` predictions; outside the warn range warns, outside the halt range halts."""
    from nhlbet.models.calibration import calibration_slope_intercept
    w, mn = int(_p("performance_window", default=200)), int(_p("min_games", default=100))
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < mn:
        return passed("calibration", "soft", f"{len(p)} resolved predictions (needs {mn})"), []
    slope = float(calibration_slope_intercept(p[-w:], y[-w:])[0])
    (wl, wh), (hl, hh) = _p("calibration_slope_warn", default=[0.6, 1.4]), _p("calibration_slope_halt", default=[0.3, 2.0])
    if slope < hl or slope > hh:
        return failed("calibration", "soft", f"rolling calibration slope {slope:.2f} is outside the halt range [{hl}, {hh}]", "The probabilities are badly calibrated: check the latest retrain and the data feeds; the halt clears after healthy runs."), []
    return passed("calibration", "soft", f"rolling calibration slope {slope:.2f}"), ([f"rolling calibration slope {slope:.2f} outside [{wl}, {wh}]"] if slope < wl or slope > wh else [])


def disagreement_check(p_model, p_market) -> tuple[CheckResult, list[str]]:
    """Mean |model - market| in probability points; a model that disagrees with the market this much is presumed broken, not brilliant."""
    mn = int(_p("min_games", default=100))
    a, b = np.asarray(p_model, float), np.asarray(p_market, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < mn:
        return passed("model_market_disagreement", "soft", f"{len(a)} games (needs {mn})"), []
    d = float(np.mean(np.abs(a - b)) * 100)
    warn, halt = float(_p("disagreement_warn_pts", default=8.0)), float(_p("disagreement_halt_pts", default=12.0))
    if d > halt:
        return failed("model_market_disagreement", "soft", f"mean |model - market| = {d:.1f} points over {len(a)} games exceeds the {halt:.0f}-point halt limit", "Likely a model or odds-parsing error: inspect the latest picks against the market; the halt clears after healthy runs."), []
    return passed("model_market_disagreement", "soft", f"mean |model - market| = {d:.1f} points"), ([f"mean |model - market| = {d:.1f} points > {warn:.0f}"] if d > warn else [])


def market_decay_check(p_model, p_market, y) -> tuple[CheckResult, list[str]]:
    """Rolling log-loss gain of the model over the decision-time market versus the baseline measured in the audit. Fails (soft) only when the gain has WORSENED by more than the margin."""
    cfg = lambda k, d: policy_get("autonomy", "decay", k, default=d)
    win, mn = int(cfg("market_gain_window_games", 150)), int(cfg("min_games", 100))
    base, margin = float(cfg("market_gain_baseline", -0.012)), float(cfg("market_gain_margin", 0.03))
    a, b, yy = np.asarray(p_model, float), np.asarray(p_market, float), np.asarray(y, float)
    ok = ~(np.isnan(a) | np.isnan(b) | np.isnan(yy))
    a, b, yy = a[ok][-win:], b[ok][-win:], yy[ok][-win:]
    if len(a) < mn:
        return passed("market_decay", "soft", f"{len(a)} games with a market price (needs {mn})"), []
    ll = lambda p: -(yy * np.log(np.clip(p, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-6, 1)))
    gain = float(np.mean(ll(b) - ll(a)))
    if gain < base - margin:
        return failed("market_decay", "soft", f"rolling {len(a)}-game log-loss gain over the market is {gain:+.4f}, below the audit baseline {base:+.3f} by more than {margin}",
                      "The model has got worse relative to the market since the audit: inspect recent picks, data feeds and the last retrain; the halt clears after healthy runs."), []
    warn = [f"rolling model-vs-market gain {gain:+.4f} (baseline {base:+.3f})"] if gain < base - margin / 2 else []
    return passed("market_decay", "soft", f"rolling {len(a)}-game gain over the market {gain:+.4f} (baseline {base:+.3f})"), warn
