"""Rolling live metrics, written to state/rolling_metrics.json every run: closing-line value of the picks, calibration of the model, and its log-loss gain over the market.
These are the decay trackers; the thresholds that act on them live in config/policy.yaml (``autonomy.kill_switch.clv``, ``autonomy.drift``, ``autonomy.decay``)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def compute(resolved_df: pd.DataFrame, now: datetime | None = None, clv_window: int = 100, cal_window: int = 200, gain_window: int = 150) -> dict:
    now = now or datetime.now(timezone.utc)
    out = {"computed_at": now.isoformat(timespec="seconds"), "games_resolved": int(len(resolved_df))}
    d = resolved_df
    if d is None or d.empty:
        return out
    d = d.sort_values("game_date") if "game_date" in d else d
    if "clv_move_pts" in d and "is_bet" in d:
        c = d[d.is_bet & d.clv_move_pts.notna()].clv_move_pts.to_numpy(float)[-clv_window:]
        if len(c) >= 5:
            out["rolling_clv_pts"] = {"n": int(len(c)), "mean": float(c.mean()), "se": float(c.std(ddof=1) / np.sqrt(len(c)))}
    if {"p_model", "home_win"} <= set(d.columns):
        from nhlbet.models.calibration import calibration_slope_intercept
        x = d[d.p_model.notna() & d.home_win.notna()].tail(cal_window)
        if len(x) >= 50 and x.home_win.nunique() == 2:
            s, i = calibration_slope_intercept(x.p_model.to_numpy(float), x.home_win.to_numpy(float))
            out["rolling_calibration"] = {"n": int(len(x)), "slope": float(s), "intercept": float(i)}
    if {"p_model", "p_market", "home_win"} <= set(d.columns):
        x = d[d.p_model.notna() & d.p_market.notna() & d.home_win.notna()].tail(gain_window)
        if len(x) >= 30:
            y = x.home_win.to_numpy(float)
            ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
            g = ll(x.p_market.to_numpy(float)) - ll(x.p_model.to_numpy(float))
            out["rolling_gain_over_market"] = {"n": int(len(x)), "mean": float(g.mean()), "se": float(g.std(ddof=1) / np.sqrt(len(g)))}
    return out


def save(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=1, sort_keys=True))
