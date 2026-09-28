"""Market features, used *only* in a way that is valid for a pre-game prediction.

Allowed: the opening line, and the latest line observed **no later than the decision time**
(``decision_time = start_utc - lead``). Forbidden: the closing line. Snapshots captured after the
decision time are ignored, so the closing line can never leak into a training feature. Closing
lines are kept in the odds store solely to measure closing-line value (CLV) afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {"game_id", "captured_at", "home_prob_novig"}


def attach_market_features(feats: pd.DataFrame, snapshots: pd.DataFrame, lead_minutes: int = 90) -> pd.DataFrame:
    """Add ``mkt_open_p`` (first no-vig home prob seen) and ``mkt_move`` (latest-before-decision minus open).

    ``snapshots``: one row per (game_id, captured_at) with ``home_prob_novig`` (consensus, vig removed).
    """
    missing = REQUIRED - set(snapshots.columns)
    if missing:
        raise ValueError(f"odds snapshots missing columns: {sorted(missing)}")
    s = snapshots.copy()
    s["captured_at"] = pd.to_datetime(s["captured_at"], utc=True)
    start = pd.to_datetime(feats["start_utc"], utc=True)
    decision = (start - pd.Timedelta(minutes=lead_minutes)).rename("decision")
    s = s.merge(decision, left_on="game_id", right_index=True, how="inner")
    s = s[s.captured_at <= s.decision].sort_values("captured_at")  # never look past the decision time
    g = s.groupby("game_id").home_prob_novig
    out = feats.copy()
    out["mkt_open_p"] = g.first().reindex(out.index)
    out["mkt_move"] = (g.last() - g.first()).reindex(out.index)
    return out


def assert_no_closing_columns(columns) -> None:
    bad = [c for c in columns if "close" in c.lower() or "closing" in c.lower()]
    if bad:
        raise ValueError(f"closing-line columns must never be model features: {bad}")
