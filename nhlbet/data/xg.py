"""Expected-goals (xG) model from shot location.

The NHL API exposes shot coordinates but no xG. This module ships a *simple, transparent*
geometry model (distance + angle + shot type) with default coefficients chosen to match
well-known league-wide shooting-percentage-by-distance curves (~17% at 10 ft, ~4% at 40 ft,
~1.4% at 60 ft). It is a placeholder for a properly fitted model: ``fit_xg`` refits the same
functional form on the ``shots`` table and ``XGModel.save/load`` persist the coefficients.

``xg`` is P(goal | shot on target); an unblocked *miss* is discounted by ``p_on_target``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GOAL_X = 89.0  # goal line, feet from centre ice
HD_MAX_DIST, HD_MAX_ANGLE = 25.0, 60.0  # "high danger" = close to the net and not too wide

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around", "bat", "poke", "cradle"]
DEFAULT_TYPE_ADJ = {"wrist": 0.0, "snap": 0.10, "slap": -0.15, "backhand": -0.10, "tip-in": 0.55,
                    "deflected": 0.45, "wrap-around": -0.55, "bat": 0.10, "poke": -0.20, "cradle": 0.0}


def geometry(x: float, y: float) -> tuple[float, float]:
    """(distance in ft to the goal mouth, shot angle in degrees off the centre line)."""
    dx = GOAL_X - abs(float(x))
    dist = math.hypot(dx, float(y))
    angle = math.degrees(math.atan2(abs(float(y)), max(dx, 0.01)))
    return dist, angle


def is_high_danger(x: float, y: float) -> bool:
    d, a = geometry(x, y)
    return d <= HD_MAX_DIST and a <= HD_MAX_ANGLE and abs(x) <= GOAL_X


@dataclass
class XGModel:
    intercept: float = -1.06
    per_ft: float = -0.053
    per_deg: float = -0.012
    type_adj: dict = field(default_factory=lambda: dict(DEFAULT_TYPE_ADJ))
    p_on_target: float = 0.70

    def p_goal_on_target(self, x: float, y: float, shot_type: str | None) -> float:
        d, a = geometry(x, y)
        z = self.intercept + self.per_ft * d + self.per_deg * a + self.type_adj.get((shot_type or "").lower(), 0.0)
        return 1.0 / (1.0 + math.exp(-z))

    def xg(self, x: float | None, y: float | None, shot_type: str | None, kind: str) -> float | None:
        """xG of one attempt. ``kind`` is 'sog' | 'goal' | 'miss'. None if it can't be located."""
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
            return None
        p = self.p_goal_on_target(x, y, shot_type)
        return p * (self.p_on_target if kind == "miss" else 1.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "XGModel":
        p = Path(path)
        return cls(**json.loads(p.read_text())) if p.exists() else cls()


def fit_xg(shots: pd.DataFrame, min_shots: int = 20_000) -> XGModel:
    """Refit coefficients from the ``shots`` table (kind in sog/goal/miss, with x, y).

    Requires ``min_shots`` on-target attempts so we never overwrite the defaults with noise.
    """
    from sklearn.linear_model import LogisticRegression

    s = shots.dropna(subset=["x", "y"])
    ot = s[s.kind.isin(["sog", "goal"])]
    if len(ot) < min_shots:
        raise ValueError(f"need >= {min_shots} located on-target shots to refit xG, have {len(ot)}")
    geo = np.array([geometry(x, y) for x, y in zip(ot.x, ot.y)])
    types = [t for t in SHOT_TYPES[1:]]
    X = np.column_stack([geo[:, 0], geo[:, 1]] + [(ot.shot_type == t).to_numpy(float) for t in types])
    lr = LogisticRegression(C=10.0, max_iter=500).fit(X, ot.is_goal.to_numpy())
    adj = {"wrist": 0.0, **{t: float(c) for t, c in zip(types, lr.coef_[0][2:])}}
    miss_rate = (s.kind == "miss").sum() / max(1, len(s[s.kind.isin(["sog", "goal", "miss"])]))
    return XGModel(float(lr.intercept_[0]), float(lr.coef_[0][0]), float(lr.coef_[0][1]), adj,
                   p_on_target=float(1 - miss_rate))
