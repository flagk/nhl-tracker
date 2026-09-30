"""The deployable unit: base models + stacker + calibrators + metadata, saved/loaded with joblib."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nhlbet.models.base import Factory
from nhlbet.models.calibration import Calibrator
from nhlbet.models.walkforward import fit_stack, predict_stack


@dataclass
class ModelBundle:
    version: str
    features: list[str]
    builder_cfg: dict
    fitted: dict
    stack: Any
    wavg: Any
    cals: dict
    online_cal: Calibrator | None = None
    meta: dict = field(default_factory=dict)

    def predict(self, F: pd.DataFrame) -> pd.DataFrame:
        """All model outputs. ``p`` is the recommended probability: online-calibrated stack when
        available (fit on the model's own resolved out-of-sample history), else the inner-OOF Platt stack."""
        out = predict_stack(self.fitted, self.stack, self.wavg, self.cals, F)
        out["p"] = self.online_cal.predict(out["stack"]) if self.online_cal is not None else out["stack__platt"]
        out["p_source"] = "online_platt" if self.online_cal is not None else "inner_oof_platt"
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        return joblib.load(path)


def train_bundle(F: pd.DataFrame, zoo: dict[str, Factory], version: str, features: list[str], builder_cfg: dict,
                 resolved_history: pd.DataFrame | None = None, min_online: int = 300, meta: dict | None = None) -> ModelBundle:
    """Fit on every completed game in ``F``. ``resolved_history`` = past out-of-sample predictions of the deployed
    stack (columns ``stack``, ``y``); when it has >= ``min_online`` rows a Platt calibrator is fit on it."""
    F = F[F.home_win.notna()].sort_values("game_date")
    y = F.home_win.astype(int).to_numpy()
    fitted, stack, wavg, cals = fit_stack(zoo, F, y, F.game_date.reset_index(drop=True))
    online = None
    if resolved_history is not None and len(resolved_history) >= min_online:
        online = Calibrator("platt").fit(resolved_history["stack"], resolved_history["y"])
    return ModelBundle(version, list(features), dict(builder_cfg), fitted, stack, wavg, cals, online, meta or {})
