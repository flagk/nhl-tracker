"""Shared configuration helpers."""
from __future__ import annotations

import json
from pathlib import Path

from nhlbet.features.builder import BuilderConfig
from nhlbet.models.base import HYPERPARAMS_PATH

FEATURE_SELECTION_PATH = Path("data/models/feature_selection.json")


def load_builder_config(path: str | Path = HYPERPARAMS_PATH, **overrides) -> BuilderConfig:
    """BuilderConfig with tuned Elo parameters (if tuned), plus explicit overrides."""
    p = Path(path)
    elo = json.loads(p.read_text()).get("elo", {}) if p.exists() else {}
    return BuilderConfig(**{**elo, **overrides})


def load_selected_features(path: str | Path = FEATURE_SELECTION_PATH) -> list[str]:
    return json.loads(Path(path).read_text())["kept"]
