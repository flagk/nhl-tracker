import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.monitor import feature_drift, performance_drift, psi
from nhlbet.registry import ModelRegistry, data_fingerprint


def test_registry_append_only(tmp_path):
    r = ModelRegistry(tmp_path / "reg.json")
    assert r.latest() is None
    for v in ("a", "b", "c", "d"):
        (tmp_path / f"{v}.bin").write_text("x")
        r.register({"version": v, "artifact": str(tmp_path / f"{v}.bin")})
    with pytest.raises(ValueError):
        r.register({"version": "a"})
    assert r.latest()["version"] == "d" and len(r.history()) == 4 and "created_at" in r.latest()
    assert len(r.prune_artifacts(keep=2)) == 2 and not (tmp_path / "a.bin").exists() and (tmp_path / "d.bin").exists()
    assert len(r.history()) == 4                                    # lineage rows are kept


def test_fingerprint_changes_with_data():
    a, b = data_fingerprint([1, 2, 3], "2024-01-01"), data_fingerprint([1, 2, 3, 4], "2024-01-01")
    assert a != b and a == data_fingerprint([3, 2, 1], "2024-01-01")


def test_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, rng.normal(size=500)) < 0.1
    assert psi(x, rng.normal(loc=1.5, size=500)) > 0.25
    assert np.isnan(psi(x[:10], x[:5]))
    d = feature_drift(pd.DataFrame({"a": x, "b": x}), pd.DataFrame({"a": rng.normal(2, 1, 300), "b": rng.normal(size=300)}), ["a", "b"])
    assert list(d["flagged"]) == ["a"] and d["status"] == "WARN"


def _hist(n, skill, seed=0, recent_skill=None):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.35, 0.65, n)
    truth = p.copy()
    if recent_skill is not None:
        truth[-200:] = 0.5 + (p[-200:] - 0.5) * recent_skill
    y = (rng.random(n) < (0.5 + (truth - 0.5) * skill)).astype(int)
    return pd.DataFrame({"game_date": pd.date_range("2024-01-01", periods=n, freq="6h"), "y": y, "p": p, "elo": 0.5 + (p - 0.5) * 0.5})


def test_performance_drift_states():
    assert performance_drift(_hist(300, 1.0), "p")["status"] == "INSUFFICIENT_DATA"
    assert performance_drift(_hist(1500, 1.0), "p", "elo")["status"] in ("OK", "WARN")
    bad = _hist(1500, 1.0, recent_skill=-2.0)                        # model turns anti-predictive recently
    r = performance_drift(bad, "p", "elo")
    assert r["status"] in ("WARN", "ALERT") and r["reasons"]


def test_retrain_end_to_end_is_idempotent_and_registers(league, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/models").mkdir(parents=True)
    (tmp_path / "data/models/feature_selection.json").write_text(json.dumps({"kept": ["d_gd_season_shrunk", "d_rest_days"]}))
    g = league["games"].copy(); g["game_date"] = g.game_date.dt.strftime("%Y-%m-%d"); g["start_utc"] = None
    store = Store(str(tmp_path / "t.db"))
    store.upsert("games", g.assign(updated_at=None).to_dict("records"), ["game_id"])
    store.upsert("team_game", league["team_game"].to_dict("records"), ["game_id", "team"])
    store.upsert("goalie_game", league["goalie_game"].to_dict("records"), ["game_id", "player_id"])
    from nhlbet.train import retrain
    r1 = retrain(str(tmp_path / "t.db"), model_dir="data/models", report_dir="reports")
    assert r1["retrained"] and Path(r1["artifact"]).exists() and (tmp_path / "reports/drift.json").exists()
    r2 = retrain(str(tmp_path / "t.db"), model_dir="data/models", report_dir="reports")
    assert not r2["retrained"] and r2["version"] == r1["version"]        # no new games -> no retrain
    r3 = retrain(str(tmp_path / "t.db"), force=True, model_dir="data/models", report_dir="reports")
    assert r3["retrained"] is True
