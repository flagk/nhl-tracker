"""Retrain-if-needed, drift-check and register. This is what the daily job calls."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from nhlbet.analysis.importance import candidate_columns
from nhlbet.config import load_builder_config, load_selected_features
from nhlbet.data.store import Store
from nhlbet.features.builder import build_features
from nhlbet.models.base import load_params, make_zoo
from nhlbet.models.bundle import ModelBundle, train_bundle
from nhlbet.models.goals import add_goal_targets, goal_feature_columns
from nhlbet.monitor import feature_drift, performance_drift
from nhlbet.registry import ModelRegistry, data_fingerprint

log = logging.getLogger("nhlbet.train")
WF_PATH = Path("reports/walkforward_predictions.csv")


def resolved_history(wf_path: Path = WF_PATH, live_log: Path | None = None) -> pd.DataFrame:
    """Resolved out-of-sample predictions of the deployed stack: walk-forward backtest, plus the live log if present."""
    parts = []
    if wf_path.exists():
        w = pd.read_csv(wf_path, index_col=0, parse_dates=["game_date"])
        parts.append(w[["game_date", "y", "stack", "elo"]])
    if live_log and live_log.exists():
        l = pd.read_csv(live_log, parse_dates=["game_date"])
        l = l[l.y.notna()][["game_date", "y", "stack", "elo"]] if {"stack", "elo", "y"} <= set(l.columns) else pd.DataFrame()
        if len(l):
            parts.append(l)
    return pd.concat(parts).sort_values("game_date") if parts else pd.DataFrame(columns=["game_date", "y", "stack", "elo"])


def retrain(db: str = "data/nhl.db", force: bool = False, model_dir: str = "data/models", report_dir: str = "reports",
            live_log: str | None = None) -> dict:
    store = Store(db)
    cfg = load_builder_config()
    F = build_features(store, cfg)
    done = F[(F.game_type == 2) & F.home_score.notna()]
    if done.empty:
        raise RuntimeError("no completed games in the store; run `python -m nhlbet.data.ingest` first")
    fp = data_fingerprint(done.index, done.game_date.max())
    reg = ModelRegistry(Path(model_dir) / "registry.json")
    last = reg.latest()
    if last and last.get("data_fingerprint") == fp and last.get("has_goals") and not force:      # a bundle without the goals model is retrained once
        log.info("no new completed games since %s - model unchanged", last["version"])
        return {"retrained": False, "version": last["version"], "drift": last.get("drift", {})}

    features = load_selected_features()
    done = add_goal_targets(done, store.df("SELECT game_id, home_score, away_score, last_period FROM games"))
    gpath = Path(report_dir) / "goals_model.json"
    galpha = json.loads(gpath.read_text()).get("alpha", 50.0) if gpath.exists() else 50.0
    cpath = Path(report_dir) / "goals_calibration.json"
    gcal = json.loads(cpath.read_text()) if cpath.exists() else None
    zoo = make_zoo(features, candidate_columns(done.drop(columns=["hr", "ar", "ot", "so", "tot", "mar"])))
    hist = resolved_history(live_log=Path(live_log) if live_log else None)
    version = f"{pd.Timestamp(done.game_date.max()).strftime('%Y%m%d')}-{fp.split('@')[0][:6]}"
    taken = {e["version"] for e in reg.history()}
    base, rev = version, 1
    while version in taken:                      # forced retrain on identical data -> distinct, ordered version ids
        rev += 1
        version = f"{base}-r{rev}"
    bundle = train_bundle(done, zoo, version, features, cfg.__dict__, hist[["stack", "y"]] if len(hist) else None,
                          goals_features=goal_feature_columns(done), goals_alpha=galpha, goals_cal=gcal,
                          meta={"n_train": len(done), "train_start": str(done.game_date.min().date()),
                                "train_end": str(done.game_date.max().date())})
    art = Path(model_dir) / f"model_{version}.joblib"
    bundle.save(art)

    # ---- drift ----
    drift = {"performance": {"status": "INSUFFICIENT_DATA"}, "features": {"status": "OK", "flagged": {}}}
    if len(hist):
        h = hist.copy()
        h["p"] = h["stack"]
        drift["performance"] = performance_drift(h, "p", "elo")
    recent_cut = done.game_date.max() - pd.Timedelta(days=60)
    ref, new = done[done.game_date < recent_cut], done[done.game_date >= recent_cut]
    if len(ref) > 300 and len(new) > 30:
        drift["features"] = feature_drift(ref, new, features)
    status = "ALERT" if "ALERT" in (drift["performance"]["status"], drift["features"]["status"]) else (
        "WARN" if "WARN" in (drift["performance"]["status"], drift["features"]["status"]) else "OK")
    drift["status"] = status

    metrics = {}
    cmp_path = Path(report_dir) / "model_comparison.json"
    if cmp_path.exists():
        sub = json.loads(cmp_path.read_text()).get("online_subset", {})
        metrics = {r["model"]: {k: r[k] for k in ("log_loss", "brier", "auc", "ece", "cal_slope")} for r in sub.get("metrics", [])
                   if r["model"] in ("stack__online", "elo", "home_rate")}
        metrics["walk_forward_n"] = sub.get("n")
    entry = reg.register({"version": version, "artifact": str(art), "data_fingerprint": fp, **bundle.meta,
                          "features": features, "hyperparameters": {m: load_params(m) for m in ("logistic", "rf", "lgbm", "xgb")},
                          "builder_config": cfg.__dict__, "p_source": "online_platt" if bundle.online_cal is not None else "inner_oof_platt",
                          "walk_forward_metrics": metrics, "drift": drift, "has_goals": bundle.goals is not None})
    reg.prune_artifacts(keep=3)
    Path(report_dir).mkdir(exist_ok=True)
    Path(report_dir, "drift.json").write_text(json.dumps({"version": version, **drift}, indent=1, default=str))
    log.info("trained %s on %d games; drift status %s", version, len(done), status)
    return {"retrained": True, "version": version, "drift": drift, "artifact": str(art)}
