"""The holdout lock (Phase 1.8).

The holdout is the most recent contiguous block of walk-forward games: the larger of 20% of all games and one half-season (656 games), cut at a date boundary.
Every decision that selects or tunes something (features, hyper-parameters, shrinkage, model choice, thresholds) must see ``research_view(F)`` only. The
holdout is evaluated exactly once (``evaluate_once``); a second evaluation raises, because a number you may look at repeatedly is a selection tool.

Honesty note: these games were examined in reports written before the lock existed (model comparison, ablations), so the lock protects FUTURE decisions; it does
not make the block pristine. The live record since 2026-09-30 is the only data nobody has tuned on.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LOCK_PATH = ROOT / "reports" / "holdout_lock.json"
EVAL_PATH = ROOT / "reports" / "holdout_eval.json"
WF_PATH = ROOT / "reports" / "walkforward_predictions.csv"
FRACTION = 0.20
HALF_SEASON_GAMES = 656             # half of an 82-game, 32-team regular season (1,312 games)


class HoldoutViolation(RuntimeError):
    """Research code touched holdout dates, or the holdout was evaluated twice."""


def _hash(df: pd.DataFrame) -> str:
    lines = sorted(f"{int(r.game_id)},{pd.Timestamp(r.game_date).date()},{int(r.y)}" for r in df.itertuples())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def compute_lock(wf_path: Path = WF_PATH) -> dict:
    w = pd.read_csv(wf_path, index_col=0, parse_dates=["game_date"]) if "game_id" not in pd.read_csv(wf_path, nrows=1).columns else pd.read_csv(wf_path, parse_dates=["game_date"])
    if "game_id" not in w.columns:
        w = w.reset_index().rename(columns={"index": "game_id"})
    w = w.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(w)
    want = max(int(np.ceil(FRACTION * n)), HALF_SEASON_GAMES)
    start = w.game_date.iloc[n - want]                                   # cut at a date boundary: the whole first day belongs to the holdout
    h = w[w.game_date >= start]
    return {"rule": f"last max({FRACTION:.0%} of walk-forward games, {HALF_SEASON_GAMES} games = one half-season), cut at a date boundary",
            "source": "reports/walkforward_predictions.csv", "n_walkforward_games": int(n), "start": str(start.date()), "end": str(h.game_date.max().date()),
            "n_games": int(len(h)), "hash_sha256": _hash(h), "hash_of": "sorted 'game_id,game_date,y' lines",
            "previously_examined": True, "examined_note": "model comparison and ablation reports were written before this lock existed; the lock constrains new decisions"}


def load_lock(path: Path = LOCK_PATH) -> dict:
    return json.loads(Path(path).read_text())


def in_holdout(dates, lock: dict | None = None) -> np.ndarray:
    lock = lock or load_lock()
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    return ((d >= pd.Timestamp(lock["start"])) & (d <= pd.Timestamp(lock["end"]))).to_numpy()


def research_view(F: pd.DataFrame, date_col: str = "game_date", lock: dict | None = None) -> pd.DataFrame:
    """``F`` without the holdout games. Selection and tuning code must use this."""
    m = in_holdout(F[date_col], lock)
    return F.loc[~m]


def assert_research_only(dates, who: str, lock: dict | None = None) -> None:
    if in_holdout(dates, lock).any():
        raise HoldoutViolation(f"{who} touched holdout dates ({(lock or load_lock())['start']} to {(lock or load_lock())['end']})")


def evaluate_once(fn, path: Path = EVAL_PATH, lock_path: Path = LOCK_PATH):
    """Run ``fn(lock)`` (it returns a JSON-able result) and record it. A second call raises ``HoldoutViolation`` and returns nothing."""
    path = Path(path)
    if path.exists():
        raise HoldoutViolation(f"the holdout was already evaluated ({path}); it can be evaluated once")
    lock = load_lock(lock_path)
    res = fn(lock)
    path.write_text(json.dumps({"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "lock_hash": lock["hash_sha256"], "result": res}, indent=1, default=float))
    return res
