"""Phase 6(c): the pre-registered confirmatory evaluation. Runs ONCE, only after the trigger in audit/preregistration.md, and only if the file still matches its recorded hash.

    python audit/scripts/p6_prereg_eval.py [--count-only]

--count-only prints how many qualifying games exist and nothing else (no metric is computed, so it cannot become a way to peek).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit/scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULT = ROOT / "audit/results/prereg_result.json"
GAIN_MIN, P_ALPHA, CLV_MIN_PICKS, CLV_MIN_POINTS = 0.007, 0.025, 300, 0.5


class PreregViolation(RuntimeError):
    pass


def check_lock(root: Path = ROOT) -> dict:
    lock = json.loads((root / "audit/prereg_lock.json").read_text())
    if hashlib.sha256((root / "audit/preregistration.md").read_bytes()).hexdigest() != lock["preregistration_sha256"]:
        raise PreregViolation("audit/preregistration.md no longer matches the hash recorded at pre-registration")
    return lock


def verdict(p1: dict, p2: dict) -> str:
    """Fixed rules from audit/preregistration.md."""
    p1_pass = p1["dm_p"] < P_ALPHA and p1["mean_gain"] >= GAIN_MIN
    p2_pass = p2["n"] >= CLV_MIN_PICKS and p2["mean"] >= CLV_MIN_POINTS and p2["ci_low"] > 0
    if p1_pass and p2_pass:
        return "EDGE SUPPORTED"
    if p1["ci_high"] < GAIN_MIN or (p1["mean_gain"] < 0 and p1["dm_p"] > 1 - P_ALPHA):
        return "EDGE UNSUPPORTED"
    return "INCONCLUSIVE"


def qualifying(store, lock: dict, verified_ids: set[int]):
    from nhlbet.report.evaluation import live_game_frame
    d = live_game_frame(store, "shin")
    if d.empty:
        return d
    d = d[pd.to_datetime(d.start_utc, utc=True) > pd.Timestamp(lock["locked_at"])]
    return d[d.game_id.isin(verified_ids)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--db", default=str(ROOT / "data/nhl.db"))
    ap.add_argument("--verification", default=str(ROOT / "audit/results/forward_verification.json"))
    a = ap.parse_args()
    lock = check_lock()
    if RESULT.exists():
        raise PreregViolation("the confirmatory evaluation was already run; its result is in audit/results/prereg_result.json and cannot be rerun")
    from nhlbet.data.store import Store
    store = Store(a.db)
    ver = json.loads(Path(a.verification).read_text())
    d = qualifying(store, lock, set(ver.get("last_run_verified_game_ids", [])))
    n = int(len(d))
    reached = n >= lock["trigger"]["qualifying_games"] or datetime.now(timezone.utc).date().isoformat() >= lock["trigger"]["or_date"]
    if a.count_only:
        print(json.dumps({"qualifying_games": n, "trigger_reached": reached}))
        return 0
    if not reached:
        raise PreregViolation(f"trigger not reached ({n} of {lock['trigger']['qualifying_games']} qualifying games, date before {lock['trigger']['or_date']}): evaluating now would be a look, not the test")
    from nhlbet.report.evaluation import dm_test, market_table, mean_ci_p, pick_frame
    y = d.home_win.astype(float).to_numpy()
    ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    lm, lk = ll(d.p_market.to_numpy(float)), ll(d.p_model.to_numpy(float))
    g = mean_ci_p(lm - lk, d.date.to_numpy(), 4000)
    dm = dm_test(lm, lk)
    p1 = {"n": n, "mean_gain": g["mean"], "ci_low": float(g["ci"][0]), "ci_high": float(g["ci"][1]), "dm_p": dm["p"]}
    pf = pick_frame(store, 15)
    pf = pf[(pf.market == "h2h") & (pf.group == "model") & (pd.to_datetime(pf.date) > pd.Timestamp(lock["locked_at"]).tz_convert("UTC").tz_localize(None).normalize())] if len(pf) else pf
    t = market_table(pf, "shin") if len(pf) else pd.DataFrame()
    p2 = ({"n": int(t.clv_n.iloc[0]), "mean": float(t.clv_pts.iloc[0]), "ci_low": float(t.clv_lo.iloc[0]), "ci_high": float(t.clv_hi.iloc[0])} if len(t) else {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
    from p6_lock_prereg import tree_hash
    changed = [k for k, v in tree_hash(ROOT).items() if v != lock["frozen_paths_sha256"].get(k)]
    out = {"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "P1": p1, "P2": p2, "verdict": verdict(p1, p2),
           "protocol_deviations_frozen_paths_changed": changed, "thresholds": {"gain_min": GAIN_MIN, "alpha_each": P_ALPHA, "clv_min_picks": CLV_MIN_PICKS, "clv_min_points": CLV_MIN_POINTS}}
    RESULT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
