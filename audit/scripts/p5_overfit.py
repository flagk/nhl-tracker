"""Phase 5 on the real database (GitHub Actions): overfitting statistics on the paper-strategy returns, and bankroll simulation under parameter uncertainty.

Only decision-time prices are used (the logged best price at the capture the run used). Live record only (since 2026-09-30): nobody has tuned on it.
Writes audit/results/phase5_real.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nhlbet.data.store import Store  # noqa: E402
from nhlbet.report.betlog import shadow_resolved  # noqa: E402
from nhlbet.report.evaluation import live_game_frame  # noqa: E402
from nhlbet.risk.uncertainty import estimate_theta, simulate_uncertain  # noqa: E402
from nhlbet.stats.overfit import deflated_sharpe, pbo, reality_check  # noqa: E402


def main() -> int:
    store = Store(str(ROOT / "data" / "nhl.db"))
    n_trials = int(subprocess.run([sys.executable, str(ROOT / "audit/scripts/p5_trials.py")], capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()[-1])
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trial_register_total": n_trials}
    d = shadow_resolved(store)
    d = d[d.is_bet & ~d.push.fillna(False)] if len(d) else d
    if d.empty:
        res["note"] = "no settled paper bets"
        (ROOT / "audit/results/phase5_real.json").write_text(json.dumps(res, indent=1)); return 0
    g = d.groupby(["strategy", "game_date"]).apply(lambda x: x.profit.sum() / x.stake.sum() if x.stake.sum() > 0 else np.nan).unstack(0)
    g = g.sort_index()
    keep = g.columns[g.notna().sum() >= 10]
    M = g[keep]
    res["matrix"] = {"strategies": int(M.shape[1]), "days": int(M.shape[0]), "dropped_for_fewer_than_10_betting_days": int(g.shape[1] - M.shape[1]),
                     "note": "daily return per unit staked; days without a bet count as 0 (no bet, no return)"}
    M0 = M.fillna(0.0)
    res["pbo"] = pbo(M0, splits=12)
    sr = (M0.mean() / M0.std(ddof=1)).replace([np.inf, -np.inf], np.nan).dropna()
    best = sr.idxmax()
    res["best_strategy_in_sample"] = {"name": str(best), "mean_daily_return": float(M0[best].mean()), "bets_days": int(M[best].notna().sum())}
    res["deflated_sharpe_matrix_trials"] = deflated_sharpe(M0[best].to_numpy(), int(M.shape[1]), float(sr.var()))
    res["deflated_sharpe_register_trials"] = deflated_sharpe(M0[best].to_numpy(), n_trials, float(sr.var()))
    res["reality_check"] = reality_check(M0, B=2000)
    res["strategies_with_positive_mean"] = [str(c) for c in M0.columns if M0[c].mean() > 0]
    # bankroll simulation: moneyline picks, decision-time prices, theta estimated from all live games with a market price
    live = live_game_frame(store, "shin")
    from nhlbet.risk.shrink import shrink_to_market
    live["p_adj"] = [shrink_to_market(a, b) for a, b in zip(live.p_model, live.p_market)]       # the logged p_adj exists only for bets; the shrinkage is a deterministic function of the two probabilities
    sr_all = shadow_resolved(store)
    sr_all = sr_all[sr_all.is_bet & (sr_all.market.fillna("h2h") == "h2h")] if len(sr_all) else sr_all
    templates = {}
    if len(sr_all):
        a = sr_all[sr_all.strategy == "no_guard"]
        templates["policy-like: the live policy without its early-season guard (3% edge floor, shrunk claims)"] = (a, "p_adj", "p_adj")
        b = sr_all[sr_all.strategy.isin(["every_game", "flat_model_side"])].drop_duplicates(["game_id", "side"])
        templates["aggressive: any model pick, claim = unshrunk p_model"] = (b, "p_model", "p_model")
    res["simulations"], res["theta"], res["templates"] = [], {}, {}
    for tname, (bets, claim, live_claim) in templates.items():
        tmpl = pd.DataFrame({"decimal": bets["decimal"].astype(float), "p_market": bets["p_market"].astype(float), "p_model": bets[claim].astype(float)}).dropna()
        lv = live[live[live_claim].notna()] if live_claim in live else live
        if len(lv) < 40 or len(tmpl) < 10:
            res["templates"][tname] = {"bets": int(len(tmpl)), "note": "too few bets or games"}
            continue
        th = estimate_theta(lv[live_claim], lv.p_market, lv.home_win.astype(int), lv.date, B=2000)
        res["theta"][tname] = th
        res["templates"][tname] = {"bets": int(len(tmpl)), "mean_decimal": float(tmpl.decimal.mean()), "mean_claimed_edge_pts": float((tmpl.p_model - tmpl.p_market).mean() * 100),
                                   "share_with_positive_claimed_edge": float((tmpl.p_model > tmpl.p_market).mean())}
        scen = {"estimated from the live record": (th["theta"], th["se"]), "no skill (theta = 0, known)": (0.0, 0.0), "optimistic prior (theta ~ N(0.5, 0.3))": (0.5, 0.3)}
        for name, (mu, se) in scen.items():
            for kf in (0.25, 0.5):
                for hc in (0.0, 0.02):
                    for horizon in (200, 500):
                        res["simulations"].append({"template": tname, "theta_scenario": name, **simulate_uncertain(tmpl, mu, se, kelly_fraction=kf, max_bet_pct=0.02, price_haircut=hc, horizon=horizon, n_paths=10000)})
    (ROOT / "audit/results/phase5_real.json").write_text(json.dumps(res, indent=1, default=float))
    print("phase 5 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
