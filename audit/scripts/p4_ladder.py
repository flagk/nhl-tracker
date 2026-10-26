"""Phase 4 on the REAL database (GitHub Actions): purged and embargoed walk-forward ablation ladder, calibration variants, and the comparison with the devigged market.

A. Research window (games up to RESEARCH_END only): home_rate, Elo, logistic, random forest, the production stack, and a hierarchical Bayesian challenger, each raw and
   with online Platt / isotonic / beta calibration; log loss, Brier, ECE, calibration slope, date-clustered intervals, Diebold-Mariano tests against Elo and the home rate.
   Embargo: training stops 1 day before each block; sensitivity at 0 and 7 days.
B. Live record (decision-time prices exist only here): the same ladder retrained daily, against the market devigged three ways (Shin, power, proportional/multiplicative)
   at the decision time and at the close, on identical games, Holm-corrected Diebold-Mariano tests, plus the logged production probability.
C. Closing-line value per bet for the live moneyline picks, with every capture and with only those whose "close" was taken within 15 and 5 minutes of puck drop.

Writes audit/results/phase4_real.json and audit/results/p4_reliability.png. Exits 1 only if a step breaks.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nhlbet.analysis.importance import candidate_columns  # noqa: E402,F401
from nhlbet.config import load_selected_features  # noqa: E402
from nhlbet.data.loaders import load_tables  # noqa: E402
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.features.builder import BuilderConfig, FeatureBuilder  # noqa: E402
from nhlbet.models.base import EloOnly, HomeRate, Logistic, RandomForest, make_zoo  # noqa: E402
from nhlbet.models.calibration import brier, calibration_slope_intercept, ece, log_loss_, online_calibrate, plot_reliability, reliability_table  # noqa: E402
from nhlbet.models.hier import HierBT  # noqa: E402
from nhlbet.models.walkforward import walk_forward  # noqa: E402
from nhlbet.report.evaluation import cluster_dist, correct, dm_test, live_game_frame, mean_ci_p, official_picks, market_table, pick_frame  # noqa: E402

RESEARCH_END = pd.Timestamp("2026-01-04")          # the day before the locked half-season; part A never reads later games
FIRST_TEST, STEP = "2024-10-04", 28
LIVE_FROM = "2026-09-30"
EPS = 1e-6
ll = lambda p, y: -(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))


def scores(p, y, d):
    p = np.asarray(p, float)
    s, i = calibration_slope_intercept(p, y)
    return {"n": int(len(y)), "log_loss": float(ll(p, y).mean()), "brier": float(brier(p, y)), "ece": float(ece(p, y)), "cal_slope": float(s), "cal_intercept": float(i)}


def gain(ref_loss, cand_loss, dates, B=4000):
    g = mean_ci_p(ref_loss - cand_loss, dates, B)              # positive = candidate better
    return {"mean_gain": g["mean"], "ci": [float(g["ci"][0]), float(g["ci"][1])], "p_not_better": float(g["p_not_above_0"])}


def part_a(F, feats):
    R = F[F.game_date <= RESEARCH_END]
    prod = make_zoo(feats)
    runs = {}
    t0 = datetime.now(timezone.utc)
    P = walk_forward(R, prod, FIRST_TEST, STEP, embargo_days=1, n_jobs=-1)
    H = walk_forward(R, {"elo": lambda: EloOnly(), "hier_bayes": lambda: HierBT()}, FIRST_TEST, STEP, embargo_days=1, n_jobs=-1)
    P["hier_bayes"] = H["hier_bayes"]
    y = P.y.to_numpy(float)
    dates = P.game_date.to_numpy()
    cols = {"home_rate": "home_rate", "elo": "elo", "logistic": "logistic", "random_forest": "rf", "production_stack": "stack", "hierarchical_bayes": "hier_bayes"}
    out = {"n_games": int(len(P)), "window": [str(P.game_date.min().date()), str(P.game_date.max().date())], "embargo_days": 1, "models": {}, "calibration": {}, "dm": [], "embargo_sensitivity": {}}
    for name, c in cols.items():
        out["models"][name] = scores(P[c].to_numpy(), y, dates)
        out["calibration"][name] = {"raw": out["models"][name]}
        for meth in ("platt", "isotonic", "beta"):
            o = online_calibrate(P.assign(y=P.y).rename(columns={c: "p"}), "p", meth).reindex(P.index)
            m = o.notna().to_numpy()
            if m.sum() > 100:
                out["calibration"][name][meth] = {**scores(o[m].to_numpy(), y[m], dates[m]), "raw_log_loss_same_games": float(ll(P[c].to_numpy()[m], y[m]).mean())}
    # gains and DM tests against two references
    for ref in ("home_rate", "elo"):
        lr = ll(P[cols[ref]].to_numpy(), y)
        rows = []
        for name, c in cols.items():
            if name == ref:
                continue
            lc = ll(P[c].to_numpy(), y)
            rows.append({"reference": ref, "model": name, **gain(lr, lc, dates), **{f"dm_{k}": v for k, v in dm_test(lr, lc).items() if k in ("stat", "p", "lag")}})
        pa = correct(pd.Series([r["dm_p"] for r in rows]))
        for r, p in zip(rows, pa):
            r["dm_p_holm"] = float(p)
        out["dm"] += rows
    plot_reliability({k: (P[c].to_numpy(), y) for k, c in (("elo", "elo"), ("logistic", "logistic"), ("stack", "stack"), ("hier_bayes", "hier_bayes"))}, str(ROOT / "audit/results/p4_reliability.png"))
    for e in (0, 7):
        Q = walk_forward(R, {"elo": lambda: EloOnly(), "logistic": lambda: Logistic(feats), "hier_bayes": lambda: HierBT()}, FIRST_TEST, STEP, embargo_days=e, n_jobs=-1)
        out["embargo_sensitivity"][str(e)] = {m: float(ll(Q[m].to_numpy(), Q.y.to_numpy(float)).mean()) for m in ("elo", "logistic", "hier_bayes")} | {"n": int(len(Q))}
    out["embargo_sensitivity"]["1"] = {m: float(ll(P[m].to_numpy(), y).mean()) for m in ("elo", "logistic", "hier_bayes")} | {"n": int(len(P))}
    out["reliability_stack"] = reliability_table(P["stack"].to_numpy(), y, 8).round(4).to_dict("records")
    out["minutes"] = round((datetime.now(timezone.utc) - t0).total_seconds() / 60, 1)
    return out


def part_b(store, F, feats):
    out = {}
    live_ids = None
    L = F[F.game_date >= LIVE_FROM]
    if len(L) < 30:
        return {"note": "too few live games"}
    zoo = {"home_rate": lambda: HomeRate(), "elo": lambda: EloOnly(), "logistic": lambda: Logistic(feats), "random_forest": lambda: RandomForest(feats), "hierarchical_bayes": lambda: HierBT()}
    W = walk_forward(F, {k: v for k, v in zoo.items()}, LIVE_FROM, 1, min_train=300, embargo_days=1, n_jobs=-1)
    for method in ("shin", "power", "proportional"):
        d = live_game_frame(store, method)
        d = d.merge(W[[c for c in zoo if c in W.columns]], left_on="game_id", right_index=True, how="inner")
        if len(d) < 30:
            out[method] = {"n": int(len(d))}
            continue
        y = d.home_win.astype(float).to_numpy()
        dates = d.date.to_numpy()
        preds = {"production_logged": d.p_model.to_numpy(float), **{k: d[k].to_numpy(float) for k in zoo}}
        mk = {"market_at_decision": d.p_market.to_numpy(float), "market_close": d.p_close.to_numpy(float)}
        res = {"n": int(len(d)), "from": str(pd.to_datetime(d.date).min().date()), "to": str(pd.to_datetime(d.date).max().date()), "scores": {}, "vs_market": []}
        for k, p in {**preds, **mk}.items():
            res["scores"][k] = scores(p, y, dates)
        for mname, mp in mk.items():
            rows = []
            for k, p in preds.items():
                g = gain(ll(mp, y), ll(p, y), dates)
                dm = dm_test(ll(mp, y), ll(p, y))
                rows.append({"market": mname, "model": k, **g, "dm_stat": dm["stat"], "dm_p": dm["p"]})
            adj = correct(pd.Series([r["dm_p"] for r in rows]))
            for r, a in zip(rows, adj):
                r["dm_p_holm"] = float(a)
            res["vs_market"] += rows
        out[method] = res
    return out


def part_c(store):
    out = {}
    for gap in (60, 15, 5):
        row = {}
        for label, fn in (("official", lambda g: official_picks(store, g)), ("model_paper_picks", lambda g: pick_frame(store, g))):
            d = fn(gap)
            if d is None or d.empty:
                row[label] = {"note": "no picks"}
                continue
            grp = "official" if label == "official" else "model"
            t = market_table(d[d.market == "h2h"], "shin", group=grp)
            row[label] = t.round(5).to_dict("records")[0] if len(t) else {"note": "no moneyline picks"}
        out[f"close_within_{gap}_min"] = row
    return out


def main() -> int:
    store = Store(str(ROOT / "data" / "nhl.db"))
    tables = load_tables(store)
    n = int(tables["games"].home_score.notna().sum())
    if n < 5000:
        sys.exit(f"only {n} completed games in the database: restore the cache first")
    feats = load_selected_features()
    F = FeatureBuilder(BuilderConfig()).build(tables)
    F = F[(F.game_type == 2) & F.home_win.notna()]
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "completed_games": n, "research_end": str(RESEARCH_END.date()), "features": feats}
    (ROOT / "audit/results").mkdir(parents=True, exist_ok=True)
    res["A_research_ladder"] = part_a(F, feats)
    (ROOT / "audit/results/phase4_real.json").write_text(json.dumps(res, indent=1, default=float))
    res["B_live_vs_market"] = part_b(store, F, feats)
    res["C_clv_by_close_quality"] = part_c(store)
    (ROOT / "audit/results/phase4_real.json").write_text(json.dumps(res, indent=1, default=float))
    print("phase 4 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
