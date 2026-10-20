"""Leakage checks that need the whole history: run on the real database in CI (scripts/phase1_realdata.py) and on a synthetic league in tests.

1. truncation invariance of the full pipeline (features + walk-forward models + calibrators): predictions for every game up to D are identical with and without
   the data after D;
2. the feature-availability guard on the real feature set, and a planted-outcome sentinel it must catch;
3. the outcome-shuffle null: results shuffled within each season; a clean pipeline finds no skill, and betting a fair market with a margin loses about the margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.models.base import EloOnly, HomeRate, Logistic
from nhlbet.models.walkforward import walk_forward
from nhlbet.validity.availability import LeakageError, assert_features_clean
from nhlbet.validity.null import shuffle_outcomes

VIG = 0.045


def _build(tables) -> pd.DataFrame:
    F = FeatureBuilder(BuilderConfig()).build(tables)
    return F[F.game_type == 2]


def _zoo(features):
    return {"home_rate": lambda: HomeRate(), "elo": lambda: EloOnly(), "logistic": lambda: Logistic(features)}


def truncate(tables: dict, D: pd.Timestamp) -> dict:
    ids = set(tables["games"].loc[tables["games"].game_date <= D, "game_id"])
    return {k: (v[v.game_id.isin(ids)] if "game_id" in v else v) for k, v in tables.items()}


def truncation_invariance(tables, features, D, first_test, step_days=28, min_train=300) -> dict:
    full, cut = _build(tables), _build(truncate(tables, D))
    feats = [f for f in features if f in full.columns]
    z = _zoo(feats)
    a = walk_forward(full, z, first_test, step_days, min_train=min_train, inner_folds=3, n_jobs=1)
    b = walk_forward(cut, z, first_test, step_days, min_train=min_train, inner_folds=3, n_jobs=1)
    a = a[a.game_date <= D].sort_index()
    same_games = set(a.index) == set(b.index)
    diff = (a.drop(columns="game_date") - b.loc[a.index].drop(columns="game_date")).abs().max() if same_games else None
    exact = [c for c in a.columns if c not in ("game_date",) and not c.startswith("wavg")]
    worst_exact = float(diff[exact].max()) if same_games else float("nan")
    worst_wavg = float(diff[[c for c in diff.index if c.startswith("wavg")]].max()) if same_games else float("nan")
    return {"truncated_at": str(pd.Timestamp(D).date()), "games_compared": int(len(a)), "same_games": bool(same_games), "max_abs_diff_models_stack_calibrators": worst_exact,
            "max_abs_diff_weighted_average": worst_wavg, "note": "walk_forward sorts ties within a date stably, so even the SLSQP weighted average is bit-for-bit reproducible",
            "pass": bool(same_games and len(a) > 0 and worst_exact < 1e-9 and worst_wavg < 1e-9), "features_used": feats}


def availability_and_sentinel(tables, features) -> dict:
    F = _build(tables)
    done = F[F.home_win.notna()]
    y = done.home_win.astype(int).to_numpy()
    feats = [f for f in features if f in done.columns]
    out = {"features_checked": len(feats), "clean": True, "sentinel_caught": {}}
    try:
        assert_features_clean(done, y, feats)
    except LeakageError as e:
        out["clean"], out["error"] = False, str(e)
    for name, col in (("outcome_copy", y.astype(float)), ("outcome_complement", 1.0 - y), ("noisy_outcome", y + np.random.default_rng(0).normal(0, 0.05, len(y)))):
        try:
            assert_features_clean(done.assign(**{name: col}), y, feats + [name])
            out["sentinel_caught"][name] = False
        except LeakageError:
            out["sentinel_caught"][name] = True
    out["pass"] = bool(out["clean"] and all(out["sentinel_caught"].values()))
    return out


def shuffle_null(tables, features, first_test, step_days=28, min_train=300, seeds=(1, 2, 3)) -> dict:
    from sklearn.metrics import roc_auc_score
    real = _build(tables)
    feats = [f for f in features if f in real.columns]
    z = _zoo(feats)
    pr = walk_forward(real, z, first_test, step_days, min_train=min_train, inner_folds=3, n_jobs=1)
    res = {"real": {"n": int(len(pr)), "auc_elo": float(roc_auc_score(pr.y, pr.elo)), "auc_logistic": float(roc_auc_score(pr.y, pr.logistic))}, "shuffled": []}
    for s in seeds:
        pn = walk_forward(_build(shuffle_outcomes(tables, seed=s)), z, first_test, step_days, min_train=min_train, inner_folds=3, n_jobs=1)
        y = pn.y.to_numpy()
        base = float(y.mean())
        ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
        dec_h, dec_a = 1 / (base * (1 + VIG)), 1 / ((1 - base) * (1 + VIG))
        home = (pn.logistic >= 0.5).to_numpy()
        roi = float(np.mean((home == (y == 1)) * np.where(home, dec_h, dec_a) - 1))
        res["shuffled"].append({"seed": s, "n": int(len(pn)), "auc_elo": float(roc_auc_score(y, pn.elo)), "auc_logistic": float(roc_auc_score(y, pn.logistic)),
                                "logloss_gain_vs_base_rate": float(ll(np.full(len(y), base)) - ll(pn.logistic.to_numpy())), "roi_vs_fair_market_with_4.5pct_margin": roi})
    se_auc = 0.5 / np.sqrt(3 * res["real"]["n"]) * 1.0
    res["pass"] = bool(all(abs(r["auc_logistic"] - 0.5) < 4 * max(se_auc, 0.01) + 0.02 and r["logloss_gain_vs_base_rate"] < 0.01 and r["roi_vs_fair_market_with_4.5pct_margin"] < 0.06 for r in res["shuffled"])
                       and res["real"]["auc_logistic"] > 0.5)
    return res


def settlement_sensitivity(store) -> dict:
    """How many settled paper bets change if the book's shootout rule is the other one? Totals only: nothing else depends on it."""
    from unittest import mock

    from nhlbet.report.betlog import shadow_resolved
    out = {}
    for name, flag in (("so_adds_goal", True), ("so_excluded", False)):
        with mock.patch("nhlbet.settle.shootout_in_total", lambda book=None, f=flag: f):
            d = shadow_resolved(store)
        out[name] = d[(d.market == "totals") & d.is_bet][["game_id", "strategy", "won", "profit"]].reset_index(drop=True) if len(d) else pd.DataFrame()
    a, b = out["so_adds_goal"], out["so_excluded"]
    if a.empty or b.empty:
        return {"totals_bets": 0}
    m = a.merge(b, on=["game_id", "strategy"], how="outer", suffixes=("_adds", "_excl"))
    changed = m[(m.won_adds != m.won_excl) & m.won_adds.notna() & m.won_excl.notna()]
    return {"totals_bets_so_adds_goal": int(len(a)), "totals_bets_so_excluded": int(len(b)), "bets_whose_result_changes": int(len(changed)),
            "bets_that_become_pushes_or_leave": int(m.won_adds.isna().sum() + m.won_excl.isna().sum()), "profit_so_adds_goal": float(a.profit.sum()), "profit_so_excluded": float(b.profit.sum())}
