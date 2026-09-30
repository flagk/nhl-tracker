"""Feature selection and importance analysis (all time-ordered; no random splits).

Pipeline: drop mostly-missing / constant columns -> drop redundant (highly correlated) columns,
keeping the more predictive one -> walk-forward permutation importance (log-loss increase on
held-out future folds) -> drop features whose importance is not reliably positive -> SHAP for
direction/interpretation on the kept set.

To avoid selection leakage, run selection on an early window only (``selection_end``) and evaluate
models on data after it (see ``scripts/select_features.py``).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from nhlbet.features.builder import feature_group_map
from nhlbet.splits import expanding_folds

log = logging.getLogger(__name__)
META = {"game_date", "season", "game_type", "home", "away", "home_score", "away_score", "home_win", "start_utc"}


def candidate_columns(F: pd.DataFrame, max_missing: float = 0.5, prefer_diff: bool = True) -> list[str]:
    """Numeric feature columns that are populated and non-constant.

    With ``prefer_diff`` the raw ``h_``/``a_`` version of a feature is excluded whenever a home-minus-away
    differential ``d_`` exists: at ~1-3k games the three near-duplicates only split importance and add noise.
    """
    cols = []
    for c in F.columns:
        if c in META or F[c].dtype == object:
            continue
        if prefer_diff and c[:2] in ("h_", "a_") and f"d_{c[2:]}" in F.columns:
            continue
        s = pd.to_numeric(F[c], errors="coerce")
        if s.isna().mean() > max_missing or s.nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


def univariate_auc(F: pd.DataFrame, y: pd.Series, cols: list[str]) -> pd.Series:
    out = {}
    for c in cols:
        m = F[c].notna()
        out[c] = abs(roc_auc_score(y[m], F.loc[m, c]) - 0.5) if m.sum() > 50 and y[m].nunique() == 2 else 0.0
    return pd.Series(out)


def prune_correlated(F: pd.DataFrame, cols: list[str], score: pd.Series, threshold: float = 0.92) -> tuple[list[str], dict[str, str]]:
    """Greedy: visit columns by descending univariate score; drop any column correlated above threshold with a kept one."""
    corr = F[cols].corr(method="spearman").abs()
    kept, dropped = [], {}
    for c in score.reindex(cols).sort_values(ascending=False).index:
        hit = next((k for k in kept if corr.loc[c, k] > threshold), None)
        if hit is None:
            kept.append(c)
        else:
            dropped[c] = f"corr {corr.loc[c, hit]:.2f} with {hit}"
    return kept, dropped


def _model(seed: int = 0):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=1)


def walk_forward_permutation_importance(F: pd.DataFrame, y: pd.Series, cols: list[str], dates: pd.Series,
                                        n_folds: int = 5, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Mean log-loss increase when a feature is shuffled *within a future test fold*."""
    rng = np.random.default_rng(seed)
    X = F[cols].reset_index(drop=True); yv = y.reset_index(drop=True)
    rows = []
    for k, (tr, te) in enumerate(expanding_folds(dates, n_folds)):
        m = _model(seed).fit(X.iloc[tr], yv.iloc[tr])
        Xt, yt = X.iloc[te], yv.iloc[te]
        base = log_loss(yt, m.predict_proba(Xt)[:, 1], labels=[0, 1])
        for c in cols:
            inc = []
            for _ in range(n_repeats):
                Xp = Xt.copy(); Xp[c] = rng.permutation(Xp[c].to_numpy())
                inc.append(log_loss(yt, m.predict_proba(Xp)[:, 1], labels=[0, 1]) - base)
            rows.append({"fold": k, "feature": c, "delta_logloss": float(np.mean(inc))})
    df = pd.DataFrame(rows)
    g = df.groupby("feature").delta_logloss
    out = pd.DataFrame({"perm_mean": g.mean(), "perm_std": g.std(ddof=1), "folds_positive": g.apply(lambda s: (s > 0).mean())})
    return out.sort_values("perm_mean", ascending=False)


def shap_importance(F: pd.DataFrame, y: pd.Series, cols: list[str], dates: pd.Series, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Mean |SHAP| of out-of-sample predictions (each fold explained by a model trained only on its past)."""
    import shap
    X = F[cols].reset_index(drop=True); yv = y.reset_index(drop=True)
    vals, sgn = [], []
    for tr, te in expanding_folds(dates, n_folds):
        m = _model(seed).fit(X.iloc[tr], yv.iloc[tr])
        sv = shap.TreeExplainer(m).shap_values(X.iloc[te])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
        vals.append(np.abs(sv)); sgn.append(sv)
    a = np.vstack(vals); s = np.vstack(sgn)
    Xt = pd.concat([X.iloc[te] for _, te in expanding_folds(dates, n_folds)])
    direction = {c: float(np.sign(np.corrcoef(Xt[c].fillna(Xt[c].median()), s[:, i])[0, 1])) if s[:, i].std() > 0 else 0.0
                 for i, c in enumerate(cols)}
    return pd.DataFrame({"shap_mean_abs": a.mean(0), "shap_direction": pd.Series(direction)}, index=cols).sort_values("shap_mean_abs", ascending=False)


@dataclass
class SelectionResult:
    kept: list[str]
    report: pd.DataFrame
    dropped: dict[str, str]

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kept": self.kept, "dropped": self.dropped, "meta": meta or {}}, indent=2))


def select_features(F: pd.DataFrame, dates: pd.Series | None = None, corr_threshold: float = 0.85, n_folds: int = 6,
                    n_repeats: int = 8, min_folds_positive: float = 0.6, use_shap: bool = True, seed: int = 0) -> SelectionResult:
    """Run the full selection pipeline on ``F`` (must contain ``home_win`` and ``game_date``; pass only the selection window)."""
    F = F.sort_values("game_date").reset_index(drop=True)
    y = F["home_win"].astype(int)
    dates = F["game_date"] if dates is None else dates
    cols = candidate_columns(F)
    dropped: dict[str, str] = {c: ("raw h_/a_ column superseded by differential" if c[:2] in ("h_", "a_") and f"d_{c[2:]}" in F.columns
                                   else "missing >50% or constant")
                               for c in F.columns if c not in META and c not in cols and F[c].dtype != object}
    auc = univariate_auc(F, y, cols)
    cols2, corr_drop = prune_correlated(F, cols, auc, corr_threshold)
    dropped.update({c: f"redundant: {r}" for c, r in corr_drop.items()})
    perm = walk_forward_permutation_importance(F, y, cols2, dates, n_folds, n_repeats, seed)
    perm["univariate_auc_edge"] = auc.reindex(perm.index)
    gmap = feature_group_map(cols2)
    perm["group"] = pd.Series(gmap).reindex(perm.index)
    if use_shap:
        perm = perm.join(shap_importance(F, y, cols2, dates, n_folds, seed))
    keep_mask = (perm.perm_mean > 0) & (perm.folds_positive >= min_folds_positive)
    kept = list(perm.index[keep_mask])
    dropped.update({c: f"noisy: permutation importance {perm.loc[c, 'perm_mean']:+.5f} in {perm.loc[c, 'folds_positive']:.0%} of folds" for c in perm.index[~keep_mask]})
    return SelectionResult(kept, perm, dropped)
