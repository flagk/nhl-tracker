"""Walk-forward evaluation of the shots-on-goal model against the player's own shrunk rate. Retrains on strictly earlier games only.

There are no historical prop lines in this project, so the test is on shot counts and on over/under hit rates at typical lines
(calibration), not against the market. The market test is the paper trading going forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from nhlbet.models.evaluate import _week_groups
from nhlbet.models.props import MIN_GAMES, ShotsModel, baseline_lam
from nhlbet.splits import walk_forward_windows

LINES_BY_STAT = {"sog": (1.5, 2.5, 3.5), "points": (0.5, 1.5), "assists": (0.5, 1.5), "goals": (0.5,), "saves": (22.5, 24.5, 26.5, 28.5)}
STAT_KMAX = {"saves": 60}          # the largest count scored separately; anything above shares the last bucket
LINES = LINES_BY_STAT["sog"]
EPS = 1e-6
KMAX = 14                       # counts at or above this share the last bucket when scoring the PMF


def walk_forward_props(feats: pd.DataFrame, first_test: str, step_days: int = 28, min_train: int = 5000, stat: str = "sog", shrink_k: float | None = None) -> pd.DataFrame:
    """``feats``: output of ``add_asof_features`` (needs ``opp_factor``). Returns out-of-sample rows with the model and baseline expectations.

    ``stat`` is 'sog', 'points', 'assists' or 'goals'; the outcome is in column ``y`` (and ``sog`` for shots, for older callers)."""
    f = feats.sort_values("game_date").reset_index(drop=True)
    dates = f.game_date
    parts = []
    for tr, te in walk_forward_windows(dates, first_test, step_days, min_train):
        m = ShotsModel(stat=stat, shrink_k=shrink_k).fit(f.iloc[tr])
        t = f.iloc[te]
        t = t[(t.n_prev >= MIN_GAMES) & t[stat].notna()]
        if t.empty:
            continue
        out = t[["game_id", "game_date", "player_id", "pos"]].copy()
        out["y"] = t[stat].to_numpy(float)
        if stat == "sog":
            out["sog"] = out["y"]
        out["lam"] = m.lam(t)
        out["base"] = baseline_lam(t, m.pos_mean_, stat, shrink_k)
        out["k"] = m.dispersion_
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _p_over(lam, line, k):
    lam = np.asarray(lam, float)
    kk = np.asarray(k, float)
    r = 1.0 / np.maximum(kk, 1e-9)
    nb = 1.0 - nbinom.cdf(np.floor(line), r, r / (r + lam))
    return np.where(kk <= 1e-9, 1.0 - poisson.cdf(np.floor(line), lam), nb)


def _pmf_matrix(lam, k, kmax: int = KMAX) -> np.ndarray:
    """(n, kmax+1) probability of 0..kmax-1 and 'kmax or more', negative binomial (Poisson when k ~ 0), one dispersion per row."""
    lam, kk = np.asarray(lam, float)[:, None], np.maximum(np.asarray(k, float), 1e-9)[:, None]
    x = np.arange(kmax)[None, :]
    r = 1.0 / kk
    pm = np.where(kk <= 1e-8, poisson.pmf(x, lam), nbinom.pmf(x, r, r / (r + lam)))
    return np.column_stack([pm, np.clip(1.0 - pm.sum(1), 0.0, 1.0)])


def rps(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score of an integer-count forecast (lower is better): squared distance between the forecast CDF and the outcome's step CDF."""
    obs = (np.arange(pmf.shape[1])[None, :] >= np.minimum(np.asarray(y, int), pmf.shape[1] - 1)[:, None]).astype(float)
    return np.sum((np.cumsum(pmf, 1) - obs) ** 2, 1) / (pmf.shape[1] - 1)


def _boot(d: np.ndarray, dates: pd.Series, B: int = 1500, seed: int = 0) -> tuple[float, float]:
    groups = _week_groups(dates)
    rng = np.random.default_rng(seed)
    means = [d[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _needed(d: np.ndarray) -> float:
    """Observations needed for a one-sided 5% test at 80% power to confirm the observed mean paired difference (inf when the model is not better)."""
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return float(np.ceil(((1.6449 + 0.8416) * sd / abs(m)) ** 2)) if m < 0 and sd > 0 else float("inf")


def evaluate(P: pd.DataFrame, stat: str = "sog") -> pd.DataFrame:
    """Count-PMF scores of the model vs the player's own shrunk rate, then one row per line (log loss of P(over)).

    Negative diff = the model beats the player's own rate. ``n_to_confirm`` is how many player-games a significance test would need at the observed
    effect size; compare it with ``n`` and read 'insufficient evidence' when n < n_to_confirm."""
    rows = []
    y = P.y.to_numpy(float) if "y" in P else P.sog.to_numpy(float)
    kmax = STAT_KMAX.get(stat, KMAX)
    pm, pb = _pmf_matrix(P.lam, P.k, kmax), _pmf_matrix(P.base, P.k, kmax)
    idx = np.minimum(y.astype(int), kmax)
    nll_m, nll_b = -np.log(np.maximum(pm[np.arange(len(y)), idx], 1e-12)), -np.log(np.maximum(pb[np.arange(len(y)), idx], 1e-12))
    d = nll_m - nll_b
    lo, hi = _boot(d, P.game_date)
    noun = {"sog": "shots", "points": "points", "assists": "assists", "goals": "goals", "saves": "saves"}[stat]
    rows.append({"what": f"{noun} per player-game (PMF log-likelihood, NLL)", "n": len(P), "model": float(nll_m.mean()), "baseline": float(nll_b.mean()), "diff": float(d.mean()),
                 "ci_low": lo, "ci_high": hi, "hit_rate": np.nan, "mean_p": np.nan, "n_to_confirm": _needed(d)})
    dr = rps(pm, y) - rps(pb, y)
    lo, hi = _boot(dr, P.game_date)
    rows.append({"what": f"{noun} per player-game (ranked probability score)", "n": len(P), "model": float(rps(pm, y).mean()), "baseline": float(rps(pb, y).mean()), "diff": float(dr.mean()),
                 "ci_low": lo, "ci_high": hi, "hit_rate": np.nan, "mean_p": np.nan, "n_to_confirm": _needed(dr)})
    for line in LINES_BY_STAT[stat]:
        sel = P[(P.lam > line * 0.45) & (P.lam < line * 1.8)] if stat in ("sog", "saves") else P[(P.lam > 0.05)]                 # players for whom this line is a realistic prop
        if len(sel) < 200:
            continue
        ys = ((sel.y if "y" in sel else sel.sog) > line).astype(float).to_numpy()
        pmo, pbo = np.clip(_p_over(sel.lam, line, sel.k), EPS, 1 - EPS), np.clip(_p_over(sel.base, line, sel.k), EPS, 1 - EPS)
        ll = lambda p: -(ys * np.log(p) + (1 - ys) * np.log(1 - p))  # noqa: E731
        dd = ll(pmo) - ll(pbo)
        lo, hi = _boot(dd, sel.game_date)
        rows.append({"what": f"over {line} (log loss)", "n": len(sel), "model": float(ll(pmo).mean()), "baseline": float(ll(pbo).mean()), "diff": float(dd.mean()), "ci_low": lo, "ci_high": hi,
                     "hit_rate": float(ys.mean()), "mean_p": float(pmo.mean()), "n_to_confirm": _needed(dd)})
    out = pd.DataFrame(rows)
    out["verdict"] = np.where(out["diff"] < 0, np.where(out["n"] >= out["n_to_confirm"], "skill vs own rate", "insufficient evidence"), "no skill vs own rate")
    return out


def calibration_table(P: pd.DataFrame, line: float = 2.5, bins: int = 8) -> pd.DataFrame:
    """Reliability of P(over ``line``): predicted vs actual by quantile bin, for one line (call it once per line)."""
    sel = P[(P.lam > line * 0.45) & (P.lam < line * 1.8)].copy() if line >= 1 else P[P.lam > 0.05].copy()
    sel["p"] = _p_over(sel.lam, line, sel.k)
    sel["yy"] = ((sel.y if "y" in sel else sel.sog) > line).astype(float)
    sel["bin"] = pd.qcut(sel.p, bins, duplicates="drop")
    t = sel.groupby("bin", observed=True).agg(n=("yy", "size"), predicted=("p", "mean"), actual=("yy", "mean")).reset_index(drop=True)
    return t
