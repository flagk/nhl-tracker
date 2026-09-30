"""Walk-forward evaluation of the goals model (totals and puck line) against simple baselines. Retrains on strictly earlier games only.

No historical sportsbook lines exist in this project, so the test is against *base rates* and calibration at typical lines, not against
the market. The market comparison is what the paper-trading bets measure going forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from nhlbet.models.evaluate import _week_groups
from nhlbet.models.goals import GoalsModel
from nhlbet.splits import walk_forward_windows

TOTAL_LINES = (5.5, 6.0, 6.5)
SPREADS = (-1.5, 1.5)            # home point
EPS = 1e-6


def walk_forward_goals(F: pd.DataFrame, features: list[str], first_test: str, step_days: int = 14, min_train: int = 500,
                       alpha: float = 50.0, p_home: pd.Series | None = None) -> pd.DataFrame:
    """``F``: features + ``hr, ar, ot, so, tot, mar`` (see ``add_goal_targets``). ``p_home``: out-of-sample moneyline probabilities by game_id."""
    F = F[F.hr.notna()].sort_values("game_date")
    dates = F.game_date.reset_index(drop=True)
    rows = []
    for tr, te in walk_forward_windows(dates, first_test, step_days, min_train):
        Ftr, Fte = F.iloc[tr], F.iloc[te]
        m = GoalsModel(features, alpha).fit(Ftr, Ftr.hr, Ftr.ar, Ftr.ot, Ftr.so)
        ph = p_home.reindex(Fte.index).to_numpy() if p_home is not None else None
        ds = m.distributions(Fte, ph)
        lam = m.predict_lambdas(Fte)
        out = pd.DataFrame({"game_date": Fte.game_date, "hr": Fte.hr, "ar": Fte.ar, "tot": Fte.tot, "mar": Fte.mar,
                            "lam_home": lam.lam_home, "lam_away": lam.lam_away, "base_home": Ftr.hr.mean(), "base_away": Ftr.ar.mean()}, index=Fte.index)
        for line in TOTAL_LINES:
            out[f"p_over_{line}"] = [d.total_probs(line)[0] / max(1e-9, 1 - d.total_probs(line)[2]) for d in ds]      # conditional on no push
            out[f"base_over_{line}"] = float((Ftr.tot > line).sum() / max(1, (Ftr.tot != line).sum()))
        for pt in SPREADS:
            out[f"p_cover_{pt}"] = [d.spread_probs(pt)[0] / max(1e-9, 1 - d.spread_probs(pt)[2]) for d in ds]
            out[f"base_cover_{pt}"] = float(((Ftr.mar + pt) > 0).mean())
        out["p_total_mean"] = [d.expected_total() for d in ds]
        out["p_tie"] = [d.p_tie for d in ds]
        rows.append(out)
    return pd.concat(rows) if rows else pd.DataFrame()


def _ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _boot(d: np.ndarray, dates: pd.Series, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    groups = _week_groups(dates)
    rng = np.random.default_rng(seed)
    means = [d[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(P: pd.DataFrame) -> pd.DataFrame:
    """One row per market/line: log loss of the model vs the training base rate, paired difference with a week-cluster bootstrap CI."""
    rows = []
    specs = [(f"total over {l}", f"p_over_{l}", f"base_over_{l}", (P.tot > l), (P.tot != l)) for l in TOTAL_LINES]
    specs += [(f"home {pt:+} cover", f"p_cover_{pt}", f"base_cover_{pt}", ((P.mar + pt) > 0), ((P.mar + pt) != 0)) for pt in SPREADS]
    for name, pc, bc, y, keep in specs:
        Q, y = P[keep], y[keep].astype(float).to_numpy()
        if len(Q) < 50:
            continue
        lm, lb = _ll(Q[pc], y), _ll(Q[bc], y)
        d = lm - lb
        lo, hi = _boot(d, Q.game_date)
        slope = float(np.polyfit(np.log(np.clip(Q[pc], EPS, 1 - EPS) / (1 - np.clip(Q[pc], EPS, 1 - EPS))), y, 1)[0]) if len(Q) > 100 else np.nan
        rows.append({"market": name, "n": len(Q), "hit_rate": float(y.mean()), "mean_p": float(Q[pc].mean()), "ll_model": float(lm.mean()),
                     "ll_base": float(lb.mean()), "diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "lin_slope": slope})
    return pd.DataFrame(rows)


def goal_rate_table(P: pd.DataFrame) -> pd.DataFrame:
    """Per-team-side goal log-likelihood (Poisson) of the model vs the training mean, and the mean predicted vs actual total."""
    rows = []
    for side, lam, base in (("home", "lam_home", "base_home"), ("away", "lam_away", "base_away")):
        y = P["hr" if side == "home" else "ar"].to_numpy()
        ll_m, ll_b = poisson.logpmf(y, P[lam]), poisson.logpmf(y, P[base])
        lo, hi = _boot(-(ll_m - ll_b), P.game_date)
        rows.append({"side": side, "n": len(P), "mean_pred": float(P[lam].mean()), "mean_actual": float(y.mean()), "nll_model": float(-ll_m.mean()),
                     "nll_base": float(-ll_b.mean()), "diff": float(-(ll_m - ll_b).mean()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
