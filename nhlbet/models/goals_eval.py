"""Walk-forward evaluation of the goals model (totals and puck line) against simple baselines. Retrains on strictly earlier games only.

No historical sportsbook lines exist in this project, so the test is against *base rates* and calibration at typical lines, not against
the market. The market comparison is what the paper-trading bets measure going forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from nhlbet.models.evaluate import _week_groups
from nhlbet.models.goals import GRID, GoalsModel, ScoreDistribution, _pmf
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
                            "lam_home": lam.lam_home, "lam_away": lam.lam_away, "base_home": Ftr.hr.mean(), "base_away": Ftr.ar.mean(), "disp": m.dispersion_}, index=Fte.index)
        for line in TOTAL_LINES:
            out[f"p_over_{line}"] = [d.total_probs(line)[0] / max(1e-9, 1 - d.total_probs(line)[2]) for d in ds]      # conditional on no push
            out[f"base_over_{line}"] = float((Ftr.tot > line).sum() / max(1, (Ftr.tot != line).sum()))
        for pt in SPREADS:
            out[f"p_cover_{pt}"] = [d.spread_probs(pt)[0] / max(1e-9, 1 - d.spread_probs(pt)[2]) for d in ds]
            out[f"base_cover_{pt}"] = float(((Ftr.mar + pt) > 0).mean())
        # alternative joint distributions for the same games, scored on the observed regulation score (see ``joint_table``): independent Poisson, independent negative
        # binomial, and the production grid with the tie correction
        hr_i, ar_i = np.clip(Fte.hr.astype(int).to_numpy(), 0, GRID - 1), np.clip(Fte.ar.astype(int).to_numpy(), 0, GRID - 1)
        variants = {"independent Poisson": lambda a, b, p: ScoreDistribution(a, b, 0.0, m.p_ot_, p, None),
                    "independent negative binomial": lambda a, b, p: ScoreDistribution(a, b, m.dispersion_, m.p_ot_, p, None)}
        out["ll_cell_production"] = [np.log(max(d.grid[i, j], 1e-12)) for d, i, j in zip(ds, hr_i, ar_i)]
        for name, mk in variants.items():
            phs = ph if ph is not None else [None] * len(Fte)
            out["ll_cell_" + name.replace(" ", "_")] = [np.log(max(mk(a, b, None if p is None or p != p else float(p)).grid[i, j], 1e-12))
                                                       for a, b, p, i, j in zip(lam.lam_home, lam.lam_away, phs, hr_i, ar_i)]
        # the whole totals PMF, scored against the observed total under the CURRENT shootout rule: the production model, the previous model (shootout goal excluded) and
        # the training base rate. Log score, ranked probability score and a randomised PIT (a discrete distribution needs the randomisation to be uniform when calibrated).
        K = 20
        obs = np.clip(Fte.tot.astype(int).to_numpy(), 0, K - 1)
        base = np.bincount(np.clip(Ftr.tot.astype(int).to_numpy(), 0, K - 1), minlength=K) + 0.5
        base = base / base.sum()
        rng = np.random.default_rng(0)
        for tag, pm in (("", [d.total_pmf(K) for d in ds]), ("_old", [d.total_pmf(K, so_in_total=False) for d in ds]), ("_base", [base] * len(ds))):
            out["ll_total" + tag] = [float(np.log(max(q[k], 1e-12))) for q, k in zip(pm, obs)]
            out["rps_total" + tag] = [_rps(q, k) for q, k in zip(pm, obs)]
            if tag == "":
                cdf = [np.cumsum(q) for q in pm]
                out["pit_total"] = [float((c[k - 1] if k > 0 else 0.0) + rng.random() * (c[k] - (c[k - 1] if k > 0 else 0.0))) for c, k in zip(cdf, obs)]
        out["p_total_mean"] = [d.expected_total() for d in ds]
        out["p_tie"] = [d.p_tie for d in ds]
        rows.append(out)
    return pd.concat(rows) if rows else pd.DataFrame()


def _rps(pmf, k: int) -> float:
    """Ranked probability score of a discrete forecast for the observed category ``k`` (0 = perfect), normalised by the number of categories minus one."""
    cdf = np.cumsum(np.asarray(pmf, float))
    obs = (np.arange(len(cdf)) >= k).astype(float)
    return float(np.sum((cdf - obs) ** 2) / (len(cdf) - 1))


def totals_pmf_table(P: pd.DataFrame, B: int = 2000) -> pd.DataFrame:
    """Whole-distribution scores of the totals forecast (higher log score and lower RPS are better) with a date-clustered interval for the paired difference to the
    training base rate: production model, previous model (shootout goal excluded), base rate."""
    from nhlbet.report.validation import cluster_bootstrap
    rows = []
    d = pd.to_datetime(P.game_date).to_numpy()
    for name, tag in (("production (current shootout rule)", ""), ("previous model (shootout goal excluded)", "_old"), ("training base rate", "_base")):
        ll, rp = P["ll_total" + tag].to_numpy(float), P["rps_total" + tag].to_numpy(float)
        dll, drp = ll - P["ll_total_base"].to_numpy(float), rp - P["rps_total_base"].to_numpy(float)
        lo, hi = cluster_bootstrap(drp, d, np.mean, B) if tag != "_base" else (0.0, 0.0)
        rows.append({"forecast": name, "n": int(len(P)), "log_score": float(ll.mean()), "rps": float(rp.mean()), "rps_minus_base": float(drp.mean()), "rps_diff_ci_low": lo, "rps_diff_ci_high": hi})
    return pd.DataFrame(rows)


def pit_table(P: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Histogram of the randomised PIT values of the totals forecast: a calibrated forecast gives equal counts. Includes a chi-square test against uniform."""
    from scipy.stats import chisquare
    u = P["pit_total"].dropna().to_numpy(float)
    h, _ = np.histogram(u, bins=bins, range=(0, 1))
    p = float(chisquare(h).pvalue)
    return pd.DataFrame({"pit_bin": [f"{i / bins:.1f}-{(i + 1) / bins:.1f}" for i in range(bins)], "count": h, "expected": len(u) / bins, "chi2_p_uniform": p})


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt_line(p, y, line) -> dict:
    """logit(p') = a + b*logit(p) + c*line: one slope b shared by all lines, an intercept that moves with the line.

    A pooled map without the line term fits the mixture of a 5.5 line (57% over) and a 6.5 line (44% over) and ends up with b near 1 although every
    single line is overconfident (per-line slopes 0.55-0.76); the line term lets b measure the shared overconfidence.
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return {"a": 0.0, "b": float("nan"), "c": 0.0, "n": int(len(y))}
    X = np.column_stack([_logit(p), np.asarray(line, float)])
    m = LogisticRegression(C=100.0).fit(X, y)
    return {"a": float(m.intercept_[0]), "b": float(m.coef_[0, 0]), "c": float(m.coef_[0, 1]), "n": int(len(y))}


def slope_se(p, y=None) -> float:
    """Standard error of a logistic calibration slope from the observed information: 1 / sqrt(sum w (x - mean_w x)^2) with w = q(1-q) and x = logit(p).

    When the model's probabilities barely vary (a totals model whose logits span about +-0.1), the slope is estimated very imprecisely (SE about 0.25 on 6,000
    games), so a slope of 0.5 versus 0.8 is not evidence of anything: read ``cal_slope`` together with this.
    """
    x = _logit(p)
    q = 1.0 / (1.0 + np.exp(-x))
    w = q * (1 - q)
    xbar = np.sum(w * x) / np.sum(w)
    return float(1.0 / np.sqrt(np.sum(w * (x - xbar) ** 2)))


def fit_platt(p, y) -> dict:
    """logit(p') = a + b*logit(p), fit by (lightly regularised) logistic regression. b near 1 = calibrated; b < 1 = overconfident; b near 0 = no signal."""
    from sklearn.linear_model import LogisticRegression
    if len(set(np.asarray(y, int).tolist())) < 2:          # a one-class outcome carries no calibration information
        return {"a": 0.0, "b": float("nan"), "n": int(len(y))}
    m = LogisticRegression(C=100.0).fit(_logit(p).reshape(-1, 1), np.asarray(y, int))
    return {"a": float(m.intercept_[0]), "b": float(m.coef_[0, 0]), "n": int(len(y))}


def _pooled(P: pd.DataFrame, kind: str, with_line: bool = False):
    """(p, y, date[, line]) stacked over the lines of one market type, pushes excluded. kind: 'totals' (5.5, 6.5) or 'spreads' (-1.5, +1.5)."""
    parts = []
    specs = [(f"p_over_{l}", P.tot > l, P.tot != l, l) for l in (5.5, 6.5)] if kind == "totals" else [(f"p_cover_{pt}", (P.mar + pt) > 0, (P.mar + pt) != 0, pt) for pt in SPREADS]
    for col, y, keep, line in specs:
        parts.append(pd.DataFrame({"p": P.loc[keep, col], "y": y[keep].astype(int), "d": P.loc[keep, "game_date"], "line": line}))
    Q = pd.concat(parts)
    return (Q.p, Q.y, Q.d, Q.line) if with_line else (Q.p, Q.y, Q.d)


def calibration_params(P: pd.DataFrame) -> dict:
    """Platt maps per market type from ALL walk-forward predictions: what production applies to live totals / puck-line probabilities."""
    out = {}
    for kind in ("totals", "spreads"):
        p, y, _, line = _pooled(P, kind, with_line=True)
        if len(p) >= 300:
            c = fit_platt_line(p, y, line)
            if np.isfinite(c["b"]):
                out[kind] = c
    return out


def online_calibrate_goals(P: pd.DataFrame, step_days: int = 14, burn_in: int = 1000) -> pd.DataFrame:
    """Add ``*_cal`` columns: each block's probabilities are mapped by a Platt fit on strictly EARLIER out-of-sample predictions only."""
    P = P.sort_values("game_date").copy()
    dates = pd.to_datetime(P.game_date)
    for kind, cols in (("totals", [f"p_over_{l}" for l in TOTAL_LINES]), ("spreads", [f"p_cover_{pt}" for pt in SPREADS])):
        p, y, d, ln = _pooled(P, kind, with_line=True)
        line_of = dict(zip(cols, (5.5, 6.0, 6.5) if kind == "totals" else SPREADS))
        for c in cols:
            P[c + "_cal"] = np.nan
        start, end = dates.min(), dates.max()
        while start <= end:
            stop = start + pd.Timedelta(days=step_days)
            past = pd.to_datetime(d) < start
            blk = ((dates >= start) & (dates < stop)).to_numpy()
            if past.sum() >= burn_in and blk.any():
                cal = fit_platt_line(p[past.to_numpy()], y[past.to_numpy()], ln[past.to_numpy()])
                for c in cols:
                    P.loc[blk, c + "_cal"] = 1.0 / (1.0 + np.exp(-(cal["a"] + cal["b"] * _logit(P.loc[blk, c]) + cal["c"] * line_of[c])))
            start = stop
    return P


def _ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _boot(d: np.ndarray, dates: pd.Series, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    groups = _week_groups(dates)
    rng = np.random.default_rng(seed)
    means = [d[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(P: pd.DataFrame, calibrated: bool = False) -> pd.DataFrame:
    """One row per market/line: log loss of the model vs the training base rate, paired difference with a week-cluster bootstrap CI.

    ``calibrated=True`` scores the ``*_cal`` columns from ``online_calibrate_goals`` (rows before the calibration burn-in are skipped)."""
    rows = []
    sfx = "_cal" if calibrated else ""
    specs = [(f"total over {l}", f"p_over_{l}{sfx}", f"base_over_{l}", (P.tot > l), (P.tot != l)) for l in TOTAL_LINES]
    specs += [(f"home {pt:+} cover", f"p_cover_{pt}{sfx}", f"base_cover_{pt}", ((P.mar + pt) > 0), ((P.mar + pt) != 0)) for pt in SPREADS]
    for name, pc, bc, y, keep in specs:
        if pc not in P:
            continue
        keep = keep & P[pc].notna()
        Q, y = P[keep], y[keep].astype(float).to_numpy()
        if len(Q) < 50:
            continue
        lm, lb = _ll(Q[pc], y), _ll(Q[bc], y)
        d = lm - lb
        lo, hi = _boot(d, Q.game_date)
        slope = fit_platt(Q[pc], y)["b"] if len(Q) > 100 else np.nan                  # logistic calibration slope (1 = calibrated)
        slope_err = slope_se(Q[pc]) if len(Q) > 100 else np.nan
        rows.append({"market": name, "n": len(Q), "hit_rate": float(y.mean()), "mean_p": float(Q[pc].mean()), "ll_model": float(lm.mean()),
                     "ll_base": float(lb.mean()), "diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "cal_slope": slope, "cal_slope_se": slope_err})
    return pd.DataFrame(rows)


def goal_rate_table(P: pd.DataFrame) -> pd.DataFrame:
    """Per-team-side goal log-likelihood (Poisson) of the model vs the training mean, and the mean predicted vs actual total."""
    rows = []
    for side, lam, base in (("home", "lam_home", "base_home"), ("away", "lam_away", "base_away")):
        y = P["hr" if side == "home" else "ar"].to_numpy()
        ll_m, ll_b = poisson.logpmf(y, P[lam]), poisson.logpmf(y, P[base])
        lo, hi = _boot(-(ll_m - ll_b), P.game_date)
        row = {"side": side, "n": len(P), "mean_pred": float(P[lam].mean()), "mean_actual": float(y.mean()), "nll_model": float(-ll_m.mean()),
               "nll_base": float(-ll_b.mean()), "diff": float(-(ll_m - ll_b).mean()), "ci_low": lo, "ci_high": hi}
        if "disp" in P:                                            # the same forecast scored as a full count distribution: ranked probability score (lower is better)
            yi = np.clip(y.astype(int), 0, GRID - 1)
            obs = (np.arange(GRID)[None, :] >= yi[:, None]).astype(float)
            for name, mu in (("rps_model", P[lam]), ("rps_base", P[base])):
                cdf = np.cumsum(np.vstack([_pmf(np.array([m_]), k_)[0] for m_, k_ in zip(mu, P["disp"])]), 1)
                row[name] = float(np.mean(np.sum((cdf - obs) ** 2, 1) / (GRID - 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def joint_table(P: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample log-likelihood of the observed regulation score (home goals, away goals) under three joint distributions fed the same expected goals.

    Higher is better. The distribution is chosen by this table, not by convention: the production grid (negative binomial plus a correction that moves probability
    onto ties, the Dixon-Coles-style fix for hockey's excess of tied regulation scores) is compared with independent Poisson and independent negative binomial.
    ``diff_vs_production`` is the per-game log-likelihood of the variant minus production (negative = production better), with a week-clustered bootstrap interval."""
    cols = {"production (negative binomial + tie correction)": "ll_cell_production", "independent negative binomial": "ll_cell_independent_negative_binomial",
            "independent Poisson": "ll_cell_independent_Poisson"}
    rows = []
    for name, c in cols.items():
        if c not in P:
            continue
        d = (P[c] - P["ll_cell_production"]).to_numpy(float)
        lo, hi = _boot(d, P.game_date) if name != next(iter(cols)) else (0.0, 0.0)
        rows.append({"joint distribution": name, "n": len(P), "mean_log_likelihood": float(P[c].mean()), "diff_vs_production": float(d.mean()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
