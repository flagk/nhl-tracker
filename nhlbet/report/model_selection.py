"""Phase 4 model selection (rule written in reports/model_comparison_protocol.md BEFORE any result existed). Pure functions on prediction frames: the caller decides
which games are visible, so this module never reads the holdout lock."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.policy import policy_get
from nhlbet.report.evaluation import correct, dm_test, mean_ci_p

EPS = 1e-6


def logloss(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cal_slope(p, y) -> float:
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6).fit(z, np.asarray(y, int)).coef_[0, 0])


def compare_candidates(P: pd.DataFrame, reference: str, candidates: list[str], B: int | None = None) -> pd.DataFrame:
    """Per candidate: log loss, Brier, calibration slope, the paired loss difference to the reference (positive = candidate better) with a date-clustered bootstrap p-value
    and a Diebold-Mariano p-value, both Holm-corrected across the candidates, and whether the candidate qualifies under the decision rule."""
    B = int(B or policy_get("evaluation", "bootstrap_resamples", default=4000))
    alpha = float(policy_get("evaluation", "alpha", default=0.05))
    y = P.y.to_numpy(float)
    ref = logloss(P[reference], y)
    rows = [{"model": reference, "role": "reference", "n": int(len(P)), "log_loss": float(ref.mean()), "brier": float(np.mean((P[reference] - y) ** 2)), "cal_slope": cal_slope(P[reference], y),
             "gain": 0.0, "gain_lo": 0.0, "gain_hi": 0.0, "p_boot": np.nan, "p_dm": np.nan}]
    for c in candidates:
        loss = logloss(P[c], y)
        g = mean_ci_p(ref - loss, P.game_date.dt.strftime("%Y-%m-%d").to_numpy(), B)
        dm = dm_test(ref, loss)
        rows.append({"model": c, "role": "candidate", "n": int(len(P)), "log_loss": float(loss.mean()), "brier": float(np.mean((P[c] - y) ** 2)), "cal_slope": cal_slope(P[c], y),
                     "gain": g["mean"], "gain_lo": g["ci"][0], "gain_hi": g["ci"][1], "p_boot": g["p_not_above_0"], "p_dm": dm["p"]})
    t = pd.DataFrame(rows)
    m = t.role == "candidate"
    t["p_boot_adj"], t["p_dm_adj"] = np.nan, np.nan
    t.loc[m, "p_boot_adj"] = correct(t.loc[m, "p_boot"].to_numpy())
    t.loc[m, "p_dm_adj"] = correct(t.loc[m, "p_dm"].to_numpy())
    t["qualifies"] = m & (t.p_boot_adj < alpha) & (t.p_dm_adj < alpha) & (t.gain > 0)
    return t


def select_model(table: pd.DataFrame) -> str:
    """The decision rule: the qualifying candidate with the lowest log loss, otherwise the reference stays."""
    q = table[table.qualifies]
    return str(q.sort_values("log_loss").model.iloc[0]) if len(q) else str(table[table.role == "reference"].model.iloc[0])


def score_models(P: pd.DataFrame, names: list[str], baseline: str = "home_rate", B: int | None = None) -> pd.DataFrame:
    """Log loss, Brier and calibration slope of each model on ``P`` and its log-loss gain over the baseline with a date-clustered interval."""
    B = int(B or policy_get("evaluation", "bootstrap_resamples", default=4000))
    y = P.y.to_numpy(float)
    base = logloss(P[baseline], y)
    rows = []
    for n in names:
        loss = logloss(P[n], y)
        g = mean_ci_p(base - loss, P.game_date.dt.strftime("%Y-%m-%d").to_numpy(), B)
        rows.append({"model": n, "n": int(len(P)), "log_loss": float(loss.mean()), "brier": float(np.mean((P[n] - y) ** 2)), "cal_slope": cal_slope(P[n], y),
                     f"gain_over_{baseline}": g["mean"], "ci_low": g["ci"][0], "ci_high": g["ci"][1]})
    return pd.DataFrame(rows)
