"""Where is the model systematically wrong? (descriptive analytics with guards against seeing patterns in noise)

For each segment (home favourites, back-to-backs, early season, each team, ...) we test calibration-in-the-large: do home teams win more
or less often than the probabilities say? With ~100 segments, several will look "significant" by pure luck, so:

1. p-values are corrected across ALL segments tested (Benjamini-Hochberg false-discovery rate),
2. a flagged segment only counts as a **tendency** if it replicates: the same-direction bias must also appear in both halves of the data
   (first half / second half by date) - a real tendency persists, a fluke usually does not,
3. nothing here changes the model or the staking policy automatically; it is a report for a human to read.

The same analysis runs on the model's probabilities (``p_col='p_model_home'``) and, once enough games carry market prices, on the
market's probabilities, which shows where the market is wrong *relative to outcomes*, a much more useful signal for betting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _z(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(z, bias) for H0: y ~ Bernoulli(p). bias = actual - predicted frequency (positive: home teams win MORE than predicted)."""
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    var = float(np.sum(p * (1 - p)))
    return (float(np.sum(y - p) / np.sqrt(var)) if var > 0 else float("nan")), float(np.mean(y - p))


def segments(df: pd.DataFrame, p_col: str) -> list[tuple[str, str, np.ndarray]]:
    """(group, label, boolean mask). Only segments whose input columns exist are produced."""
    m: list[tuple[str, str, np.ndarray]] = []
    has = lambda *c: all(x in df and df[x].notna().any() for x in c)  # noqa: E731
    p = df[p_col]
    for lo, hi in ((0, .45), (.45, .5), (.5, .55), (.55, .6), (.6, 1.01)):
        m.append(("home win probability", f"{lo:.2f} to {min(hi, 1):.2f}", ((p >= lo) & (p < hi)).to_numpy()))
    m.append(("favourite", "home team favoured", (p >= .5).to_numpy()))
    m.append(("favourite", "away team favoured", (p < .5).to_numpy()))
    if has("home_b2b", "away_b2b"):
        hb, ab = df.home_b2b.fillna(0) == 1, df.away_b2b.fillna(0) == 1
        m += [("rest", "home on a back-to-back", hb.to_numpy()), ("rest", "away on a back-to-back", ab.to_numpy()),
              ("rest", "both on a back-to-back", (hb & ab).to_numpy())]
    if has("home_rest_days", "away_rest_days"):
        d = df.home_rest_days - df.away_rest_days
        m += [("rest", "home has 1+ more days of rest", (d >= 1).to_numpy()), ("rest", "away has 1+ more days of rest", (d <= -1).to_numpy())]
    if has("home_games_played", "away_games_played"):
        g = np.minimum(df.home_games_played, df.away_games_played)
        m += [("season stage", "first 15 games (ratings lean on last year)", (g < 15).to_numpy()), ("season stage", "games 15 to 60", ((g >= 15) & (g < 60)).to_numpy()),
              ("season stage", "after game 60", (g >= 60).to_numpy())]
    if has("is_rivalry"):
        m.append(("matchup", "division / rival game", (df.is_rivalry == 1).to_numpy()))
    if "game_date" in df:
        dt = pd.to_datetime(df.game_date)
        for mo in sorted(dt.dt.month.unique()):
            m.append(("month", dt[dt.dt.month == mo].dt.strftime("%B").iloc[0], (dt.dt.month == mo).to_numpy()))
        for dow in range(7):
            m.append(("weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow], (dt.dt.dayofweek == dow).to_numpy()))
    if "home" in df and "away" in df:
        for t in sorted(set(df.home) | set(df.away)):
            m.append(("team at home", t, (df.home == t).to_numpy()))
            m.append(("team away", t, (df.away == t).to_numpy()))
    return m


@dataclass
class TendencyResult:
    table: pd.DataFrame
    n_games: int
    n_tests: int
    alpha: float
    overall_bias: float


def analyze(df: pd.DataFrame, p_col: str = "p_model_home", y_col: str = "home_win", alpha: float = 0.10, min_n: int = 40) -> TendencyResult:
    """``df`` needs ``p_col``, ``y_col``, ``game_date`` and (optionally) the context columns used in :func:`segments`."""
    d = df[df[p_col].notna() & df[y_col].notna()].reset_index(drop=True)
    if len(d) == 0:
        return TendencyResult(pd.DataFrame(), 0, 0, alpha, float("nan"))
    y, p = d[y_col].to_numpy(float), d[p_col].to_numpy(float)
    cut = pd.to_datetime(d.game_date).sort_values().iloc[len(d) // 2]
    first = (pd.to_datetime(d.game_date) < cut).to_numpy()
    rows = []
    for group, label, mask in segments(d, p_col):
        n = int(mask.sum())
        if n < min_n:
            continue
        z, bias = _z(y[mask], p[mask])
        z1, b1 = _z(y[mask & first], p[mask & first])
        z2, b2 = _z(y[mask & ~first], p[mask & ~first])
        rows.append({"group": group, "segment": label, "n": n, "predicted": float(p[mask].mean()), "actual": float(y[mask].mean()), "bias": bias, "z": z,
                     "p": float(2 * stats.norm.sf(abs(z))), "n_first": int((mask & first).sum()), "bias_first": b1, "z_first": z1,
                     "n_second": int((mask & ~first).sum()), "bias_second": b2, "z_second": z2})
    t = pd.DataFrame(rows)
    if t.empty:
        return TendencyResult(t, len(d), 0, alpha, float(np.mean(y - p)))
    t["q"] = bh_qvalues(t.p.to_numpy())
    same = (np.sign(t.bias_first) == np.sign(t.bias)) & (np.sign(t.bias_second) == np.sign(t.bias))
    replicated = same & (t.z_second.abs() > 1.28) & (np.sign(t.z_second) == np.sign(t.z))        # second half shows it at roughly p < 0.2 two-sided
    t["verdict"] = np.where((t.q < alpha) & replicated, "TENDENCY (replicated)", np.where(t.q < alpha, "flagged, did not replicate", "no reliable tendency"))
    t = t.sort_values(["q", "p"]).reset_index(drop=True)
    return TendencyResult(t, len(d), len(t), alpha, float(np.mean(y - p)))


def render_markdown(results: dict[str, TendencyResult], generated: str) -> str:
    L = ["# Model tendencies", "",
         "> Research/education only. Descriptive analytics, **not** betting advice, and nothing here changes the model or the staking rules automatically.", "",
         f"*Generated {generated}.* For each segment we test whether home teams win more or less often than the probabilities say "
         "(`bias` = actual minus predicted home-win frequency, in percentage points). With many segments, some look significant by luck, so every "
         "p-value is corrected across all tests (Benjamini-Hochberg false-discovery rate) and a segment is only a **tendency** if the same-direction bias also "
         "shows up in both halves of the data (first/second half by date).", ""]
    for name, r in results.items():
        L += [f"## {name}", ""]
        if r.n_games == 0 or r.table.empty:
            L += ["Not enough data yet.", ""]
            continue
        t = r.table
        flagged = t[t.verdict == "TENDENCY (replicated)"]
        L += [f"**{r.n_games:,} games, {r.n_tests} segments tested.** Overall bias {r.overall_bias * 100:+.2f} pts. At a {r.alpha:.0%} false-discovery rate about "
              f"{max(1, round(r.alpha * max(len(t[t.q < r.alpha]), 1)))} of the flagged segments could still be flukes.", ""]
        if flagged.empty:
            L += ["**No replicated tendency.** Every apparent pattern is either consistent with chance after correcting for the number of tests, or did not hold in both halves. "
                  "That is the normal, expected result and is good news for the model's calibration.", ""]
        else:
            L += ["### Replicated tendencies", "", "| Segment | Games | Predicted | Actual | Bias | q-value | Bias: 1st half | 2nd half |", "|---|---|---|---|---|---|---|---|"]
            for x in flagged.itertuples():
                L += [f"| {x.group}: {x.segment} | {x.n} | {x.predicted:.1%} | {x.actual:.1%} | {x.bias * 100:+.1f} pts | {x.q:.3f} | {x.bias_first * 100:+.1f} | {x.bias_second * 100:+.1f} |"]
            L += [""]
        top = t.head(8)
        L += ["### Strongest signals (for context; most are noise)", "", "| Segment | Games | Bias | p | q | Verdict |", "|---|---|---|---|---|---|"]
        for x in top.itertuples():
            L += [f"| {x.group}: {x.segment} | {x.n} | {x.bias * 100:+.1f} pts | {x.p:.3f} | {x.q:.3f} | {x.verdict} |"]
        L += [""]
    return "\n".join(L)
