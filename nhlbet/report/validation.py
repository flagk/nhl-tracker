"""Honest evaluation of the picks from the committed logs only (no database, no network): everything here can be re-run by an outsider.

Inputs are the files the daily job already publishes:
  data/export/games_predictions.csv  one row per game (latest pre-game run): model / market (at pick time) / closing probabilities, result
  data/export/paper_trading.csv      every settled fake bet (flat stake), with price, result, closing-line value
  reports/walkforward_predictions.csv  the out-of-sample backtest (no market prices exist for it)

Nothing is tuned here and nothing is chosen to look good: every number comes with its sample size, a clustered confidence interval and, for
"is this just luck?" questions, a permutation test against random picks at the same prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


# ------------------------------------------------------------------------------------------------ helpers
def _ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat=np.mean, B: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% interval of ``stat(values)`` resampling whole clusters (days) with replacement, because bets on one slate are not independent."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if len(values) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = {c: np.flatnonzero(clusters == c) for c in np.unique(clusters)}
    keys = list(idx)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        out[b] = stat(values[np.concatenate([idx[keys[i]] for i in pick])])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def longest_losing_streak(won: np.ndarray) -> int:
    best = cur = 0
    for w in np.asarray(won, float):
        cur = cur + 1 if w == 0 else 0
        best = max(best, cur)
    return int(best)


def max_drawdown(profit: np.ndarray) -> float:
    """Largest peak-to-trough fall of cumulative profit (same units as ``profit``)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(profit, float))])
    return float(np.max(np.maximum.accumulate(c) - c))


def required_n(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Observations needed to detect a true mean ``effect`` with a one-sided test, given a per-observation standard deviation ``sd``."""
    if effect <= 0 or sd <= 0:
        return 0
    return int(np.ceil(((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * sd / effect) ** 2))


# ------------------------------------------------------------------------------------------------ pick table
def pick_table(gp: pd.DataFrame, paper: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """One row per settled moneyline paper bet of ``strategy`` joined with the game's model / market probabilities.

    side probabilities: p_model, p_market (no-vig, at the time the pick was made), p_close (no-vig closing); edge = p_model - p_market.
    """
    b = paper[(paper.strategy == strategy) & (paper.market == "moneyline") & paper.won.notna()].copy()
    g = gp[["game_id", "p_model_home", "p_market_home", "p_close_home", "home_win"]]
    d = b.merge(g, on="game_id", how="inner")
    home = d.side == "home"
    d["p_model"] = np.where(home, d.p_model_home, 1 - d.p_model_home)
    d["p_market"] = np.where(home, d.p_market_home, 1 - d.p_market_home)
    d["p_close"] = np.where(home, d.p_close_home, 1 - d.p_close_home)
    d["edge"] = d.p_model - d.p_market
    d["clv_prob"] = d.p_close - d.p_market                      # did the line move toward our side after we picked?
    d["units"] = d.profit / d.stake
    d["overround"] = (1.0 / d.price) / d.p_market               # price paid vs the fair price (1.0 = fair, >1 = bookmaker margin paid)
    return d.sort_values(["date", "game_id"]).reset_index(drop=True)


def summarize_picks(d: pd.DataFrame, B: int = 4000) -> dict:
    if d.empty:
        return {"n": 0}
    stake, profit = d.stake.to_numpy(float), d.profit.to_numpy(float)
    roi = float(profit.sum() / stake.sum())
    lo, hi = cluster_bootstrap(profit / stake, d.date.to_numpy(), np.mean, B)
    t = stats.ttest_1samp(d.units, 0.0) if len(d) > 2 else None
    out = {"n": int(len(d)), "days": int(d.date.nunique()), "win_rate": float(d.won.mean()), "avg_price": float(d.price.mean()), "units": float(d.units.sum()),
           "roi": roi, "roi_ci": [lo, hi], "p_roi_not_above_0": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2) if t is not None else float("nan"),
           "max_drawdown_units": max_drawdown(d.units.to_numpy()), "longest_losing_streak": longest_losing_streak(d.won.to_numpy()),
           "mean_edge_vs_no_vig": float(d.edge.mean()), "break_even_win_rate": float(np.mean(1.0 / d.price)),
           "expected_win_rate_if_market_right": float(d.p_market.mean())}
    if d.clv_ev.notna().any():
        c = d.clv_ev.dropna().to_numpy(float)
        clo, chi = cluster_bootstrap(c, d.loc[d.clv_ev.notna(), "date"].to_numpy(), np.mean, B)
        cp = d.clv_prob.dropna().to_numpy(float)
        tt = stats.ttest_1samp(c, 0.0) if len(c) > 2 else None
        out["clv"] = {"n": int(len(c)), "mean_ev": float(c.mean()), "ci": [clo, chi], "share_positive": float((c > 0).mean()),
                      "mean_prob_points": float(cp.mean() * 100), "p_clv_not_above_0": float(tt.pvalue / 2 if tt.statistic > 0 else 1 - tt.pvalue / 2) if tt is not None else float("nan"),
                      "sd": float(c.std(ddof=1)) if len(c) > 1 else float("nan")}
    out["sd_unit_return"] = float(d.units.std(ddof=1)) if len(d) > 1 else float("nan")
    return out


def split_table(d: pd.DataFrame, by: str, bins=None) -> pd.DataFrame:
    """ROI / win rate / CLV by a grouping column ('month', 'side', 'role', 'odds_bucket')."""
    x = d.copy()
    x["month"] = x.date.str[:7]
    x["role"] = np.where(x.p_market >= 0.5, "favorite", "underdog")
    x["odds_bucket"] = pd.cut(x.price, [1.0, 1.6, 2.0, 2.5, 99], labels=["<1.60", "1.60-2.00", "2.00-2.50", ">2.50"], right=False)
    rows = []
    for k, g in x.groupby(by, observed=True):
        rows.append({by: str(k), "n": len(g), "win_rate": g.won.mean(), "roi": g.profit.sum() / g.stake.sum(), "units": g.units.sum(),
                     "avg_edge_pts": g.edge.mean() * 100, "clv_ev": g.clv_ev.mean() if g.clv_ev.notna().any() else float("nan")})
    return pd.DataFrame(rows)


def random_pick_baseline(d: pd.DataFrame, gp: pd.DataFrame, B: int = 10000, seed: int = 0) -> dict:
    """Random side in the same games at the same margin: how often does chance alone match the observed ROI?

    For each game the price of a side is rebuilt as 1 / (no-vig probability of that side x the margin actually paid on the real pick), so the only
    thing that differs between the real picks and the random ones is which side was chosen.
    """
    g = gp.set_index("game_id")
    d = d[d.game_id.isin(g.index)]
    if d.empty:
        return {"n": 0}
    ph = g.loc[d.game_id, "p_market_home"].to_numpy(float)
    hw = g.loc[d.game_id, "home_win"].to_numpy(float)
    r = d.overround.to_numpy(float)
    stake = d.stake.to_numpy(float)
    rng = np.random.default_rng(seed)
    dec_home, dec_away = 1.0 / (ph * r), 1.0 / ((1 - ph) * r)
    rois = np.empty(B)
    for b in range(B):
        home = rng.random(len(d)) < 0.5
        dec = np.where(home, dec_home, dec_away)
        win = np.where(home, hw == 1, hw == 0)
        rois[b] = np.sum(np.where(win, stake * (dec - 1), -stake)) / stake.sum()
    obs = float(d.profit.sum() / d.stake.sum())
    return {"n": int(len(d)), "observed_roi": obs, "random_mean_roi": float(rois.mean()), "random_roi_95": [float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))],
            "p_random_at_least_as_good": float((rois >= obs).mean())}


# ------------------------------------------------------------------------------------------------ calibration / scoring
def score_vs_market(gp: pd.DataFrame, B: int = 4000, seed: int = 0) -> dict:
    """Model vs the market's no-vig probability on the same games: log loss, Brier, calibration slope, paired clustered difference."""
    d = gp[gp.home_win.notna() & gp.p_model_home.notna() & gp.p_close_home.notna()].copy()
    if len(d) < 20:
        return {"n": int(len(d))}
    y = d.home_win.astype(int).to_numpy()
    cols = {"model": d.p_model_home, "market_at_pick": d.p_market_home, "market_close": d.p_close_home, "elo": d.elo_prob_home}
    base = float(np.mean(y))                                                          # in-sample base rate: generous to the baseline
    out = {"n": int(len(d)), "home_win_rate": base, "models": {}}
    for k, p in cols.items():
        p = p.to_numpy(float)
        ok = ~np.isnan(p)
        lr = stats.linregress(np.log(np.clip(p[ok], EPS, 1 - EPS) / (1 - np.clip(p[ok], EPS, 1 - EPS))), y[ok]) if ok.sum() > 10 else None
        out["models"][k] = {"log_loss": float(_ll(p[ok], y[ok]).mean()), "brier": float(np.mean((p[ok] - y[ok]) ** 2)), "n": int(ok.sum()),
                            "linear_slope_on_logit": float(lr.slope) if lr else float("nan")}
    out["models"]["home_rate"] = {"log_loss": float(_ll(np.full(len(y), base), y).mean()), "brier": float(np.mean((base - y) ** 2)), "n": len(y)}
    diff = _ll(d.p_model_home, y) - _ll(d.p_close_home, y)
    lo, hi = cluster_bootstrap(diff, d.game_date.to_numpy(), np.mean, B, seed)
    out["model_minus_close_log_loss"] = {"diff": float(diff.mean()), "ci": [lo, hi], "p_value_model_better": float(stats.norm.cdf(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))))}
    d["bin"] = pd.qcut(d.p_model_home, 5, duplicates="drop")
    out["reliability"] = [{"bin": str(k), "n": len(g), "model": g.p_model_home.mean(), "market_close": g.p_close_home.mean(), "actual": g.home_win.mean()}
                          for k, g in d.groupby("bin", observed=True)]
    return out


def backtest_by_period(P: pd.DataFrame, model: str, base: str = "elo", home: str = "home_rate") -> pd.DataFrame:
    """Walk-forward log loss by season half so a model that only worked in one stretch cannot hide in the pooled number."""
    P = P[P[model].notna()].copy()
    P["game_date"] = pd.to_datetime(P.game_date)
    P["period"] = np.where(P.game_date.dt.month >= 8, P.game_date.dt.year.astype(str) + "-" + (P.game_date.dt.year + 1).astype(str).str[2:],
                           (P.game_date.dt.year - 1).astype(str) + "-" + P.game_date.dt.year.astype(str).str[2:])
    rows = []
    for k, g in P.groupby("period"):
        y = g.y.to_numpy()
        a, b, c = _ll(g[model], y), _ll(g[base], y), _ll(g[home], y)
        lo, hi = cluster_bootstrap(a - b, g.game_date.dt.strftime("%Y-%W").to_numpy(), np.mean, 2000)
        rows.append({"period": k, "n": len(g), f"{model}": a.mean(), base: b.mean(), home: c.mean(), f"{model}-{base}": (a - b).mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ other markets and log integrity
def control_names() -> set[str]:
    """Strategies that are no-skill controls by construction (their description starts with CONTROL)."""
    from nhlbet.risk.parlays import PARLAY_STRATEGIES
    from nhlbet.risk.shadow import PROP_STRATEGIES, STRATEGIES
    return {s.name for s in (*STRATEGIES, *PROP_STRATEGIES, *PARLAY_STRATEGIES) if s.description.startswith("CONTROL")}


def market_summary(paper: pd.DataFrame, controls: bool = False) -> dict:
    """Per market type, one row per DISTINCT pick (a pick shared by several strategies counts once): win rate, ROI with a clustered interval, a one-sided t-test,
    the number of picks a 5% ROI would need to be detectable, and a plain verdict. CLV is only measurable for moneylines (one pre-game price capture exists for the rest)."""
    ctrl = control_names()
    oth = paper[paper.won.notna() & (paper.strategy.isin(ctrl) == controls)].copy()
    oth["key"] = oth.game_id.astype(str) + "|" + oth.team.astype(str) + "|" + oth.market
    oth = oth.drop_duplicates("key")
    out = {}
    for m, g in oth.groupby("market"):
        r = (g.profit / g.stake).to_numpy(float)
        lo, hi = cluster_bootstrap(r, g.date.to_numpy(), np.mean)
        t = stats.ttest_1samp(r, 0.0) if len(r) > 2 else None
        sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        verdict = ("too few picks (<30) to say anything" if len(g) < 30 else "insufficient evidence (interval spans 0)" if lo <= 0 <= hi else "credible edge" if lo > 0 else "credibly LOSING")
        out[m] = {"picks": int(len(g)), "win_rate": float(g.won.mean()), "roi": float(r.mean()), "roi_ci": [lo, hi], "units": float(g.profit.sum() / g.stake.mean()),
                  "p_roi_not_above_0": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2) if t is not None else float("nan"),
                  "picks_needed_for_5pct_roi": required_n(0.05, sd), "picks_needed_for_2pct_roi": required_n(0.02, sd), "verdict": verdict}
    return out


def log_integrity(logs_root, archive_dir) -> dict:
    """Checks on the append-only pick logs: nothing logged at or after puck drop, odds captured before the run and fresh, no duplicate rows, a model version on every pick.

    Start times come from the archived pages (the schedule is the same for every run of a day)."""
    import glob
    import json
    from pathlib import Path
    starts = {}
    for f in sorted(glob.glob(str(Path(archive_dir) / "*.html"))):
        h = Path(f).read_text()
        i = h.find('id="payload"')
        j, k = h.find(">", i) + 1, h.find("</script>", h.find(">", i) + 1)
        try:
            for g in json.loads(h[j:k].replace("<\\/", "</"))["games"]:
                starts[g["game_id"]] = pd.Timestamp(g["start_utc"])
        except (ValueError, KeyError):
            continue

    def load(d):
        fs = glob.glob(str(Path(logs_root) / d / "*.csv"))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

    res = {"games_with_known_start": len(starts)}
    keys = {"recommendations": ["run_id", "game_id"], "shadow_bets": ["run_id", "game_id", "strategy"], "prop_bets": ["run_id", "game_id", "strategy", "player_id", "side"],
           "alt_quotes": ["run_id", "game_id", "market", "side"], "parlay_bets": ["run_id", "strategy", "idx"]}
    for t, k in keys.items():
        r = load(t)
        if r.empty:
            continue
        row = {"rows": int(len(r)), "duplicate_keys": int(r.duplicated(k).sum())}
        if "game_id" in r:
            r["start"] = r.game_id.map(starts)
            ra = pd.to_datetime(r.run_at, utc=True)
            known = r.start.notna()
            row["rows_with_known_start"] = int(known.sum())
            row["logged_at_or_after_puck_drop"] = int((ra[known] >= r.start[known]).sum())
        if t == "recommendations":
            cap = pd.to_datetime(r.odds_captured_at, utc=True, errors="coerce")
            age = (pd.to_datetime(r.run_at, utc=True) - cap).dt.total_seconds() / 60
            row.update({"odds_age_minutes_median": float(age.median()), "odds_age_minutes_max": float(age.max()), "odds_captured_after_run": int((age < 0).sum()),
                        "rows_without_odds": int(cap.isna().sum()), "rows_without_model_version": int(r.model_version.isna().sum()), "model_versions": int(r.model_version.nunique())})
        res[t] = row
    return res


def run_validation(root=".", export_dir=None, report_dir=None, logs_dir=None, archive_dir=None) -> dict:
    """Everything above, from the committed files under ``root`` (see the module docstring). Returns a JSON-able dict; cheap enough to run on every daily job."""
    from pathlib import Path
    root = Path(root)
    export_dir, report_dir = Path(export_dir or root / "data/export"), Path(report_dir or root / "reports")
    logs_dir, archive_dir = Path(logs_dir or root / "data/logs"), Path(archive_dir or root / "site/archive")
    gp = pd.read_csv(export_dir / "games_predictions.csv")
    paper = pd.read_csv(export_dir / "paper_trading.csv")
    res: dict = {"data_through": str(gp.game_date.max()), "games_with_results": int(gp.home_win.notna().sum())}
    res["game_level"] = score_vs_market(gp)
    res["picks"], res["splits"], res["random_baseline"] = {}, {}, {}
    for strat in ("every_game", "flat_model_side", "market_favorite", "home_ml_control"):
        d = pick_table(gp, paper, strat)
        res["picks"][strat] = summarize_picks(d)
        if strat in ("every_game", "flat_model_side"):
            res["splits"][strat] = {k: split_table(d, k).round(4).to_dict("records") for k in ("month", "side", "role", "odds_bucket")}
            res["random_baseline"][strat] = random_pick_baseline(d, gp)
        if strat == "every_game":
            sd = res["picks"][strat].get("sd_unit_return", np.nan)
            csd = res["picks"][strat].get("clv", {}).get("sd", np.nan)
            res["required_bets"] = {"roi": {f"{e:.0%}": required_n(e, sd) for e in (0.01, 0.02, 0.03, 0.05)},
                                    "clv_ev": {f"{e:.1%}": required_n(e, csd) for e in (0.005, 0.01, 0.02)}}
    res["by_market"] = market_summary(paper)                           # the model's picks only
    res["controls_by_market"] = market_summary(paper, controls=True)   # no-skill controls: the price of the bookmaker margin alone
    res["log_integrity"] = log_integrity(logs_dir, archive_dir)
    wf = report_dir / "walkforward_predictions.csv"
    if wf.exists():
        P = pd.read_csv(wf, index_col=0)
        res["backtest_by_season"] = {m: backtest_by_period(P, m).round(4).to_dict("records") for m in ("logistic", "stack")}
    return res
