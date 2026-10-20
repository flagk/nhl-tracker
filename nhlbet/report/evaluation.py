"""Phase 3: honest evaluation of every pick, from decision-time prices, with closing-line value as the primary evidence.

* No-vig probabilities are recomputed here from the raw per-book quotes with the policy's PRIMARY method (proportional normalisation) and re-done with the sensitivity
  methods (power, Shin). Production edges use Shin, so this module never trusts a stored market probability.
* CLV for a bet = how far the closing no-vig probability of the bet's side moved from the decision-time no-vig probability (probability points and log-odds), margin free.
  It exists for moneylines, totals and puck lines whose line is still offered at the close; player props have no closing capture and are labelled so.
* Results are at decision-time prices (primary); the same bets re-priced at the best closing price are secondary.
* Intervals resample whole game dates. Significance is one-sided against zero and corrected for multiplicity (Holm by default) across markets and across split cells.
* Baselines: random sides in the same games, and the market favourite in the same games (a zero-edge pick).
* Below the policy's minimum sample a result is labelled "insufficient evidence" and never given a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from nhlbet.odds.math import devig
from nhlbet.policy import policy_get
from nhlbet.report.validation import control_names, required_n

EPS = 1e-6
PRICE_MARKETS = ("h2h", "totals", "spreads")


# ------------------------------------------------------------------------------------------------ multiplicity
def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(p, float)
    order = np.argsort(p)[::-1]
    m = len(p)
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = min(1.0, run)
    return adj


def correct(p, how: str | None = None) -> np.ndarray:
    how = how or policy_get("evaluation", "multiple_comparisons", default="holm")
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    if ok.any():
        out[ok] = (benjamini_hochberg if how in ("bh", "benjamini_hochberg") else holm)(p[ok])
    return out


# ------------------------------------------------------------------------------------------------ market probabilities from raw quotes
def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def leg_pairs(rows: pd.DataFrame, home: str, away: str, market: str, side: str, point) -> dict:
    """book -> (decimal of the leg, decimal of the other side) at the leg's own line, from the quotes of ONE capture."""
    from nhlbet.review.odds_check import _two_way
    return _two_way(rows.to_dict("records"), {"home": home, "away": away}, market, side, 0.0 if point is None or (isinstance(point, float) and np.isnan(point)) else float(point))


def leg_prob(pairs: dict, method: str) -> tuple[float, float, int]:
    """(consensus no-vig probability of the leg, best decimal price for the leg, number of books)."""
    if not pairs:
        return float("nan"), float("nan"), 0
    ps = [float(devig([a, b], method)[0]) for a, b in pairs.values()]
    return float(np.mean(ps)), float(max(a for a, _ in pairs.values())), len(pairs)


class Quotes:
    """Raw per-book quotes indexed by game for fast as-of lookups. ``df`` = odds_snapshots rows (captured_at, game_id, home, away, book, market, outcome, point, price)."""

    def __init__(self, df: pd.DataFrame) -> None:
        d = df[df.market.isin(PRICE_MARKETS) & df.game_id.notna()].copy()
        d["cap"] = pd.to_datetime(d.captured_at, utc=True, errors="coerce", format="ISO8601")
        d["game_id"] = d.game_id.astype(int)
        self.by_game = {g: x.sort_values("cap") for g, x in d.groupby("game_id")}

    def capture(self, game_id: int, market: str, when, before=None, after=None) -> pd.DataFrame | None:
        """The quotes of the latest capture of ``market`` at or before ``when`` (and strictly before ``before``, strictly after ``after``); None if there is none."""
        g = self.by_game.get(int(game_id))
        if g is None:
            return None
        g = g[g.market == market]
        g = g[g.cap <= pd.Timestamp(when)]
        if before is not None:
            g = g[g.cap < pd.Timestamp(before)]
        if after is not None:
            g = g[g.cap > pd.Timestamp(after)]
        if g.empty:
            return None
        return g[g.cap == g.cap.max()]


def prob_at(q: Quotes, game_id, home, away, market, side, point, when, method, after=None):
    """(probability, best price, books, capture time) of the leg at the latest capture <= when."""
    rows = q.capture(game_id, market, when, after=after)
    if rows is None:
        return float("nan"), float("nan"), 0, None
    p, best, n = leg_prob(leg_pairs(rows, home, away, market, side, point), method)
    return p, best, n, rows.cap.iloc[0]


def clv_between(p_decision: float, p_close: float) -> dict:
    """Margin-free closing-line value of a bet: how far the no-vig probability of the bet's side moved between the decision and the close."""
    if np.isnan(p_decision) or np.isnan(p_close):
        return {"clv_pts": np.nan, "clv_logodds": np.nan}
    return {"clv_pts": 100.0 * (p_close - p_decision), "clv_logodds": float(_logit(p_close) - _logit(p_decision))}


# ------------------------------------------------------------------------------------------------ the pick table
def pick_frame(store, max_close_gap_min: float | None = None) -> pd.DataFrame:
    """One row per DISTINCT settled paper pick (a pick shared by several strategies counts once, in the group of its first strategy), with decision-time and
    closing no-vig probabilities under every method in the policy, CLV, and the result at decision-time prices and at the best closing price."""
    from nhlbet.report.betlog import shadow_resolved
    gap = float(max_close_gap_min if max_close_gap_min is not None else policy_get("evaluation", "max_close_gap_min", default=60.0))
    d = shadow_resolved(store)
    if d.empty:
        return d
    d = d[d.is_bet | d.push].copy()
    ctrl = control_names()
    d["group"] = np.where(d.strategy.isin(ctrl), "control", "model")
    d["market"] = d.market.fillna("h2h")
    d["player_id"] = d.player_id.fillna(0)
    d["point"] = d.point.astype(float) if "point" in d else np.nan
    d = d.sort_values("run_at").drop_duplicates(["group", "game_id", "market", "side", "point", "player_id"], keep="last").reset_index(drop=True)
    return _attach_market(d, store, gap)


def _attach_market(d: pd.DataFrame, store, gap: float) -> pd.DataFrame:
    """Add date, month, decision-time and closing no-vig probabilities for every policy method, CLV, the result at decision-time and closing prices."""
    g = store.df("SELECT game_id, home, away, start_utc, game_date AS date FROM games")
    d = d.drop(columns=[c for c in ("home", "away", "start_utc", "date", "month") if c in d]).merge(g, on="game_id", how="left")
    d["date"] = d.date.fillna(d.game_date).astype(str)
    d["month"] = d.date.str[:7]
    q = Quotes(store.df("SELECT captured_at, game_id, home, away, book, market, outcome, point, price FROM odds_snapshots WHERE market IN ('h2h','totals','spreads')"))
    methods = [policy_get("evaluation", "devig", "primary", default="proportional")] + list(policy_get("evaluation", "devig", "sensitivity", default=["power", "shin"]))
    recs = []
    for r in d.itertuples():
        mk = {"h2h": "h2h", "totals": "totals", "spreads": "spreads"}.get(r.market)
        out = {}
        if mk and pd.notna(r.start_utc):
            start = pd.Timestamp(r.start_utc)
            when = min(pd.Timestamp(r.run_at), start)
            for m in methods:
                p0, _, n0, t0 = prob_at(q, r.game_id, r.home, r.away, mk, r.side, r.point, when, m)
                close_rows = q.capture(r.game_id, mk, start, after=t0 if t0 is not None else None)
                p1, best1, t1 = float("nan"), float("nan"), None
                if close_rows is not None and (start - close_rows.cap.iloc[0]).total_seconds() / 60 <= gap:      # a "closing" capture hours before the start is not a closing price
                    p1, best1, _ = leg_prob(leg_pairs(close_rows, r.home, r.away, mk, r.side, r.point), m)
                    t1 = close_rows.cap.iloc[0]
                c = clv_between(p0, p1)
                out.update({f"p_dec_{m}": p0, f"p_close_{m}": p1, f"clv_pts_{m}": c["clv_pts"], f"clv_lo_{m}": c["clv_logodds"], "best_close": best1, "n_books_dec": n0})
        recs.append(out)
    d = pd.concat([d, pd.DataFrame(recs, index=d.index)], axis=1)
    d["settled"] = d.is_bet
    d["unit_return"] = np.where(d.is_bet, d.profit / d.stake, np.nan)
    won = d.won.to_numpy(float)
    d["unit_return_at_close"] = np.where(d.is_bet & d.best_close.notna() if "best_close" in d else False, np.where(won == 1, d.best_close - 1, -1.0), np.nan)
    d["edge_primary"] = d.p_model - d[f"p_dec_{methods[0]}"] if f"p_dec_{methods[0]}" in d else d.p_model - d.p_market
    d["favorite"] = d.p_market > 0.5
    return d


def official_picks(store, max_close_gap_min: float | None = None) -> pd.DataFrame:
    """The live policy's own bets (recommendations with action BET, one per game), run through the same market-probability and CLV machinery."""
    from nhlbet.report.betlog import resolved
    gap = float(max_close_gap_min if max_close_gap_min is not None else policy_get("evaluation", "max_close_gap_min", default=60.0))
    r = resolved(store)
    if r.empty:
        return r
    r = r[r.is_bet].copy()
    if r.empty:
        return r
    r["group"], r["market"], r["strategy"], r["push"], r["player_id"], r["point"] = "official", "h2h", "official", False, 0, np.nan
    r["p_model"] = r.p_adj.fillna(r.p_model) if "p_adj" in r else r.p_model
    r["p_market"] = np.where(r.side == "home", r.p_market, 1 - r.p_market)
    r["won"] = r.bet_won
    return _attach_market(r.rename(columns={"game_date": "game_date"}), store, gap)

# ------------------------------------------------------------------------------------------------ summaries
def cluster_dist(values, clusters, B: int, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling whole clusters (game dates)."""
    v, c = np.asarray(values, float), np.asarray(clusters)
    keys = np.unique(c)
    idx = {k: np.flatnonzero(c == k) for k in keys}
    rng = np.random.default_rng(seed)
    sums = np.array([v[idx[k]].sum() for k in keys])
    cnt = np.array([len(idx[k]) for k in keys], float)
    draws = rng.integers(0, len(keys), (B, len(keys)))
    return sums[draws].sum(1) / cnt[draws].sum(1)


def mean_ci_p(values, clusters, B: int, seed: int = 0) -> dict:
    """Mean with a 95% clustered interval and a one-sided p-value for 'mean > 0' (share of resampled means at or below 0)."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    v, c = v[ok], np.asarray(clusters)[ok]
    if len(v) < 5:
        return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan, "ci": [np.nan, np.nan], "p_not_above_0": np.nan}
    dist = cluster_dist(v, c, B, seed)
    return {"n": int(len(v)), "mean": float(v.mean()), "ci": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))], "p_not_above_0": float((np.sum(dist <= 0) + 1) / (B + 1))}


def verdict(n: int, p_adj: float, mean: float, minimum: int, what: str) -> str:
    alpha = float(policy_get("evaluation", "alpha", default=0.05))
    if n < minimum:
        return f"insufficient evidence (n={n} < {minimum})"
    if np.isnan(p_adj):
        return "insufficient evidence"
    if p_adj < alpha and mean > 0:
        return f"credible positive {what} (adjusted p={p_adj:.3f})"
    if p_adj < alpha and mean < 0:
        return f"credibly negative {what}"
    return f"no demonstrable edge on {what} (adjusted p={p_adj:.2f})"


def market_table(d: pd.DataFrame, method: str, group: str = "model") -> pd.DataFrame:
    """Per market: ROI and CLV with clustered intervals, Holm-corrected p-values across markets, and the policy's sample-size label."""
    B = int(policy_get("evaluation", "bootstrap_resamples", default=4000))
    mn = int(policy_get("evaluation", "min_picks_per_market", default=100))
    mnc = int(policy_get("evaluation", "min_clv_picks", default=30))
    rows = []
    for m, x in d[d.group == group].groupby("market"):
        s = x[x.settled]
        roi = mean_ci_p(s.unit_return, s.date, B)
        clv = mean_ci_p(s.get(f"clv_pts_{method}", pd.Series(np.nan, index=s.index)), s.date, B)
        lo = mean_ci_p(s.get(f"clv_lo_{method}", pd.Series(np.nan, index=s.index)), s.date, B)
        sd = float(np.nanstd(s.unit_return, ddof=1)) if len(s) > 1 else np.nan
        rows.append({"market": m, "picks": int(len(s)), "pushes_or_void": int(len(x) - len(s)), "win_rate": float(s.won.mean()) if len(s) else np.nan, "roi": roi["mean"], "roi_lo": roi["ci"][0], "roi_hi": roi["ci"][1],
                     "p_roi": roi["p_not_above_0"], "clv_n": clv["n"], "clv_pts": clv["mean"], "clv_lo": clv["ci"][0], "clv_hi": clv["ci"][1], "p_clv": clv["p_not_above_0"],
                     "clv_logodds": lo["mean"], "roi_at_closing_price": float(s.unit_return_at_close.mean()) if s.unit_return_at_close.notna().any() else np.nan,
                     "picks_for_5pct_roi": required_n(0.05, sd) if sd == sd else np.nan, "min_picks": mn, "min_clv_picks": mnc})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p_roi_adj"], t["p_clv_adj"] = correct(t.p_roi), correct(t.p_clv)
    t["verdict_roi"] = [verdict(n, p, m, mn, "ROI") for n, p, m in zip(t.picks, t.p_roi_adj, t.roi)]
    t["verdict_clv"] = ["CLV not measurable (no closing price captured)" if c == 0 else verdict(c, p, m, mnc, "CLV") for c, p, m in zip(t.clv_n, t.p_clv_adj, t.clv_pts)]
    return t


def split_cells(d: pd.DataFrame, method: str, group: str = "model") -> pd.DataFrame:
    """ROI and CLV by market, month, side (home/away or over/under), favourite/underdog and odds bucket, with Holm correction over ALL cells that have the minimum size."""
    B = int(policy_get("evaluation", "bootstrap_resamples", default=4000))
    mnc = int(policy_get("evaluation", "min_clv_picks", default=30))
    s = d[(d.group == group) & d.settled].copy()
    s["home_away"] = np.where(s.side.isin(["home", "over"]), "home/over", "away/under")
    s["price_bucket"] = pd.cut(s.decimal, [1.0, 1.5, 1.8, 2.2, 3.0, 100.0], labels=["<1.5", "1.5-1.8", "1.8-2.2", "2.2-3.0", ">3.0"]).astype(str)
    s["fav_dog"] = np.where(s.favorite, "favourite", "underdog")
    rows = []
    for dim in ("market", "month", "home_away", "fav_dog", "price_bucket"):
        for k, x in s.groupby(dim):
            roi = mean_ci_p(x.unit_return, x.date, B)
            clv = mean_ci_p(x.get(f"clv_pts_{method}", pd.Series(np.nan, index=x.index)), x.date, B)
            rows.append({"split": dim, "cell": str(k), "n": int(len(x)), "roi": roi["mean"], "roi_lo": roi["ci"][0], "roi_hi": roi["ci"][1], "p_roi": roi["p_not_above_0"],
                         "clv_n": clv["n"], "clv_pts": clv["mean"], "p_clv": clv["p_not_above_0"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    big = t.n >= mnc
    t["p_roi_adj"] = np.nan
    t["p_clv_adj"] = np.nan
    t.loc[big, "p_roi_adj"] = correct(t.loc[big, "p_roi"].to_numpy())
    t.loc[big & (t.clv_n >= mnc), "p_clv_adj"] = correct(t.loc[big & (t.clv_n >= mnc), "p_clv"].to_numpy())
    t["label"] = np.where(~big, f"insufficient evidence (n<{mnc})", np.where(t.p_roi_adj < float(policy_get("evaluation", "alpha", default=0.05)), "significant after correction", "not significant after correction"))
    t["family_size"] = int(big.sum())
    return t


def edge_threshold_table(d: pd.DataFrame, method: str, group: str = "model", market: str | None = None) -> pd.DataFrame:
    """ROI and CLV of the picks whose model edge (model minus the primary no-vig market probability) is at least t, for each policy threshold t."""
    B = int(policy_get("evaluation", "bootstrap_resamples", default=2000))
    s = d[(d.group == group) & d.settled & d.edge_primary.notna()]
    if market:
        s = s[s.market == market]
    rows = []
    for t in policy_get("evaluation", "edge_thresholds", default=[0.0, 0.03]):
        x = s[s.edge_primary >= t]
        roi = mean_ci_p(x.unit_return, x.date, B)
        clv = mean_ci_p(x.get(f"clv_pts_{method}", pd.Series(np.nan, index=x.index)), x.date, B)
        rows.append({"min_edge": t, "picks": int(len(x)), "roi": roi["mean"], "roi_lo": roi["ci"][0], "roi_hi": roi["ci"][1], "clv_n": clv["n"], "clv_pts": clv["mean"], "clv_lo": clv["ci"][0], "clv_hi": clv["ci"][1]})
    return pd.DataFrame(rows)


def method_sensitivity(d: pd.DataFrame, group: str = "model") -> pd.DataFrame:
    """The same CLV under each no-vig method: if the sign or size depends on the method, the result is not robust."""
    B = int(policy_get("evaluation", "bootstrap_resamples", default=2000))
    s = d[(d.group == group) & d.settled]
    rows = []
    for m in [policy_get("evaluation", "devig", "primary", default="proportional")] + list(policy_get("evaluation", "devig", "sensitivity", default=["power", "shin"])):
        for mk, x in s.groupby("market"):
            c = mean_ci_p(x.get(f"clv_pts_{m}", pd.Series(np.nan, index=x.index)), x.date, B)
            if c["n"]:
                rows.append({"method": m, "market": mk, "clv_n": c["n"], "clv_pts": c["mean"], "clv_lo": c["ci"][0], "clv_hi": c["ci"][1]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ baselines
def baselines(store, d: pd.DataFrame, method: str, B: int = 5000, seed: int = 0) -> dict:
    """Random sides in the same games at the real best prices, and the market favourite in the same games, over the model's moneyline / total / puck-line picks.

    Needs both sides' prices, so it is rebuilt from the raw quotes at each pick's decision time. ``p_random_at_least_as_good`` is the share of random draws whose ROI
    (or CLV) matched or beat the model picks'."""
    s = d[(d.group == "model") & d.settled & d.market.isin(PRICE_MARKETS) & d.get(f"p_dec_{method}", pd.Series(np.nan, index=d.index)).notna()]
    if s.empty:
        return {"n": 0}
    q = Quotes(store.df("SELECT captured_at, game_id, home, away, book, market, outcome, point, price FROM odds_snapshots WHERE market IN ('h2h','totals','spreads')"))
    games = store.df("SELECT game_id, home, away, start_utc FROM games").set_index("game_id")
    opp = {"home": "away", "away": "home", "over": "under", "under": "over"}
    rows = []
    for r in s.itertuples():
        when = min(pd.Timestamp(r.run_at), pd.Timestamp(r.start_utc))
        cap = q.capture(r.game_id, r.market, when)
        if cap is None:
            continue
        pt_opp = r.point if r.market == "totals" or r.market == "h2h" else -r.point
        a = leg_pairs(cap, r.home, r.away, r.market, r.side, r.point)
        b = leg_pairs(cap, r.home, r.away, r.market, opp[r.side], pt_opp)
        pa, ba, _ = leg_prob(a, method)
        pb, bb, _ = leg_prob(b, method)
        if np.isnan(pa) or np.isnan(pb):
            continue
        rows.append({"game_id": r.game_id, "date": r.date, "market": r.market, "dec_pick": ba, "dec_other": bb, "p_pick": pa, "won_pick": r.won, "unit_pick": r.unit_return,
                     "won_other": 1.0 - r.won if r.won in (0.0, 1.0) else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"n": 0}
    t["unit_other"] = np.where(t.won_other == 1, t.dec_other - 1, -1.0)
    rng = np.random.default_rng(seed)
    pick_roi = float(t.unit_pick.mean())
    flips = rng.random((B, len(t))) < 0.5
    rand = np.where(flips, t.unit_pick.to_numpy()[None, :], t.unit_other.to_numpy()[None, :]).mean(1)
    fav_pick = t.p_pick >= 0.5
    fav = np.where(fav_pick, t.unit_pick, t.unit_other)
    return {"n": int(len(t)), "model_roi": pick_roi, "random_side_mean_roi": float(rand.mean()), "random_side_roi_95": [float(np.percentile(rand, 2.5)), float(np.percentile(rand, 97.5))],
            "p_random_at_least_as_good": float((np.sum(rand >= pick_roi) + 1) / (B + 1)), "market_favourite_roi": float(fav.mean()),
            "note": "random side and favourite are priced at the best decision-time price of that side; the market favourite is the side with the higher no-vig probability"}


# ------------------------------------------------------------------------------------------------ model vs market scoring (game level)
def score_games(store, method: str, B: int = 4000, seed: int = 0) -> dict:
    """Log loss and Brier of the model versus the no-vig market (at the decision and at the close) on the same finished games, with a paired date-clustered interval
    for the log-loss difference and reliability of both."""
    d = live_game_frame(store, method)
    if len(d) < 20:
        return {"n": int(len(d))}
    return _score_frame(d, method, B)


def live_game_frame(store, method: str) -> pd.DataFrame:
    """Finished games with the model's final pre-game probability, the no-vig market probability at the decision and at the close (columns p_model, p_market, p_close, home_win, date)."""
    from nhlbet.report.betlog import final_recommendations
    r = final_recommendations(store)
    if r.empty:
        return pd.DataFrame()
    g = store.df("SELECT game_id, game_date AS date, home, away, start_utc, home_win FROM games WHERE home_win IS NOT NULL")
    d = r.drop(columns=[c for c in ("home", "away", "start_utc") if c in r]).merge(g, on="game_id")
    q = Quotes(store.df("SELECT captured_at, game_id, home, away, book, market, outcome, point, price FROM odds_snapshots WHERE market='h2h'"))
    pm, pc = [], []
    for x in d.itertuples():
        start = pd.Timestamp(x.start_utc)
        p0, _, _, t0 = prob_at(q, x.game_id, x.home, x.away, "h2h", "home", None, min(pd.Timestamp(x.run_at), start), method)
        cl = q.capture(x.game_id, "h2h", start)
        p1 = leg_prob(leg_pairs(cl, x.home, x.away, "h2h", "home", None), method)[0] if cl is not None and (start - cl.cap.iloc[0]).total_seconds() / 60 <= 60 else np.nan
        pm.append(p0)
        pc.append(p1)
    d["p_market"], d["p_close"] = pm, pc
    return d[d.p_market.notna() & d.p_close.notna() & d.p_model.notna()]


def _score_frame(d: pd.DataFrame, method: str, B: int) -> dict:
    y = d.home_win.astype(float).to_numpy()
    ll = lambda p: -(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
    out = {"n": int(len(d)), "method": method}
    for name, p in (("model", d.p_model.to_numpy(float)), ("market_at_decision", d.p_market.to_numpy(float)), ("market_close", d.p_close.to_numpy(float))):
        out[name] = {"log_loss": float(ll(p).mean()), "brier": float(np.mean((p - y) ** 2))}
    for a, b in (("model", "market_at_decision"), ("model", "market_close")):
        diff = ll(d.p_model.to_numpy(float)) - ll(d.p_market.to_numpy(float) if b == "market_at_decision" else d.p_close.to_numpy(float))
        res = mean_ci_p(-diff, d.date.to_numpy(), B)                      # positive = the model scored better than that market
        out[f"{a}_vs_{b}_log_loss_gain"] = {"mean_gain": res["mean"], "ci": res["ci"], "p_model_not_better": res["p_not_above_0"]}
    bins = pd.qcut(d.p_model, 5, duplicates="drop")
    out["reliability"] = [{"bin": str(k), "n": int(len(x)), "model": float(x.p_model.mean()), "market_at_decision": float(x.p_market.mean()), "actual": float(x.home_win.mean())} for k, x in d.groupby(bins, observed=True)]
    return out


def blend_scores(d: pd.DataFrame, weights=(0.0, 0.25, 0.5, 0.75, 1.0), B: int = 4000) -> pd.DataFrame:
    """Log loss of p = (1 - w) * market + w * model for fixed weights w (the market at the decision time), with a date-clustered interval for the gain over the market alone."""
    y = d.home_win.astype(float).to_numpy()
    ll = lambda p: -(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
    base = ll(d.p_market.to_numpy(float))
    rows = []
    for w in weights:
        p = (1 - w) * d.p_market.to_numpy(float) + w * d.p_model.to_numpy(float)
        loss = ll(p)
        g = mean_ci_p(base - loss, d.date.to_numpy(), B)
        rows.append({"model_weight": w, "n": int(len(d)), "log_loss": float(loss.mean()), "gain_over_market": g["mean"], "ci_low": g["ci"][0], "ci_high": g["ci"][1], "p_not_better_than_market": g["p_not_above_0"]})
    return pd.DataFrame(rows)


def dm_test(loss_ref, loss_cand, lag: int | None = None) -> dict:
    """One-sided Diebold-Mariano test that the candidate's loss is lower than the reference's (H1: mean(ref - cand) > 0), Bartlett-kernel HAC variance, default lag floor(n^(1/3))."""
    d = np.asarray(loss_ref, float) - np.asarray(loss_cand, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "mean_diff": float(d.mean()) if n else np.nan, "stat": np.nan, "p": np.nan, "lag": 0}
    L = int(np.floor(n ** (1 / 3))) if lag is None else int(lag)
    x = d - d.mean()
    var = float(np.dot(x, x) / n)
    for k in range(1, L + 1):
        var += 2.0 * (1 - k / (L + 1)) * float(np.dot(x[k:], x[:-k]) / n)
    stat = float(d.mean() / np.sqrt(max(var, 1e-18) / n))
    return {"n": n, "mean_diff": float(d.mean()), "stat": stat, "p": float(1 - stats.norm.cdf(stat)), "lag": L}
