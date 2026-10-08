"""Totals (over/under) and puck-line (spread) quotes: the goals-model probability against the market's no-vig price.

For each market we take the *main line* (the point most books agree on in the latest capture), remove the vig per book at that exact
line, average across books, and compare with the model. Whole-number lines can push (stake refunded): probabilities are compared
conditional on no push, and EV uses the true win / lose / push split.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.models.goals import ScoreDistribution
from nhlbet.odds.consensus import pregame_only
from nhlbet.odds.math import best_price, devig

MARKETS = ("h2h", "spreads", "totals")
ALT_MAX_AGE_MIN = 120.0     # totals / puck-line prices captured longer ago than this (relative to the run) are not used


@dataclass
class MarketQuote:
    market: str               # 'totals' | 'spreads'
    side: str                 # 'over' | 'under' | 'home' | 'away'
    label: str                # 'Over 6.5', 'BOS -1.5'
    point: float              # total line, or this side's handicap
    model_prob: float         # P(win | no push)
    market_prob: float        # consensus no-vig P(win | no push)
    p_push: float
    best_book: str
    best_decimal: float
    edge: float
    ev: float                 # per $1 at the best price, using the model's win / lose / push probabilities
    n_books: int
    player_id: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _ev(p_win: float, p_push: float, decimal: float) -> float:
    return p_win * (decimal - 1.0) - (1.0 - p_win - p_push)


def latest_alt_prices(store: Store, game_id: int, max_book_age_min: float = 90.0, now=None, max_capture_age_min: float = ALT_MAX_AGE_MIN) -> dict[str, pd.DataFrame]:
    """Latest capture's per-book pairs. ``totals``: book, point, over, under. ``spreads``: book, home_point, home, away (decimal odds)."""
    df = pregame_only(store.df("SELECT * FROM odds_snapshots WHERE game_id=? AND market IN ('spreads','totals')", [game_id]))
    out = {"totals": pd.DataFrame(), "spreads": pd.DataFrame()}
    if df.empty:
        return out
    last = df.captured_at.max()
    df = df[df.captured_at == last]
    cap = pd.to_datetime(last, utc=True)
    if now is not None:
        nowts = pd.Timestamp(now)
        nowts = nowts.tz_localize("UTC") if nowts.tzinfo is None else nowts
        if nowts - cap > pd.Timedelta(minutes=max_capture_age_min):
            return out                                       # these totals / puck-line prices are too old to call current: price nothing rather than price old lines
    upd = pd.to_datetime(df.book_updated, utc=True, errors="coerce")
    df = df[upd.isna() | ((cap - upd) <= pd.Timedelta(minutes=max_book_age_min))]
    t = df[df.market == "totals"]
    if len(t):
        w = t.groupby(["book", "point", "outcome"]).price.max().unstack("outcome").reset_index()
        if {"Over", "Under"} <= set(w.columns):
            out["totals"] = w.dropna(subset=["Over", "Under"]).rename(columns={"Over": "over", "Under": "under"})[["book", "point", "over", "under"]]
    s = df[df.market == "spreads"]
    if len(s):
        home, away = s.home.iloc[0], s.away.iloc[0]
        h = s[s.outcome == home][["book", "point", "price"]].rename(columns={"point": "home_point", "price": "home"})
        a = s[s.outcome == away][["book", "point", "price"]].rename(columns={"point": "away_point", "price": "away"})
        m = h.merge(a, on="book")
        m = m[np.isclose(m.home_point, -m.away_point)]
        out["spreads"] = m[["book", "home_point", "home", "away"]].drop_duplicates(["book", "home_point"])
    for v in out.values():
        v.attrs["captured_at"] = last
    return out


def _modal(points: pd.Series, prefer: float) -> float:
    """The point most books quote; ties go to the one closest to the league's usual line (``prefer``, compared on magnitude)."""
    c = points.value_counts()
    top = list(c[c == c.max()].index)
    return float(min(top, key=lambda p: abs(abs(p) - abs(prefer))))


def recalibrate(cond: float, cal: dict | None) -> float:
    """Apply a fitted Platt map (logit(p') = a + b*logit(p)) to a conditional win probability; identity when no map is given."""
    if not cal:
        return cond
    z = np.log(np.clip(cond, 1e-6, 1 - 1e-6) / (1 - np.clip(cond, 1e-6, 1 - 1e-6)))
    return float(1.0 / (1.0 + np.exp(-(cal["a"] + cal["b"] * z))))


def _pair_quotes(market: str, pairs: pd.DataFrame, cols: tuple[str, str], labels: tuple[str, str], sides: tuple[str, str], points: tuple[float, float],
                 probs: tuple[float, float, float], method: str, cal: dict | None = None) -> list[MarketQuote]:
    """``probs`` = model (P(A wins), P(B wins), P(push)). ``cal`` recalibrates the conditional probability. Returns the two sides' quotes."""
    pa, pb, pp = probs
    cond_a = recalibrate(pa / max(1e-12, pa + pb), cal)
    pa, pb = cond_a * (1.0 - pp), (1.0 - cond_a) * (1.0 - pp)
    mk = float(np.mean([devig([x, y], method)[0] for x, y in zip(pairs[cols[0]], pairs[cols[1]])]))
    out = []
    for i, (side, label, point, p_win, p_lose, cond, mprob) in enumerate(
            ((sides[0], labels[0], points[0], pa, pb, cond_a, mk), (sides[1], labels[1], points[1], pb, pa, 1 - cond_a, 1 - mk))):
        book, dec = best_price(dict(zip(pairs.book, pairs[cols[i]])))
        out.append(MarketQuote(market, side, label, float(point), float(cond), float(mprob), float(pp), str(book), float(dec), float(cond - mprob),
                               float(_ev(p_win, pp, dec)), int(len(pairs))))
    return out


def alt_quotes(prices: dict[str, pd.DataFrame], dist: ScoreDistribution, home: str, away: str, method: str = "shin", cal: dict | None = None) -> list[MarketQuote]:
    quotes: list[MarketQuote] = []
    t = prices.get("totals", pd.DataFrame())
    t = t[(t.over > 1) & (t.under > 1)] if len(t) else t
    if len(t):
        pt = _modal(t.point, 6.0)
        pairs = t[t.point == pt]
        o, u, p = dist.total_probs(pt)
        quotes += _pair_quotes("totals", pairs, ("over", "under"), (f"Over {pt:g}", f"Under {pt:g}"), ("over", "under"), (pt, pt), (o, u, p), method, (cal or {}).get("totals"))
    s = prices.get("spreads", pd.DataFrame())
    s = s[(s.home > 1) & (s.away > 1)] if len(s) else s
    if len(s):
        hp = _modal(s.home_point, 1.5)
        pairs = s[s.home_point == hp]
        c, a, p = dist.spread_probs(hp)
        quotes += _pair_quotes("spreads", pairs, ("home", "away"), (f"{home} {hp:+g}", f"{away} {-hp:+g}"), ("home", "away"), (hp, -hp), (c, a, p), method, (cal or {}).get("spreads"))
    return quotes
