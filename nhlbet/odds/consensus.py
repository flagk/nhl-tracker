"""Turn raw per-book snapshots into (a) consensus no-vig probabilities over time, (b) the latest best prices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.odds.math import devig


def h2h_wide(store: Store, since: str | None = None) -> pd.DataFrame:
    """One row per (captured_at, event, book): ``home``/``away`` = decimal odds of each side (moneyline only);
    ``home_team``/``away_team`` = abbreviations."""
    q = "SELECT * FROM odds_snapshots WHERE market='h2h'" + (" AND captured_at >= ?" if since else "")
    df = store.df(q, [since] if since else [])
    if df.empty:
        return df
    df["side"] = np.where(df.outcome == df.home, "home", np.where(df.outcome == df.away, "away", "other"))
    df = df[df.side != "other"].rename(columns={"home": "home_team", "away": "away_team"})
    w = df.pivot_table(index=["captured_at", "event_id", "book", "game_id", "commence_time", "home_team", "away_team", "book_updated"],
                       columns="side", values="price", aggfunc="max", dropna=False).reset_index()
    w.columns.name = None
    return w.dropna(subset=["home", "away"]) if {"home", "away"} <= set(w.columns) else pd.DataFrame()


def consensus_snapshots(store: Store, method: str = "shin", since: str | None = None) -> pd.DataFrame:
    """game_id, captured_at, home_prob_novig (mean across books), n_books, best_home, best_away."""
    w = h2h_wide(store, since)
    if w.empty:
        return pd.DataFrame(columns=["game_id", "event_id", "captured_at", "home_prob_novig", "n_books", "best_home", "best_away"])
    w = w[w.game_id.notna()].copy()
    w["p_home"] = [devig([h, a], method)[0] for h, a in zip(w.home, w.away)]
    g = w.groupby(["game_id", "event_id", "captured_at"])
    out = g.agg(home_prob_novig=("p_home", "mean"), n_books=("book", "nunique"), best_home=("home", "max"), best_away=("away", "max")).reset_index()
    out["game_id"] = out.game_id.astype(int)
    return out.sort_values(["game_id", "captured_at"])


def latest_book_prices(store: Store, game_id: int, max_book_age_min: float = 90.0, fetch_max_age_min: float | None = None) -> pd.DataFrame:
    """Most recent snapshot for a game: per-book (home, away) decimals, dropping books whose line looks stale.

    A book line is stale if its own ``book_updated`` is older than ``max_book_age_min`` before capture - such
    prices would otherwise show up as a fake 'best price' after the market has already moved.
    """
    w = h2h_wide(store)
    if w.empty:
        return w
    w = w[w.game_id == game_id]
    if w.empty:
        return w
    last = w.captured_at.max()
    cur = w[w.captured_at == last].copy()
    cap = pd.to_datetime(last, utc=True)
    upd = pd.to_datetime(cur.book_updated, utc=True, errors="coerce")
    fresh = upd.isna() | ((cap - upd) <= pd.Timedelta(minutes=max_book_age_min))
    cur = cur[fresh]
    cur.attrs["captured_at"] = last
    return cur
