"""Turn raw per-book snapshots into (a) consensus no-vig probabilities over time, (b) the latest best prices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.odds.math import devig


def pregame_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop quotes captured after the game's scheduled start: those are in-play prices (a 2% favourite in the third period), not the pre-game market.
    Rows without a commence time are kept."""
    if df.empty or "commence_time" not in df:
        return df
    cap = pd.to_datetime(df.captured_at, utc=True, errors="coerce", format="ISO8601")
    start = pd.to_datetime(df.commence_time, utc=True, errors="coerce", format="ISO8601")
    return df[~(cap > start)]


def h2h_wide(store: Store, since: str | None = None) -> pd.DataFrame:
    """One row per (captured_at, event, book): ``home``/``away`` = decimal odds of each side (moneyline only);
    ``home_team``/``away_team`` = abbreviations. Pre-game quotes only."""
    q = "SELECT * FROM odds_snapshots WHERE market='h2h'" + (" AND captured_at >= ?" if since else "")
    df = pregame_only(store.df(q, [since] if since else []))
    if df.empty:
        return df
    df["side"] = np.where(df.outcome == df.home, "home", np.where(df.outcome == df.away, "away", "other"))
    df = df[df.side != "other"].rename(columns={"home": "home_team", "away": "away_team"})
    keys = ["captured_at", "event_id", "book", "game_id", "commence_time", "home_team", "away_team", "book_updated"]
    # groupby+unstack only materialises combinations that exist (pivot_table(dropna=False) builds the full cartesian product
    # of all index columns and ran out of memory on a real 15-game slate); dropna=False keeps rows with NULL game_id/book_updated
    w = df.groupby(keys + ["side"], dropna=False).price.max().unstack("side").reset_index()
    w.columns.name = None
    return w.dropna(subset=["home", "away"]) if {"home", "away"} <= set(w.columns) else pd.DataFrame()


COLS = ["game_id", "event_id", "captured_at", "home_prob_novig", "n_books", "best_home", "best_away"]


def _consensus_from_raw(store: Store, method: str = "shin", since: str | None = None) -> pd.DataFrame:
    w = h2h_wide(store, since)
    if w.empty:
        return pd.DataFrame(columns=COLS)
    w = w[w.game_id.notna()].copy()
    if w.empty:
        return pd.DataFrame(columns=COLS)
    w["p_home"] = [devig([h, a], method)[0] for h, a in zip(w.home, w.away)]
    g = w.groupby(["game_id", "event_id", "captured_at"])
    out = g.agg(home_prob_novig=("p_home", "mean"), n_books=("book", "nunique"), best_home=("home", "max"), best_away=("away", "max")).reset_index()
    out["game_id"] = out.game_id.astype(int)
    return out


def consensus_snapshots(store: Store, method: str = "shin", since: str | None = None) -> pd.DataFrame:
    """game_id, captured_at, home_prob_novig (mean across books), n_books [, event_id, best_home, best_away when raw quotes exist].

    Raw per-book quotes (kept only in the local database) take precedence; the stored, publishable ``odds_consensus`` rows fill in
    captures whose raw quotes are not available (e.g. after restoring a fresh database from the public-safe git logs).
    """
    raw = _consensus_from_raw(store, method, since)
    stored = store.df("SELECT game_id, captured_at, home_prob_novig, n_books FROM odds_consensus" + (" WHERE captured_at >= ?" if since else ""),
                      [since] if since else [])
    if not stored.empty:
        have = set(zip(raw.game_id, raw.captured_at)) if not raw.empty else set()
        stored = stored[[(g, c) not in have for g, c in zip(stored.game_id, stored.captured_at)]]
        for c in ("event_id", "best_home", "best_away"):
            stored[c] = None
        raw = pd.concat([raw, stored[COLS]], ignore_index=True) if not raw.empty else stored[COLS]
    if raw.empty:
        return pd.DataFrame(columns=COLS)
    raw["game_id"] = raw.game_id.astype(int)
    return raw.sort_values(["game_id", "captured_at"]).reset_index(drop=True)


def persist_consensus(store: Store, method: str = "shin") -> int:
    """Store the derived consensus of every linked capture (idempotent). This, not the raw quotes, is what gets committed."""
    c = _consensus_from_raw(store, method)
    rows = [{"game_id": int(r.game_id), "captured_at": r.captured_at, "home_prob_novig": float(r.home_prob_novig), "n_books": int(r.n_books)}
            for r in c.itertuples()]
    return store.upsert("odds_consensus", rows, ["game_id", "captured_at"])


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
