"""Player-prop (shots on goal over/under) prices: latest per-book pairs, player matching, and model-vs-market quotes."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.odds.consensus import pregame_only
from nhlbet.odds.math import best_price, devig

SOG_MARKET = "player_shots_on_goal"
STORE_MARKET = "player_sog"


@dataclass
class PropQuote:
    game_id: int
    player_id: int
    name: str
    team: str
    opp: str
    point: float
    lam: float                     # model's expected shots tonight
    p_over: float                  # model P(shots > point)
    p_over_market: float           # consensus no-vig P(over)
    over_price: float
    under_price: float
    edge_over: float
    ev_over: float
    ev_under: float
    n_books: int
    n_prev: int
    history: list = field(default_factory=list)     # last games: [{"date", "opp", "sog"}]
    avg_season: float | None = None
    avg_l10: float | None = None
    hit_l10: float | None = None                    # share of the last 10 games over this line
    hit_l20: float | None = None

    @property
    def take(self) -> str | None:
        """The side with positive expected value at the best price (None when neither)."""
        best = "over" if self.ev_over >= self.ev_under else "under"
        return best if max(self.ev_over, self.ev_under) > 0 else None

    @property
    def best_side(self) -> str:
        return "over" if self.ev_over >= self.ev_under else "under"

    def edge(self, side: str) -> float:
        return self.edge_over if side == "over" else -self.edge_over


def norm_name(n: str) -> tuple[str, str]:
    """(first initial, last name) with accents, punctuation and suffixes removed: 'Connor McDavid' and 'C. McDavid' both give ('c', 'mcdavid')."""
    s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii)\b\.?", "", s)
    s = re.sub(r"[^a-z\s-]", " ", s).replace("-", "")
    toks = s.split()
    return (toks[0][0], toks[-1]) if toks else ("", "")


def latest_prop_prices(store: Store, game_id: int, market: str = SOG_MARKET, max_book_age_min: float = 90.0) -> pd.DataFrame:
    """Latest pre-game capture: one row per (book, player, line) with decimal ``over`` / ``under`` prices."""
    df = pregame_only(store.df("SELECT * FROM odds_snapshots WHERE game_id=? AND market=?", [game_id, market]))
    if df.empty:
        return pd.DataFrame(columns=["book", "player", "point", "over", "under"])
    last = df.captured_at.max()
    df = df[df.captured_at == last]
    cap = pd.to_datetime(last, utc=True)
    upd = pd.to_datetime(df.book_updated, utc=True, errors="coerce")
    df = df[upd.isna() | ((cap - upd) <= pd.Timedelta(minutes=max_book_age_min))]
    split = df.outcome.str.split("|", n=1, expand=True)
    if split.shape[1] < 2:
        return pd.DataFrame(columns=["book", "player", "point", "over", "under"])
    df = df.assign(side=split[0], player=split[1])
    w = df.groupby(["book", "player", "point", "side"]).price.max().unstack("side").reset_index()
    if not {"Over", "Under"} <= set(w.columns):
        return pd.DataFrame(columns=["book", "player", "point", "over", "under"])
    w = w.dropna(subset=["Over", "Under"]).rename(columns={"Over": "over", "Under": "under"})
    w.attrs["captured_at"] = last
    return w[["book", "player", "point", "over", "under"]]


def roster_candidates(store: Store, game_id: int, lookback_days: int = 45) -> pd.DataFrame:
    """Skaters who played for either team recently: (player_id, name, team, opp, is_home, position), most recent row per player."""
    g = store.df("SELECT game_date, home, away FROM games WHERE game_id=?", [game_id])
    if g.empty:
        return pd.DataFrame(columns=["player_id", "name", "team", "opp", "is_home", "position"])
    d, home, away = pd.Timestamp(g.game_date.iloc[0]), g.home.iloc[0], g.away.iloc[0]
    r = store.df("""SELECT s.player_id, s.name, s.team, s.position, g.game_date FROM skater_game s JOIN games g ON g.game_id = s.game_id
                    WHERE s.team IN (?, ?) AND g.game_date < ? AND g.game_date >= ? AND g.game_type = 2""",
                 [home, away, str(d.date()), str((d - pd.Timedelta(days=lookback_days)).date())])
    if r.empty:
        return r.assign(opp=None, is_home=None)
    r = r.sort_values("game_date").groupby("player_id", as_index=False).tail(1)
    r["is_home"] = (r.team == home).astype(int)
    r["opp"] = np.where(r.team == home, away, home)
    return r[["player_id", "name", "team", "opp", "is_home", "position"]].reset_index(drop=True)


def match_players(names: list[str], roster: pd.DataFrame) -> dict[str, int]:
    """odds-feed name -> player_id by (first initial, last name) among the two teams' recent skaters; ambiguous or unknown names are left out."""
    idx: dict[tuple[str, str], list[int]] = {}
    for pid, nm in zip(roster.player_id, roster.name):
        idx.setdefault(norm_name(nm), []).append(int(pid))
    out = {}
    for n in names:
        c = idx.get(norm_name(n), [])
        if len(c) == 1:
            out[n] = c[0]
    return out


def consensus_pair(rows: pd.DataFrame, method: str = "shin") -> tuple[float, float, float, int]:
    """(p_over_market, best over price, best under price, n_books) for one player's quotes at one line."""
    ps = [devig([o, u], method)[0] for o, u in zip(rows.over, rows.under) if o > 1 and u > 1]
    bo = best_price(dict(zip(rows.book, rows.over)))[1]
    bu = best_price(dict(zip(rows.book, rows.under)))[1]
    return float(np.mean(ps)), float(bo), float(bu), len(ps)


def modal_line(points: pd.Series) -> float:
    c = points.value_counts()
    return float(sorted(c[c == c.max()].index)[0])
