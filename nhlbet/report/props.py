"""Player-prop quotes for a slate: model expectation vs the market, with each player's recent history for context."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.features.players import add_asof_features, load_player_games, team_allowance
from nhlbet.models.props import MIN_GAMES, ShotsModel
from nhlbet.odds.props import STATS, PropQuote, consensus_pair, latest_prop_prices, match_players, modal_line, roster_candidates

log = logging.getLogger(__name__)
HISTORY_DAYS = 1100          # about three seasons of history is plenty for shot rates and keeps the live fit fast
MIN_TRAIN_ROWS = 3000


class PropEngine:
    """Fits the shots model once on all earlier games, then prices any game's props. ``None`` from ``create`` means there is not enough shot data yet."""

    def __init__(self, store: Store, pg: pd.DataFrame, model: ShotsModel, factors: dict[str, float], models: dict | None = None) -> None:
        self.store, self.pg, self.model, self.factors = store, pg, model, factors
        self.models = models or {"sog": model}

    @classmethod
    def create(cls, store: Store, as_of: pd.Timestamp | None = None) -> "PropEngine | None":
        pg = load_player_games(store)
        if pg.empty:
            return None
        end = pd.Timestamp(as_of) if as_of is not None else pg.game_date.max() + pd.Timedelta(days=1)
        pg = pg[(pg.game_date < end) & (pg.game_date >= end - pd.Timedelta(days=HISTORY_DAYS))]
        if len(pg) < MIN_TRAIN_ROWS:
            log.info("player props: only %d player-games with shot data so far (need %d)", len(pg), MIN_TRAIN_ROWS)
            return None
        allowance = team_allowance(store)
        feats = add_asof_features(pg, allowance)
        model = ShotsModel().fit(feats)
        models = {"sog": model, "points": ShotsModel(stat="points").fit(feats)}
        a = allowance.dropna(subset=["allow"]).sort_values("game_date").groupby("team").allow.last()
        lg = float(allowance.lg.dropna().iloc[-1]) if allowance.lg.notna().any() else float(a.mean())
        factors = {t: float(np.clip(v / lg, 0.7, 1.4)) for t, v in a.items()}
        return cls(store, pg, model, factors, models)

    def all_quotes_for_game(self, game_id: int, date: pd.Timestamp) -> list[PropQuote]:
        """Quotes for every individual-player market we price (shots, points), best edge first."""
        out = [q for stat in STATS for q in self.quotes_for_game(game_id, date, stat=stat)]
        return sorted(out, key=lambda q: -max(abs(q.edge_over), 0))

    def quotes_for_game(self, game_id: int, date: pd.Timestamp, now: pd.Timestamp | None = None, stat: str = "sog") -> list[PropQuote]:
        spec, model = STATS[stat], self.models[stat]
        col = spec["col"]
        prices = latest_prop_prices(self.store, game_id, spec["odds"])
        if prices.empty:
            return []
        roster = roster_candidates(self.store, game_id)
        names = list(prices.player.unique())
        ids = match_players(names, roster)
        if not ids:
            return []
        info = roster.set_index("player_id")
        future = pd.DataFrame([{"game_id": -game_id, "game_date": pd.Timestamp(date), "player_id": pid, "name": info.loc[pid, "name"], "team": info.loc[pid, "team"],
                                "position": info.loc[pid, "position"], "toi_sec": np.nan, "sog": np.nan, "points": np.nan, "is_home": int(info.loc[pid, "is_home"]), "opp": info.loc[pid, "opp"]}
                               for pid in set(ids.values())])
        hist = self.pg[self.pg.player_id.isin(future.player_id)]
        f = add_asof_features(pd.concat([hist, future], ignore_index=True))
        tonight = f[f.game_id == -game_id].copy()
        tonight["opp_factor"] = tonight.opp.map(self.factors).fillna(1.0)
        tonight["lam"] = model.lam(tonight)
        by_pid = tonight.set_index("player_id")
        out: list[PropQuote] = []
        season_start = pd.Timestamp(date) - pd.Timedelta(days=int((pd.Timestamp(date).month - 10) % 12 * 30.5 + pd.Timestamp(date).day))   # roughly 1 Oct of this season
        for nm, pid in ids.items():
            rows = prices[prices.player == nm]
            line = modal_line(rows.point)
            if abs(line - round(line)) < 1e-9:
                continue                                     # whole-number lines can push; props are normally .5 lines
            r = by_pid.loc[pid]
            if r.n_prev < MIN_GAMES:
                continue
            pm, bo, bu, nb = consensus_pair(rows[rows.point == line])
            lam = float(r.lam)
            po = float(model.p_over(lam, line))
            h = self.pg[self.pg.player_id == pid].sort_values("game_date")
            last10, last20 = h.tail(10), h.tail(20)
            season = h[h.game_date >= season_start]
            out.append(PropQuote(
                game_id=game_id, player_id=int(pid), name=str(r["name"]), team=str(r.team), opp=str(r.opp), point=float(line), lam=lam, p_over=po, p_over_market=pm,
                over_price=bo, under_price=bu, edge_over=po - pm, ev_over=po * (bo - 1) - (1 - po), ev_under=(1 - po) * (bu - 1) - po, n_books=nb, n_prev=int(r.n_prev),
                history=[{"date": str(d.date()), "opp": o, "sog": int(s), "val": int(s)} for d, o, s in zip(last10.game_date, last10.opp, last10[col])],
                avg_season=float(season[col].mean()) if len(season) >= 5 else None,          # a one-game 'season average' early in the year would only mislead
                avg_l10=float(last10[col].mean()) if len(last10) else None,
                hit_l10=float((last10[col] > line).mean()) if len(last10) else None, hit_l20=float((last20[col] > line).mean()) if len(last20) else None, stat=stat))
        return sorted(out, key=lambda q: -max(abs(q.edge_over), 0))
