"""Build one day's slate: features -> model probability -> odds/edge -> risk policy -> persisted recommendations."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nhlbet.data.goalies import goalie_display, starter_overrides
from nhlbet.data.loaders import load_tables
from nhlbet.data.store import Store
from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.models.bundle import ModelBundle
from nhlbet.odds.consensus import latest_book_prices
from nhlbet.odds.edge import SideQuote, evaluate_game
from nhlbet.risk.policy import Recommendation, RiskConfig, recommend_slate

log = logging.getLogger(__name__)


@dataclass
class SlateGame:
    game_id: int
    start_utc: pd.Timestamp | None
    home: str
    away: str
    home_goalie: str
    away_goalie: str
    goalie_status: str                      # 'confirmed' | 'probable' | 'mixed' | 'unknown'
    p_home: float                           # calibrated model probability (the one used for decisions)
    p_raw: float
    quotes: dict[str, SideQuote] | None
    rec: Recommendation
    odds_captured_at: str | None
    odds_stale: bool
    notes: list[str] = field(default_factory=list)


def context_notes(row: pd.Series, home: str, away: str) -> list[str]:
    """Plain-language drivers taken from the game's as-of features (max 4). Purely descriptive."""
    g = lambda k: row.get(k, np.nan)  # noqa: E731
    notes = []
    if g("d_gd_season_shrunk") == g("d_gd_season_shrunk") and abs(g("d_gd_season_shrunk")) >= 0.25:
        better = home if g("d_gd_season_shrunk") > 0 else away
        notes.append(f"{better} have the clearly better goal differential this season (gap {abs(g('d_gd_season_shrunk')):.2f} goals/game).")
    if g("h_b2b") == 1 or g("a_b2b") == 1:
        tired = [t for t, k in ((home, "h_b2b"), (away, "a_b2b")) if g(k) == 1]
        notes.append(f"{' and '.join(tired)} {'is' if len(tired) == 1 else 'are'} on the second night of a back-to-back.")
    if g("a_travel_km") == g("a_travel_km") and g("a_travel_km") > 2500 and g("a_tz_shift") not in (0, np.nan):
        notes.append(f"{away} travelled {g('a_travel_km'):,.0f} km across {abs(g('a_tz_shift')):.0f} time zone(s) since their last game.")
    if g("d_xg_share") == g("d_xg_share") and abs(g("d_xg_share")) >= 0.03:
        notes.append(f"{home if g('d_xg_share') > 0 else away} are creating the better share of expected goals recently.")
    if g("d_g_gsax100_season") == g("d_g_gsax100_season") and abs(g("d_g_gsax100_season")) >= 0.5:
        notes.append(f"Goaltending edge to {home if g('d_g_gsax100_season') > 0 else away} (goals saved above expected).")
    if g("is_rivalry") == 1:
        notes.append("Division/rival game.")
    if g("h_gp_season") == g("h_gp_season") and min(g("h_gp_season"), g("a_gp_season")) < 15:
        notes.append("Early in the season: team ratings lean on last year's results and are less reliable.")
    return notes[:4]


def build_slate(store: Store, bundle: ModelBundle, date: str, cfg: RiskConfig, run_type: str = "morning", model_status: str = "OK",
                max_odds_age_min: float = 240.0, now: datetime | None = None, persist: bool = True) -> list[SlateGame]:
    now = now or datetime.now(timezone.utc)
    tables = load_tables(store)
    games = tables["games"]
    today = games[(games.game_date == pd.Timestamp(date)) & (games.source == "nhl_api") & (games.home_score.isna())]
    if today.empty:
        log.info("no unplayed games on %s", date)
        return []
    overrides = starter_overrides(store, today)
    bcfg = BuilderConfig(**{**bundle.builder_cfg, "goalie_mode": "actual" if overrides else "probable"})
    F = FeatureBuilder(bcfg).build(tables, starter_override=overrides)
    rows = F.loc[today.game_id]
    preds = bundle.predict(rows)

    log_row = store.df("SELECT source, captured_at FROM odds_fetch_log ORDER BY captured_at DESC LIMIT 1")
    fetch_stale = bool(len(log_row) and log_row.source.iloc[0] == "stale_cache")
    inputs, meta = [], {}
    for g in today.itertuples():
        p_home = float(preds.loc[g.game_id, "p"]); p_raw = float(preds.loc[g.game_id, "stack"])
        prices = latest_book_prices(store, g.game_id)
        cap = prices.attrs.get("captured_at") if len(prices) else None
        age_min = (now - pd.to_datetime(cap, utc=True)).total_seconds() / 60 if cap else None
        stale = bool(cap and (age_min > max_odds_age_min or fetch_stale))
        quotes = evaluate_game(p_home, g.home, g.away, prices) if len(prices) else None
        hg, hs = goalie_display(store, g.home, date); ag, as_ = goalie_display(store, g.away, date)
        status = hs if hs == as_ else "mixed"
        row = rows.loc[g.game_id]
        ctx = {"model_status": model_status, "odds_stale": stale, "goalie_confirmed": hs == "confirmed" and as_ == "confirmed",
               "games_played_min": float(min(row.get("h_gp_season", np.nan), row.get("a_gp_season", np.nan)))
               if row.get("h_gp_season") == row.get("h_gp_season") else None}
        inputs.append((int(g.game_id), g.home, g.away, quotes, ctx))
        meta[int(g.game_id)] = dict(start=g.start_utc, hg=hg, ag=ag, status=status, p=p_home, p_raw=p_raw, quotes=quotes, cap=cap, stale=stale,
                                    notes=context_notes(row, g.home, g.away), ctx=ctx)
    recs = recommend_slate(inputs, cfg)
    slate = []
    for rec in recs:
        m = meta[rec.game_id]
        slate.append(SlateGame(rec.game_id, m["start"], rec.home, rec.away, m["hg"], m["ag"], m["status"], m["p"], m["p_raw"], m["quotes"], rec,
                               m["cap"], m["stale"], m["notes"]))
    epoch = pd.Timestamp(0, tz="UTC")
    slate.sort(key=lambda s: (pd.isna(s.start_utc), epoch if pd.isna(s.start_utc) else s.start_utc, s.game_id))
    if persist:
        run_id = f"{date}-{run_type}-{now:%H%M}"
        store.upsert("recommendations", [{
            "run_id": run_id, "run_at": now.isoformat(timespec="seconds"), "run_type": run_type, "game_id": s.game_id, "game_date": date,
            "home": s.home, "away": s.away, "home_goalie": s.home_goalie, "away_goalie": s.away_goalie, "goalie_status": s.goalie_status,
            "model_version": bundle.version, "p_model": s.p_home, "p_adj": s.rec.adj_prob,
            "p_market": (s.quotes["home"].market_prob if s.quotes else None), "p_stack_raw": s.p_raw, "action": s.rec.action, "side": s.rec.side,
            "team": s.rec.team, "book": s.rec.book, "decimal": s.rec.decimal, "stake": s.rec.stake, "edge": s.rec.edge, "ev": s.rec.ev,
            "reasons": " | ".join(s.rec.reasons), "odds_captured_at": s.odds_captured_at, "odds_stale": int(s.odds_stale), "model_status": model_status}
            for s in slate], ["run_id", "game_id"])
    return slate
