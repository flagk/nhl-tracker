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
from nhlbet.odds.props import STATS
from nhlbet.report.props import PropEngine
from nhlbet.report.ranking import rank_picks
from nhlbet.risk.parlays import build_parlays
from nhlbet.report.stats import game_stats
from nhlbet.odds.consensus import latest_book_prices
from nhlbet.odds.edge import SideQuote, evaluate_game
from nhlbet.odds.markets import MarketQuote, alt_quotes, latest_alt_prices
from nhlbet.risk.policy import Recommendation, RiskConfig, recommend_slate
from nhlbet.risk.shadow import prop_shadow_bets, shadow_bets

log = logging.getLogger(__name__)
IN_PLAY_WINDOW_H = 12.0


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
    books: list[dict] = field(default_factory=list)      # per-book decimal prices [{book, home, away}] for the site/parlay maths
    ctx: dict = field(default_factory=dict)              # the policy context this game was evaluated under
    alt: list[MarketQuote] = field(default_factory=list)  # totals / puck-line quotes from the goals model (experimental: paper-traded only)
    goals: dict = field(default_factory=dict)            # {lam_home, lam_away, exp_total} from the goals model
    stats: list = field(default_factory=list)            # both teams' as-of stats for the side-by-side table (nhlbet.report.stats.game_stats)
    drivers: list = field(default_factory=list)          # biggest pulls on the logistic component for this game
    props: list = field(default_factory=list)            # player shots over/under quotes (nhlbet.odds.props.PropQuote), experimental


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
    if len(today):                       # a game that started within the last 12h is in progress / awaiting its score: its odds are in-play prices
        start = pd.to_datetime(today.start_utc, utc=True, errors="coerce")
        live = (start <= pd.Timestamp(now)) & (start > pd.Timestamp(now) - pd.Timedelta(hours=IN_PLAY_WINDOW_H))
        if live.any():
            log.info("skipping %d game(s) already under way: %s", int(live.sum()), ", ".join(f"{a}@{h}" for a, h in zip(today[live].away, today[live].home)))
        today = today[~live.to_numpy()]
    if today.empty:
        log.info("no unplayed games on %s", date)
        return []
    overrides = starter_overrides(store, today)
    bcfg = BuilderConfig(**{**bundle.builder_cfg, "goalie_mode": "actual" if overrides else "probable"})
    F = FeatureBuilder(bcfg).build(tables, starter_override=overrides)
    rows = F.loc[today.game_id]
    preds = bundle.predict(rows)
    dists = bundle.score_distributions(rows, preds["p"].to_numpy())
    dist_by_game = dict(zip(rows.index, dists)) if dists is not None else {}
    drivers_by_game = bundle.drivers(rows) if hasattr(bundle, "drivers") else {}
    has_props = len(store.df("SELECT 1 FROM odds_snapshots WHERE market LIKE 'player_%%' AND game_id IN (%s) LIMIT 1" % ",".join(str(int(x)) for x in today.game_id)))
    engine = PropEngine.create(store, as_of=pd.Timestamp(date)) if has_props else None

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
        books = [{"book": r.book, "home": float(r.home), "away": float(r.away)} for r in prices.itertuples()] if len(prices) else []
        dist = dist_by_game.get(g.game_id)
        alt = alt_quotes(latest_alt_prices(store, g.game_id, now=now), dist, g.home, g.away, cal=getattr(bundle.goals, "cal", None)) if dist is not None else []
        goals = {"lam_home": dist.lam_home, "lam_away": dist.lam_away, "exp_total": dist.expected_total()} if dist is not None else {}
        inputs.append((int(g.game_id), g.home, g.away, quotes, ctx))
        meta[int(g.game_id)] = dict(start=g.start_utc, hg=hg, ag=ag, status=status, p=p_home, p_raw=p_raw, quotes=quotes, cap=cap, stale=stale,
                                    notes=context_notes(row, g.home, g.away), ctx=ctx, books=books, alt=alt, goals=goals, stats=game_stats(row), drivers=drivers_by_game.get(g.game_id, []),
                                    props=engine.all_quotes_for_game(int(g.game_id), pd.Timestamp(date), now=pd.Timestamp(now)) if engine is not None else [])
    recs = recommend_slate(inputs, cfg)
    slate = []
    for rec in recs:
        m = meta[rec.game_id]
        slate.append(SlateGame(rec.game_id, m["start"], rec.home, rec.away, m["hg"], m["ag"], m["status"], m["p"], m["p_raw"], m["quotes"], rec,
                               m["cap"], m["stale"], m["notes"], m["books"], m["ctx"], m["alt"], m["goals"], m["stats"], m["drivers"], m["props"]))
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
        store.upsert("alt_quotes", [{"run_id": run_id, "run_at": now.isoformat(timespec="seconds"), "game_id": s.game_id, "game_date": date, "market": q.market,
                                    "side": q.side, "label": q.label, "point": q.point, "p_model": q.model_prob, "p_market": q.market_prob, "p_push": q.p_push,
                                    "book": q.best_book, "decimal": q.best_decimal, "edge": q.edge, "ev": q.ev, "n_books": q.n_books,
                                    "lam_home": s.goals.get("lam_home"), "lam_away": s.goals.get("lam_away"), "exp_total": s.goals.get("exp_total"),
                                    "odds_captured_at": s.odds_captured_at} for s in slate for q in s.alt], ["run_id", "game_id", "market", "side"])
        prop_rows = []
        for s_ in slate:
            for q in s_.props:
                for side in ("over", "under"):
                    if not q.has_side(side):
                        continue
                    over = side == "over"
                    prop_rows.append({"run_id": run_id, "run_at": now.isoformat(timespec="seconds"), "game_id": s_.game_id, "game_date": date, "market": STATS[q.stat]["store"], "side": f"{side}:{q.player_id}",
                                      "label": (f"{q.name} to score (anytime)" if q.one_sided else f"{q.name} {'Over' if over else 'Under'} {q.point:g}"), "point": q.point, "p_model": q.p_over if over else 1 - q.p_over,
                                      "p_market": q.p_over_market if over else 1 - q.p_over_market, "p_push": 0.0, "book": None, "decimal": q.over_price if over else q.under_price,
                                      "edge": q.edge(side), "ev": q.ev_over if over else q.ev_under, "n_books": q.n_books, "lam_home": None, "lam_away": None, "exp_total": q.lam,
                                      "odds_captured_at": s_.odds_captured_at, "player_id": q.player_id})
        if prop_rows:
            store.upsert("alt_quotes", prop_rows, ["run_id", "game_id", "market", "side"])
            store.upsert("prop_bets", prop_shadow_bets(slate, cfg, run_id, now.isoformat(timespec="seconds"), date), ["run_id", "game_id", "strategy", "player_id", "side"])
        store.upsert("shadow_bets", shadow_bets(slate, cfg, run_id, now.isoformat(timespec="seconds"), date), ["run_id", "game_id", "strategy"])
        ranked = rank_picks(slate, cfg, now=pd.Timestamp(now))
        parlays = build_parlays(ranked, slate, run_id, now.isoformat(timespec="seconds"), date, now=pd.Timestamp(now))
        if parlays:
            store.upsert("parlay_bets", parlays, ["run_id", "strategy", "idx"])
    return slate
