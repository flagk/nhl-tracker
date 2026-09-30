"""Paper-trading ("shadow") strategies: fake money, evaluated on every slate, settled like real bets.

Purpose: MEASUREMENT, not training. The model already learns from every game result; its own picks carry no extra label
(training on them would be circular). What the real policy lacks is *evidence*: it is deliberately selective, so it produces few
bets. Running several strategies on the same games gives many more observations of profit and closing-line value, so questions
like "does the early-season guard help?" or "is shrinking toward the market worth it?" can be answered from data.

``market_favorite`` is a CONTROL: flat bets on the market favourite. It has no skill, so its ROI is what the bookmaker margin
alone costs. A strategy only means something if it beats the control by more than noise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from nhlbet.odds.edge import SideQuote
from nhlbet.risk.policy import RiskConfig, game_block_reason, recommend_slate
from nhlbet.risk.shrink import shrink_to_market

FLAT_PCT = 0.01   # flat strategies stake 1% of bankroll per bet


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    kind: str                    # 'policy' | 'flat_edge' | 'model_side' | 'market_favorite' | 'alt_edge' | 'alt_every' | 'alt_control' (other markets)
    overrides: tuple = ()        # ((RiskConfig field, value), ...)


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("no_guard", "Live policy without the early-season guard (does the guard help?)", "policy", (("min_games_played", 0),)),
    Strategy("edge_1pct", "Live policy at a 1% raw-edge threshold instead of 3%", "policy",
             (("min_edge", 0.01), ("min_edge_unconfirmed_goalie", 0.01), ("min_ev", 0.0))),
    Strategy("no_shrink", "Live policy trusting the raw model fully (no shrinkage toward the market)", "policy",
             (("trust_w0", 1.0), ("trust_d0", 1e9), ("max_disagreement", 1.0))),
    Strategy("flat_model_side", "Flat 1% on every game where the model sees any positive edge", "flat_edge"),
    Strategy("every_game", "Flat 1% on the model's preferred side of EVERY game that has fresh odds (no edge filter, even at a negative edge)", "model_side"),
    Strategy("totals_edge", "Goals model: flat 1% on the over/under side with a 3%+ raw edge vs the market (experimental market)", "alt_edge", (("market", "totals"),)),
    Strategy("puckline_edge", "Goals model: flat 1% on the puck-line side with a 3%+ raw edge vs the market (experimental market)", "alt_edge", (("market", "spreads"),)),
    Strategy("every_total", "Goals model: flat 1% on its over/under side of EVERY game with fresh totals odds, no edge filter", "alt_every", (("market", "totals"),)),
    Strategy("every_puckline", "Goals model: flat 1% on its puck-line side of EVERY game with fresh spread odds, no edge filter", "alt_every", (("market", "spreads"),)),
    Strategy("always_over", "CONTROL: flat 1% on the Over of every game (no skill; shows what totals vig plus base rate cost)", "alt_control", (("market", "totals"),)),
    Strategy("market_favorite", "CONTROL: flat 1% on the market favourite (no skill; shows what the bookmaker margin costs)", "market_favorite"),
)


def _row(run_id, run_at, date, game_id, strategy, action="NO_BET", q: SideQuote | None = None, stake=0.0, p_adj=None, ev=None) -> dict:
    return {"market": None, "point": None, "label": None, "run_id": run_id, "run_at": run_at, "game_id": game_id, "strategy": strategy, "game_date": date, "action": action,
            "side": q.side if q else None, "team": q.team if q else None, "book": q.best_book if q else None,
            "decimal": q.best_decimal if q else None, "stake": stake, "p_model": q.model_prob if q else None, "p_adj": p_adj,
            "p_market": q.market_prob if q else None, "edge": q.edge if q else None, "ev": ev}


def _usable(quotes, cfg: RiskConfig, ctx) -> bool:
    """Flat strategies only need odds that exist and are fresh, and a model that is not flagged ALERT."""
    if not quotes or len(quotes) < 2:
        return False
    if ctx.get("model_status") == "ALERT":
        return False
    return not (ctx.get("odds_stale") and not cfg.allow_stale_odds)


def _alt_row(run_id, run_at, date, game_id, strategy, quote=None, stake=0.0) -> dict:
    r = _row(run_id, run_at, date, game_id, strategy)
    if quote is None:
        return r
    r.update({"action": "BET", "side": quote.side, "team": quote.label, "book": quote.best_book, "decimal": quote.best_decimal, "stake": stake,
              "p_model": quote.model_prob, "p_adj": quote.model_prob, "p_market": quote.market_prob, "edge": quote.edge, "ev": quote.ev,
              "market": quote.market, "point": quote.point, "label": quote.label})
    return r


def _alt_usable(g, cfg: RiskConfig) -> bool:
    return not (g.ctx.get("model_status") == "ALERT" or (g.ctx.get("odds_stale") and not cfg.allow_stale_odds))


def shadow_bets(games: Sequence, cfg: RiskConfig, run_id: str, run_at: str, date: str,
                strategies: Sequence[Strategy] = STRATEGIES) -> list[dict]:
    """``games``: objects with ``game_id, home, away, quotes, ctx`` (``SlateGame``). One row per (game, strategy)."""
    rows: list[dict] = []
    inputs = [(g.game_id, g.home, g.away, g.quotes, g.ctx) for g in games]
    for st in strategies:
        if st.kind == "policy":
            c = replace(cfg, **dict(st.overrides))
            for rec in recommend_slate(inputs, c):
                q = None
                g = next(x for x in games if x.game_id == rec.game_id)
                if rec.action == "BET" and g.quotes:
                    q = g.quotes[rec.side]
                rows.append(_row(run_id, run_at, date, rec.game_id, st.name, rec.action, q, rec.stake, rec.adj_prob, rec.ev))
            continue
        if st.kind.startswith("alt_"):
            market = dict(st.overrides)["market"]
            for g in games:
                qs = [q for q in (getattr(g, "alt", None) or []) if q.market == market]
                if len(qs) < 2 or not _alt_usable(g, cfg):
                    rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name))
                    continue
                if st.kind == "alt_control":
                    q = next(x for x in qs if x.side == "over")
                else:
                    q = max(qs, key=lambda x: x.edge if st.kind == "alt_edge" else x.model_prob)
                if st.kind == "alt_edge" and q.edge < cfg.min_edge:
                    rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name))
                    continue
                rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name, q, round(cfg.bankroll * FLAT_PCT, 2)))
            continue
        for g in games:
            if not _usable(g.quotes, cfg, g.ctx):
                rows.append(_row(run_id, run_at, date, g.game_id, st.name))
                continue
            key = {"flat_edge": lambda x: x.edge, "model_side": lambda x: x.model_prob}.get(st.kind, lambda x: x.market_prob)
            q = max(g.quotes.values(), key=key)
            if st.kind == "flat_edge" and q.edge <= 0:
                rows.append(_row(run_id, run_at, date, g.game_id, st.name))
                continue
            p_adj = shrink_to_market(q.model_prob, q.market_prob, cfg.trust_w0, cfg.trust_d0)
            rows.append(_row(run_id, run_at, date, g.game_id, st.name, "BET", q, round(cfg.bankroll * FLAT_PCT, 2), p_adj, q.ev))
    return rows
