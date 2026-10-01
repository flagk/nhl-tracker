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

PAPER_MIN, PAPER_MAX = 5.0, 30.0     # pretend stakes scale with the model's conviction between these
CONTROL_STAKE = 10.0                 # no-skill controls always stake the same, so their ROI is a clean baseline


def paper_stake(conviction: float) -> float:
    """$5 at no conviction up to $30 at full conviction, rounded to whole dollars. conviction is clipped to [0, 1]."""
    c = min(1.0, max(0.0, float(conviction)))
    return float(round(PAPER_MIN + (PAPER_MAX - PAPER_MIN) * c))


def edge_conviction(edge: float) -> float:
    """A 3-point edge is about $12, a 10-point edge or more is the $30 maximum."""
    return edge / 0.10


def side_conviction(p: float) -> float:
    """Bets on the model's favoured side of a two-way market: 50% -> $5, 75% or more -> $30."""
    return (p - 0.5) / 0.25


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
    Strategy("flat_model_side", "$5-$30 (more for a bigger edge) on every game where the model sees any positive edge", "flat_edge"),
    Strategy("every_game", "$5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge", "model_side"),
    Strategy("totals_edge", "Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market)", "alt_edge", (("market", "totals"),)),
    Strategy("puckline_edge", "Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market)", "alt_edge", (("market", "spreads"),)),
    Strategy("every_total", "Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter", "alt_every", (("market", "totals"),)),
    Strategy("every_puckline", "Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter", "alt_every", (("market", "spreads"),)),
    Strategy("always_over", "CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost)", "alt_control", (("market", "totals"),)),
    Strategy("market_favorite", "CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs)", "market_favorite"),
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
                    rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name, q, CONTROL_STAKE))
                    continue
                else:
                    q = max(qs, key=lambda x: x.edge if st.kind == "alt_edge" else x.model_prob)
                if st.kind == "alt_edge" and q.edge < cfg.min_edge:
                    rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name))
                    continue
                conv = edge_conviction(q.edge) if st.kind == "alt_edge" else side_conviction(q.model_prob)
                rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name, q, paper_stake(conv)))
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
            stake = (CONTROL_STAKE if st.kind == "market_favorite" else
                     paper_stake(edge_conviction(q.edge)) if st.kind == "flat_edge" else paper_stake(side_conviction(q.model_prob)))
            rows.append(_row(run_id, run_at, date, g.game_id, st.name, "BET", q, stake, p_adj, q.ev))
    return rows


# ---------------------------------------------------------------- player props (several bets per game, so they live in their own table)
PROP_MAX_PER_DAY = 8
PROP_MIN_GAMES = 10
PROP_STRATEGIES: tuple[Strategy, ...] = (
    Strategy("sog_edge", f"Player shots: $5-$30 on the over/under the model likes (3%+ edge, 10+ games of history), best {PROP_MAX_PER_DAY} of the day (experimental market)", "prop_edge"),
    Strategy("sog_over_control", f"CONTROL: flat $10 on the Over for the {PROP_MAX_PER_DAY} players with the highest shots lines (no model; shows what prop vig costs)", "prop_control"),
)


def _prop_row(run_id, run_at, date, game_id, strategy, q, side: str, stake: float) -> dict:
    over = side == "over"
    p_side = q.p_over if over else 1 - q.p_over
    m_side = q.p_over_market if over else 1 - q.p_over_market
    return {"run_id": run_id, "run_at": run_at, "game_id": game_id, "strategy": strategy, "game_date": date, "player_id": q.player_id, "name": q.name, "side": side,
            "label": f"{q.name} {'Over' if over else 'Under'} {q.point:g}", "point": q.point, "book": None, "decimal": q.over_price if over else q.under_price, "stake": stake,
            "p_model": p_side, "p_market": m_side, "edge": p_side - m_side, "ev": q.ev_over if over else q.ev_under, "lam": q.lam}


def prop_shadow_bets(games: Sequence, cfg: RiskConfig, run_id: str, run_at: str, date: str, strategies: Sequence[Strategy] = PROP_STRATEGIES) -> list[dict]:
    """Paper bets on player shots across the whole slate (capped per day). ``games``: objects with ``game_id``, ``props`` (PropQuote list) and ``ctx``."""
    cands = [(g, q) for g in games if _alt_usable(g, cfg) for q in (getattr(g, "props", None) or [])]
    rows: list[dict] = []
    for st in strategies:
        if st.kind == "prop_edge":
            scored = [(q.edge(q.best_side), g, q) for g, q in cands if q.n_prev >= PROP_MIN_GAMES]
            picks = sorted([x for x in scored if x[0] >= cfg.min_edge], key=lambda x: -x[0])[:PROP_MAX_PER_DAY]
            for e, g, q in picks:
                rows.append(_prop_row(run_id, run_at, date, g.game_id, st.name, q, q.best_side, paper_stake(edge_conviction(e))))
        elif st.kind == "prop_control":
            for g, q in sorted(cands, key=lambda x: (-x[1].point, x[1].player_id))[:PROP_MAX_PER_DAY]:
                rows.append(_prop_row(run_id, run_at, date, g.game_id, st.name, q, "over", CONTROL_STAKE))
    return rows
