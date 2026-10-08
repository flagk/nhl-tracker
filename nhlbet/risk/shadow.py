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
from nhlbet.odds.props import STATS
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


TRUST_FULL_EDGE = 0.03     # experimental markets: a weighted edge (trust weight x raw edge) of 3 points earns the $30 maximum


def trust_conviction(edge: float, weight: float) -> float:
    """Conviction for an experimental market: the raw edge scaled by how much that kind of model has earned trust (see nhlbet.report.ranking.learn_weights)."""
    return edge * weight / TRUST_FULL_EDGE


def _trust(weights: dict | None, key: str) -> float:
    from nhlbet.report.ranking import WEIGHTS
    return float((weights or {}).get(key, WEIGHTS.get(key, 0.2)))


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
    Strategy("always_over", "CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost)", "alt_control", (("market", "totals"), ("role", "over"))),
    Strategy("under_edge", "Goals model: $5-$30 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?)", "alt_role_edge", (("market", "totals"), ("role", "under"))),
    Strategy("over_edge", "Goals model: $5-$30 on the Over only, when it has a 2%+ edge", "alt_role_edge", (("market", "totals"), ("role", "over"))),
    Strategy("puckline_dog_edge", "Goals model: $5-$30 on the +1.5 underdog side of the puck line, when it has a 2%+ edge", "alt_role_edge", (("market", "spreads"), ("role", "dog"))),
    Strategy("puckline_fav_edge", "Goals model: $5-$30 on the -1.5 favourite side of the puck line, when it has a 2%+ edge", "alt_role_edge", (("market", "spreads"), ("role", "fav"))),
    Strategy("always_under", "CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over)", "alt_control", (("market", "totals"), ("role", "under"))),
    Strategy("puckline_dog_control", "CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs)", "alt_control", (("market", "spreads"), ("role", "dog"))),
    Strategy("underdog_ml", "$5-$30 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does", "ml_dog_edge"),
    Strategy("underdog_ml_control", "CONTROL: flat $10 on the moneyline underdog of every game (no skill)", "ml_dog_control"),
    Strategy("home_ml_control", "CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?)", "ml_home_control"),
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


ALT_MIN_EDGE = 0.02      # the role-specific strategies bet at a 2% edge, a bit looser than the live policy, to collect more evidence on each bet type


def _role_ok(q, role: str) -> bool:
    """over / under for totals; dog (+ handicap) / fav (- handicap) for the puck line."""
    return q.side == role if role in ("over", "under") else (q.point > 0 if role == "dog" else q.point < 0)


def _alt_usable(g, cfg: RiskConfig) -> bool:
    return not (g.ctx.get("model_status") == "ALERT" or (g.ctx.get("odds_stale") and not cfg.allow_stale_odds))


def shadow_bets(games: Sequence, cfg: RiskConfig, run_id: str, run_at: str, date: str,
                strategies: Sequence[Strategy] = STRATEGIES, weights: dict | None = None) -> list[dict]:
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
                role = dict(st.overrides).get("role")
                if st.kind in ("alt_control", "alt_role_edge"):
                    cand = [x for x in qs if _role_ok(x, role)]
                    if not cand:
                        rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name))
                        continue
                    q = max(cand, key=lambda x: x.edge)
                    if st.kind == "alt_control":
                        rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name, q, CONTROL_STAKE))
                    elif q.edge >= ALT_MIN_EDGE:
                        rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name, q, paper_stake(trust_conviction(q.edge, _trust(weights, market)))))
                    else:
                        rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name))
                    continue
                else:
                    q = max(qs, key=lambda x: x.edge if st.kind == "alt_edge" else x.model_prob)
                if st.kind == "alt_edge" and q.edge < cfg.min_edge:
                    rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name))
                    continue
                w = _trust(weights, market)
                conv = (trust_conviction(q.edge, w) if st.kind == "alt_edge"
                        else side_conviction(q.market_prob + w * (q.model_prob - q.market_prob)))      # conviction from the trust-blended chance, not the raw model chance
                rows.append(_alt_row(run_id, run_at, date, g.game_id, st.name, q, paper_stake(conv)))
            continue
        for g in games:
            if not _usable(g.quotes, cfg, g.ctx):
                rows.append(_row(run_id, run_at, date, g.game_id, st.name))
                continue
            if st.kind in ("ml_dog_edge", "ml_dog_control", "ml_home_control"):
                qs = list(g.quotes.values())
                q = (g.quotes.get("home") or qs[0]) if st.kind == "ml_home_control" else min(qs, key=lambda x: x.market_prob)
                if st.kind == "ml_dog_edge" and q.edge < ALT_MIN_EDGE:
                    rows.append(_row(run_id, run_at, date, g.game_id, st.name))
                    continue
                stake = paper_stake(edge_conviction(q.edge)) if st.kind == "ml_dog_edge" else CONTROL_STAKE
                rows.append(_row(run_id, run_at, date, g.game_id, st.name, "BET", q, stake, shrink_to_market(q.model_prob, q.market_prob, cfg.trust_w0, cfg.trust_d0), q.ev))
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
    Strategy("sog_over_edge", f"Player shots, OVERS only: $5-$30 when the model sees a 2%+ edge on the over (10+ games of history), best {PROP_MAX_PER_DAY} of the day", "prop_edge_over"),
    Strategy("sog_under_edge", f"Player shots, UNDERS only: $5-$30 when the model sees a 2%+ edge on the under (10+ games of history), best {PROP_MAX_PER_DAY} of the day", "prop_edge_under"),
    Strategy("pts_edge", f"Player POINTS (goals + assists): $5-$30 on the over/under the model likes (3%+ edge, 10+ games of history), best {PROP_MAX_PER_DAY} of the day (experimental market)", "prop_edge", (("stat", "points"),)),
    Strategy("pts_over_edge", f"Player POINTS, OVERS only: $5-$30 at a 2%+ edge, best {PROP_MAX_PER_DAY} of the day", "prop_edge_over", (("stat", "points"),)),
    Strategy("pts_under_edge", f"Player POINTS, UNDERS only: $5-$30 at a 2%+ edge, best {PROP_MAX_PER_DAY} of the day", "prop_edge_under", (("stat", "points"),)),
    Strategy("pts_over_control", f"CONTROL: flat $10 on the Over for {PROP_MAX_PER_DAY} players priced for points, highest line first (no model; shows what prop vig costs)", "prop_control", (("stat", "points"),)),
    Strategy("ast_edge", f"Player ASSISTS: $5-$30 on the over/under the model likes (3%+ edge, 10+ games of history), best {PROP_MAX_PER_DAY} of the day (experimental market)", "prop_edge", (("stat", "assists"),)),
    Strategy("ast_over_edge", f"Player ASSISTS, OVERS only: $5-$30 at a 2%+ edge, best {PROP_MAX_PER_DAY} of the day", "prop_edge_over", (("stat", "assists"),)),
    Strategy("ast_under_edge", f"Player ASSISTS, UNDERS only: $5-$30 at a 2%+ edge, best {PROP_MAX_PER_DAY} of the day", "prop_edge_under", (("stat", "assists"),)),
    Strategy("ast_over_control", f"CONTROL: flat $10 on the Over for {PROP_MAX_PER_DAY} players priced for assists (no model; shows what prop vig costs)", "prop_control", (("stat", "assists"),)),
    Strategy("goal_edge", f"ANYTIME GOALSCORER: $5-$30 on players the model rates 3%+ likelier to score than the market, best {PROP_MAX_PER_DAY} of the day (experimental; the market's margin is estimated)", "prop_edge_over", (("stat", "goals"),)),
    Strategy("goal_control", f"CONTROL: flat $10 on the {PROP_MAX_PER_DAY} most likely anytime scorers by market price (no model; shows what the margin costs)", "prop_control", (("stat", "goals"),)),
    Strategy("sog_over_control", f"CONTROL: flat $10 on the Over for the {PROP_MAX_PER_DAY} players with the highest shots lines (no model; shows what prop vig costs)", "prop_control"),
)


def _prop_row(run_id, run_at, date, game_id, strategy, q, side: str, stake: float) -> dict:
    over = side == "over"
    p_side = q.p_over if over else 1 - q.p_over
    m_side = q.p_over_market if over else 1 - q.p_over_market
    return {"run_id": run_id, "run_at": run_at, "game_id": game_id, "strategy": strategy, "game_date": date, "player_id": q.player_id, "name": q.name, "side": side,
            "label": (f"{q.name} to score (anytime)" if q.one_sided else f"{q.name} {'Over' if over else 'Under'} {q.point:g}"), "market": STATS[q.stat]["store"], "point": q.point, "book": None, "decimal": q.over_price if over else q.under_price, "stake": stake,
            "p_model": p_side, "p_market": m_side, "edge": p_side - m_side, "ev": q.ev_over if over else q.ev_under, "lam": q.lam}


def prop_shadow_bets(games: Sequence, cfg: RiskConfig, run_id: str, run_at: str, date: str, strategies: Sequence[Strategy] = PROP_STRATEGIES, weights: dict | None = None) -> list[dict]:
    """Paper bets on player shots across the whole slate (capped per day). ``games``: objects with ``game_id``, ``props`` (PropQuote list) and ``ctx``."""
    all_cands = [(g, q) for g in games if _alt_usable(g, cfg) for q in (getattr(g, "props", None) or [])]
    rows: list[dict] = []
    for st in strategies:
        stat = dict(st.overrides).get("stat", "sog")
        cands = [(g, q) for g, q in all_cands if q.stat == stat]
        if st.kind == "prop_edge":
            scored = [(q.edge(q.best_side), g, q) for g, q in cands if q.n_prev >= PROP_MIN_GAMES and q.has_side(q.best_side)]
            picks = sorted([x for x in scored if x[0] >= cfg.min_edge], key=lambda x: -x[0])[:PROP_MAX_PER_DAY]
            for e, g, q in picks:
                rows.append(_prop_row(run_id, run_at, date, g.game_id, st.name, q, q.best_side, paper_stake(trust_conviction(e, _trust(weights, STATS[q.stat]['store'])))))
        elif st.kind in ("prop_edge_over", "prop_edge_under"):
            side = st.kind.rsplit("_", 1)[1]
            floor = cfg.min_edge if st.name == "goal_edge" else ALT_MIN_EDGE
            scored = [((q.edge(side)), g, q) for g, q in cands if q.n_prev >= PROP_MIN_GAMES and q.has_side(side)]
            for e, g, q in sorted([x for x in scored if x[0] >= floor], key=lambda x: -x[0])[:PROP_MAX_PER_DAY]:
                rows.append(_prop_row(run_id, run_at, date, g.game_id, st.name, q, side, paper_stake(trust_conviction(e, _trust(weights, STATS[q.stat]['store'])))))
        elif st.kind == "prop_control":
            for g, q in sorted(cands, key=lambda x: (-x[1].point, -x[1].p_over_market, x[1].player_id))[:PROP_MAX_PER_DAY]:
                rows.append(_prop_row(run_id, run_at, date, g.game_id, st.name, q, "over", CONTROL_STAKE))
    return rows
