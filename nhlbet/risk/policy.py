"""Bet selection and sizing. "No bet" is the default and the common outcome - picks are never forced.

Pipeline per game: both sides' best price/market probability come from ``nhlbet.odds.edge``. Then

 1. hard screens on the RAW edge (model prob - no-vig market prob) and other guards -> otherwise NO_BET, with reasons
 2. shrink the model probability toward the market (bigger disagreement -> less trust)
 3. size with fractional Kelly on the *shrunk* probability at the best available price, capped per bet
 4. portfolio caps: max bets and total daily exposure (stakes are scaled down proportionally to fit)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from nhlbet.odds.edge import SideQuote
from nhlbet.odds.math import expected_value
from nhlbet.risk.kelly import kelly_fraction
from nhlbet.risk.shrink import shrink_to_market


@dataclass
class RiskConfig:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25           # quarter Kelly
    max_bet_pct: float = 0.02              # hard cap per bet (share of bankroll)
    max_daily_exposure_pct: float = 0.05   # hard cap on total stakes per day
    max_bets_per_day: int = 5
    min_edge: float = 0.03                 # raw edge: model prob - no-vig market prob
    min_edge_unconfirmed_goalie: float = 0.04
    min_ev: float = 0.01                   # EV per $1 after shrinkage
    min_prob: float = 0.40                 # minimum model win probability for the chosen side (no long-shot noise)
    max_decimal: float = 4.0
    max_disagreement: float = 0.12         # beyond this the model is presumed broken, not brilliant
    trust_w0: float = 0.5
    trust_d0: float = 0.06
    min_stake: float = 1.0
    allow_stale_odds: bool = False


@dataclass
class Recommendation:
    game_id: int | None
    home: str
    away: str
    action: str = "NO_BET"                 # 'BET' | 'NO_BET'
    side: str | None = None
    team: str | None = None
    book: str | None = None
    decimal: float | None = None
    stake: float = 0.0
    stake_pct: float = 0.0
    model_prob: float | None = None
    adj_prob: float | None = None
    market_prob: float | None = None
    edge: float | None = None
    ev: float | None = None                # EV/$ using the shrunk probability
    kelly_full: float | None = None
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        if self.action != "BET":
            return "No bet: " + "; ".join(self.reasons) if self.reasons else "No bet."
        return (f"Bet ${self.stake:,.2f} ({self.stake_pct:.2%} of bankroll) on {self.team} at {self.decimal:.2f} ({self.book}). "
                f"The model gives {self.team} {self.model_prob:.1%} vs the market's no-vig {self.market_prob:.1%} (edge {self.edge:+.1%}); "
                f"after shrinking toward the market the working probability is {self.adj_prob:.1%}, worth {self.ev:+.1%} per $1 at that price."
                + (" " + " ".join(self.reasons) if self.reasons else ""))


def _screen(q: SideQuote, cfg: RiskConfig, ctx: Mapping) -> list[str]:
    fails = []
    need = cfg.min_edge if ctx.get("goalie_confirmed", True) else cfg.min_edge_unconfirmed_goalie
    if q.edge < need:
        fails.append(f"edge {q.edge:+.1%} below the {need:.0%} minimum" + ("" if ctx.get("goalie_confirmed", True) else " (starting goalie unconfirmed)"))
    if q.model_prob < cfg.min_prob:
        fails.append(f"model win probability {q.model_prob:.1%} below the {cfg.min_prob:.0%} confidence floor")
    if q.best_decimal > cfg.max_decimal:
        fails.append(f"price {q.best_decimal:.2f} above the {cfg.max_decimal:.1f} long-shot cap")
    if abs(q.edge) > cfg.max_disagreement:
        fails.append(f"model disagrees with the market by {abs(q.edge):.0%} - more likely a model error than an edge")
    return fails


def recommend_game(game_id, home: str, away: str, quotes: Mapping[str, SideQuote] | None, cfg: RiskConfig,
                   ctx: Mapping | None = None) -> Recommendation:
    """Evaluate one game. ``ctx`` keys: model_status ('OK'|'WARN'|'ALERT'), odds_stale (bool), goalie_confirmed (bool)."""
    ctx = ctx or {}
    rec = Recommendation(game_id, home, away)
    if ctx.get("model_status") == "ALERT":
        rec.reasons.append("model health check is ALERT (performance drift) - recommendations suspended")
        return rec
    if not quotes:
        rec.reasons.append("no usable odds available")
        return rec
    if ctx.get("odds_stale") and not cfg.allow_stale_odds:
        rec.reasons.append("odds are stale (API unavailable) - will not bet on old prices")
        return rec
    cands = []
    for q in quotes.values():
        fails = _screen(q, cfg, ctx)
        p_adj = shrink_to_market(q.model_prob, q.market_prob, cfg.trust_w0, cfg.trust_d0)
        ev = expected_value(p_adj, q.best_decimal)
        if not fails and ev < cfg.min_ev:
            fails = [f"expected value after shrinkage toward the market is only {ev:+.1%} (< {cfg.min_ev:.0%})"]
        cands.append((q, p_adj, ev, fails))
    passing = [c for c in cands if not c[3]]
    if not passing:
        q, _, _, fails = max(cands, key=lambda c: c[0].edge)          # explain using the side closest to qualifying
        rec.model_prob, rec.market_prob, rec.edge = q.model_prob, q.market_prob, q.edge
        rec.reasons = fails if q.edge > 0 else ["no side has a positive edge over the market"]
        return rec
    best, rec.adj_prob, best_ev, _ = max(passing, key=lambda c: c[2])
    q = best
    full = kelly_fraction(rec.adj_prob, q.best_decimal)
    pct = min(cfg.kelly_fraction * full, cfg.max_bet_pct)
    amount = round(cfg.bankroll * pct, 2)
    if amount < cfg.min_stake:
        rec.reasons = [f"Kelly stake ${amount:.2f} is below the ${cfg.min_stake:.0f} minimum"]
        return rec
    rec.action, rec.side, rec.team, rec.book, rec.decimal = "BET", q.side, q.team, q.best_book, q.best_decimal
    rec.stake, rec.stake_pct, rec.model_prob, rec.market_prob, rec.edge, rec.ev, rec.kelly_full = amount, amount / cfg.bankroll, q.model_prob, q.market_prob, q.edge, best_ev, full
    rec.reasons = ["Stake capped at the per-bet limit."] if cfg.kelly_fraction * full > cfg.max_bet_pct else []
    return rec


def apply_portfolio_caps(recs: list[Recommendation], cfg: RiskConfig) -> list[Recommendation]:
    """Enforce max bets/day (keep best EV) and total daily exposure (scale stakes down proportionally)."""
    bets = sorted((r for r in recs if r.action == "BET"), key=lambda r: -(r.ev or 0))
    for r in bets[cfg.max_bets_per_day:]:
        r.action, r.stake, r.stake_pct = "NO_BET", 0.0, 0.0
        r.reasons = [f"more than {cfg.max_bets_per_day} qualifying bets today; kept only the highest-EV ones"]
    kept = [r for r in bets[:cfg.max_bets_per_day]]
    total, cap = sum(r.stake for r in kept), cfg.bankroll * cfg.max_daily_exposure_pct
    if total > cap > 0:
        scale = cap / total
        for r in kept:
            r.stake = math.floor(r.stake * scale * 100) / 100                                   # round DOWN: never exceed the cap
            r.stake_pct = r.stake / cfg.bankroll
            r.reasons.append(f"Scaled by {scale:.2f} to respect the {cfg.max_daily_exposure_pct:.0%} daily exposure cap.")
            if r.stake < cfg.min_stake:
                r.action, r.stake, r.stake_pct = "NO_BET", 0.0, 0.0
                r.reasons = ["stake fell below the minimum after the daily exposure cap"]
    return recs


def recommend_slate(games: Iterable[tuple], cfg: RiskConfig | None = None) -> list[Recommendation]:
    """``games``: iterable of (game_id, home, away, quotes_or_None, ctx). Returns one Recommendation per game."""
    cfg = cfg or RiskConfig()
    recs = [recommend_game(gid, h, a, q, cfg, ctx) for gid, h, a, q, ctx in games]
    return apply_portfolio_caps(recs, cfg)
