"""Bet selection and sizing. "No bet" is the default and the common outcome - picks are never forced.

Pipeline per game (after the model-health, early-season and odds-availability guards): both sides' best price/market probability come from ``nhlbet.odds.edge``. Then

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


def _pol(key: str, default):
    from nhlbet.policy import policy_get
    return policy_get("staking", key, default=default)


def _checked_kelly() -> float:
    """The policy's Kelly fraction, refusing a value above the policy's own ceiling."""
    k, cap = float(_pol("kelly_fraction", 0.25)), float(_pol("kelly_fraction_max", 0.25))
    if not 0 < k <= cap:
        raise ValueError(f"config/policy.yaml: staking.kelly_fraction={k} must be in (0, kelly_fraction_max={cap}]")
    return k


@dataclass
class RiskConfig:
    """Every default comes from config/policy.yaml (``staking:``); explicit arguments override it (tests, what-if runs), the live pipeline uses the defaults."""
    bankroll: float = 1000.0
    kelly_fraction: float = field(default_factory=_checked_kelly)           # fraction of full Kelly
    max_bet_pct: float = field(default_factory=lambda: float(_pol("max_bet_pct", 0.02)))
    max_game_exposure_pct: float = field(default_factory=lambda: float(_pol("max_game_exposure_pct", 0.03)))     # all markets on one game together
    max_daily_exposure_pct: float = field(default_factory=lambda: float(_pol("max_daily_exposure_pct", 0.05)))
    max_bets_per_day: int = field(default_factory=lambda: int(_pol("max_bets_per_day", 5)))
    min_edge: float = field(default_factory=lambda: float(_pol("min_edge", 0.03)))                  # raw edge: model prob - no-vig market prob
    min_edge_unconfirmed_goalie: float = field(default_factory=lambda: float(_pol("min_edge_unconfirmed_goalie", 0.04)))
    min_ev: float = field(default_factory=lambda: float(_pol("min_ev", 0.01)))                      # EV per $1 after shrinkage
    min_prob: float = field(default_factory=lambda: float(_pol("min_prob", 0.40)))                  # minimum model win probability for the chosen side
    max_decimal: float = field(default_factory=lambda: float(_pol("max_decimal", 4.0)))
    max_disagreement: float = field(default_factory=lambda: float(_pol("max_disagreement", 0.12)))  # beyond this the model is presumed broken, not brilliant
    trust_w0: float = field(default_factory=lambda: float(_pol("trust_w0", 0.5)))
    trust_d0: float = field(default_factory=lambda: float(_pol("trust_d0", 0.06)))
    min_stake: float = field(default_factory=lambda: float(_pol("min_stake", 1.0)))
    allow_stale_odds: bool = field(default_factory=lambda: bool(_pol("allow_stale_odds", False)))
    min_games_played: int = field(default_factory=lambda: int(_pol("min_games_played", 10)))        # early-season sample-size guard


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

    def explain(self, show_book: bool = True) -> str:
        if self.action != "BET":
            return "No bet: " + "; ".join(self.reasons) if self.reasons else "No bet."
        at = f"{self.decimal:.2f} ({self.book})" if show_book and self.book else f"{self.decimal:.2f}"
        return (f"Bet ${self.stake:,.2f} ({self.stake_pct:.2%} of bankroll) on {self.team} at {at}. "
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


def game_block_reason(quotes, cfg: RiskConfig, ctx: Mapping) -> str | None:
    """Game-level reasons to pass regardless of price (model health, early season, no/stale odds)."""
    if ctx.get("model_status") == "ALERT":
        return "model health check is ALERT (performance drift) - recommendations suspended"
    if ctx.get("games_played_min") is not None and ctx["games_played_min"] < cfg.min_games_played:
        return f"early-season sample size: a team has only played {int(ctx['games_played_min'])} games (< {cfg.min_games_played})"
    if not quotes:
        return "no usable odds available"
    if ctx.get("odds_stale") and not cfg.allow_stale_odds:
        return "odds are stale (API unavailable) - will not bet on old prices"
    return None


def assess_sides(quotes: Mapping[str, SideQuote], cfg: RiskConfig, ctx: Mapping) -> list[tuple]:
    """Per side: (quote, shrunk prob, EV at best price using the shrunk prob, list of failed screens). Single source of truth
    for both the recommender and the website payload."""
    cands = []
    for q in quotes.values():
        fails = _screen(q, cfg, ctx)
        p_adj = shrink_to_market(q.model_prob, q.market_prob, cfg.trust_w0, cfg.trust_d0)
        ev = expected_value(p_adj, q.best_decimal)
        if not fails and ev < cfg.min_ev:
            fails = [f"expected value after shrinkage toward the market is only {ev:+.1%} (< {cfg.min_ev:.0%})"]
        cands.append((q, p_adj, ev, fails))
    return cands


def recommend_game(game_id, home: str, away: str, quotes: Mapping[str, SideQuote] | None, cfg: RiskConfig,
                   ctx: Mapping | None = None) -> Recommendation:
    """Evaluate one game. ``ctx`` keys: model_status ('OK'|'WARN'|'ALERT'), odds_stale (bool), goalie_confirmed (bool)."""
    ctx = ctx or {}
    rec = Recommendation(game_id, home, away)
    blocked = game_block_reason(quotes, cfg, ctx)
    if blocked:
        rec.reasons.append(blocked)
        return rec
    cands = assess_sides(quotes, cfg, ctx)
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
    """Enforce max bets/day (keep best EV), the per-game cap (every market on a game counts together) and total daily exposure (stakes scaled down proportionally)."""
    from nhlbet.risk.caps import apply_caps
    bets = [r for r in recs if r.action == "BET"]
    out = apply_caps([{"id": i, "game_id": r.game_id, "stake": r.stake, "ev": r.ev or 0.0} for i, r in enumerate(bets)], cfg.bankroll, cfg.max_bet_pct, cfg.max_game_exposure_pct,
                     cfg.max_daily_exposure_pct, cfg.max_bets_per_day, cfg.min_stake)
    for o, r in zip(out, bets):
        if o["dropped"]:
            r.action, r.stake, r.stake_pct = "NO_BET", 0.0, 0.0
            r.reasons = [o["reason"]]
        else:
            r.stake, r.stake_pct = o["stake"], o["stake"] / cfg.bankroll
            r.reasons += o["notes"]
    return recs


def recommend_slate(games: Iterable[tuple], cfg: RiskConfig | None = None) -> list[Recommendation]:
    """``games``: iterable of (game_id, home, away, quotes_or_None, ctx). Returns one Recommendation per game."""
    cfg = cfg or RiskConfig()
    recs = [recommend_game(gid, h, a, q, cfg, ctx) for gid, h, a, q, ctx in games]
    return apply_portfolio_caps(recs, cfg)
