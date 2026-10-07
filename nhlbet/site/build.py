"""Build the self-contained picks page (``site/index.html`` + ``site/picks.json``) from a day's slate.

The page is static: the data is embedded, all unit/bankroll/stake/parlay maths runs in the browser (``core.js``, parity-tested
against the Python policy), and no API key or server is involved. Open ``site/index.html`` from disk or any static host.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from nhlbet.privacy import is_public_safe
from nhlbet.report.markdown import DISCLAIMER
from nhlbet.report.ranking import rank_picks
from nhlbet.report.slate import SlateGame
from nhlbet.report.stats import feature_label, model_inputs
from nhlbet.risk.policy import RiskConfig, assess_sides, game_block_reason

HERE = Path(__file__).parent
PARLAY_MAX_LEGS = 4
PARLAY_MAX_PCT = 0.005          # never suggest more than 0.5% of bankroll on a parlay


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python, Timestamps -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "isoformat"):
        try:
            return None if str(o) == "NaT" else o.isoformat()
        except (ValueError, AttributeError):
            return None
    return o


def game_payload(s: SlateGame, cfg: RiskConfig, public_safe: bool = False) -> dict:
    from nhlbet.risk.kelly import kelly_fraction

    blocked = game_block_reason(s.quotes, cfg, s.ctx)
    sides: dict[str, dict] = {}
    if s.quotes:
        for q, p_adj, ev, fails in assess_sides(s.quotes, cfg, s.ctx):
            books = {b["book"]: b[q.side] for b in s.books if b.get(q.side)}
            sides[q.side] = {"team": q.team, "p_model": q.model_prob, "p_market": q.market_prob, "p_adj": p_adj, "edge": q.edge, "ev": ev,
                             "kelly_full": kelly_fraction(p_adj, q.best_decimal), "best_book": None if public_safe else q.best_book,
                             "best_decimal": q.best_decimal, "books": {} if public_safe else books, "fails": ([blocked] if blocked else fails), "qualifies": blocked is None and not fails}
    return {"game_id": s.game_id, "start_utc": s.start_utc, "home": s.home, "away": s.away, "home_goalie": s.home_goalie,
            "away_goalie": s.away_goalie, "goalie_status": s.goalie_status, "p_home": s.p_home, "blocked": blocked, "notes": s.notes,
            "model_side": ("home" if s.p_home >= 0.5 else "away") if sides else None, "odds_stale": s.odds_stale, "odds_captured_at": s.odds_captured_at, "sides": sides,
            "alt": [{"market": q.market, "side": q.side, "label": q.label, "point": q.point, "p_model": q.model_prob, "p_market": q.market_prob,
                     "p_push": q.p_push, "edge": q.edge, "ev": q.ev, "best_decimal": q.best_decimal, "best_book": None if public_safe else q.best_book,
                     "n_books": q.n_books} for q in (getattr(s, "alt", None) or [])], "goals": getattr(s, "goals", None) or None,
            "props": [{"player_id": q.player_id, "name": q.name, "team": q.team, "opp": q.opp, "point": q.point, "lam": q.lam, "p_over": q.p_over, "p_over_market": q.p_over_market,
                       "over_price": q.over_price, "under_price": q.under_price, "ev_over": q.ev_over, "ev_under": q.ev_under, "edge_over": q.edge_over, "take": q.take, "best_side": q.best_side,
                       "n_books": q.n_books, "n_prev": q.n_prev, "history": q.history, "avg_season": q.avg_season, "avg_l10": q.avg_l10, "hit_l10": q.hit_l10, "hit_l20": q.hit_l20, "stat": getattr(q, "stat", "sog"), "one_sided": getattr(q, "one_sided", False), "captured_at": getattr(q, "captured_at", None)}
                      for q in sorted((getattr(s, "props", None) or []), key=lambda x: -max(x.ev_over, x.ev_under))[:15]],
            "stats": getattr(s, "stats", None) or [],
            "drivers": [{"feature": d["feature"], "label": feature_label(d["feature"]), "value": d["value"], "raw": d.get("raw")} for d in (getattr(s, "drivers", None) or [])],
            "server_action": s.rec.action, "server_side": s.rec.side, "server_stake": s.rec.stake}


def build_payload(slate: list[SlateGame], cfg: RiskConfig, perf: dict | None, model_entry: dict, odds_meta: dict | None, date: str,
                  run_type: str, now: datetime | None = None, notes: list[str] | None = None, shadow=None,
                  public_safe: bool | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ps = is_public_safe() if public_safe is None else public_safe
    perf = perf or {}
    drift = model_entry.get("drift", {})
    track = {"resolved_games": perf.get("resolved_games", 0), "bets": perf.get("bets"), "model": perf.get("model"), "vs_market": perf.get("vs_market"),
             "clv": perf.get("clv"), "drawdown": perf.get("drawdown"), "no_bet_rate": perf.get("no_bet_rate")}
    payload = {
        "generated_at": now.isoformat(timespec="seconds"), "public_safe": ps, "date": date, "run_type": run_type, "notes": notes or [],
        "model": {"version": model_entry.get("version"), "p_source": model_entry.get("p_source"), "status": drift.get("status", "UNKNOWN"),
                  "reasons": drift.get("performance", {}).get("reasons", []), "train_end": model_entry.get("train_end"),
                  "inputs": model_inputs(list(model_entry.get("features") or []))},
        "odds": odds_meta or {"enabled": False},
        "policy": {"bankroll": cfg.bankroll, "max_bet_pct": cfg.max_bet_pct, "max_daily_exposure_pct": cfg.max_daily_exposure_pct,
                   "max_bets_per_day": cfg.max_bets_per_day, "min_stake": cfg.min_stake, "min_edge": cfg.min_edge,
                   "parlay_max_legs": PARLAY_MAX_LEGS, "parlay_max_pct": PARLAY_MAX_PCT},
        "games": [game_payload(s, cfg, ps) for s in slate],
        "ranking": rank_picks(slate, cfg),
        "track": track,
        "paper": ([] if shadow is None or len(shadow) == 0 else
                  [{"strategy": k, **{c: (None if v != v else v) for c, v in r.items()}} for k, r in shadow.iterrows()]),
        "disclaimer": DISCLAIMER.replace("> ", "").replace("**", ""),
    }
    return _clean(payload)


def render_html(payload: dict) -> str:
    tpl = (HERE / "template.html").read_text()
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")      # cannot close the <script> tag
    return (tpl.replace("/*__THEME__*/", (HERE / "theme.css").read_text()).replace("/*__CORE__*/", (HERE / "core.js").read_text())
               .replace("/*__CHROME__*/", (HERE / "chrome.js").read_text()).replace("/*__UI__*/", (HERE / "ui.js").read_text())
               .replace("__DATA__", data).replace("__TITLE__", f"NHL picks {payload['date']}"))


def build_site(payload: dict, out_dir: str | Path = "site") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(payload, indent=1, allow_nan=False))
    page = render_html(payload)
    (out / "index.html").write_text(page)
    (out / "archive").mkdir(exist_ok=True)
    (out / "archive" / f"{payload['date']}.html").write_text(page)            # the last run of each day stays browsable as history
    return out / "index.html"
