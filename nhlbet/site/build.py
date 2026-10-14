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
from nhlbet.report.ranking import WEIGHTS, rank_picks
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


def ai_parlays(slate: list[SlateGame], cfg: RiskConfig, now: datetime, weights: dict | None = None) -> list[dict]:
    """Today's AI-built fake-money parlays for display (the same builder the paper trading uses; nothing is stored here)."""
    import pandas as pd
    from nhlbet.risk.parlays import PARLAY_STRATEGIES, build_parlays
    ts = pd.Timestamp(now)
    rows = build_parlays(rank_picks(slate, cfg, now=ts, weights=weights), slate, "", "", "", now=ts)
    desc = {s.name: s.description for s in PARLAY_STRATEGIES}
    return [{"strategy": r["strategy"], "description": desc[r["strategy"]], "n_legs": r["n_legs"], "decimal": r["decimal"], "p_model": r["p_model"], "p_market": r["p_market"],
             "stake": r["stake"], "ev": r["ev"], "legs": [{"label": l["label"], "game": l["game"], "decimal": l["decimal"], "p_model": l["p_model"], "p_market": l["p_market"]}
                                                          for l in json.loads(r["legs"])]} for r in rows]


def build_payload(slate: list[SlateGame], cfg: RiskConfig, perf: dict | None, model_entry: dict, odds_meta: dict | None, date: str,
                  run_type: str, now: datetime | None = None, notes: list[str] | None = None, shadow=None,
                  public_safe: bool | None = None, rank_weights: dict | None = None, evidence: list | None = None) -> dict:
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
        "ranking": rank_picks(slate, cfg, weights=rank_weights),
        "evidence": evidence or [],
        "ranking_weights": {**WEIGHTS, **(rank_weights or {})},
        "ai_parlays": ai_parlays(slate, cfg, now, rank_weights),
        "track": track,
        "paper": ([] if shadow is None or len(shadow) == 0 else
                  [{"strategy": k, **{c: (None if v != v else v) for c, v in r.items()}} for k, r in shadow.iterrows()]),
        "disclaimer": DISCLAIMER.replace("> ", "").replace("**", ""),
    }
    return _clean(payload)


def static_games_table(payload: dict) -> str:
    """The core table as plain HTML for readers without JavaScript: every game with model and market probabilities, the best price, the edge, when the prices
    were captured and the decision under the default settings. Everything is escaped."""
    import html
    e = html.escape
    rows = []
    for g in payload.get("games", []):
        sides = g.get("sides") or {}
        best = max(sides, key=lambda k: sides[k]["ev"]) if sides else None
        sd = sides.get(best) if best else None
        mkt = f"{sides['home']['p_market'] * 100:.1f}%" if sides.get("home") else "no odds"
        if sd:
            side_cells = (f"<td data-label='Best side'>{e(str(sd['team']))}</td><td class='num' data-label='Best price'>{sd['best_decimal']:.2f}</td>"
                          f"<td class='num' data-label='Edge'>{sd['edge'] * 100:+.1f} pts</td><td class='num' data-label='EV per $1'>{sd['ev'] * 100:+.1f}%</td>")
            decision = "Qualifies under the default rules" if sd["qualifies"] else "No bet: " + e(str(g.get("blocked") or (sd["fails"][0] if sd["fails"] else "below the rules")))
        else:
            side_cells = "<td>-</td><td>-</td><td>-</td><td>-</td>"
            decision = "No bet: " + e(str(g.get("blocked") or "no odds"))
        start = e(str(g.get("start_utc") or ""))[:16].replace("T", " ")
        rows.append(f"<tr><td data-label='Game'><b>{e(g['away'])} @ {e(g['home'])}</b><br><span class='muted'>{start} UTC</span></td>"
                    f"<td class='num' data-label='Model P(home)'>{g['p_home'] * 100:.1f}%</td><td class='num' data-label='Market P(home), no vig'>{mkt}</td>{side_cells}"
                    f"<td data-label='Prices as of'>{e(str(g.get('odds_captured_at') or '-'))[:16].replace('T', ' ')}</td><td data-label='Decision'>{decision}</td></tr>")
    model = payload.get("model", {})
    head = ("<tr><th>Game</th><th class='num'>Model P(home)</th><th class='num'>Market P(home), no vig</th><th>Best side</th><th class='num'>Best price</th><th class='num'>Edge</th>"
            "<th class='num'>EV per $1</th><th>Prices as of</th><th>Decision</th></tr>")
    return (f"<div class='banner'>JavaScript is off, so this is the basic table. Model {e(str(model.get('version')))} ({e(str(model.get('status')))}), generated {e(str(payload.get('generated_at')))}. "
            "Research and education only; not betting advice. See <a href='validation.html'>the evidence page</a> for whether any of this has an edge.</div>"
            f"<div class='card flat scroll'><table class='rtable'><thead>{head}</thead><tbody>{''.join(rows) or '<tr><td colspan=9>No games to show.</td></tr>'}</tbody></table></div>")


def _review_text(rv: dict | None) -> str:
    """Review status, reason and sources of one pick as escaped HTML (https links only)."""
    import html
    import re
    e = html.escape
    if not rv:
        return "-"
    label = "UNREVIEWED" if rv.get("status") == "UNREVIEWED" else {"flag": "Flagged", "veto": "Vetoed"}.get(rv.get("decision"), "Kept")
    out = f"<b>{label}</b>"
    note = rv.get("warning") or rv.get("reason")
    if note:
        out += f"<br><span class='muted'>{e(str(note))}</span>"
    links = [f"<a href='{e(x['url'], quote=True)}' rel='noopener noreferrer'>{e(x['url'].split('//', 1)[1].split('/', 1)[0])}</a>" for x in (rv.get("sources") or [])
             if isinstance(x.get("url"), str) and re.fullmatch(r"https://\S+", x["url"])]
    if links:
        out += "<br><span class='muted'>Sources: " + ", ".join(links) + "</span>"
    if rv.get("reviewed_at"):
        out += f"<br><span class='muted'>checked {e(str(rv['reviewed_at']))[:16].replace('T', ' ')} UTC{' (rule-based)' if rv.get('decided_by') == 'code' else ''}</span>"
    return out


def static_picks_table(payload: dict) -> str:
    """Today's picks that went through the pre-game review, and the vetoed ones with the reason, as plain HTML (no JavaScript needed). Everything is escaped."""
    import html
    e = html.escape

    def rows(items):
        return "".join(f"<tr><td data-label='Pick'><b>{e(str(r['pick']))}</b><br><span class='muted'>{e(str(r['game']))}</span></td><td class='num' data-label='Model'>{r['p_model'] * 100:.1f}%</td>"
                       f"<td class='num' data-label='Market, no vig'>{r['p_market'] * 100:.1f}%</td><td class='num' data-label='Best price'>{r['price']:.2f}</td>"
                       f"<td data-label='Pre-game review'>{_review_text(r.get('review'))}</td></tr>" for r in items)
    head = "<tr><th>Pick</th><th class='num'>Model</th><th class='num'>Market, no vig</th><th class='num'>Best price</th><th>Pre-game review</th></tr>"
    picks = [r for r in payload.get("ranking", []) if r.get("review")]
    out = ""
    if picks:
        out += (f"<h2>Picks and their pre-game review</h2><div class='card flat scroll'><table class='rtable'><thead>{head}</thead><tbody>{rows(picks)}</tbody></table></div>")
    if payload.get("vetoed"):
        out += (f"<h2>Vetoed by the pre-game review ({len(payload['vetoed'])})</h2><div class='card flat scroll'><table class='rtable'><thead>{head}</thead>"
                f"<tbody>{rows(payload['vetoed'])}</tbody></table></div>")
    return out


def render_html(payload: dict) -> str:
    tpl = (HERE / "template.html").read_text()
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")      # cannot close the <script> tag
    return (tpl.replace("/*__THEME__*/", (HERE / "theme.css").read_text()).replace("/*__CORE__*/", (HERE / "core.js").read_text())
               .replace("/*__CHROME__*/", (HERE / "chrome.js").read_text()).replace("/*__UI__*/", (HERE / "ui.js").read_text())
               .replace("__STATIC__", static_games_table(payload) + static_picks_table(payload)).replace("__DATA__", data).replace("__TITLE__", f"NHL picks {payload['date']}"))


def build_site(payload: dict, out_dir: str | Path = "site") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(payload, indent=1, allow_nan=False))
    page = render_html(payload)
    (out / "index.html").write_text(page)
    (out / "archive").mkdir(exist_ok=True)
    (out / "archive" / f"{payload['date']}.html").write_text(page)            # the last run of each day stays browsable as history
    return out / "index.html"
