"""Build ``site/bets.html``: a personal bet tracker (stored only in the viewer's browser) plus the fake-money paper-trading bets.

Personal bets never touch the repository. The fake bets come from the ``shadow_bets`` table (pretend stakes, settled like real bets).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.report.markdown import DISCLAIMER
from nhlbet.risk.shadow import STRATEGIES
from nhlbet.site.build import HERE, _clean

MAX_BETS = 4000          # keep the page small: the most recent paper bets only


def paper_bets(store: Store, limit: int = MAX_BETS) -> pd.DataFrame:
    """Latest pretend bet per (game, strategy) including unsettled ones. result: pending | won | lost."""
    r = store.df("SELECT * FROM shadow_bets")
    if r.empty:
        return pd.DataFrame()
    final = r.sort_values("run_at").groupby(["game_id", "strategy"], as_index=False).tail(1)
    final = final[(final.action == "BET") & (final.stake > 0)]
    g = store.df("SELECT game_id, home, away, home_win FROM games")
    d = final.merge(g, on="game_id", how="left")
    d["result"] = np.where(d.home_win.isna(), "pending", np.where((d.side == "home") == (d.home_win == 1), "won", "lost"))
    d["game"] = d.away.fillna("?") + " @ " + d.home.fillna("?")
    d["pick"] = d.team.fillna("?") + " moneyline"
    d = d.rename(columns={"game_date": "date"})
    return d.sort_values(["date", "game_id"]).tail(limit)[["date", "strategy", "game_id", "game", "pick", "decimal", "stake", "result"]].reset_index(drop=True)


def build_bets_payload(store: Store, games_payload: list[dict], bankroll: float, generated_at: str, date: str) -> dict:
    pb = paper_bets(store)
    return _clean({"generated_at": generated_at, "date": date, "bankroll": bankroll, "games": [
                       {"away": g["away"], "home": g["home"], "model_side": g.get("model_side"),
                        "sides": {k: {"team": v["team"], "best_decimal": v["best_decimal"]} for k, v in g["sides"].items()}} for g in games_payload],
                   "paper_strategies": [{"name": s.name, "description": s.description} for s in STRATEGIES],
                   "paper_bets": pb.to_dict("records") if len(pb) else [],
                   "disclaimer": DISCLAIMER.replace("> ", "").replace("**", "")})


def render_bets_html(payload: dict) -> str:
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return ((HERE / "bets_template.html").read_text().replace("/*__CORE__*/", (HERE / "bets_core.js").read_text())
            .replace("/*__UI__*/", (HERE / "bets_ui.js").read_text()).replace("__DATA__", data))


def build_bets_page(store: Store, games_payload: list[dict], bankroll: float, generated_at: str, date: str, out_dir: str | Path = "site") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bets.html").write_text(render_bets_html(build_bets_payload(store, games_payload, bankroll, generated_at, date)))
    return out / "bets.html"
