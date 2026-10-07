"""Tidy, stable datasets for BI tools (Power BI, Excel, Tableau): ``data/export/*.csv``, rebuilt on every daily run.

Design rules (so a dashboard built today keeps working): fixed column names and order, ISO dates, one row per game / day / bet,
empty cell = unknown, and public-safe (no bookmaker names or per-book prices; the single ``bet_price`` is the best price at the time
of the recommendation). Schemas are pinned by tests/test_export.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.report.betlog import final_recommendations, performance, resolved, shadow_resolved
from nhlbet.report.betlog import shadow_performance  # noqa: F401  (re-exported for callers)

CONTEXT_COLS = ["h_rest_days", "a_rest_days", "h_b2b", "a_b2b", "h_gp_season", "a_gp_season", "elo_prob", "is_rivalry", "days_into_season"]

GAMES_COLS = ["game_id", "game_date", "away", "home", "home_goalie", "away_goalie", "goalie_status", "model_version", "p_model_home", "p_market_home",
              "p_close_home", "edge_home", "home_win", "decision", "bet_side", "bet_team", "bet_price", "stake", "edge", "ev", "bet_won", "profit",
              "clv_ev", "reasons", "home_rest_days", "away_rest_days", "home_b2b", "away_b2b", "home_games_played", "away_games_played", "elo_prob_home",
              "is_rivalry", "days_into_season"]
DAILY_COLS = ["date", "games", "bets", "staked", "profit", "roi", "cum_staked", "cum_profit", "model_log_loss", "market_close_log_loss", "avg_clv"]
PAPER_COLS = ["date", "strategy", "game_id", "market", "point", "away", "home", "side", "team", "price", "stake", "won", "profit", "clv_ev"]
ALT_COLS = ["game_id", "game_date", "away", "home", "market", "side", "label", "point", "p_model", "p_market", "p_push", "edge", "ev", "best_price", "n_books",
            "lam_home", "lam_away", "exp_total", "outcome"]
PAPER_SUMMARY_COLS = ["strategy", "bets", "staked", "profit", "roi", "roi_lo", "roi_hi", "win_rate", "avg_clv", "n_clv"]
MODEL_COLS = ["version", "created_at", "n_train", "train_start", "train_end", "p_source", "drift_status", "log_loss", "brier", "auc", "ece", "cal_slope"]
OOS_COLS = ["game_id", "game_date", "home_win", "home_rate", "elo", "logistic", "rf", "lgbm", "xgb", "stack", "wavg", "stack__online"]


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def games_predictions(store: Store, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per game with a logged recommendation (passes included): model vs market vs outcome, and the decision."""
    r = final_recommendations(store)
    if r.empty:
        return pd.DataFrame(columns=GAMES_COLS)
    d = resolved(store)
    res = (d.set_index("game_id")[["home_win", "close_home_prob", "bet_won", "profit", "clv_ev"]] if len(d)
           else pd.DataFrame(columns=["home_win", "close_home_prob", "bet_won", "profit", "clv_ev"]))
    g = r.merge(res, left_on="game_id", right_index=True, how="left")
    is_bet = (g.action == "BET") & (g.stake > 0)
    out = pd.DataFrame({
        "game_id": g.game_id, "game_date": g.game_date, "away": g.away, "home": g.home, "home_goalie": g.home_goalie, "away_goalie": g.away_goalie,
        "goalie_status": g.goalie_status, "model_version": g.model_version, "p_model_home": g.p_model, "p_market_home": g.p_market,
        "p_close_home": g.close_home_prob, "edge_home": g.p_model - g.p_market, "home_win": g.home_win, "decision": np.where(is_bet, "BET", "NO_BET"),
        "bet_side": np.where(is_bet, g.side, None), "bet_team": np.where(is_bet, g.team, None), "bet_price": np.where(is_bet, g.decimal, np.nan),
        "stake": np.where(is_bet, g.stake, 0.0), "edge": np.where(is_bet, g.edge, np.nan), "ev": np.where(is_bet, g.ev, np.nan),
        "bet_won": np.where(is_bet, g.bet_won, np.nan), "profit": np.where(is_bet, g.profit, np.nan), "clv_ev": np.where(is_bet, g.clv_ev, np.nan),
        "reasons": g.reasons})
    ctx = {c: np.nan for c in CONTEXT_COLS}
    if features is not None and len(features):
        f = features.reindex(g.game_id)
        ctx = {c: f[c].to_numpy() if c in f else np.nan for c in CONTEXT_COLS}
    out["home_rest_days"], out["away_rest_days"] = ctx["h_rest_days"], ctx["a_rest_days"]
    out["home_b2b"], out["away_b2b"] = ctx["h_b2b"], ctx["a_b2b"]
    out["home_games_played"], out["away_games_played"] = ctx["h_gp_season"], ctx["a_gp_season"]
    out["elo_prob_home"], out["is_rivalry"], out["days_into_season"] = ctx["elo_prob"], ctx["is_rivalry"], ctx["days_into_season"]
    return out[GAMES_COLS].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def daily_summary(store: Store) -> pd.DataFrame:
    g = games_predictions(store)
    if g.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    rows = []
    for date, part in g.groupby("game_date"):
        bets = part[part.decision == "BET"]
        settled = bets[bets.profit.notna()]
        done = part[part.home_win.notna()]
        withc = done[done.p_close_home.notna()]
        rows.append({"date": date, "games": len(part), "bets": len(bets), "staked": settled.stake.sum() if len(settled) else np.nan,
                     "profit": settled.profit.sum() if len(settled) else np.nan,
                     "model_log_loss": float(_ll(done.p_model_home, done.home_win).mean()) if len(done) else np.nan,
                     "market_close_log_loss": float(_ll(withc.p_close_home, withc.home_win).mean()) if len(withc) else np.nan,
                     "avg_clv": float(bets.clv_ev.mean()) if bets.clv_ev.notna().any() else np.nan})
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    d["roi"] = d.profit / d.staked
    d["cum_staked"], d["cum_profit"] = d.staked.fillna(0).cumsum(), d.profit.fillna(0).cumsum()
    return d[DAILY_COLS]


def paper_trading(store: Store) -> pd.DataFrame:
    d = shadow_resolved(store)
    if d.empty:
        return pd.DataFrame(columns=PAPER_COLS)
    games = store.df("SELECT game_id, home, away FROM games")
    d = d[d.is_bet].merge(games, on="game_id", how="left")
    mk = d["market"].fillna("moneyline").replace({"h2h": "moneyline"})
    out = pd.DataFrame({"date": d.game_date, "strategy": d.strategy, "game_id": d.game_id, "market": mk, "point": d["point"], "away": d.away, "home": d.home, "side": d.side,
                        "team": d["label"].fillna(d.team),
                        "price": d.decimal, "stake": d.stake, "won": d.won, "profit": d.profit, "clv_ev": d.clv_ev})
    return out[PAPER_COLS].sort_values(["date", "strategy", "game_id"]).reset_index(drop=True)


def alt_market_predictions(store: Store) -> pd.DataFrame:
    """Totals / puck-line quotes from the latest run per game (passes included) with the settled outcome (won / lost / push): for calibration."""
    from nhlbet.report.betlog import alt_value
    q = store.df("SELECT * FROM alt_quotes WHERE market NOT LIKE 'player_%'")
    if q.empty:
        return pd.DataFrame(columns=ALT_COLS)
    q = q.sort_values("run_at").groupby(["game_id", "market", "side"], as_index=False).tail(1)
    g = store.df("SELECT game_id, home, away, home_score, away_score, last_period, home_win FROM games")
    d = q.merge(g, on="game_id", how="left")
    done = d.home_win.notna()
    val = alt_value(d.market.to_numpy(), d.side.to_numpy(), d.point.to_numpy(), d.home_score, d.away_score, d.last_period)
    d["outcome"] = np.where(done, np.where(val > 0, "won", np.where(val < 0, "lost", "push")), None)
    d["best_price"] = d["decimal"]
    return d[ALT_COLS].sort_values(["game_date", "game_id", "market", "side"]).reset_index(drop=True)


PROP_COLS = ["game_id", "game_date", "away", "home", "player_id", "player", "stat", "side", "line", "p_model", "p_market", "edge", "ev", "price", "expected_shots", "actual_shots", "outcome"]
# note: for the 'points' stat the columns expected_shots / actual_shots hold expected / actual POINTS (names kept so existing reports do not break)


def prop_predictions(store: Store) -> pd.DataFrame:
    """Every player-shots quote the model priced (latest run per game), both sides, with the actual shots and won / lost / void once the game is final."""
    q = store.df("SELECT * FROM alt_quotes WHERE market LIKE 'player_%'")
    if q.empty:
        return pd.DataFrame(columns=PROP_COLS)
    q = q.sort_values("run_at").groupby(["game_id", "market", "side"], as_index=False).tail(1)
    g = store.df("SELECT game_id, home, away, home_win FROM games")
    sk = store.df("SELECT game_id, player_id, sog, points FROM skater_game WHERE sog IS NOT NULL")
    d = q.merge(g, on="game_id", how="left").merge(sk, on=["game_id", "player_id"], how="left")
    d["stat"] = np.where(d.market == "player_points", "points", "sog")
    d["sog"] = np.where(d.stat == "points", d.points, d.sog)
    have = set(sk.game_id)
    over = d.side.str.startswith("over")
    settled = d.home_win.notna() & d.game_id.isin(have)
    win = np.where(over, d.sog > d.point, d.sog < d.point)
    d["outcome"] = np.where(~settled, None, np.where(d.sog.isna(), "void", np.where(win, "won", "lost")))
    d["player"] = d.label.str.replace(r" (Over|Under) [0-9.]+$", "", regex=True)
    d["side"] = np.where(over, "over", "under")
    d = d.rename(columns={"point": "line", "decimal": "price", "exp_total": "expected_shots", "sog": "actual_shots"})
    return d[PROP_COLS].sort_values(["game_date", "game_id", "player_id", "side"]).reset_index(drop=True)


def paper_summary(store: Store) -> pd.DataFrame:
    p = shadow_performance(store)
    if p.empty:
        return pd.DataFrame(columns=PAPER_SUMMARY_COLS)
    return p.reset_index()[PAPER_SUMMARY_COLS]


def model_versions(registry_path: str | Path = "data/models/registry.json") -> pd.DataFrame:
    p = Path(registry_path)
    if not p.exists():
        return pd.DataFrame(columns=MODEL_COLS)
    rows = []
    for e in json.loads(p.read_text()):
        m = (e.get("walk_forward_metrics") or {}).get("stack__online") or {}
        rows.append({"version": e.get("version"), "created_at": e.get("created_at"), "n_train": e.get("n_train"), "train_start": e.get("train_start"),
                     "train_end": e.get("train_end"), "p_source": e.get("p_source"), "drift_status": (e.get("drift") or {}).get("status"),
                     "log_loss": m.get("log_loss"), "brier": m.get("brier"), "auc": m.get("auc"), "ece": m.get("ece"), "cal_slope": m.get("cal_slope")})
    return pd.DataFrame(rows, columns=MODEL_COLS)


def oos_predictions(path: str | Path = "reports/walkforward_predictions.csv") -> pd.DataFrame:
    """The walk-forward out-of-sample predictions (thousands of games, no odds): for calibration and segment analysis in a BI tool."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=OOS_COLS)
    w = pd.read_csv(p, index_col=0)
    w.index.name = "game_id"
    w = w.reset_index().rename(columns={"y": "home_win"})
    for c in OOS_COLS:
        if c not in w:
            w[c] = np.nan
    return w[OOS_COLS]


def export_dataset(store: Store, root: str | Path = "data/export", features: pd.DataFrame | None = None,
                   registry_path: str | Path = "data/models/registry.json", oos_path: str | Path = "reports/walkforward_predictions.csv") -> dict[str, int]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tables = {"games_predictions": games_predictions(store, features), "daily_summary": daily_summary(store), "paper_trading": paper_trading(store),
              "paper_summary": paper_summary(store), "alt_market_predictions": alt_market_predictions(store), "prop_predictions": prop_predictions(store), "model_versions": model_versions(registry_path), "oos_predictions": oos_predictions(oos_path)}
    for name, df in tables.items():
        df.to_csv(root / f"{name}.csv", index=False, float_format="%.6g")
    return {k: len(v) for k, v in tables.items()}
