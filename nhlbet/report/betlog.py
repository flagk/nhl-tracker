"""Recommendation/bet log: settle results, running performance (ROI, win rate, CLV, Brier, calibration), CSV export/restore.

Everything in the log that cannot be re-fetched later (odds snapshots, recommendations) is exported to git-tracked CSVs
and restored into a fresh database at the start of every run; the SQLite file itself is rebuildable cache.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nhlbet.data.store import Store
from nhlbet.models.calibration import brier, log_loss_, reliability_table
from nhlbet.odds.clv import bet_clv, closing_consensus, summarize_clv
from nhlbet.odds.consensus import consensus_snapshots
from nhlbet.risk.bankroll import drawdown_stats, equity_curve, summarize_bets

log = logging.getLogger(__name__)


def final_recommendations(store: Store) -> pd.DataFrame:
    """The last recommendation made for each game (the one that would actually have been acted on)."""
    r = store.df("SELECT * FROM recommendations")
    if r.empty:
        return r
    return r.sort_values("run_at").groupby("game_id", as_index=False).tail(1).reset_index(drop=True)


def resolved(store: Store, cons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Final recommendations joined with results, profit and closing-line value. One row per resolved game."""
    r = final_recommendations(store)
    if r.empty:
        return r
    g = store.df("SELECT game_id, game_date AS gd, start_utc, home_win FROM games WHERE home_win IS NOT NULL")
    d = r.merge(g, on="game_id", how="inner")
    if d.empty:
        return d
    cons = consensus_snapshots(store) if cons is None else cons
    d["close_home_prob"] = [closing_consensus(cons, gid, st) if len(cons) else None for gid, st in zip(d.game_id, d.start_utc)]
    d["is_bet"] = (d.action == "BET") & (d.stake > 0)
    d["bet_won"] = np.where(d.is_bet, ((d.side == "home") == (d.home_win == 1)).astype(float), np.nan)
    d["profit"] = np.where(d.is_bet, np.where(d.bet_won == 1, d.stake * (d.decimal - 1), -d.stake), 0.0)
    clv_rows = [bet_clv(s, dec, c) if (b and c is not None and not pd.isna(c)) else {"clv_ev": np.nan, "clv_prob_pts": np.nan}
                for b, s, dec, c in zip(d.is_bet, d.side, d.decimal, d.close_home_prob)]
    d["clv_ev"] = [x["clv_ev"] for x in clv_rows]
    d["clv_prob_pts"] = [x["clv_prob_pts"] for x in clv_rows]
    d["date"] = pd.to_datetime(d.game_date)
    return d.sort_values("date").reset_index(drop=True)


def performance(store: Store, start_bankroll: float = 1000.0, cons: pd.DataFrame | None = None) -> dict:
    """Running metrics over all resolved games. Every number is out-of-sample: the recommendation existed before the game."""
    d = resolved(store, cons)
    out: dict = {"resolved_games": 0, "bets": summarize_bets(pd.DataFrame({"stake": [], "profit": [], "decimal": []}))}
    if d.empty:
        return {**out, "bankroll": start_bankroll, "note": "no resolved recommendations yet"}
    y = d.home_win.astype(int).to_numpy()
    p = d.p_model.astype(float).to_numpy()
    out["resolved_games"] = int(len(d))
    out["model"] = {"brier": brier(p, y), "log_loss": log_loss_(p, y), "accuracy": float(np.mean((p > 0.5) == y))}
    m = d[d.close_home_prob.notna()]
    if len(m) >= 20:
        pc, ym = m.close_home_prob.astype(float).to_numpy(), m.home_win.astype(int).to_numpy()
        out["vs_market"] = {"n": int(len(m)), "model_log_loss": log_loss_(m.p_model.astype(float), ym), "market_close_log_loss": log_loss_(pc, ym),
                            "model_brier": brier(m.p_model.astype(float), ym), "market_close_brier": brier(pc, ym)}
    if len(d) >= 50:
        out["calibration"] = reliability_table(p, y, bins=min(8, len(d) // 25)).round(4).to_dict("records")
    bets = d[d.is_bet]
    out["bets"] = summarize_bets(bets[["stake", "profit", "decimal"]]) if len(bets) else out["bets"]
    curve = equity_curve(bets[["date", "profit"]], start_bankroll) if len(bets) else pd.DataFrame()
    out["bankroll"] = float(start_bankroll + bets.profit.sum())
    out["drawdown"] = drawdown_stats(curve) if len(curve) else drawdown_stats(pd.DataFrame())
    out["clv"] = summarize_clv(bets.clv_ev) if len(bets) else {"n": 0}
    out["no_bet_rate"] = float(1 - d.is_bet.mean())
    out["curve"] = curve
    return out


def all_shadow_rows(store: Store) -> pd.DataFrame:
    """Paper bets from both tables in one frame: game-level bets (``shadow_bets``) and player props (``prop_bets``, market ``player_sog``)."""
    r = store.df("SELECT * FROM shadow_bets")
    if "player_id" not in r:
        r["player_id"] = np.nan
    p = store.df("SELECT * FROM prop_bets")
    if len(p):
        p = p.assign(action="BET", team=p.label, p_adj=p.p_model, market=p["market"].fillna("player_sog") if "market" in p else "player_sog")
        r = pd.concat([r, p[[c for c in p.columns if c in set(r.columns) | {"player_id"}]]], ignore_index=True)
    return r


def alt_value(market, side, point, home_score, away_score, last_period, sog=None) -> np.ndarray:
    """>0 the bet won, <0 lost, 0 pushed, for totals / puck-line bets (arrays). Totals exclude the shootout goal; the puck line uses the official margin."""
    hs, as_ = np.asarray(home_score, float), np.asarray(away_score, float)
    side, market, point = np.asarray(side), np.asarray(market), np.asarray(point, float)
    total = hs + as_ - (np.asarray(last_period) == "SO").astype(float)
    margin = np.where(side == "away", as_ - hs, hs - as_)
    val = np.where(market == "totals", np.where(side == "over", total - point, point - total), margin + point)
    if sog is not None:                                   # player shots: over wins when shots exceed the line; a player who did not dress voids the bet (value 0)
        sg = np.asarray(sog, float)
        shots = np.where(side == "over", sg - point, point - sg)
        val = np.where(np.char.startswith(market.astype(str), "player_"), np.where(np.isnan(sg), 0.0, shots), val)
    return val


def shadow_resolved(store: Store, cons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Paper-trading rows joined with results, profit and CLV (latest run per game and strategy).

    Moneyline rows settle on the winner. Totals settle on regulation + overtime goals (shootout goal excluded) and the puck line on the
    official margin; a push (whole-number line landing exactly) refunds the stake and is not counted as a bet. CLV exists for moneylines only.
    """
    r = all_shadow_rows(store)
    if r.empty:
        return r
    r["player_id"] = r["player_id"].fillna(0)               # game-level bets have no player; props are one row per (game, strategy, player, side)
    final = r.sort_values("run_at").groupby(["game_id", "strategy", "player_id"], as_index=False).tail(1)     # latest run per (game, strategy, player); a re-run may flip the side
    g = store.df("SELECT game_id, start_utc, home_win, home_score, away_score, last_period FROM games WHERE home_win IS NOT NULL")
    d = final.merge(g, on="game_id", how="inner")
    if d.empty:
        return d
    sog = np.full(len(d), np.nan)
    is_prop = d["market"].astype(str).str.startswith("player_") if "market" in d else pd.Series(False, index=d.index)
    if is_prop.any():
        sk = store.df("SELECT game_id, player_id, sog, points, goals, assists FROM skater_game WHERE sog IS NOT NULL")
        have = set(sk.game_id)                                # games whose shot data is loaded: a prop can only settle once its game has it
        d = d[~is_prop | d.game_id.isin(have)].reset_index(drop=True)
        m = d[["game_id", "player_id"]].merge(sk, on=["game_id", "player_id"], how="left")      # missing = did not dress -> void
        col = d["market"].astype(str).map({"player_points": "points", "player_goals": "goals", "player_assists": "assists"}).fillna("sog")
        sog = np.select([col.eq("points"), col.eq("goals"), col.eq("assists")], [m.points.to_numpy(float), m.goals.to_numpy(float), m.assists.to_numpy(float)], m.sog.to_numpy(float))
    cons = consensus_snapshots(store) if cons is None else cons
    mk = d["market"].fillna("h2h") if "market" in d else pd.Series("h2h", index=d.index)
    ml = (mk == "h2h").to_numpy()
    d["close_home_prob"] = [closing_consensus(cons, gid, st) if (len(cons) and is_ml) else None for gid, st, is_ml in zip(d.game_id, d.start_utc, ml)]
    d["is_bet"] = (d.action == "BET") & (d.stake > 0)
    won_ml = ((d.side == "home") == (d.home_win == 1)).astype(float).to_numpy()
    point = d["point"].astype(float).to_numpy() if "point" in d else np.full(len(d), np.nan)
    val = alt_value(mk.to_numpy(), d.side.to_numpy(), point, d.home_score, d.away_score, d.last_period, sog)
    push = (~ml) & d.is_bet.to_numpy() & (val == 0)
    d["push"] = push
    d["is_bet"] = d.is_bet & ~push                        # a push refunds the stake: not a settled bet
    d["won"] = np.where(d.is_bet, np.where(ml, won_ml, (val > 0).astype(float)), np.nan)
    d["profit"] = np.where(d.is_bet, np.where(d.won == 1, d.stake * (d.decimal - 1), -d.stake), 0.0)
    d["clv_ev"] = [bet_clv(s, dec, c)["clv_ev"] if (b and m and c is not None and not pd.isna(c)) else np.nan
                   for b, m, s, dec, c in zip(d.is_bet, ml, d.side, d.decimal, d.close_home_prob)]
    return d


def shadow_performance(store: Store, cons: pd.DataFrame | None = None, B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per paper-trading strategy: bets, staked, profit, ROI (+bootstrap CI), win rate, mean CLV. Fake money."""
    d = shadow_resolved(store, cons)
    if d.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for name, part in d.groupby("strategy"):
        b = part[part.is_bet]
        if b.empty:
            rows.append({"strategy": name, "bets": 0, "staked": 0.0, "profit": 0.0, "roi": np.nan, "roi_lo": np.nan, "roi_hi": np.nan,
                         "win_rate": np.nan, "avg_clv": np.nan, "n_clv": 0})
            continue
        stake, profit = b.stake.to_numpy(), b.profit.to_numpy()
        lo = hi = np.nan
        if len(b) >= 20:
            idx = rng.integers(0, len(b), (B, len(b)))
            rois = profit[idx].sum(1) / stake[idx].sum(1)
            lo, hi = np.percentile(rois, [2.5, 97.5])
        rows.append({"strategy": name, "bets": int(len(b)), "staked": float(stake.sum()), "profit": float(profit.sum()), "roi": float(profit.sum() / stake.sum()),
                     "roi_lo": float(lo), "roi_hi": float(hi), "win_rate": float((profit > 0).mean()), "avg_clv": float(b.clv_ev.mean()) if b.clv_ev.notna().any() else np.nan,
                     "n_clv": int(b.clv_ev.notna().sum())})
    return pd.DataFrame(rows).set_index("strategy")


def plot_performance(store: Store, path: str | Path, start_bankroll: float = 1000.0) -> bool:
    """Bankroll/drawdown, cumulative CLV and model-vs-market log loss over time. Returns False if there is nothing to plot."""
    d = resolved(store)
    if d.empty:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    bets = d[d.is_bet]
    if len(bets):
        c = equity_curve(bets[["date", "profit"]], start_bankroll)
        ax[0].plot(c.date, c.bankroll, color="#1565C0"); ax[0].fill_between(c.date, c.peak, c.bankroll, color="#C62828", alpha=0.25, label="drawdown")
        ax[0].legend()
        ax[1].plot(bets.date, bets.clv_ev.fillna(0).cumsum() / np.arange(1, len(bets) + 1), color="#2E7D32")
    ax[0].set_ylabel("bankroll"); ax[0].set_title("Performance (out-of-sample, logged before each game)"); ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("running mean CLV (EV/$)"); ax[1].grid(alpha=0.3)
    ll = -(d.home_win * np.log(d.p_model.clip(1e-6, 1)) + (1 - d.home_win) * np.log((1 - d.p_model).clip(1e-6, 1)))
    ax[2].plot(d.date, ll.expanding().mean(), label="model", color="#1565C0")
    if d.close_home_prob.notna().sum() > 20:
        m = d.close_home_prob.astype(float).clip(1e-6, 1 - 1e-6)
        mll = -(d.home_win * np.log(m) + (1 - d.home_win) * np.log(1 - m))
        ax[2].plot(d.date, mll.expanding().mean(), label="market close", color="#EF6C00")
    ax[2].axhline(0.6931, color="k", ls="--", lw=0.8, label="coin flip"); ax[2].set_ylabel("cumulative log loss"); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=120); plt.close(fig)
    return True


# ---------------------------------------------------------------- CSV export / restore (irreplaceable data lives in git)
# Layout: every odds capture and every recommendation run gets its OWN small file, so two concurrent jobs (morning run, late run,
# closing-line snapshot) add different files and git never has to merge lines of the same file. Only derived files (bet_log.csv)
# and the tiny goalie_confirmations.csv are shared, and the workflows resolve those in favour of the newest output.
PARTITIONED = {"recommendations": ("run_id", ["run_id", "game_id"]), "odds_fetch_log": ("captured_at", ["captured_at"]),
               "shadow_bets": ("run_id", ["run_id", "game_id", "strategy"]), "odds_consensus": ("captured_at", ["game_id", "captured_at"]),
               "alt_quotes": ("run_id", ["run_id", "game_id", "market", "side"]),
               "prop_bets": ("run_id", ["run_id", "game_id", "strategy", "player_id", "side"])}
SINGLE = {"goalie_confirmations": ["game_date", "team"]}
ODDS_KEYS = ["captured_at", "event_id", "book", "market", "outcome", "point"]


def _slug(value: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z]+", "-", str(value)).strip("-")


def export_logs(store: Store, root: str | Path = "data/logs", public_safe: bool | None = None) -> list[Path]:
    """Write the irreplaceable logs as small per-capture/per-run CSVs.

    In public-safe mode (the default unless the workflow has confirmed the repo is private) nothing written contains per-bookmaker
    quotes or bookmaker names: raw odds captures are NOT exported (only the derived no-vig consensus is), and ``book`` columns are blanked.
    """
    from nhlbet.privacy import is_public_safe
    public_safe = is_public_safe() if public_safe is None else public_safe
    root = Path(root)
    written = []
    for table, (part_col, keys) in PARTITIONED.items():
        df = store.df(f"SELECT * FROM {table}")
        if public_safe and "book" in df.columns:
            df = df.assign(book=None)
        for value, part in df.groupby(part_col):
            p = root / table / f"{_slug(value)}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            part.sort_values(keys).to_csv(p, index=False)
            written.append(p)
    if not public_safe:                                        # raw per-book quotes only ever leave the database for a private repo
        snaps = store.df("SELECT * FROM odds_snapshots")
        for cap, part in snaps.groupby("captured_at"):
            p = root / "odds" / f"{_slug(cap)}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            part.sort_values(["event_id", "book", "market", "outcome", "point"]).to_csv(p, index=False)
            written.append(p)
    for table, keys in SINGLE.items():
        df = store.df(f"SELECT * FROM {table}")
        if len(df):
            root.mkdir(parents=True, exist_ok=True)
            p = root / f"{table}.csv"; df.sort_values(keys).to_csv(p, index=False); written.append(p)
    bl = resolved(store)
    if len(bl):
        cols = ["game_date", "home", "away", "team", "side", "book", "decimal", "stake", "p_model", "p_adj", "p_market", "edge", "ev", "close_home_prob",
                "clv_ev", "bet_won", "profit", "home_win", "model_version"]
        if public_safe:
            cols.remove("book")
        root.mkdir(parents=True, exist_ok=True)
        p = root / "bet_log.csv"; bl[bl.is_bet][cols].to_csv(p, index=False); written.append(p)
    return written


def _read(p: Path) -> list[dict]:
    df = pd.read_csv(p, float_precision="round_trip").astype(object)
    return df.where(df.notna(), None).to_dict("records")


def restore_logs(store: Store, root: str | Path = "data/logs") -> dict[str, int]:
    """Upsert every exported file back into the database. Also reads the legacy single-file layout (recommendations.csv,
    odds_fetch_log.csv, odds/<YYYY-MM>.csv) so history committed before the partitioned layout is not lost."""
    root = Path(root); n: dict[str, int] = {}
    def load(table: str, files: list[Path], keys: list[str]):
        total = sum(store.upsert(table, _read(p), keys) for p in files if p.exists())
        if total:
            n[table] = n.get(table, 0) + total
    for table, (_, keys) in PARTITIONED.items():
        load(table, [root / f"{table}.csv"] + sorted((root / table).glob("*.csv")), keys)
    load("odds_snapshots", sorted((root / "odds").glob("*.csv")), ODDS_KEYS)
    for table, keys in SINGLE.items():
        load(table, [root / f"{table}.csv"], keys)
    return n
