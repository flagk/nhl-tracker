"""Weekly tendencies report: where is the model's probability systematically off? (descriptive; never changes the model or staking)

    python scripts/tendencies.py            # writes reports/TENDENCIES.md and data/export/tendencies.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nhlbet.analysis.tendencies import analyze, render_markdown
from nhlbet.data.loaders import load_tables
from nhlbet.data.store import Store
from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.report.export import games_predictions

MIN_MARKET_GAMES = 300


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/nhl.db")
    ap.add_argument("--oos", default="reports/walkforward_predictions.csv")
    ap.add_argument("--out", default="reports/TENDENCIES.md")
    ap.add_argument("--csv", default="data/export/tendencies.csv")
    a = ap.parse_args()
    store = Store(a.db)
    feats = FeatureBuilder(BuilderConfig()).build(load_tables(store))
    results, tables = {}, []
    oos = pd.read_csv(a.oos, index_col=0)
    oos.index.name = "game_id"
    g = store.df("SELECT game_id, home, away FROM games").set_index("game_id")
    df = oos.join(g, how="left").join(feats[[c for c in ("h_rest_days", "a_rest_days", "h_b2b", "a_b2b", "h_gp_season", "a_gp_season", "is_rivalry") if c in feats]], how="left")
    df = df.rename(columns={"y": "home_win", "stack__online": "p_model_home", "h_rest_days": "home_rest_days", "a_rest_days": "away_rest_days", "h_b2b": "home_b2b",
                            "a_b2b": "away_b2b", "h_gp_season": "home_games_played", "a_gp_season": "away_games_played"})
    if "game_date" not in df:
        df["game_date"] = store.df("SELECT game_id, game_date FROM games").set_index("game_id").game_date
    results["Model vs outcomes (walk-forward, out of sample)"] = analyze(df.dropna(subset=["p_model_home", "home_win"]))
    live = games_predictions(store, feats)
    live = live[live.home_win.notna() & live.p_market_home.notna()]
    if len(live) >= MIN_MARKET_GAMES:
        results["Betting market vs outcomes (live odds)"] = analyze(live.rename(columns={"home_rest_days": "home_rest_days"}), p_col="p_market_home")
    md = render_markdown(results, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    if len(live) < MIN_MARKET_GAMES:
        md += f"\n*Market comparison starts at {MIN_MARKET_GAMES} settled games with odds; so far {len(live)}.*\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    for name, r in results.items():
        if len(r.table):
            tables.append(r.table.assign(analysis=name))
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    (pd.concat(tables) if tables else pd.DataFrame()).to_csv(a.csv, index=False, float_format="%.6g")
    print(md[:600])


if __name__ == "__main__":
    main()
