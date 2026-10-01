# Power BI dashboard for the NHL model

Research/education only, not betting advice. Stake only what you can afford to lose.

The daily run publishes tidy CSVs to `data/export/`. Power BI reads them straight from GitHub, so the dashboard refreshes itself after each run.

| File | One row per | Use |
|---|---|---|
| `games_predictions.csv` | game (bets and passes) | model vs market vs outcome, decision, stake, profit, CLV, rest/b2b context |
| `daily_summary.csv` | day | bets, staked, profit, ROI, cumulative profit, log loss vs market |
| `paper_trading.csv` / `paper_summary.csv` | fake-money bet / strategy | which strategies work, with ROI confidence interval |
| `model_versions.csv` | trained model | log loss, Brier, AUC, calibration slope, drift status |
| `oos_predictions.csv` | historical game | walk-forward out-of-sample predictions for calibration charts |
| `prop_predictions.csv` | player x line x side | player shots-on-goal quotes (model vs market) with the actual shots and won / lost / void |
| `alt_market_predictions.csv` | game x market x side | totals / puck-line model vs market with the settled outcome (won / lost / push) |
| `tendencies.csv` | tested segment | weekly tendency analysis (q-values, replication verdict) |

## Connect (about 5 minutes)

1. Power BI Desktop → **Get data → Web** for each file, with URL
   `https://raw.githubusercontent.com/flagk/nhl-tracker/main/data/export/games_predictions.csv` (repeat per file; ready-made `.pbids` files are in this folder, double-click one).
2. Choose **Anonymous** access. Power Query detects it as CSV; click **Transform data**, confirm the header row and column types (dates as Date), then **Close & Apply**.
3. Model view: relate `games_predictions[game_id]` → `oos_predictions[game_id]` (optional) and `paper_trading[game_id]`, many-to-one is fine; use a Date table on `game_date`/`date`.
4. Paste the measures from `measures.dax` into a new table.
5. **Publish** to the Power BI service → dataset settings → Scheduled refresh, credentials: Anonymous, privacy level Public. Refresh daily after the 14:30 UTC run.

If the repository is private the raw URLs need authentication (a GitHub token header), so Power BI cannot refresh them anonymously. Either keep the repo public (exports are public-safe: no bookmaker names or per-book prices), or clone the repo and point Power BI at the local `data/export/` folder (Get data → Folder) with a gateway.

## Suggested pages

- **Overview**: cards for total profit, ROI, bets, average CLV; line of `cum_profit` by date.
- **Model vs market**: calibration scatter (`oos_predictions` binned predicted vs actual), log loss by month, `p_model_home` vs `p_market_home`.
- **Bets**: table of decisions with edge, stake, profit; slicers for team, goalie status, decision.
- **Paper trading**: bar of ROI per strategy with `roi_lo`/`roi_hi` error bars. `market_favorite` is the no-skill control.
- **Tendencies**: table filtered to `verdict = "TENDENCY (replicated)"`; most weeks it should be empty, which is normal.

Read ROI and CLV with caution: hundreds of bets cannot separate a 1-3% edge from luck.
