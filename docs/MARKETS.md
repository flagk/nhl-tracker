# Other bet types: totals and puck line

Research/education only. No model guarantees profit; stake only what you can afford to lose.

## What is supported

| Bet type | Model | Real-stake recommendations? |
|---|---|---|
| Moneyline | win-probability stack (see RESULTS.md) | Yes, under the normal risk policy |
| Total (over/under) | goals model | **No. Paper-traded only** |
| Puck line (spread) | goals model | **No. Paper-traded only** |
| Player shots on goal (over/under) | player shots model | **No. Paper-traded only** |
| Parlay, other props, anything else | none | Not modelled. You can log them on the My bets page |

Totals and puck lines are experimental on purpose. The goals model has no record against sportsbook prices yet, so the site and report show
its edges for information and run fake-money strategies on it. They become candidates for real recommendations only after
weeks of settled paper bets beat the no-skill control (`always_over`) by more than noise.

## How the goals model works (`nhlbet/models/goals.py`)
- Each team's **regulation goals** are negative-binomial (Poisson with a small fitted overdispersion). The means come from a regularised
  Poisson regression on the same leak-free as-of features; the regularisation strength is chosen by the walk-forward backtest.
- Hockey is tied after 60 minutes more often than independent scoring implies (about 23% vs 16%), so the diagonal of the score grid is
  boosted to match the training tie rate.
- Regulation ties are split using the production moneyline probability, so moneyline, puck line and totals stay consistent.
- Settlement follows sportsbook rules: **totals exclude the shootout goal** (a game decided in overtime adds one goal, a shootout adds none);
  the **puck line uses the official margin** (an overtime or shootout winner wins by one).
- Whole-number lines can push (stake refunded): probabilities are compared conditional on no push; EV uses the real win/lose/push split.

## Evidence
First honest walk-forward results (6,565 games, 2021-22 to 2025-26; `reports/GOALS.md` has the live numbers):
- **Goal rates:** about 1.9% better Poisson log-likelihood than the training average (CI excludes zero). Hockey scoring is mostly noise, so this is a small real signal.
- **Puck line:** about 0.019 lower log loss than the base rate after recalibration (CI excludes zero); calibration slope about 0.8.
- **Totals:** calibrated (slope about 1.0) but essentially no skill: 0.001-0.002 better than the base rate, CI touching zero. Expect the market to price totals better than this model.
- A first version of this report was wrong (outcome columns leaked into the features); it was caught because the gains were implausibly large, fixed, and is covered by regression tests.

`reports/GOALS.md` (workflow "Goals model backtest") holds the walk-forward test against base rates. There are **no historical sportsbook lines**, so
it cannot say whether the model beats the market, only whether it beats naive rates and is calibrated. The market test is the paper trading:

| Strategy | What it does |
|---|---|
| `totals_edge`, `puckline_edge` | flat 1% when the goals model's raw edge is 3%+ |
| `every_total`, `every_puckline` | flat 1% on the model's side of every game, no filter |
| `always_over` | control: Over every game |

## Odds and credits
The daily morning and late runs fetch `h2h,spreads,totals` together (3 credits per call). With the three h2h-only closing snapshots that is about 9
credits/day (~270 of the 500 free monthly credits). Override with `ODDS_MARKETS=h2h`. Closing-line value is only computed for moneylines.

## Pages
- Picks page: a "Totals & puck line (experimental)" table with model vs market, edge and EV, and no stake suggestion.
- Bets page: choose a bet type (moneyline, puck line, total, parlay, player prop, other) for your own bets; the fake-bets tab lists every
  strategy's pretend bets by type.
- BI export: `alt_market_predictions.csv` (every game's quotes with outcome) and a `market` column in `paper_trading.csv`.


## Player props: shots on goal
- **Data:** the boxscore's per-skater shots on goal are now stored (`skater_game.sog`). Games ingested earlier are filled from their cached boxscores (`python -m nhlbet.data.ingest --reparse-skaters`; the daily run catches up automatically).
- **Model (`nhlbet/models/props.py`, features in `nhlbet/features/players.py`):** each player's shots are negative-binomial. The mean is a Poisson regression on his own recent shot rate (exponentially weighted and shrunk toward his position's average), how many shots tonight's opponent has been allowing, home ice, recent ice time and rest. Everything is as of the game date.
- **Prices:** The Odds API serves player props per game and charges one credit per game, so only the first few games to start are priced (`ODDS_PROPS_MAX_GAMES`, default 3, late run only). The morning run fetches moneylines only. That keeps the plan near 12 credits a day (about 360 of the free 500 a month). If your plan has no prop access, the run records that and carries on.
- **What the page shows:** each priced player with the line, the model's expected shots, model vs market probability of the over, the suggested side, the price, and a strip of his last 10 games (green bars went over the line) with averages and hit rates.
- **Paper strategies:** `sog_edge` ($5-$30 on the side with a 3%+ edge, at most 8 a day, 10+ games of history) and the control `sog_over_control` (flat $10 on the Over for the 8 highest lines, no model). A player who does not dress voids the bet.
- **How current the totals / puck-line prices are:** they come from the latest capture that included those markets and are used only if that capture is at most 2 hours old at report time; otherwise totals and puck lines are simply not priced (the morning run, which only fetches moneylines, therefore shows none).
- **Player points (goals + assists):** same recipe as shots (own shrunk recent scoring rate, opponent, home ice, ice time, rest) against the `player_points` over/under market (usually 0.5 or 1.5). Strategies: `pts_edge`, `pts_over_edge`, `pts_under_edge` and the control `pts_over_control`; settled from the player's actual points. It costs one more odds credit per priced game (about 3 a day).
- **Player assists:** same recipe against the `player_assists` over/under market. Strategies `ast_edge`, `ast_over_edge`, `ast_under_edge`, control `ast_over_control`.
- **Anytime goalscorer:** the model's chance of scoring at least once (from its expected goals) against the `player_goal_scorer_anytime` price. Only the "yes" side is priced, so there is no opposite side to remove the bookmaker margin with; the market's chance is estimated as the implied probability less an assumed 7% margin (shown on the site). EV is always exact at the real price. Strategies `goal_edge` and control `goal_control` (the 8 likeliest scorers).
- **How current the player lines are:** player prices are fetched fresh in each late run (only live answers are stored, never an old cached one), only the latest pre-game capture is used, books that have not updated in 90 minutes are dropped, and any capture more than 2 hours old at report time is ignored. The site shows when each player's prices were captured.
- **Credits:** each market costs one credit per priced game. Shots and points are always fetched; assists and anytime goals only while at least 250 credits remain, and no player prices at all below 100, so moneylines are protected.
- **Not covered yet:** goalie saves (needs a goalie-saves model and per-goalie data), first goalscorer, and player markets for games more than 6 hours away.
- **Evidence:** `reports/PROPS.md` (workflow "Player props backtest") tests the model against the player's own rate and checks calibration; there are no historical prop lines, so the market test is the paper trading.
