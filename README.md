# NHL Model

**Estimates each NHL game's win chances, compares them with real sportsbook prices, and tells you plainly when there is nothing worth betting.**
Updated automatically every day. Most days the honest answer is "no bet".

> **Disclaimer.** This is a **research and educational project**. No model guarantees profit, and back-tested or past results
> do not predict future results. Sports betting carries a real risk of loss: only stake money you can afford to lose, and check
> that betting is legal where you live. Nothing in this repository is financial advice.

<!-- PICKS:START -->

### Today's picks: 2026-10-07

**3 game(s) · 0 recommended bet(s)** · late run · model health **WARN** · updated 2026-10-07 23:01 UTC

| | |
|---|---|
| 📄 **[Today's full report](reports/latest.md)** | every game: model vs market, edge, stake and the reason for each decision |
| 🌐 **[Interactive picks page](https://flagk.github.io/nhl-tracker/)** | choose your unit size and staking, top picks, parlay calculator |
| 🗂️ **[Pick history](reports/HISTORY.md)** | every day's picks and how they settled ([web version](https://flagk.github.io/nhl-tracker/history.html)) |
| 🧾 **[My bets & fake bets](https://flagk.github.io/nhl-tracker/bets.html)** | log your own bets (kept only in your browser) and see the pretend-money bets on every game |
| 📈 [Bet log (CSV)](data/logs/bet_log.csv) | all settled recommended bets |

| Game | Model: home win | Market (no-vig) | Decision |
|---|---|---|---|
| PIT @ WSH | 49.8% | 59.5% | no bet |
| COL @ WPG | 41.7% | 36.1% | no bet |
| EDM @ ANA | 55.9% | 44.9% | no bet |

*No bets today is normal: the model only bets when it sees a sizeable edge over the market.*

*Research and educational project; no guarantee of profit. The numbers above are not betting advice.*

<!-- PICKS:END -->

> If a link above shows a 404, GitHub Pages is not switched on yet: open Settings -> Pages and set Source to *GitHub Actions* (once). The repository is
> public, so everything published is **public-safe**: model probabilities, edges and stakes, but no bookmaker names or per-bookmaker prices.

## Start here

| I want to... | Open |
|---|---|
| See today's picks and how much to stake | the **[picks page](https://flagk.github.io/nhl-tracker/)** (set your unit size once; it remembers) |
| Check a pick against the price in my own betting app | type it into the **"Price in your app"** box on any pick |
| Keep track of the bets I actually place | the **[Bets page](https://flagk.github.io/nhl-tracker/bets.html)** -> *My bets* (stored only in your browser) |
| See how the model's pretend-money bets did | the **[Bets page](https://flagk.github.io/nhl-tracker/bets.html)** -> *Fake bets* |
| Look back at past days | **[History](https://flagk.github.io/nhl-tracker/history.html)** |
| Read the full reasoning for every game | **[`reports/latest.md`](reports/latest.md)** |
| Get the data into Excel or Power BI | **[`powerbi/`](powerbi/README.md)** and the CSVs in **[`data/export/`](data/export)** |

## How to use it (two minutes a day)

1. **Open the picks page.** The top strip shows how many games there are and how many bets are recommended. Zero is normal.
2. **Set your units once** (the "Your betting units" bar): how many dollars is one unit, and how cautious you want to be.
3. **Read the picks.** Each card shows the model's chance vs the market's chance, the edge, the price, and a stake. *Recommended* means every safety check passed; *Lean* means the model likes it but the checks did not pass, so it is information only.
4. **Compare the price in your app.** Each card says the worst price at which the bet is still worth taking. If your app offers less, skip it.
5. **Log what you bet** on the Bets page so you can see your real results, not just the model's.

## What the numbers mean

| Term | In plain words |
|---|---|
| **Model probability** | How likely the model thinks it is that a team wins. |
| **Market (no-vig) probability** | What the sportsbooks' prices imply once their built-in margin ("vig") is removed. This is the number to beat. |
| **Edge** | Model probability minus market probability. A 3-point edge means the model thinks the team is 3 points likelier than the market does. |
| **EV (expected value)** | Average profit per $1 staked if the model is right about the probabilities. +5% means $0.05 per $1 over many bets. |
| **Unit / stake** | Your own bet size. The page suggests a stake as a fraction of your bankroll (quarter Kelly, capped at 2% per bet). |
| **No bet** | The most common result. The model only bets when it sees a big enough edge that survives its safety checks. |
| **CLV (closing-line value)** | Did you get a better price than the final price before puck drop? The fastest honest sign of a real edge. |
| **Puck line / total / player props** | The -1.5 / +1.5 goal spread, the over/under on goals, and players' shots-on-goal over/unders. These are **experimental here and paper-traded only**. |
| **Fake bets (paper trading)** | Pretend-money strategies run on every game to measure what works, including a no-skill control. They never touch real recommendations. |
| **Parlay** | Several bets that must all win. The margin of every leg stacks up, so the calculator warns you when singles are better. |

## Honest status

| | |
|---|---|
| **Has an edge over the betting market been shown?** | **No.** The model has never been compared with enough real closing lines yet. Treat it as a research system that *measures* whether it has an edge, not one that has it. |
| **What the model can do (walk-forward on real NHL API data: trained on 8 seasons, 2,269 out-of-sample games with calibration history, Oct 2024 to Sep 2026)** | Production model log loss **0.6759** vs 0.6896 for "always the base rate" and 0.6812 for plain Elo. The gap to Elo (-0.0053, 95% CI -0.0100 to -0.0003) and to the base rate (-0.0137) both exclude zero. A coin flip is 0.6931. **Against the market: unknown.** |
| **Player props (shots on goal)** | A separate player-level model, shown with each player's recent history. Paper-traded only until real results exist; see [docs/MARKETS.md](docs/MARKETS.md). |
| **Totals and puck line** | A separate goals model. It is calibrated and beats naive base rates on the puck line (about 0.019 better log loss), but has **essentially no skill on totals**. Both are paper-traded only. See [docs/MARKETS.md](docs/MARKETS.md). |
| **The original README's "55.8% win rate / +11.69% ROI / proven edge"** | **Not valid.** It assumed even-money payouts (no odds, no vig), had corrupted features, and did no better than always picking the home team. See [AUDIT.md](AUDIT.md). |
| **Most likely daily output** | **"No bet."** By design. |

Full numbers, protocol and caveats: [docs/RESULTS.md](docs/RESULTS.md) · [reports/model_comparison.md](reports/model_comparison.md) · [reports/GOALS.md](reports/GOALS.md).

## Questions people ask

**Why does it say "no bet" so often?** Because real edges are small and rare. A model that bet every game would mostly be paying the sportsbooks' margin. Early in the season the model also refuses to bet until teams have played about 10 games.

**Are the prices real?** Yes: they come from The Odds API, which collects live lines from US sportsbooks. The "best price" is the highest payout any book offered at the time of the snapshot, so check the price in your own app, which is why the "Price in your app" box exists.

**Why don't I see which sportsbook has a price?** The repository is public and the odds feed's terms restrict republishing per-book prices, so only derived numbers are published. A private repository would show book names.

**Where are my own bets stored?** Only in your browser, on the device you used. Nothing is sent anywhere. Use *Export JSON* on the Bets page to back them up or move them.

**Does it cost anything?** No. GitHub Actions and Pages are free for public repositories, the NHL data API is free, and the odds API has a free tier (500 credits a month; this project uses roughly 270).

**Can I trust the results?** Read the "Honest status" table above. Past results, including this model's, do not predict future results, and the model has not been shown to beat the market.

## Under the hood (for developers)

<details>
<summary><b>What is in the repository</b></summary>

```
nhlbet/
  data/       NHL API client (cache, retries, circuit breaker) -> parsers -> SQLite store (idempotent upserts), xG model, goalie inputs
  features/   leak-free as-of-date feature builder (Elo, form, rest/travel, xG/Corsi/PDO, special teams, goalies, lineup, situational, market)
  analysis/   correlation pruning, permutation importance, SHAP, tendencies (multiple-testing-corrected)
  models/     home-rate, Elo, logistic, Random Forest, LightGBM, XGBoost; stacking; calibration; walk-forward engine; goals model for totals / puck line
  odds/       The Odds API client (quota-aware), de-vig, best price, edge/EV, totals and spread quotes, append-only snapshots, CLV
  risk/       fractional Kelly, caps, shrinkage toward the market, "no bet" policy, bankroll/drawdown, Monte Carlo, paper-trading strategies
  report/     slate builder, daily report, bet log (settlement, ROI, CLV), history, BI exports, CSV export/restore
  site/       the picks, bets and history pages (static HTML, shared theme)
  gate.py     decides whether a scheduled workflow trigger should do any work
scripts/      daily.py  fetch_odds.py  train.py  backtest.py  goals_backtest.py  tendencies.py  gate.py  tune.py  select_features.py  risk_sim.py  check_api.py
tests/        no-leakage property tests, parsers, ingestion, odds math, Kelly, risk invariants, goals model, reports, site, workflows
docs/         FEATURES  ODDS  RISK  RESULTS  MARKETS  PAPER_TRADING  SITE      AUDIT.md = audit of the original pipeline
```
</details>

<details>
<summary><b>Setup and running it yourself</b></summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                   # the test suite, about a minute
```

**1. Data**
```bash
python scripts/check_api.py --season 20242025 --team BOS --n 5       # validates the parsers against the LIVE NHL API first
python -m nhlbet.data.ingest --seasons 20232024 20242025 20252026    # backfill (cached + idempotent; first run takes a while)
```
Without network access the daily job bootstraps from `data/history/nhl_history.csv` (2,987 games, 28 of 32 teams, scores only).

**2. Research pipeline** (feature selection, tuning, walk-forward backtest, model)
```bash
python scripts/select_features.py --selection-end 2024-06-30
python scripts/tune.py --tune-end 2024-06-30
python scripts/backtest.py --eval-start 2024-10-01     # writes reports/model_comparison.md + reliability.png
python scripts/train.py --force                        # trains, drift-checks, registers a version
python scripts/goals_backtest.py                       # totals / puck-line model vs base rates -> reports/GOALS.md
```
Selection and tuning only see games up to `--selection-end`; evaluation starts later, so nothing is chosen on the data it is judged on.

**3. Odds API key** (optional, but needed for any recommendation)
1. Get a free key at <https://the-odds-api.com> (500 credits a month).
2. Locally: `export ODDS_API_KEY=<your key>`. **Never commit it.** The code reads it only from the environment and never writes it to disk or logs.
3. GitHub Actions: Settings -> Secrets and variables -> Actions -> New repository secret named `ODDS_API_KEY` (optional variable `BANKROLL`, default 1000).

Without a key every game is reported as "no odds -> no bet". Details: [docs/ODDS.md](docs/ODDS.md).

**4. Daily report**
```bash
python scripts/daily.py --run-type morning     # refresh data, retrain if new games, fetch odds, settle, report
python scripts/daily.py --run-type late        # later in the day: confirmed goalies + fresh odds
```
Output: `reports/latest.md`, `reports/daily/<date>-<run>.md`, `site/`, `data/export/`, and the logs in `data/logs/`.
</details>

<details>
<summary><b>Automation (GitHub Actions)</b></summary>

| Workflow | Does |
|---|---|
| `daily.yml` | One morning run per game day (data refresh, retrain if new games, odds, report, commit) and one late run before the first puck drop. |
| `odds-close.yml` | Fetch-only odds snapshots just before games start, for closing-line value. |
| `pages.yml` | Publishes `site/` to GitHub Pages after every daily run. |
| `tendencies.yml` | Weekly report on where the model is systematically off (descriptive only). |
| `goals.yml`, `backfill.yml` | Manual: goals-model backtest; backfill seasons, retune and retrain. |
| `ci.yml` | Tests on every push. |

GitHub's scheduled triggers can arrive hours late, so the daily and closing-line workflows are triggered often and `scripts/gate.py` lets only a useful trigger do any work and a `heartbeat` workflow checks every few minutes and starts whatever is due (see [docs/SITE.md](docs/SITE.md)).
The SQLite database is a rebuildable cache; data that can never be re-fetched (odds snapshots, recommendations) is committed as append-only CSV in `data/logs/`.
</details>

<details>
<summary><b>How the model is judged and what keeps it safe</b></summary>

1. **Log loss / Brier / calibration**, out-of-sample and time-ordered (no random splits anywhere).
2. **Closing-line value** and log loss **against the market close**, as odds snapshots accumulate: the only fast, trustworthy signal of real edge.
3. ROI last: with 1-3% edges you need thousands of bets to tell skill from luck (about 6,900 bets to detect a true 3% ROI at 80% power).

Safety rails: quarter Kelly, 2% per-bet cap, 5% daily cap, max 5 bets a day, 3% minimum raw edge (4% with unconfirmed goalies), model probability shrunk toward the market
(gaps over 12 points are treated as model error), no bets on stale odds, on games already under way, or when the model is unhealthy, and an early-season guard. A drift monitor
suspends all recommendations if the model's recent performance degrades (`ALERT`). See [docs/RISK.md](docs/RISK.md).
</details>

## Known limitations
- No long real-odds history: ROI from the backtest is only scenario-based; live closing-line value accumulates from the first snapshots (moneylines only).
- Goaltending and lineup features are implemented and tested, but the model is still close to Elo-level strength.
- No official injury feed in the free API. Confirmed goalies are a manual input (`data/manual/confirmed_goalies.csv`).
- Playoffs are excluded from training and evaluation. Only moneylines are eligible for real-stake recommendations.

## License
MIT
