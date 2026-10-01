# NHL game-outcome model: calibrated probabilities, market prices, risk-managed staking

> **Disclaimer.** This is a **research and educational project**. No model guarantees profit, and back-tested or past results
> do not predict future results. Sports betting carries a real risk of loss: only stake money you can afford to lose, and check
> that betting is legal where you live. Nothing in this repository is financial advice.

<!-- PICKS:START -->

### Today's picks: 2026-09-30

**3 game(s) · 0 recommended bet(s)** · late run · model health **OK** · updated 2026-10-01 00:32 UTC

| | |
|---|---|
| 📄 **[Today's full report](reports/latest.md)** | every game: model vs market, edge, stake and the reason for each decision |
| 🌐 **[Interactive picks page](https://flagk.github.io/nhl-tracker/)** | choose your unit size and staking, top picks, parlay calculator |
| 🗂️ **[Pick history](reports/HISTORY.md)** | every day's picks and how they settled ([web version](https://flagk.github.io/nhl-tracker/history.html)) |
| 🧾 **[My bets & fake bets](https://flagk.github.io/nhl-tracker/bets.html)** | log your own bets (kept only in your browser) and see the pretend-money bets on every game |
| 📈 Bet log (CSV) | appears here after the first recommended bet settles |

| Game | Model: home win | Market (no-vig) | Decision |
|---|---|---|---|
| PIT @ PHI | 49.4% | 21.8% | no bet |
| NYI @ TOR | 38.6% | 86.1% | no bet |
| LAK @ COL | 62.7% | 64.5% | no bet |

*No bets today is normal: the model only bets when it sees a sizeable edge over the market.*

*Research and educational project; no guarantee of profit. The numbers above are not betting advice.*

<!-- PICKS:END -->

> The **Interactive picks page** link uses GitHub Pages. Enable it once (Settings -> Pages -> Source: *GitHub Actions*); after that it updates itself
> after every daily run. The repository is public, so published pages and logs are **public-safe**: model probabilities, edges and stakes, but
> no bookmaker names or per-bookmaker prices.

## Honest status

| | |
|---|---|
| **Has an edge over the betting market been shown?** | **No.** The model has never been compared to real closing lines (none were available). Treat it as a research system that *measures* whether it has edge, not one that has it. |
| **What the model can do (walk-forward on real NHL API data: trained on 8 seasons, 2,269 out-of-sample games with calibration history, Oct 2024 to Sep 2026)** | Production model log loss **0.6759** vs 0.6896 for "always the base rate" and 0.6812 for plain Elo. The gap to Elo (-0.0053, 95% CI -0.0100 to -0.0003) and to the base rate (-0.0137) both exclude zero. A coin flip is 0.6931. **Against the market: unknown.** |
| **The original README's "55.8% win rate / +11.69% ROI / proven edge"** | **Not valid.** It assumed even-money payouts (no odds, no vig), had corrupted features, and did no better than always picking the home team. See [AUDIT.md](AUDIT.md). |
| **Most likely daily output** | **"No bet."** By design. |

Full numbers, protocol and caveats: [docs/RESULTS.md](docs/RESULTS.md) · [reports/model_comparison.md](reports/model_comparison.md).

## What's here

```
nhlbet/
  data/       NHL API client (cache, retries, circuit breaker) -> parsers -> SQLite store (idempotent upserts), xG model, goalie inputs
  features/   leak-free as-of-date feature builder (Elo, form, rest/travel, xG/Corsi/PDO, special teams, goalies, lineup, situational, market)
  analysis/   correlation pruning, walk-forward permutation importance, SHAP
  models/     home-rate, Elo-only, logistic, Random Forest, LightGBM, XGBoost; stacking/weighted ensembles; Platt/isotonic/online calibration;
              time-series-CV tuning; walk-forward engine; model bundle
  odds/       The Odds API client (quota-aware), odds math (Shin/proportional de-vig), best price, edge/EV, append-only snapshots, CLV
  risk/       fractional Kelly, caps, shrinkage toward the market, "no bet" policy, bankroll/drawdown, Monte Carlo risk of ruin
  report/     slate builder, markdown daily report, bet log (settlement, ROI, CLV, Brier, calibration), CSV export/restore
  registry.py model registry;  monitor.py drift checks;  train.py retrain-if-needed;  pipeline.py the daily job
scripts/      daily.py  fetch_odds.py  train.py  backtest.py  tune.py  select_features.py  risk_sim.py  check_api.py
tests/        108 tests (no-leakage property tests, parsers, ingestion, odds math, Kelly, risk invariants, reports, workflows)
docs/         FEATURES.md  ODDS.md  RISK.md  RESULTS.md      AUDIT.md = audit of the original pipeline
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                   # 108 tests, ~30 s
```

### 1. Data
```bash
python scripts/check_api.py --season 20242025 --team BOS --n 5       # validates the parsers against the LIVE NHL API first
python -m nhlbet.data.ingest --seasons 20232024 20242025 20252026    # backfill (cached + idempotent; first run takes a while)
```
Without network access the daily job bootstraps from `data/history/nhl_history.csv` (2,987 games, 28 of 32 teams, scores only). That is enough
to run everything except the advanced/goalie/lineup features, which are NaN on that file.

> The NHL API parsers follow the documented payload shape and are tested against hand-built fixtures; **they have not been validated against the
> live API from the development sandbox** (blocked by network policy). `scripts/check_api.py` (also the first step of the backfill workflow) cross-checks
> parsed play-by-play against official scores and shot totals and fails loudly if reality differs.

### 2. Research pipeline (feature selection, tuning, walk-forward backtest, model)
```bash
python scripts/select_features.py --selection-end 2024-06-30
python scripts/tune.py --tune-end 2024-06-30
python scripts/backtest.py --eval-start 2024-10-01     # writes reports/model_comparison.md + reliability.png
python scripts/train.py --force                        # trains, drift-checks, registers a version
```
Selection and tuning only see games up to `--selection-end`; evaluation starts later, so nothing is chosen on the data it is judged on.

### 3. Odds API key (optional but needed for any recommendation)
1. Sign up for a free key at <https://the-odds-api.com> (500 credits/month; one moneyline call = 1 credit).
2. Locally: `export ODDS_API_KEY=<your key>`. **Never commit it.** The code reads it only from the environment and never writes it to disk or logs.
3. GitHub Actions: *Settings -> Secrets and variables -> Actions -> New repository secret* named `ODDS_API_KEY`.
   (Optional variable `BANKROLL`, default 1000.)

Without a key every game is reported as "no odds -> no bet". Details, quota plan and failure behaviour: [docs/ODDS.md](docs/ODDS.md).

### 4. Daily report
```bash
python scripts/daily.py --run-type morning     # refresh data, retrain if new games, fetch odds, settle, report
python scripts/daily.py --run-type late        # later in the day: confirmed goalies + fresh odds
```
Output: `reports/latest.md`, `reports/daily/<date>-<run>.md`, `reports/performance.png`, and the logs in `data/logs/`.
Each game lists teams, goalies (confirmed/probable), model win probability, best price and book, no-vig market probability, edge, EV per $1,
stake and a plain-language reason. Confirmed goalies: add rows to `data/manual/confirmed_goalies.csv` (`date,team,name`); the free NHL API does
not publish them, so this is an input, not automatic.

## Picks page
`site/index.html` is rebuilt by every daily run: pick your unit size and risk level, see the top 5 games by expected value (recommended vs lean),
and use a parlay calculator with guardrails. See [docs/SITE.md](docs/SITE.md). It contains bookmaker prices, so keep the repo private.

## Paper trading
Every slate is also run through five fake-money strategies (including a no-skill control that bets the market favourite), settled automatically, to
measure what works faster than the selective live policy can. It is for measurement, not training. See [docs/PAPER_TRADING.md](docs/PAPER_TRADING.md).

## Automation (GitHub Actions)
| Workflow | When | Does |
|---|---|---|
| `daily.yml` | ~10:30 ET **and** ~17:15 ET | morning: refresh data, retrain if there are new games, fetch odds, report, commit. Late: rerun after starting goalies are usually confirmed. |
| `odds-close.yml` | ~18:45 and ~21:45 ET | fetch-only odds snapshots for closing-line value |
| `backfill.yml` | manual | validate parsers on the live API (gate), backfill seasons, reselect features, retune, backtest, retrain |
| `ci.yml` | every push | tests |

The SQLite database is rebuildable cache (`actions/cache`); data that can never be re-fetched (odds snapshots, recommendations) is committed as
append-only CSV in `data/logs/` and restored at the start of every run. The first run after setup should be `backfill.yml`.

## How the model is judged
1. **Log loss / Brier / calibration** out-of-sample and time-ordered (no random splits anywhere); calibration curves in `reports/reliability.png`.
2. **Closing-line value (CLV)** and log loss **against the market close**, as odds snapshots accumulate: the only fast, trustworthy signal of real edge.
3. ROI last: with ~1-3% edges you need thousands of bets to tell skill from luck (about 6,900 bets to detect a true 3% ROI at 80% power).

Drift monitor (`nhlbet/monitor.py`): each retrain compares recent log loss/calibration to history and to Elo, and PSI on features. `ALERT` suspends all
recommendations; `WARN` is shown at the top of the report. (At the time of writing it reports `WARN`: the last 200 games show no skill.)

## Safety rails
Quarter Kelly, 2% per-bet cap, 5% daily cap, max 5 bets/day, 3% minimum raw edge (4% with unconfirmed goalies), model probability shrunk toward the
market (large gaps distrusted, >12 points = "model error"), no bets on stale odds or when the model is unhealthy, early-season guard. See [docs/RISK.md](docs/RISK.md),
including a Monte Carlo of how often even a *real* 3% edge finishes a 500-bet stretch down (40%).

## Known limitations
- No real historical odds: backtest ROI is reported as scenarios only; live CLV will accumulate from the first fetched snapshot.
- Advanced-metric, goaltending and lineup features are implemented and tested but untested on real API data; the score-only model is at Elo level.
- No official injury feed in the free API; lineup features are off by default. Confirmed goalies are a manual input.
- Playoffs are excluded from training/evaluation. Only moneylines are bet (puck line/totals are stored, not modelled).
- Playoff-race (`cutoff_gap`) and rivalry features are approximations (see docs/FEATURES.md).

## License
MIT

## Dashboards and tendencies

- **Power BI / Excel:** the daily run publishes tidy CSVs to [`data/export/`](data/export); setup steps, `.pbids` connectors and DAX measures are in [`powerbi/`](powerbi/README.md).
- **Tendencies:** a weekly job ([`reports/TENDENCIES.md`](reports/TENDENCIES.md)) looks for segments where the model is systematically off, corrected for multiple testing and required to replicate across both halves of the data. It is descriptive only and never changes the model or staking.

## Other bet types
Beyond moneylines, a separate goals model prices **totals (over/under)** and the **puck line**. These are experimental and paper-traded only (no real stake suggestions) until settled results support them; see [`docs/MARKETS.md`](docs/MARKETS.md). On the [bets page](site/bets.html) you can log your own bets of any type (moneyline, puck line, total, parlay, prop).
