# NHL model report: 2026-10-07 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-07 22:47 UTC · model `20261006-8663f0` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: d_elo, h_early_w, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-07T22:47:04+00:00 · API credits left: 375*

*Player props: 3 game(s) priced*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PIT @ WSH (19:30 ET) | S. Skinner / L. Thompson (probable) | 49.8% | 59.6% | PIT 2.38 | +9.8% | - | - | no bet |
| COL @ WPG (19:30 ET) | S. Wedgewood / C. Hellebuyck (probable) | 41.7% | 36.1% | WPG 2.70 | +5.6% | - | - | no bet |
| EDM @ ANA (22:00 ET) | C. Ingram / L. Dostal (probable) | 55.9% | 45.0% | ANA 2.15 | +10.9% | - | - | no bet |

## Why

**PIT @ WSH (19:30 ET)**: No bet: early-season sample size: a team has only played 2 games (< 10)
  - Context: PIT have the clearly better goal differential this season (gap 0.32 goals/game). PIT are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**COL @ WPG (19:30 ET)**: No bet: early-season sample size: a team has only played 2 games (< 10)
  - Context: COL have the clearly better goal differential this season (gap 1.11 goals/game). COL are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**EDM @ ANA (22:00 ET)**: No bet: early-season sample size: a team has only played 2 games (< 10)
  - Context: ANA are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **47** (bet on 0, passed on 100%); model log loss 0.6493, Brier 0.2286 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (47 games with odds): model log loss 0.6493 vs market 0.6408 (market better)

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| PIT @ WSH (19:30 ET) | Puck line | PIT +1.5 | 73.2% | 61.9% | +11.3 pts | +16.3% | 1.59 |
| EDM @ ANA (22:00 ET) | Total | Under 6.5 | 54.4% | 43.3% | +11.1 pts | +24.0% | 2.28 |
| EDM @ ANA (22:00 ET) | Puck line | ANA +1.5 | 73.9% | 65.9% | +8.0 pts | +10.1% | 1.49 |
| COL @ WPG (19:30 ET) | Puck line | WPG +1.5 | 63.2% | 57.6% | +5.6 pts | +6.8% | 1.69 |
| PIT @ WSH (19:30 ET) | Total | Under 6.5 | 54.5% | 49.4% | +5.1 pts | +7.3% | 1.97 |
| COL @ WPG (19:30 ET) | Total | Under 6.5 | 53.9% | 50.3% | +3.6 pts | +4.5% | 1.94 |
| COL @ WPG (19:30 ET) | Total | Over 6.5 | 46.1% | 49.7% | -3.6 pts | -9.5% | 1.96 |
| PIT @ WSH (19:30 ET) | Total | Over 6.5 | 45.5% | 50.6% | -5.1 pts | -13.0% | 1.91 |
| COL @ WPG (19:30 ET) | Puck line | COL -1.5 | 36.8% | 42.4% | -5.6 pts | -15.3% | 2.30 |
| EDM @ ANA (22:00 ET) | Puck line | EDM -1.5 | 26.1% | 34.1% | -8.0 pts | -26.9% | 2.80 |
| EDM @ ANA (22:00 ET) | Total | Over 6.5 | 45.6% | 56.7% | -11.1 pts | -20.2% | 1.75 |
| PIT @ WSH (19:30 ET) | Puck line | WSH -1.5 | 26.8% | 38.1% | -11.3 pts | -31.5% | 2.55 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| M. Ferraro | COL @ WPG (19:30 ET) | goal 0.5 | Over | 0.18 | 16.4% | 7.1% | +9.4 pts | +195.9% | 0 0 0 0 0 1 0 1 0 0 | - |
| D. DeMelo | COL @ WPG (19:30 ET) | goal 0.5 | Over | 0.12 | 11.5% | 6.7% | +4.7 pts | +152.3% | 0 0 0 0 0 0 0 1 0 0 | - |
| T. van Riemsdyk | PIT @ WSH (19:30 ET) | goal 0.5 | Over | 0.10 | 9.4% | 6.0% | +3.4 pts | +145.3% | 0 0 0 0 0 1 0 0 0 0 | - |
| D. Carlile | PIT @ WSH (19:30 ET) | goal 0.5 | Over | 0.14 | 13.1% | 7.8% | +5.3 pts | +110.0% | 0 0 0 0 0 0 0 0 1 1 | - |
| M. Fehérváry | PIT @ WSH (19:30 ET) | goal 0.5 | Over | 0.10 | 9.8% | 6.4% | +3.4 pts | +96.2% | 0 0 0 0 1 0 0 0 0 0 | - |
| R. Shea | EDM @ ANA (22:00 ET) | goal 0.5 | Over | 0.16 | 15.0% | 9.4% | +5.6 pts | +80.1% | 0 0 1 0 0 0 0 0 1 0 | - |
| J. Manson | COL @ WPG (19:30 ET) | goal 0.5 | Over | 0.11 | 10.5% | 7.1% | +3.4 pts | +78.2% | 0 0 0 0 0 0 0 0 0 1 | - |
| T. Liljegren | PIT @ WSH (19:30 ET) | goal 0.5 | Over | 0.09 | 8.2% | 5.9% | +2.3 pts | +71.4% | 0 0 1 0 0 0 0 0 0 0 | - |
| S. Malinski | COL @ WPG (19:30 ET) | goal 0.5 | Over | 0.18 | 16.1% | 10.2% | +5.9 pts | +61.1% | 0 0 0 0 0 1 0 0 0 0 | - |
| P. Kelly | COL @ WPG (19:30 ET) | goal 0.5 | Over | 0.21 | 18.9% | 13.1% | +5.8 pts | +60.6% | 0 1 0 0 0 0 0 1 1 0 | - |
| C. Murphy | EDM @ ANA (22:00 ET) | goal 0.5 | Over | 0.08 | 8.0% | 5.9% | +2.0 pts | +51.1% | 0 0 0 0 0 0 0 0 0 0 | - |
| B. Burns | COL @ WPG (19:30 ET) | goal 0.5 | Over | 0.12 | 11.4% | 9.5% | +1.9 pts | +37.2% | 0 1 0 0 0 0 0 0 0 0 | - |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 45 | -1.7% | -29% to +25% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 47 | +10.1% | -14% to +32% | 60% | -1.95% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 47 | -9.1% | -31% to +12% | 60% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 45 | -12.4% | -41% to +16% | 44% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 47 | -4.4% | -40% to +32% | 43% | -2.88% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `market_favorite` | 47 | -5.0% | -28% to +19% | 57% | -2.06% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 6 | +6.1% | - | 50% | -1.44% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `puckline_edge` | 29 | -25.4% | -56% to +5% | 48% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `sog_edge` | 38 | -2.6% | -30% to +27% | 55% | - |  |
| `sog_over_control` | 40 | +3.3% | -30% to +38% | 48% | - |  |
| `totals_edge` | 24 | -17.3% | -56% to +22% | 42% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PIT @ WSH | Moneyline | PIT moneyline | $5.00 | 2.38 | 50.2% | 40.4% |
| COL @ WPG | Moneyline | COL moneyline | $13.00 | 1.53 | 58.3% | 63.9% |
| EDM @ ANA | Moneyline | ANA moneyline | $11.00 | 2.15 | 55.9% | 45.0% |
| PIT @ WSH | Total | Under 6.5 | $9.00 | 1.97 | 54.5% | 49.4% |
| COL @ WPG | Total | Under 6.5 | $9.00 | 1.94 | 53.9% | 50.3% |
| EDM @ ANA | Total | Under 6.5 | $9.00 | 2.28 | 54.4% | 43.3% |
| PIT @ WSH | Puck line | PIT +1.5 | $28.00 | 1.59 | 73.2% | 61.9% |
| COL @ WPG | Puck line | WPG +1.5 | $18.00 | 1.69 | 63.2% | 57.6% |
| EDM @ ANA | Puck line | ANA +1.5 | $29.00 | 1.49 | 73.9% | 65.9% |

Total pretend stake $131.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
