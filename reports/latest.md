# NHL model report: 2026-10-07 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-07 21:46 UTC · model `20261006-8663f0` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: d_elo, h_early_w, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-07T21:46:16+00:00 · API credits left: 399*

*Player props: 3 game(s) priced*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PIT @ WSH (19:30 ET) | S. Skinner / L. Thompson (probable) | 49.8% | 59.5% | PIT 2.38 | +9.7% | - | - | no bet |
| COL @ WPG (19:30 ET) | S. Wedgewood / C. Hellebuyck (probable) | 41.7% | 36.0% | WPG 2.70 | +5.6% | - | - | no bet |
| EDM @ ANA (22:00 ET) | C. Ingram / L. Dostal (probable) | 55.9% | 44.9% | ANA 2.15 | +11.1% | - | - | no bet |

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
| EDM @ ANA (22:00 ET) | Total | Under 6.5 | 54.4% | 44.1% | +10.3 pts | +20.2% | 2.21 |
| EDM @ ANA (22:00 ET) | Puck line | ANA +1.5 | 73.9% | 65.9% | +8.0 pts | +10.1% | 1.49 |
| COL @ WPG (19:30 ET) | Puck line | WPG +1.5 | 63.2% | 57.6% | +5.6 pts | +6.8% | 1.69 |
| PIT @ WSH (19:30 ET) | Total | Under 6.5 | 54.5% | 49.5% | +4.9 pts | +6.8% | 1.96 |
| COL @ WPG (19:30 ET) | Total | Under 6.5 | 53.9% | 50.1% | +3.7 pts | +4.5% | 1.94 |
| COL @ WPG (19:30 ET) | Total | Over 6.5 | 46.1% | 49.9% | -3.7 pts | -10.5% | 1.94 |
| PIT @ WSH (19:30 ET) | Total | Over 6.5 | 45.5% | 50.5% | -4.9 pts | -13.0% | 1.91 |
| COL @ WPG (19:30 ET) | Puck line | COL -1.5 | 36.8% | 42.4% | -5.6 pts | -15.3% | 2.30 |
| EDM @ ANA (22:00 ET) | Puck line | EDM -1.5 | 26.1% | 34.1% | -8.0 pts | -26.9% | 2.80 |
| EDM @ ANA (22:00 ET) | Total | Over 6.5 | 45.6% | 55.9% | -10.3 pts | -19.3% | 1.77 |
| PIT @ WSH (19:30 ET) | Puck line | WSH -1.5 | 26.8% | 38.1% | -11.3 pts | -31.5% | 2.55 |


## Player props: shots on goal and points (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| J. Chychrun | PIT @ WSH (19:30 ET) | sog 2.5 | Under | 1.79 | 27.0% | 47.1% | +20.1 pts | +33.5% | 2 2 2 2 3 0 1 1 3 2 | - |
| K. Kapanen | EDM @ ANA (22:00 ET) | sog 1.5 | Under | 1.36 | 38.9% | 55.5% | +16.6 pts | +31.4% | 3 1 2 1 1 1 1 3 1 1 | - |
| V. Podkolzin | EDM @ ANA (22:00 ET) | pts 0.5 | Under | 0.60 | 45.2% | 59.5% | +14.3 pts | +28.8% | 1 0 1 2 0 0 0 2 2 1 | - |
| K. Kapanen | EDM @ ANA (22:00 ET) | pts 0.5 | Under | 0.37 | 31.2% | 47.0% | +15.9 pts | +26.0% | 0 0 1 0 0 0 0 0 2 0 | - |
| C. Gauthier | EDM @ ANA (22:00 ET) | pts 0.5 | Under | 0.75 | 52.7% | 63.9% | +11.2 pts | +25.4% | 0 0 1 1 1 2 1 1 1 3 | - |
| E. Bouchard | EDM @ ANA (22:00 ET) | sog 2.5 | Under | 2.39 | 41.8% | 51.9% | +10.1 pts | +24.5% | 2 4 0 0 1 3 2 6 4 2 | - |
| P. Dubois | PIT @ WSH (19:30 ET) | sog 1.5 | Under | 1.34 | 38.0% | 52.0% | +14.0 pts | +24.0% | 1 0 4 1 1 1 0 2 1 1 | - |
| B. Sennecke | EDM @ ANA (22:00 ET) | pts 0.5 | Under | 0.60 | 45.0% | 58.8% | +13.8 pts | +23.8% | 0 0 1 0 2 0 0 0 1 1 | - |
| C. Gauthier | EDM @ ANA (22:00 ET) | sog 3.5 | Under | 2.96 | 34.5% | 48.5% | +14.1 pts | +22.6% | 2 3 5 5 1 4 4 6 5 4 | - |
| B. Kindel | PIT @ WSH (19:30 ET) | sog 1.5 | Under | 1.65 | 47.8% | 59.0% | +11.2 pts | +21.0% | 1 3 0 1 0 1 1 1 5 2 | - |
| L. Draisaitl | EDM @ ANA (22:00 ET) | pts 1.5 | Under | 1.08 | 29.2% | 44.2% | +15.0 pts | +21.0% | 5 1 1 2 1 1 1 1 3 1 | - |
| D. Strome | PIT @ WSH (19:30 ET) | pts 0.5 | Under | 0.59 | 44.5% | 54.6% | +10.1 pts | +19.3% | 0 0 1 0 2 1 0 0 2 0 | - |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 45 | -1.7% | -31% to +27% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 47 | +10.1% | -13% to +33% | 60% | -1.95% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 47 | -9.1% | -31% to +11% | 60% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 45 | -12.4% | -42% to +15% | 44% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 47 | -4.4% | -42% to +31% | 43% | -2.88% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `market_favorite` | 47 | -5.0% | -28% to +20% | 57% | -2.06% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 6 | +6.1% | - | 50% | -1.44% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `puckline_edge` | 29 | -25.4% | -54% to +3% | 48% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `sog_edge` | 38 | -2.6% | -34% to +26% | 55% | - |  |
| `sog_over_control` | 40 | +3.3% | -32% to +37% | 48% | - |  |
| `totals_edge` | 24 | -17.3% | -53% to +21% | 42% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PIT @ WSH | Moneyline | PIT moneyline | $5.00 | 2.38 | 50.2% | 40.5% |
| COL @ WPG | Moneyline | COL moneyline | $13.00 | 1.53 | 58.3% | 64.0% |
| EDM @ ANA | Moneyline | ANA moneyline | $11.00 | 2.15 | 55.9% | 44.9% |
| PIT @ WSH | Total | Under 6.5 | $9.00 | 1.96 | 54.5% | 49.5% |
| COL @ WPG | Total | Under 6.5 | $9.00 | 1.94 | 53.9% | 50.1% |
| EDM @ ANA | Total | Under 6.5 | $9.00 | 2.21 | 54.4% | 44.1% |
| PIT @ WSH | Puck line | PIT +1.5 | $28.00 | 1.59 | 73.2% | 61.9% |
| COL @ WPG | Puck line | WPG +1.5 | $18.00 | 1.69 | 63.2% | 57.6% |
| EDM @ ANA | Puck line | ANA +1.5 | $29.00 | 1.49 | 73.9% | 65.9% |

Total pretend stake $131.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
