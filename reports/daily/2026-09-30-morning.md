# NHL model report: 2026-09-30 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-09-30 22:28 UTC · model `20260929-3184e7-r2` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

*Odds snapshot: 2026-09-30T22:28:38+00:00 · API credits left: 481*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PIT @ PHI (19:30 ET) | S. Skinner / D. Vladar (probable) | 49.4% | 57.2% | PIT 2.25 | +7.8% | - | - | no bet |
| NYI @ TOR (19:30 ET) | I. Sorokin / J. Woll (probable) | 38.6% | 55.7% | NYI 2.20 | +17.1% | - | - | no bet |
| LAK @ COL (22:00 ET) | A. Forsberg / S. Wedgewood (probable) | 62.7% | 64.1% | LAK 2.71 | +1.5% | - | - | no bet |

## Why

**PIT @ PHI (19:30 ET)**: No bet: early-season sample size: a team has only played 0 games (< 10)
  - Context: PIT are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**NYI @ TOR (19:30 ET)**: No bet: early-season sample size: a team has only played 0 games (< 10)
  - Context: NYI have the clearly better goal differential this season (gap 0.28 goals/game). TOR is on the second night of a back-to-back. NYI are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**LAK @ COL (22:00 ET)**: No bet: early-season sample size: a team has only played 0 games (< 10)
  - Context: COL have the clearly better goal differential this season (gap 0.73 goals/game). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

No resolved recommendations yet. Odds snapshots and recommendations are logged from the first run so this table fills in automatically; **judge the model by closing-line value and log loss vs the market, not by early profit.**

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| NYI @ TOR (19:30 ET) | Puck line | NYI +1.5 | 81.9% | 67.2% | +14.7 pts | +19.6% | 1.46 |
| PIT @ PHI (19:30 ET) | Puck line | PIT +1.5 | 76.0% | 65.8% | +10.2 pts | +13.3% | 1.49 |
| LAK @ COL (22:00 ET) | Puck line | LAK +1.5 | 65.5% | 59.0% | +6.5 pts | +8.1% | 1.65 |
| LAK @ COL (22:00 ET) | Total | Under 6 | 52.8% | 46.8% | +5.9 pts | +8.1% | 2.07 |
| PIT @ PHI (19:30 ET) | Total | Under 6 | 53.0% | 48.6% | +4.5 pts | +4.9% | 1.99 |
| NYI @ TOR (19:30 ET) | Total | Under 6 | 53.6% | 50.7% | +2.9 pts | +2.1% | 1.91 |
| NYI @ TOR (19:30 ET) | Total | Over 6 | 46.4% | 49.3% | -2.9 pts | -7.5% | 1.97 |
| PIT @ PHI (19:30 ET) | Total | Over 6 | 47.0% | 51.4% | -4.5 pts | -10.7% | 1.87 |
| LAK @ COL (22:00 ET) | Total | Over 6 | 47.2% | 53.2% | -5.9 pts | -12.3% | 1.82 |
| LAK @ COL (22:00 ET) | Puck line | COL -1.5 | 34.5% | 41.0% | -6.5 pts | -18.9% | 2.35 |
| PIT @ PHI (19:30 ET) | Puck line | PHI -1.5 | 24.0% | 34.2% | -10.2 pts | -32.2% | 2.83 |
| NYI @ TOR (19:30 ET) | Puck line | TOR -1.5 | 18.1% | 32.8% | -14.7 pts | -47.3% | 2.92 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

No settled paper bets yet.

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PIT @ PHI | Moneyline | PIT moneyline | $10.00 | 2.25 | 50.6% | 42.8% |
| NYI @ TOR | Moneyline | NYI moneyline | $10.00 | 2.20 | 61.4% | 44.3% |
| LAK @ COL | Moneyline | COL moneyline | $10.00 | 1.53 | 62.7% | 64.1% |
| PIT @ PHI | Total | Under 6 | $10.00 | 1.99 | 53.0% | 48.6% |
| NYI @ TOR | Total | Under 6 | $10.00 | 1.91 | 53.6% | 50.7% |
| LAK @ COL | Total | Under 6 | $10.00 | 2.07 | 52.8% | 46.8% |
| PIT @ PHI | Puck line | PIT +1.5 | $10.00 | 1.49 | 76.0% | 65.8% |
| NYI @ TOR | Puck line | NYI +1.5 | $10.00 | 1.46 | 81.9% | 67.2% |
| LAK @ COL | Puck line | LAK +1.5 | $10.00 | 1.65 | 65.5% | 59.0% |

Total pretend stake $90.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
