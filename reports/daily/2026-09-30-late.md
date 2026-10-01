# NHL model report: 2026-09-30 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-01 00:32 UTC · model `20260929-3184e7-r2` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

*Odds snapshot: 2026-10-01T00:32:42+00:00 · API credits left: 497*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PIT @ PHI (19:30 ET) | S. Skinner / D. Vladar (probable) | 49.4% | 21.8% | PHI 4.30 | +27.5% | - | - | no bet |
| NYI @ TOR (19:30 ET) | I. Sorokin / J. Woll (probable) | 38.6% | 86.1% | NYI 6.20 | +47.5% | - | - | no bet |
| LAK @ COL (22:00 ET) | A. Forsberg / S. Wedgewood (probable) | 62.7% | 64.5% | LAK 2.73 | +1.8% | - | - | no bet |

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
| NYI @ TOR (19:30 ET) | Puck line | NYI +2.5 | 90.7% | 47.4% | +43.3 pts | +86.0% | 2.05 |
| PIT @ PHI (19:30 ET) | Puck line | PHI +1.5 | 74.9% | 42.6% | +32.2 pts | +66.2% | 2.22 |
| LAK @ COL (22:00 ET) | Puck line | LAK +1.5 | 65.5% | 58.6% | +6.9 pts | +9.4% | 1.67 |
| PIT @ PHI (19:30 ET) | Total | Under 6.5 | 58.9% | 52.3% | +6.6 pts | +9.0% | 1.85 |
| LAK @ COL (22:00 ET) | Total | Under 6.5 | 58.7% | 53.1% | +5.7 pts | +7.5% | 1.83 |
| NYI @ TOR (19:30 ET) | Total | Under 6.5 | 59.4% | 54.2% | +5.2 pts | +5.2% | 1.77 |
| NYI @ TOR (19:30 ET) | Total | Over 6.5 | 40.6% | 45.8% | -5.2 pts | -16.4% | 2.06 |
| LAK @ COL (22:00 ET) | Total | Over 6.5 | 41.3% | 46.9% | -5.7 pts | -15.0% | 2.06 |
| PIT @ PHI (19:30 ET) | Total | Over 6.5 | 41.1% | 47.7% | -6.6 pts | -17.8% | 2.00 |
| LAK @ COL (22:00 ET) | Puck line | COL -1.5 | 34.5% | 41.4% | -6.9 pts | -19.3% | 2.34 |
| PIT @ PHI (19:30 ET) | Puck line | PIT -1.5 | 25.1% | 57.4% | -32.2 pts | -58.5% | 1.65 |
| NYI @ TOR (19:30 ET) | Puck line | TOR -2.5 | 9.3% | 52.6% | -43.3 pts | -83.3% | 1.80 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

No settled paper bets yet.

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PIT @ PHI | Moneyline | PIT moneyline | $10.00 | 1.24 | 50.6% | 78.2% |
| NYI @ TOR | Moneyline | NYI moneyline | $10.00 | 6.20 | 61.4% | 13.9% |
| LAK @ COL | Moneyline | COL moneyline | $10.00 | 1.53 | 62.7% | 64.5% |
| PIT @ PHI | Total | Under 6.5 | $10.00 | 1.85 | 58.9% | 52.3% |
| NYI @ TOR | Total | Under 6.5 | $10.00 | 1.77 | 59.4% | 54.2% |
| LAK @ COL | Total | Under 6.5 | $10.00 | 1.83 | 58.7% | 53.1% |
| PIT @ PHI | Puck line | PHI +1.5 | $10.00 | 2.22 | 74.9% | 42.6% |
| NYI @ TOR | Puck line | NYI +2.5 | $10.00 | 2.05 | 90.7% | 47.4% |
| LAK @ COL | Puck line | LAK +1.5 | $10.00 | 1.67 | 65.5% | 58.6% |

Total pretend stake $90.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
