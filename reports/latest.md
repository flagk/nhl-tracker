# NHL model report: 2026-11-10 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-10 21:04 UTC · model `20261109-720fd7` · probability source `online_platt` · bankroll $983.82 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-10T21:02:12+00:00 · API credits left: 336*

*Player props: 3 game(s) priced*

## Summary

- 7 games · **1 recommended bet(s)** · total stake $2.61 (0.27% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| COL @ TOR (19:00 ET) | M. Blackwood / S. Bobrovsky (probable) | 43.9% | 39.2% | TOR 2.50 | +4.7% | +9.8% | $2.61 | **BET TOR** |
| MIN @ MTL (19:00 ET) | J. Wallstedt / J. Fowler (probable) | 49.0% | 46.1% | MTL 2.10 | +2.8% | - | - | no bet |
| WSH @ OTT (19:00 ET) | L. Thompson / L. Ullmark (probable) | 54.6% | 49.8% | OTT 1.96 | +4.7% | - | - | no bet |
| NJD @ WPG (19:00 ET) | J. Allen / S. Skinner (probable) | 43.7% | 42.9% | WPG 2.27 | +0.8% | - | - | no bet |
| FLA @ CBJ (19:30 ET) | J. Markstrom / J. Greaves (probable) | 49.5% | 46.9% | CBJ 2.07 | +2.6% | - | - | no bet |
| DET @ STL (20:00 ET) | J. Gibson / J. Hofer (probable) | 53.3% | 56.7% | DET 2.23 | +3.3% | - | - | no bet |
| NSH @ LAK (22:00 ET) | J. Saros / D. Kuemper (probable) | 57.9% | 54.7% | LAK 1.78 | +3.2% | - | - | no bet |

## Why

**COL @ TOR (19:00 ET)**: Bet $2.61 (0.27% of bankroll) on TOR at 2.50. The model gives TOR 43.9% vs the market's no-vig 39.2% (edge +4.7%); after shrinking toward the market the working probability is 40.6%, worth +1.6% per $1 at that price.
  - Context: COL have the clearly better goal differential this season (gap 0.86 goals/game).

**MIN @ MTL (19:00 ET)**: No bet: edge +2.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: MIN have the clearly better goal differential this season (gap 0.36 goals/game). MIN are creating the better share of expected goals recently.

**WSH @ OTT (19:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.5% (< 1%)
  - Context: Goaltending edge to OTT (goals saved above expected).

**NJD @ WPG (19:00 ET)**: No bet: edge +0.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: NJD have the clearly better goal differential this season (gap 0.89 goals/game). NJD are creating the better share of expected goals recently.

**FLA @ CBJ (19:30 ET)**: No bet: edge +2.6% below the 4% minimum (starting goalie unconfirmed)
  - Context: FLA are creating the better share of expected goals recently.

**DET @ STL (20:00 ET)**: No bet: edge +3.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: Goaltending edge to STL (goals saved above expected).

**NSH @ LAK (22:00 ET)**: No bet: edge +3.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: NSH have the clearly better goal differential this season (gap 0.29 goals/game). NSH travelled 2,862 km across 2 time zone(s) since their last game. LAK are creating the better share of expected goals recently. Goaltending edge to NSH (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **280** (bet on 8, passed on 97%); model log loss 0.6528, Brier 0.2305 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (280 games with odds): model log loss 0.6528 vs market 0.6388 (market better)
- Bets: 8 · staked $21.06 · profit $-16.18 · **ROI -76.8%** · win rate 12.5% · avg price 2.52
- Bankroll $983.82 · max drawdown 1.6% · current drawdown 1.6%
- **CLV**: mean -2.25% per $1 (95% CI -3.89% to -0.05%), beat the close on 12% of 8 bets
- ⚠️ 8 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.4% | 42.9% | 35 |
| 46.5% | 42.9% | 35 |
| 50.0% | 45.7% | 35 |
| 52.8% | 60.0% | 35 |
| 55.5% | 48.6% | 35 |
| 58.7% | 68.6% | 35 |
| 62.2% | 68.6% | 35 |
| 68.5% | 80.0% | 35 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| COL @ TOR (19:00 ET) | Total | Under 6.5 | 55.7% | 46.9% | +8.8 pts | +16.4% | 2.09 |
| NJD @ WPG (19:00 ET) | Puck line | NJD -1.5 | 40.9% | 35.5% | +5.3 pts | +10.3% | 2.70 |
| NJD @ WPG (19:00 ET) | Total | Under 6.5 | 55.0% | 50.1% | +4.9 pts | +8.3% | 1.97 |
| DET @ STL (20:00 ET) | Total | Over 5.5 | 58.2% | 53.6% | +4.6 pts | +5.3% | 1.81 |
| WSH @ OTT (19:00 ET) | Puck line | OTT -1.5 | 32.7% | 29.1% | +3.6 pts | +7.8% | 3.30 |
| COL @ TOR (19:00 ET) | Puck line | TOR +1.5 | 63.7% | 60.5% | +3.2 pts | +3.3% | 1.62 |
| WSH @ OTT (19:00 ET) | Total | Under 6.5 | 54.5% | 51.5% | +3.0 pts | +3.0% | 1.89 |
| FLA @ CBJ (19:30 ET) | Total | Over 6 | 52.4% | 49.4% | +3.0 pts | +1.8% | 1.95 |
| DET @ STL (20:00 ET) | Puck line | DET +1.5 | 67.7% | 65.6% | +2.0 pts | +2.9% | 1.52 |
| MIN @ MTL (19:00 ET) | Total | Under 6.5 | 55.1% | 53.6% | +1.5 pts | -0.8% | 1.80 |
| FLA @ CBJ (19:30 ET) | Puck line | FLA -1.5 | 31.7% | 30.3% | +1.4 pts | +0.3% | 3.16 |
| MIN @ MTL (19:00 ET) | Puck line | MIN -1.5 | 32.6% | 31.8% | +0.8 pts | -2.2% | 3.00 |
| NSH @ LAK (22:00 ET) | Total | Over 6 | 52.3% | 52.0% | +0.3 pts | -2.0% | 1.87 |
| NSH @ LAK (22:00 ET) | Puck line | NSH +1.5 | 67.6% | 67.3% | +0.3 pts | -2.0% | 1.45 |
| NSH @ LAK (22:00 ET) | Puck line | LAK -1.5 | 32.4% | 32.7% | -0.3 pts | -5.4% | 2.92 |
| NSH @ LAK (22:00 ET) | Total | Under 6 | 47.7% | 48.0% | -0.3 pts | -3.2% | 2.02 |
| MIN @ MTL (19:00 ET) | Puck line | MTL +1.5 | 67.4% | 68.2% | -0.8 pts | -3.6% | 1.43 |
| FLA @ CBJ (19:30 ET) | Puck line | CBJ +1.5 | 68.3% | 69.7% | -1.4 pts | -2.4% | 1.43 |
| MIN @ MTL (19:00 ET) | Total | Over 6.5 | 44.9% | 46.4% | -1.5 pts | -6.2% | 2.09 |
| DET @ STL (20:00 ET) | Puck line | STL -1.5 | 32.3% | 34.4% | -2.0 pts | -9.2% | 2.81 |
| FLA @ CBJ (19:30 ET) | Total | Under 6 | 47.6% | 50.6% | -3.0 pts | -7.9% | 1.91 |
| WSH @ OTT (19:00 ET) | Total | Over 6.5 | 45.5% | 48.5% | -3.0 pts | -8.1% | 2.02 |
| COL @ TOR (19:00 ET) | Puck line | COL -1.5 | 36.3% | 39.5% | -3.2 pts | -10.1% | 2.48 |
| WSH @ OTT (19:00 ET) | Puck line | WSH +1.5 | 67.3% | 70.9% | -3.6 pts | -5.8% | 1.40 |
| DET @ STL (20:00 ET) | Total | Under 5.5 | 41.8% | 46.4% | -4.6 pts | -11.4% | 2.12 |
| NJD @ WPG (19:00 ET) | Total | Over 6.5 | 45.0% | 49.9% | -4.9 pts | -12.2% | 1.95 |
| NJD @ WPG (19:00 ET) | Puck line | WPG +1.5 | 59.1% | 64.5% | -5.3 pts | -10.1% | 1.52 |
| COL @ TOR (19:00 ET) | Total | Over 6.5 | 44.3% | 53.1% | -8.8 pts | -18.9% | 1.83 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Z. Bogosian | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.06 | 5.8% | 5.0% | +0.8 pts | +68.7% | 0 0 0 0 0 0 0 0 1 0 | 0.1 |
| B. Duhaime | COL @ TOR (19:00 ET) | goal 0.5 | Over | 0.13 | 11.9% | 8.9% | +3.1 pts | +55.3% | 0 0 1 0 0 0 0 0 0 0 | 0.1 |
| K. Guhle | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.11 | 10.3% | 7.6% | +2.7 pts | +53.8% | 0 0 1 1 0 0 0 0 0 0 | 0.2 |
| A. Carrier | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.08 | 7.2% | 6.0% | +1.2 pts | +51.9% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| T. Stecher | COL @ TOR (19:00 ET) | goal 0.5 | Over | 0.09 | 8.4% | 6.7% | +1.8 pts | +51.8% | 0 0 1 1 0 0 0 0 0 0 | 0.1 |
| J. Evans | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.21 | 18.6% | 13.0% | +5.6 pts | +48.8% | 0 0 2 0 0 0 0 0 0 0 | 0.1 |
| T. Blueger | COL @ TOR (19:00 ET) | goal 0.5 | Over | 0.18 | 16.3% | 10.9% | +5.4 pts | +47.1% | 0 0 1 0 0 0 1 0 0 0 | 0.1 |
| O. Maatta | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.07 | 6.5% | 5.8% | +0.7 pts | +36.1% | 0 0 1 0 0 0 0 0 0 0 | 0.1 |
| S. Malinski | COL @ TOR (19:00 ET) | sog 1.5 | Under | 1.41 | 40.3% | 57.7% | +17.5 pts | +34.4% | 2 0 1 0 3 3 2 1 3 0 | 1.3 |
| P. Danault | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.21 | 19.0% | 13.9% | +5.1 pts | +33.1% | 0 0 0 0 0 1 0 0 0 0 | 0.1 |
| J. Anderson | MIN @ MTL (19:00 ET) | goal 0.5 | Over | 0.24 | 21.7% | 17.9% | +3.8 pts | +27.9% | 2 0 0 0 0 0 0 2 1 0 | 0.4 |
| P. Danault | MIN @ MTL (19:00 ET) | pts 0.5 | Over | 0.48 | 38.3% | 27.3% | +11.1 pts | +27.7% | 0 1 1 0 0 2 1 0 0 0 | 0.4 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 267 | -1.7% | -14% to +10% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 222 | -4.5% | -19% to +8% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 112 | -4.6% | -20% to +11% | 60% | - |  |
| `ast_over_control` | 111 | +7.5% | -11% to +25% | 56% | - |  |
| `ast_over_edge` | 94 | -11.6% | -40% to +20% | 29% | - |  |
| `ast_under_edge` | 112 | -0.3% | -15% to +13% | 66% | - |  |
| `edge_1pct` | 22 | -32.4% | -73% to +10% | 32% | -1.30% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 280 | +1.8% | -8% to +12% | 61% | -1.94% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 279 | -4.3% | -13% to +5% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 267 | -6.9% | -18% to +5% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 280 | -19.6% | -31% to -8% | 39% | -2.61% | $10 on every game where the model sees any positive edge |
| `goal_control` | 112 | -7.0% | -32% to +19% | 33% | - |  |
| `goal_edge` | 70 | -21.5% | -70% to +40% | 11% | - |  |
| `home_ml_control` | 233 | +2.9% | -9% to +14% | 57% | -2.52% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 280 | +3.4% | -7% to +13% | 62% | -2.01% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 23 | -26.0% | -68% to +20% | 30% | -1.67% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 31 | -10.4% | -48% to +27% | 42% | -2.43% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 57 | -7.1% | -30% to +16% | 49% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 33 | -5.2% | -61% to +58% | 24% | - |  |
| `parlay_players3` | 15 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 33 | -72.0% | -100% to -27% | 6% | - |  |
| `parlay_top3` | 33 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 159 | -11.5% | -26% to +4% | 47% | - |  |
| `pts_over_control` | 163 | -2.3% | -16% to +10% | 58% | - |  |
| `pts_over_edge` | 133 | -28.5% | -48% to -8% | 31% | - |  |
| `pts_under_edge` | 159 | -3.9% | -18% to +11% | 53% | - |  |
| `puckline_dog_control` | 232 | -6.0% | -16% to +4% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 87 | -3.7% | -20% to +13% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 127 | -17.0% | -33% to -1% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 53 | -28.1% | -57% to +3% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 214 | -1.9% | -14% to +11% | 53% | - |  |
| `sog_over_control` | 218 | +0.1% | -13% to +14% | 50% | - |  |
| `sog_over_edge` | 143 | -9.9% | -26% to +6% | 45% | - |  |
| `sog_under_edge` | 175 | -3.3% | -18% to +10% | 52% | - |  |
| `totals_edge` | 158 | -9.5% | -24% to +6% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 103 | -6.8% | -26% to +13% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 121 | -33.7% | -52% to -15% | 28% | -2.87% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 233 | -16.6% | -31% to -2% | 36% | -3.00% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| COL @ TOR | Moneyline | COL moneyline | $10.00 | 1.63 | 56.1% | 60.8% |
| MIN @ MTL | Moneyline | MIN moneyline | $10.00 | 1.80 | 51.0% | 53.9% |
| WSH @ OTT | Moneyline | OTT moneyline | $10.00 | 1.96 | 54.6% | 49.8% |
| NJD @ WPG | Moneyline | NJD moneyline | $10.00 | 1.71 | 56.3% | 57.1% |
| FLA @ CBJ | Moneyline | FLA moneyline | $10.00 | 1.83 | 50.5% | 53.1% |
| DET @ STL | Moneyline | STL moneyline | $10.00 | 1.71 | 53.3% | 56.7% |
| NSH @ LAK | Moneyline | LAK moneyline | $10.00 | 1.78 | 57.9% | 54.7% |
| COL @ TOR | Total | Under 6.5 | $10.00 | 2.09 | 55.7% | 46.9% |
| MIN @ MTL | Total | Under 6.5 | $10.00 | 1.80 | 55.1% | 53.6% |
| WSH @ OTT | Total | Under 6.5 | $10.00 | 1.89 | 54.5% | 51.5% |
| NJD @ WPG | Total | Under 6.5 | $10.00 | 1.97 | 55.0% | 50.1% |
| FLA @ CBJ | Total | Over 6 | $10.00 | 1.95 | 52.4% | 49.4% |
| DET @ STL | Total | Over 5.5 | $10.00 | 1.81 | 58.2% | 53.6% |
| NSH @ LAK | Total | Over 6 | $10.00 | 1.87 | 52.3% | 52.0% |
| COL @ TOR | Puck line | TOR +1.5 | $10.00 | 1.62 | 63.7% | 60.5% |
| MIN @ MTL | Puck line | MTL +1.5 | $10.00 | 1.43 | 67.4% | 68.2% |
| WSH @ OTT | Puck line | WSH +1.5 | $10.00 | 1.40 | 67.3% | 70.9% |
| NJD @ WPG | Puck line | WPG +1.5 | $10.00 | 1.52 | 59.1% | 64.5% |
| FLA @ CBJ | Puck line | CBJ +1.5 | $10.00 | 1.43 | 68.3% | 69.7% |
| DET @ STL | Puck line | DET +1.5 | $10.00 | 1.52 | 67.7% | 65.6% |
| NSH @ LAK | Puck line | NSH +1.5 | $10.00 | 1.45 | 67.6% | 67.3% |

Total pretend stake $210.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
