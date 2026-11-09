# NHL model report: 2026-11-09 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-09 22:04 UTC · model `20261108-f8bf42` · probability source `online_platt` · bankroll $986.76 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-09T22:02:27+00:00 · API credits left: 353*

*Player props: 3 game(s) priced*

## Summary

- 3 games · **1 recommended bet(s)** · total stake $2.94 (0.30% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| NYR @ EDM (20:00 ET) | D. Garand / D. Levi (probable) | 55.2% | 60.3% | NYR 2.47 | +5.0% | +10.6% | $2.94 | **BET NYR** |
| PHI @ VAN (21:30 ET) | D. Vladar / L. Meriläinen (probable) | 50.4% | 44.5% | VAN 2.18 | +6.0% | - | - | no bet |
| NYI @ SJS (22:00 ET) | I. Sorokin / Y. Askarov (probable) | 42.3% | 43.6% | NYI 1.73 | +1.3% | - | - | no bet |

## Why

**NYR @ EDM (20:00 ET)**: Bet $2.94 (0.30% of bankroll) on NYR at 2.47. The model gives NYR 44.8% vs the market's no-vig 39.7% (edge +5.0%); after shrinking toward the market the working probability is 41.2%, worth +1.7% per $1 at that price.
  - Context: EDM have the clearly better goal differential this season (gap 0.57 goals/game). Goaltending edge to NYR (goals saved above expected).

**PHI @ VAN (21:30 ET)**: No bet: expected value after shrinkage toward the market is only +0.2% (< 1%)
  - Context: PHI have the clearly better goal differential this season (gap 0.54 goals/game). PHI is on the second night of a back-to-back. Goaltending edge to PHI (goals saved above expected).

**NYI @ SJS (22:00 ET)**: No bet: edge +1.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: NYI have the clearly better goal differential this season (gap 0.64 goals/game). NYI travelled 4,137 km across 3 time zone(s) since their last game. NYI are creating the better share of expected goals recently. Goaltending edge to NYI (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **277** (bet on 7, passed on 97%); model log loss 0.6521, Brier 0.2302 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (277 games with odds): model log loss 0.6521 vs market 0.6388 (market better)
- Bets: 7 · staked $18.12 · profit $-13.24 · **ROI -73.1%** · win rate 14.3% · avg price 2.53
- Bankroll $986.76 · max drawdown 1.3% · current drawdown 1.3%
- **CLV**: mean -2.38% per $1 (95% CI -4.14% to +0.05%), beat the close on 14% of 7 bets
- ⚠️ 7 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.4% | 40.0% | 35 |
| 46.6% | 42.9% | 35 |
| 50.1% | 47.1% | 34 |
| 52.9% | 60.0% | 35 |
| 55.6% | 50.0% | 34 |
| 58.8% | 65.7% | 35 |
| 62.2% | 70.6% | 34 |
| 68.5% | 80.0% | 35 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| NYR @ EDM (20:00 ET) | Total | Under 6.5 | 54.9% | 49.6% | +5.3 pts | +7.6% | 1.96 |
| NYI @ SJS (22:00 ET) | Puck line | NYI -1.5 | 39.3% | 34.4% | +4.8 pts | +10.0% | 2.80 |
| NYI @ SJS (22:00 ET) | Total | Under 6.5 | 54.4% | 51.8% | +2.5 pts | +1.7% | 1.87 |
| NYR @ EDM (20:00 ET) | Puck line | NYR +1.5 | 63.8% | 61.9% | +2.0 pts | +0.9% | 1.58 |
| PHI @ VAN (21:30 ET) | Puck line | VAN +1.5 | 67.1% | 66.2% | +0.8 pts | -0.7% | 1.48 |
| PHI @ VAN (21:30 ET) | Total | Under 6 | 47.8% | 47.6% | +0.2 pts | -2.6% | 2.03 |
| PHI @ VAN (21:30 ET) | Total | Over 6 | 52.2% | 52.4% | -0.2 pts | -3.5% | 1.84 |
| PHI @ VAN (21:30 ET) | Puck line | PHI -1.5 | 32.9% | 33.8% | -0.8 pts | -7.8% | 2.80 |
| NYR @ EDM (20:00 ET) | Puck line | EDM -1.5 | 36.2% | 38.1% | -2.0 pts | -7.4% | 2.56 |
| NYI @ SJS (22:00 ET) | Total | Over 6.5 | 45.6% | 48.2% | -2.5 pts | -6.9% | 2.04 |
| NYI @ SJS (22:00 ET) | Puck line | SJS +1.5 | 60.7% | 65.6% | -4.8 pts | -8.9% | 1.50 |
| NYR @ EDM (20:00 ET) | Total | Over 6.5 | 45.1% | 50.4% | -5.3 pts | -13.0% | 1.93 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| M. Joseph | NYR @ EDM (20:00 ET) | goal 0.5 | Over | 0.14 | 13.4% | 10.9% | +2.6 pts | +61.4% | 1 0 0 0 1 0 0 0 0 0 | 0.2 |
| N. Laba | NYR @ EDM (20:00 ET) | goal 0.5 | Over | 0.13 | 12.5% | 10.0% | +2.5 pts | +50.3% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| M. Robertson | NYR @ EDM (20:00 ET) | goal 0.5 | Over | 0.06 | 6.1% | 5.7% | +0.4 pts | +46.2% | 0 0 0 0 0 0 0 1 0 0 | - |
| L. Glendening | NYR @ EDM (20:00 ET) | goal 0.5 | Over | 0.11 | 10.1% | 8.0% | +2.1 pts | +41.5% | 0 0 0 1 0 0 0 0 0 0 | - |
| N. Acciari | PHI @ VAN (21:30 ET) | goal 0.5 | Over | 0.18 | 16.0% | 13.7% | +2.4 pts | +32.3% | 0 1 0 0 0 0 0 0 1 1 | 0.2 |
| K. Kapanen | NYR @ EDM (20:00 ET) | goal 0.5 | Over | 0.30 | 26.1% | 20.5% | +5.5 pts | +25.1% | 0 0 0 0 1 0 0 0 1 0 | 0.3 |
| K. Kapanen | NYR @ EDM (20:00 ET) | pts 0.5 | Over | 0.69 | 49.8% | 39.6% | +10.3 pts | +22.1% | 0 0 0 1 1 1 0 2 1 0 | 0.6 |
| S. Couturier | PHI @ VAN (21:30 ET) | goal 0.5 | Over | 0.19 | 17.3% | 14.7% | +2.6 pts | +20.8% | 0 0 0 0 0 1 0 0 0 0 | 0.1 |
| R. Shea | NYR @ EDM (20:00 ET) | goal 0.5 | Over | 0.09 | 8.6% | 8.7% | -0.1 pts | +20.4% | 0 0 0 0 0 0 0 0 0 0 | 0.1 |
| L. Draisaitl | NYR @ EDM (20:00 ET) | sog 3.5 | Over | 3.92 | 52.7% | 43.2% | +9.5 pts | +20.1% | 4 4 7 3 2 2 3 4 6 7 | 3.8 |
| D. O'Connor | PHI @ VAN (21:30 ET) | goal 0.5 | Over | 0.19 | 17.1% | 15.3% | +1.8 pts | +19.4% | 0 0 0 0 0 1 0 0 0 0 | 0.1 |
| J. DeBrusk | PHI @ VAN (21:30 ET) | sog 2.5 | Under | 2.07 | 34.1% | 44.8% | +10.7 pts | +18.7% | 0 1 2 0 3 2 2 3 1 2 | 2.2 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 264 | -0.6% | -12% to +11% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 219 | -5.8% | -19% to +6% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 104 | -5.2% | -22% to +13% | 59% | - |  |
| `ast_over_control` | 103 | +13.7% | -5% to +33% | 59% | - |  |
| `ast_over_edge` | 86 | -12.8% | -44% to +22% | 28% | - |  |
| `ast_under_edge` | 104 | -2.0% | -17% to +12% | 64% | - |  |
| `edge_1pct` | 21 | -29.1% | -70% to +17% | 33% | -1.30% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 277 | +2.3% | -8% to +11% | 61% | -1.93% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 276 | -3.8% | -12% to +5% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 264 | -7.2% | -18% to +5% | 48% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 277 | -18.7% | -30% to -7% | 40% | -2.61% | $10 on every game where the model sees any positive edge |
| `goal_control` | 104 | -3.9% | -30% to +22% | 34% | - |  |
| `goal_edge` | 68 | -19.2% | -72% to +43% | 12% | - |  |
| `home_ml_control` | 230 | +2.6% | -10% to +14% | 57% | -2.51% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 277 | +3.3% | -7% to +12% | 62% | -2.02% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 22 | -22.7% | -68% to +28% | 32% | -1.69% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 29 | -4.2% | -44% to +37% | 45% | -2.38% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 57 | -7.1% | -33% to +17% | 49% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 32 | -2.2% | -62% to +60% | 25% | - |  |
| `parlay_players3` | 14 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 32 | -71.1% | -100% to -25% | 6% | - |  |
| `parlay_top3` | 32 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 151 | -12.5% | -28% to +3% | 46% | - |  |
| `pts_over_control` | 155 | -0.6% | -14% to +13% | 59% | - |  |
| `pts_over_edge` | 125 | -27.8% | -47% to -8% | 31% | - |  |
| `pts_under_edge` | 151 | -7.0% | -21% to +8% | 51% | - |  |
| `puckline_dog_control` | 229 | -5.5% | -16% to +4% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 87 | -3.7% | -20% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 126 | -16.3% | -32% to -0% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 52 | -26.7% | -58% to +6% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 206 | -2.0% | -14% to +11% | 53% | - |  |
| `sog_over_control` | 210 | -3.0% | -17% to +10% | 49% | - |  |
| `sog_over_edge` | 135 | -10.7% | -28% to +7% | 45% | - |  |
| `sog_under_edge` | 167 | -2.1% | -17% to +13% | 53% | - |  |
| `totals_edge` | 157 | -10.2% | -25% to +6% | 46% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 101 | -8.8% | -28% to +10% | 47% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 119 | -32.6% | -52% to -12% | 29% | -2.87% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 230 | -16.5% | -31% to -2% | 37% | -3.00% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| NYR @ EDM | Moneyline | EDM moneyline | $10.00 | 1.62 | 55.2% | 60.3% |
| PHI @ VAN | Moneyline | VAN moneyline | $10.00 | 2.18 | 50.4% | 44.5% |
| NYI @ SJS | Moneyline | NYI moneyline | $10.00 | 1.73 | 57.7% | 56.4% |
| NYR @ EDM | Total | Under 6.5 | $10.00 | 1.96 | 54.9% | 49.6% |
| PHI @ VAN | Total | Over 6 | $10.00 | 1.84 | 52.2% | 52.4% |
| NYI @ SJS | Total | Under 6.5 | $10.00 | 1.87 | 54.4% | 51.8% |
| NYR @ EDM | Puck line | NYR +1.5 | $10.00 | 1.58 | 63.8% | 61.9% |
| PHI @ VAN | Puck line | VAN +1.5 | $10.00 | 1.48 | 67.1% | 66.2% |
| NYI @ SJS | Puck line | SJS +1.5 | $10.00 | 1.50 | 60.7% | 65.6% |

Total pretend stake $90.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
