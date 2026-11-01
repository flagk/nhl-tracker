# NHL model report: 2026-11-01 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-01 15:03 UTC · model `20261031-0854bc` · probability source `online_platt` · bankroll $990.96 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-01T15:02:13+00:00 · API credits left: 495*

*Player props: 2 game(s) priced*

## Summary

- 5 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| WPG @ CAR (13:00 ET) | S. Skinner / P. Kochetkov (probable) | 72.0% | 71.9% | CAR 1.36 | +0.1% | - | - | no bet |
| MIN @ WSH (15:30 ET) | J. Wallstedt / L. Thompson (probable) | 50.9% | 50.0% | WSH 1.94 | +0.9% | - | - | no bet |
| UTA @ PHI (16:00 ET) | K. Vejmelka / D. Vladar (probable) | 42.5% | 40.4% | PHI 2.39 | +2.1% | - | - | no bet |
| EDM @ NYR (18:00 ET) | T. Jarry / D. Garand (probable) | 59.3% | 53.7% | NYR 1.83 | +5.6% | - | - | no bet |
| OTT @ ANA (20:00 ET) | L. Ullmark / L. Dostal (probable) | 53.8% | 48.0% | ANA 2.02 | +5.8% | - | - | no bet |

## Why

**WPG @ CAR (13:00 ET)**: No bet: edge +0.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: CAR have the clearly better goal differential this season (gap 1.41 goals/game). CAR are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**MIN @ WSH (15:30 ET)**: No bet: edge +0.9% below the 4% minimum (starting goalie unconfirmed)
  - Context: MIN are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**UTA @ PHI (16:00 ET)**: No bet: edge +2.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: UTA have the clearly better goal differential this season (gap 0.96 goals/game). UTA are creating the better share of expected goals recently. Goaltending edge to PHI (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**EDM @ NYR (18:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.9% (< 1%)
  - Context: EDM have the clearly better goal differential this season (gap 0.58 goals/game). EDM is on the second night of a back-to-back. Early in the season: team ratings lean on last year's results and are less reliable.

**OTT @ ANA (20:00 ET)**: No bet: expected value after shrinkage toward the market is only -0.1% (< 1%)
  - Context: OTT have the clearly better goal differential this season (gap 0.51 goals/game). OTT is on the second night of a back-to-back. Goaltending edge to OTT (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **221** (bet on 5, passed on 98%); model log loss 0.6567, Brier 0.2322 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (221 games with odds): model log loss 0.6567 vs market 0.6439 (market better)
- Bets: 5 · staked $13.92 · profit $-9.04 · **ROI -65.0%** · win rate 20.0% · avg price 2.41
- Bankroll $990.96 · max drawdown 0.9% · current drawdown 0.9%
- **CLV**: mean -1.94% per $1 (95% CI -4.33% to +1.39%), beat the close on 20% of 5 bets
- ⚠️ 5 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.1% | 42.9% | 28 |
| 46.6% | 42.9% | 28 |
| 50.3% | 48.1% | 27 |
| 52.9% | 64.3% | 28 |
| 55.6% | 44.4% | 27 |
| 58.9% | 71.4% | 28 |
| 62.0% | 74.1% | 27 |
| 67.5% | 71.4% | 28 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| OTT @ ANA (20:00 ET) | Total | Under 6.5 | 55.0% | 45.0% | +10.0 pts | +18.7% | 2.16 |
| UTA @ PHI (16:00 ET) | Puck line | UTA -1.5 | 44.6% | 36.8% | +7.7 pts | +18.1% | 2.65 |
| EDM @ NYR (18:00 ET) | Total | Under 6.5 | 54.6% | 49.3% | +5.3 pts | +6.4% | 1.95 |
| WPG @ CAR (13:00 ET) | Puck line | CAR -1.5 | 54.8% | 49.7% | +5.2 pts | +6.9% | 1.95 |
| UTA @ PHI (16:00 ET) | Total | Over 6 | 53.1% | 50.0% | +3.1 pts | +2.2% | 1.93 |
| WPG @ CAR (13:00 ET) | Total | Over 6 | 53.4% | 51.1% | +2.3 pts | +0.8% | 1.89 |
| OTT @ ANA (20:00 ET) | Puck line | ANA +1.5 | 70.6% | 68.6% | +2.0 pts | +1.6% | 1.44 |
| EDM @ NYR (18:00 ET) | Puck line | EDM +1.5 | 68.1% | 67.7% | +0.4 pts | -1.2% | 1.45 |
| MIN @ WSH (15:30 ET) | Puck line | MIN +1.5 | 71.1% | 70.7% | +0.4 pts | -1.9% | 1.38 |
| MIN @ WSH (15:30 ET) | Total | Over 6 | 52.0% | 51.8% | +0.1 pts | -2.5% | 1.87 |
| MIN @ WSH (15:30 ET) | Total | Under 6 | 48.0% | 48.2% | -0.1 pts | -3.5% | 2.00 |
| MIN @ WSH (15:30 ET) | Puck line | WSH -1.5 | 28.9% | 29.3% | -0.4 pts | -4.7% | 3.30 |
| EDM @ NYR (18:00 ET) | Puck line | NYR -1.5 | 31.9% | 32.3% | -0.4 pts | -2.8% | 3.05 |
| OTT @ ANA (20:00 ET) | Puck line | OTT -1.5 | 29.4% | 31.4% | -2.0 pts | -10.3% | 3.05 |
| WPG @ CAR (13:00 ET) | Total | Under 6 | 46.6% | 48.9% | -2.3 pts | -6.0% | 2.00 |
| UTA @ PHI (16:00 ET) | Total | Under 6 | 46.9% | 50.0% | -3.1 pts | -8.3% | 1.93 |
| WPG @ CAR (13:00 ET) | Puck line | WPG +1.5 | 45.2% | 50.3% | -5.2 pts | -11.9% | 1.95 |
| EDM @ NYR (18:00 ET) | Total | Over 6.5 | 45.4% | 50.7% | -5.3 pts | -12.8% | 1.92 |
| UTA @ PHI (16:00 ET) | Puck line | PHI +1.5 | 55.4% | 63.2% | -7.7 pts | -13.5% | 1.56 |
| OTT @ ANA (20:00 ET) | Total | Over 6.5 | 45.0% | 55.0% | -10.0 pts | -20.3% | 1.77 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| O. Maatta | MIN @ WSH (15:30 ET) | goal 0.5 | Over | 0.08 | 7.5% | 6.0% | +1.6 pts | +73.5% | 0 0 0 0 0 0 1 0 0 0 | 0.1 |
| D. Samberg | WPG @ CAR (13:00 ET) | goal 0.5 | Over | 0.08 | 7.6% | 6.4% | +1.2 pts | +43.8% | 0 0 0 0 0 0 0 0 1 0 | 0.1 |
| A. Lowry | WPG @ CAR (13:00 ET) | goal 0.5 | Over | 0.15 | 14.3% | 11.7% | +2.6 pts | +42.9% | 0 0 0 0 0 0 0 1 0 0 | 0.1 |
| M. Ferraro | WPG @ CAR (13:00 ET) | goal 0.5 | Over | 0.06 | 6.1% | 6.0% | +0.0 pts | +39.8% | 0 0 0 0 0 0 0 0 0 0 | 0.1 |
| S. Gostisbehere | WPG @ CAR (13:00 ET) | pts 0.5 | Under | 0.37 | 31.2% | 53.8% | +22.6 pts | +38.4% | 0 1 0 2 0 0 2 0 0 0 | 0.6 |
| B. Jenner | MIN @ WSH (15:30 ET) | goal 0.5 | Over | 0.26 | 22.9% | 16.9% | +5.9 pts | +37.3% | 0 1 1 1 0 1 0 0 0 1 | 0.4 |
| S. Gostisbehere | WPG @ CAR (13:00 ET) | ast 0.5 | Under | 0.28 | 24.2% | 46.3% | +22.1 pts | +34.1% | 0 0 0 2 0 0 2 0 0 0 | 0.5 |
| N. Pionk | WPG @ CAR (13:00 ET) | goal 0.5 | Over | 0.14 | 12.7% | 10.7% | +2.0 pts | +26.8% | 0 0 1 0 0 0 0 0 2 0 | 0.2 |
| M. Barron | WPG @ CAR (13:00 ET) | goal 0.5 | Over | 0.13 | 12.5% | 11.1% | +1.4 pts | +25.4% | 0 0 0 0 0 0 0 0 0 1 | 0.1 |
| D. DeMelo | WPG @ CAR (13:00 ET) | goal 0.5 | Over | 0.06 | 5.4% | 6.0% | -0.6 pts | +25.1% | 0 0 0 0 0 0 0 0 0 0 | 0.1 |
| A. Tuch | MIN @ WSH (15:30 ET) | goal 0.5 | Over | 0.48 | 37.7% | 29.9% | +7.8 pts | +24.4% | 1 0 0 0 2 0 0 2 2 1 | 0.8 |
| T. Liljegren | MIN @ WSH (15:30 ET) | goal 0.5 | Over | 0.06 | 5.6% | 6.2% | -0.6 pts | +23.8% | 0 1 0 0 0 0 0 0 0 0 | 0.1 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 212 | -2.8% | -16% to +11% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 167 | -3.4% | -18% to +11% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -36% to +22% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -26% to +34% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +27% | 65% | - |  |
| `edge_1pct` | 13 | -33.9% | - | 31% | -0.81% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 221 | +2.3% | -9% to +14% | 61% | -1.97% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 220 | -7.2% | -17% to +3% | 60% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 212 | -10.7% | -24% to +3% | 47% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 221 | -20.1% | -33% to -6% | 39% | -2.40% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -47% to +32% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +91% | 10% | - |  |
| `home_ml_control` | 174 | +4.3% | -11% to +18% | 57% | -2.55% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 221 | +4.7% | -6% to +15% | 63% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 20 | -15.0% | -64% to +38% | 35% | -1.51% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 16 | -5.4% | - | 44% | -2.05% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 42 | -23.8% | -51% to +3% | 40% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 24 | +30.4% | -41% to +111% | 33% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 24 | -61.5% | -100% to -0% | 8% | - |  |
| `parlay_top3` | 24 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +5% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -25% to +11% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -64% to -4% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -33% to +6% | 47% | - |  |
| `puckline_dog_control` | 173 | -6.4% | -18% to +4% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 72 | -4.6% | -23% to +13% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 107 | -14.5% | -32% to +3% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 36 | -27.8% | -65% to +12% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 142 | +1.7% | -14% to +16% | 56% | - |  |
| `sog_over_control` | 146 | -4.5% | -21% to +12% | 47% | - |  |
| `sog_over_edge` | 72 | -12.6% | -35% to +9% | 46% | - |  |
| `sog_under_edge` | 103 | +3.9% | -14% to +22% | 56% | - |  |
| `totals_edge` | 131 | -17.6% | -33% to -1% | 43% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 77 | -15.8% | -38% to +5% | 43% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 92 | -27.7% | -50% to -3% | 30% | -2.44% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 174 | -17.7% | -34% to -0% | 36% | -2.72% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| WPG @ CAR | Moneyline | CAR moneyline | $10.00 | 1.36 | 72.0% | 71.9% |
| MIN @ WSH | Moneyline | WSH moneyline | $10.00 | 1.94 | 50.9% | 50.0% |
| UTA @ PHI | Moneyline | UTA moneyline | $10.00 | 1.65 | 57.5% | 59.6% |
| EDM @ NYR | Moneyline | NYR moneyline | $10.00 | 1.83 | 59.3% | 53.7% |
| OTT @ ANA | Moneyline | ANA moneyline | $10.00 | 2.02 | 53.8% | 48.0% |
| WPG @ CAR | Total | Over 6 | $10.00 | 1.89 | 53.4% | 51.1% |
| MIN @ WSH | Total | Over 6 | $10.00 | 1.87 | 52.0% | 51.8% |
| UTA @ PHI | Total | Over 6 | $10.00 | 1.93 | 53.1% | 50.0% |
| EDM @ NYR | Total | Under 6.5 | $10.00 | 1.95 | 54.6% | 49.3% |
| OTT @ ANA | Total | Under 6.5 | $10.00 | 2.16 | 55.0% | 45.0% |
| WPG @ CAR | Puck line | CAR -1.5 | $10.00 | 1.95 | 54.8% | 49.7% |
| MIN @ WSH | Puck line | MIN +1.5 | $10.00 | 1.38 | 71.1% | 70.7% |
| UTA @ PHI | Puck line | PHI +1.5 | $10.00 | 1.56 | 55.4% | 63.2% |
| EDM @ NYR | Puck line | EDM +1.5 | $10.00 | 1.45 | 68.1% | 67.7% |
| OTT @ ANA | Puck line | ANA +1.5 | $10.00 | 1.44 | 70.6% | 68.6% |

Total pretend stake $150.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
