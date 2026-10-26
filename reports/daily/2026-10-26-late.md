# NHL model report: 2026-10-26 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-26 20:04 UTC · model `20261025-249957` · probability source `online_platt` · bankroll $997.66 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-26T20:03:43+00:00 · API credits left: 93*

*Player props: 0 game(s) priced; **unavailable**: only 93 odds credits left: player prices skipped to protect moneylines*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| LAK @ NYR (19:00 ET) | A. Forsberg / D. Garand (probable) | 56.1% | 55.1% | NYR 1.77 | +1.0% | - | - | no bet |
| TOR @ CGY (20:30 ET) | S. Bobrovsky / D. Wolf (probable) | 51.6% | 44.3% | CGY 2.19 | +7.3% | - | - | no bet |
| DET @ EDM (21:30 ET) | J. Gibson / C. Ingram (probable) | 54.0% | 61.5% | DET 2.50 | +7.5% | - | - | no bet |

## Why

**LAK @ NYR (19:00 ET)**: No bet: early-season sample size: a team has only played 9 games (< 10)
  - Context: NYR have the clearly better goal differential this season (gap 0.71 goals/game). Early in the season: team ratings lean on last year's results and are less reliable.

**TOR @ CGY (20:30 ET)**: No bet: expected value after shrinkage toward the market is only +0.2% (< 1%)
  - Context: TOR have the clearly better goal differential this season (gap 0.98 goals/game). Goaltending edge to TOR (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**DET @ EDM (21:30 ET)**: No bet: expected value after shrinkage toward the market is only -0.2% (< 1%)
  - Context: EDM have the clearly better goal differential this season (gap 0.66 goals/game). DET are creating the better share of expected goals recently. Goaltending edge to EDM (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **179** (bet on 1, passed on 99%); model log loss 0.6590, Brier 0.2333 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (179 games with odds): model log loss 0.6590 vs market 0.6452 (market better)
- Bets: 1 · staked $2.34 · profit $-2.34 · **ROI -100.0%** · win rate 0.0% · avg price 2.71
- Bankroll $997.66 · max drawdown 0.2% · current drawdown 0.2%
- ⚠️ 1 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.1% | 42.3% | 26 |
| 46.8% | 52.0% | 25 |
| 50.9% | 50.0% | 26 |
| 53.9% | 64.0% | 25 |
| 57.6% | 65.4% | 26 |
| 60.9% | 72.0% | 25 |
| 66.8% | 69.2% | 26 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| DET @ EDM (21:30 ET) | Total | Under 6.5 | 55.6% | 45.7% | +9.8 pts | +17.8% | 2.12 |
| TOR @ CGY (20:30 ET) | Total | Under 6.5 | 54.6% | 47.7% | +6.9 pts | +11.9% | 2.05 |
| DET @ EDM (21:30 ET) | Puck line | DET +1.5 | 64.6% | 59.8% | +4.8 pts | +6.0% | 1.64 |
| LAK @ NYR (19:00 ET) | Puck line | NYR -1.5 | 36.9% | 32.6% | +4.2 pts | +7.7% | 2.92 |
| LAK @ NYR (19:00 ET) | Total | Over 5.5 | 57.8% | 53.6% | +4.2 pts | +5.1% | 1.82 |
| TOR @ CGY (20:30 ET) | Puck line | TOR -1.5 | 35.3% | 34.8% | +0.5 pts | -1.0% | 2.80 |
| TOR @ CGY (20:30 ET) | Puck line | CGY +1.5 | 64.7% | 65.2% | -0.5 pts | -2.4% | 1.51 |
| LAK @ NYR (19:00 ET) | Total | Under 5.5 | 42.2% | 46.4% | -4.2 pts | -11.3% | 2.10 |
| LAK @ NYR (19:00 ET) | Puck line | LAK +1.5 | 63.1% | 67.4% | -4.2 pts | -8.5% | 1.45 |
| DET @ EDM (21:30 ET) | Puck line | EDM -1.5 | 35.4% | 40.2% | -4.8 pts | -15.1% | 2.40 |
| TOR @ CGY (20:30 ET) | Total | Over 6.5 | 45.4% | 52.3% | -6.9 pts | -16.0% | 1.85 |
| DET @ EDM (21:30 ET) | Total | Over 6.5 | 44.4% | 54.3% | -9.8 pts | -20.0% | 1.80 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 171 | -5.8% | -20% to +9% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 126 | +1.2% | -16% to +18% | 52% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -35% to +25% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +35% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +27% | 65% | - |  |
| `edge_1pct` | 3 | -31.3% | - | 33% | -2.99% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 179 | -0.6% | -13% to +11% | 59% | -2.09% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 178 | -7.6% | -19% to +4% | 60% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 171 | -16.3% | -30% to -2% | 44% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 179 | -19.2% | -34% to -5% | 39% | -2.55% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -51% to +33% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +87% | 10% | - |  |
| `home_ml_control` | 132 | +9.9% | -7% to +26% | 60% | -2.67% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 179 | +3.0% | -9% to +15% | 62% | -2.10% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 16 | -10.4% | - | 38% | -1.55% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 3 | -36.3% | - | 33% | -4.24% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 28 | -38.8% | -72% to -5% | 32% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 18 | -0.5% | - | 28% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 18 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 18 | -100.0% | - | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +5% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -24% to +10% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -65% to -6% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -33% to +7% | 47% | - |  |
| `puckline_dog_control` | 131 | -6.6% | -19% to +7% | 60% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 61 | -4.4% | -25% to +15% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 92 | -13.2% | -32% to +6% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 24 | -12.1% | -60% to +42% | 33% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 142 | +1.7% | -13% to +17% | 56% | - |  |
| `sog_over_control` | 146 | -4.5% | -22% to +12% | 47% | - |  |
| `sog_over_edge` | 72 | -12.6% | -36% to +10% | 46% | - |  |
| `sog_under_edge` | 103 | +3.9% | -14% to +22% | 56% | - |  |
| `totals_edge` | 102 | -22.8% | -41% to -4% | 40% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 59 | -17.7% | -41% to +8% | 42% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 69 | -26.3% | -52% to +1% | 30% | -2.72% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 132 | -15.5% | -35% to +4% | 36% | -2.81% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| LAK @ NYR | Moneyline | NYR moneyline | $10.00 | 1.77 | 56.1% | 55.1% |
| TOR @ CGY | Moneyline | CGY moneyline | $10.00 | 2.19 | 51.6% | 44.3% |
| DET @ EDM | Moneyline | EDM moneyline | $10.00 | 1.59 | 54.0% | 61.5% |
| LAK @ NYR | Total | Over 5.5 | $10.00 | 1.82 | 57.8% | 53.6% |
| TOR @ CGY | Total | Under 6.5 | $10.00 | 2.05 | 54.6% | 47.7% |
| DET @ EDM | Total | Under 6.5 | $10.00 | 2.12 | 55.6% | 45.7% |
| LAK @ NYR | Puck line | LAK +1.5 | $10.00 | 1.45 | 63.1% | 67.4% |
| TOR @ CGY | Puck line | CGY +1.5 | $10.00 | 1.51 | 64.7% | 65.2% |
| DET @ EDM | Puck line | DET +1.5 | $10.00 | 1.64 | 64.6% | 59.8% |

Total pretend stake $90.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
