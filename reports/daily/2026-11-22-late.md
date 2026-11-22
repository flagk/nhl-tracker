# NHL model report: 2026-11-22 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-22 14:34 UTC · model `20261121-c3c0b5` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-22T14:32:31+00:00 · API credits left: 169*

*Player props: 1 game(s) priced*

## Summary

- 5 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CAR @ NYR (12:30 ET) | P. Kochetkov / D. Garand (probable) | 42.9% | 40.4% | NYR 2.40 | +2.5% | - | - | no bet |
| BUF @ TBL (17:00 ET) | U. Luukkonen / A. Vasilevskiy (probable) | 67.1% | 65.5% | TBL 1.49 | +1.6% | - | - | no bet |
| EDM @ STL (19:00 ET) | D. Levi / J. Hofer (probable) | 55.2% | 49.8% | STL 1.95 | +5.3% | - | - | no bet |
| PIT @ DAL (19:00 ET) | A. Silovs / J. Oettinger (probable) | 64.5% | 67.5% | PIT 2.95 | +3.1% | - | - | no bet |
| CHI @ VAN (20:00 ET) | S. Knight / L. Meriläinen (probable) | 50.0% | 49.7% | VAN 1.95 | +0.4% | - | - | no bet |

## Why

**CAR @ NYR (12:30 ET)**: No bet: edge +2.5% below the 4% minimum (starting goalie unconfirmed)
  - Context: CAR have the clearly better goal differential this season (gap 1.05 goals/game). CAR is on the second night of a back-to-back. CAR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**BUF @ TBL (17:00 ET)**: No bet: edge +1.6% below the 4% minimum (starting goalie unconfirmed)
  - Context: TBL have the clearly better goal differential this season (gap 0.94 goals/game). BUF is on the second night of a back-to-back. TBL are creating the better share of expected goals recently. Goaltending edge to TBL (goals saved above expected).

**EDM @ STL (19:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.1% (< 1%)
  - Context: EDM have the clearly better goal differential this season (gap 0.60 goals/game). STL and EDM are on the second night of a back-to-back. STL are creating the better share of expected goals recently. Goaltending edge to EDM (goals saved above expected).

**PIT @ DAL (19:00 ET)**: No bet: edge +3.1% below the 4% minimum (starting goalie unconfirmed); model win probability 35.5% below the 40% confidence floor
  - Context: DAL have the clearly better goal differential this season (gap 0.63 goals/game). PIT is on the second night of a back-to-back. DAL are creating the better share of expected goals recently. Goaltending edge to DAL (goals saved above expected).

**CHI @ VAN (20:00 ET)**: No bet: edge +0.4% below the 4% minimum (starting goalie unconfirmed)
  - Context: VAN and CHI are on the second night of a back-to-back. Goaltending edge to CHI (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **367** (bet on 11, passed on 97%); model log loss 0.6664, Brier 0.2370 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (367 games with odds): model log loss 0.6664 vs market 0.6535 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.48% to -0.52%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.6% | 41.3% | 46 |
| 46.7% | 39.1% | 46 |
| 50.0% | 39.1% | 46 |
| 52.7% | 56.5% | 46 |
| 55.2% | 46.7% | 45 |
| 58.1% | 56.5% | 46 |
| 61.4% | 71.7% | 46 |
| 68.4% | 71.7% | 46 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| CAR @ NYR (12:30 ET) | Puck line | CAR -1.5 | 42.3% | 37.1% | +5.2 pts | +9.6% | 2.59 |
| CHI @ VAN (20:00 ET) | Total | Under 6.5 | 55.1% | 51.0% | +4.1 pts | +5.3% | 1.91 |
| BUF @ TBL (17:00 ET) | Total | Under 6.5 | 54.5% | 51.1% | +3.4 pts | +3.0% | 1.89 |
| BUF @ TBL (17:00 ET) | Puck line | TBL -1.5 | 46.4% | 43.1% | +3.3 pts | +4.4% | 2.25 |
| PIT @ DAL (19:00 ET) | Puck line | PIT +1.5 | 57.3% | 54.8% | +2.5 pts | +1.4% | 1.77 |
| EDM @ STL (19:00 ET) | Total | Over 6 | 51.5% | 50.1% | +1.4 pts | -1.0% | 1.92 |
| EDM @ STL (19:00 ET) | Puck line | STL +1.5 | 71.7% | 70.9% | +0.8 pts | +0.4% | 1.40 |
| CAR @ NYR (12:30 ET) | Total | Over 6 | 52.9% | 52.3% | +0.6 pts | -1.8% | 1.85 |
| CHI @ VAN (20:00 ET) | Puck line | CHI -1.5 | 29.9% | 29.4% | +0.6 pts | -2.8% | 3.25 |
| PIT @ DAL (19:00 ET) | Total | Over 6 | 52.6% | 52.3% | +0.3 pts | -2.9% | 1.84 |
| PIT @ DAL (19:00 ET) | Total | Under 6 | 47.4% | 47.7% | -0.3 pts | -3.2% | 2.03 |
| CHI @ VAN (20:00 ET) | Puck line | VAN +1.5 | 70.1% | 70.6% | -0.6 pts | -1.9% | 1.40 |
| CAR @ NYR (12:30 ET) | Total | Under 6 | 47.1% | 47.7% | -0.6 pts | -4.3% | 2.02 |
| EDM @ STL (19:00 ET) | Puck line | EDM -1.5 | 28.3% | 29.1% | -0.8 pts | -5.2% | 3.35 |
| EDM @ STL (19:00 ET) | Total | Under 6 | 48.5% | 49.9% | -1.4 pts | -5.1% | 1.94 |
| PIT @ DAL (19:00 ET) | Puck line | DAL -1.5 | 42.7% | 45.2% | -2.5 pts | -7.7% | 2.16 |
| BUF @ TBL (17:00 ET) | Puck line | BUF +1.5 | 53.6% | 56.9% | -3.3 pts | -8.3% | 1.71 |
| BUF @ TBL (17:00 ET) | Total | Over 6.5 | 45.5% | 48.9% | -3.4 pts | -9.0% | 2.00 |
| CHI @ VAN (20:00 ET) | Total | Over 6.5 | 44.9% | 49.0% | -4.1 pts | -11.2% | 1.98 |
| CAR @ NYR (12:30 ET) | Puck line | NYR +1.5 | 57.7% | 62.9% | -5.2 pts | -10.0% | 1.56 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| S. Gostisbehere | CAR @ NYR (12:30 ET) | pts 0.5 | Under | 0.45 | 36.3% | 50.9% | +14.6 pts | +24.2% | 0 0 2 1 0 2 0 2 0 2 | 0.8 |
| G. Perreault | CAR @ NYR (12:30 ET) | pts 0.5 | Under | 0.48 | 38.2% | 47.9% | +9.7 pts | +13.2% | 0 0 0 0 1 1 0 0 2 0 | 0.5 |
| S. Jarvis | CAR @ NYR (12:30 ET) | sog 2.5 | Under | 2.00 | 32.3% | 40.8% | +8.5 pts | +13.0% | 0 3 3 1 0 2 1 2 1 2 | 1.8 |
| S. Jarvis | CAR @ NYR (12:30 ET) | pts 0.5 | Under | 0.61 | 45.5% | 54.5% | +9.0 pts | +12.8% | 0 2 2 0 0 1 0 1 0 0 | 0.8 |
| A. Lafrenière | CAR @ NYR (12:30 ET) | sog 2.5 | Under | 1.95 | 31.1% | 38.5% | +7.4 pts | +9.5% | 7 2 4 2 1 2 0 2 2 3 | 2.3 |
| T. Hall | CAR @ NYR (12:30 ET) | pts 0.5 | Under | 0.41 | 33.9% | 41.6% | +7.7 pts | +9.1% | 2 1 0 0 0 1 0 1 0 2 | 0.7 |
| M. Zibanejad | CAR @ NYR (12:30 ET) | sog 2.5 | Under | 1.92 | 30.3% | 37.6% | +7.3 pts | +6.6% | 1 1 5 1 1 1 1 2 3 2 | 1.8 |
| W. Cuylle | CAR @ NYR (12:30 ET) | sog 1.5 | Under | 1.74 | 50.5% | 55.9% | +5.4 pts | +6.5% | 1 4 0 3 0 0 1 1 2 4 | 2.0 |
| J. Blake | CAR @ NYR (12:30 ET) | sog 2.5 | Under | 2.19 | 37.1% | 42.9% | +5.8 pts | +6.4% | 4 1 1 1 2 2 1 4 3 2 | 2.5 |
| G. Perreault | CAR @ NYR (12:30 ET) | sog 1.5 | Under | 1.62 | 47.0% | 53.4% | +6.4 pts | +5.9% | 3 1 2 1 4 2 0 2 2 0 | 2.0 |
| A. Svechnikov | CAR @ NYR (12:30 ET) | sog 2.5 | Under | 2.53 | 45.1% | 50.1% | +5.1 pts | +4.9% | 2 1 3 0 7 5 3 4 2 2 | 3.0 |
| S. Aho | CAR @ NYR (12:30 ET) | pts 0.5 | Over | 1.10 | 66.6% | 60.4% | +6.2 pts | +4.5% | 1 0 2 0 2 3 2 1 0 3 | 1.5 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 352 | -1.4% | -11% to +9% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 307 | -4.6% | -15% to +6% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -8% to +23% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -41% to +7% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +1% | 60% | - |  |
| `edge_1pct` | 43 | -30.2% | -60% to +2% | 33% | -1.49% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 367 | -0.3% | -9% to +8% | 59% | -2.01% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 366 | -5.2% | -13% to +3% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 352 | -4.7% | -15% to +6% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 367 | -15.1% | -26% to -4% | 40% | -2.73% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -20% to +22% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -56% to +37% | 14% | - |  |
| `home_ml_control` | 320 | -5.2% | -16% to +5% | 52% | -2.55% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 367 | +1.8% | -7% to +10% | 61% | -2.05% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -74% to +7% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 44 | -25.4% | -54% to +6% | 36% | -2.55% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 74 | -5.4% | -26% to +16% | 50% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 44 | +15.4% | -42% to +74% | 27% | - |  |
| `parlay_players3` | 25 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 44 | -60.2% | -93% to -16% | 9% | - |  |
| `parlay_top3` | 44 | -93.0% | -100% to -79% | 2% | - |  |
| `pts_edge` | 246 | -13.6% | -25% to -1% | 46% | - |  |
| `pts_over_control` | 251 | +0.8% | -10% to +12% | 59% | - |  |
| `pts_over_edge` | 210 | -25.6% | -40% to -10% | 33% | - |  |
| `pts_under_edge` | 244 | -6.3% | -18% to +5% | 52% | - |  |
| `puckline_dog_control` | 319 | -5.0% | -13% to +3% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 107 | -3.1% | -18% to +11% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 156 | -11.8% | -28% to +5% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 78 | -21.5% | -48% to +6% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 301 | -5.0% | -16% to +6% | 51% | - |  |
| `sog_over_control` | 305 | +2.7% | -8% to +14% | 51% | - |  |
| `sog_over_edge` | 219 | -8.9% | -22% to +5% | 46% | - |  |
| `sog_under_edge` | 260 | -5.4% | -17% to +6% | 51% | - |  |
| `totals_edge` | 209 | -6.5% | -20% to +7% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 143 | -6.7% | -23% to +9% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 167 | -25.9% | -43% to -8% | 31% | -2.98% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 320 | -10.5% | -23% to +2% | 38% | -2.93% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CAR @ NYR | Moneyline | CAR moneyline | $10.00 | 1.63 | 57.1% | 59.6% |
| BUF @ TBL | Moneyline | TBL moneyline | $10.00 | 1.49 | 67.1% | 65.5% |
| EDM @ STL | Moneyline | STL moneyline | $10.00 | 1.95 | 55.2% | 49.8% |
| PIT @ DAL | Moneyline | DAL moneyline | $10.00 | 1.45 | 64.5% | 67.5% |
| CHI @ VAN | Moneyline | VAN moneyline | $10.00 | 1.95 | 50.0% | 49.7% |
| CAR @ NYR | Total | Over 6 | $10.00 | 1.85 | 52.9% | 52.3% |
| BUF @ TBL | Total | Under 6.5 | $10.00 | 1.89 | 54.5% | 51.1% |
| EDM @ STL | Total | Over 6 | $10.00 | 1.92 | 51.5% | 50.1% |
| PIT @ DAL | Total | Over 6 | $10.00 | 1.84 | 52.6% | 52.3% |
| CHI @ VAN | Total | Under 6.5 | $10.00 | 1.91 | 55.1% | 51.0% |
| CAR @ NYR | Puck line | NYR +1.5 | $10.00 | 1.56 | 57.7% | 62.9% |
| BUF @ TBL | Puck line | BUF +1.5 | $10.00 | 1.71 | 53.6% | 56.9% |
| EDM @ STL | Puck line | STL +1.5 | $10.00 | 1.40 | 71.7% | 70.9% |
| PIT @ DAL | Puck line | PIT +1.5 | $10.00 | 1.77 | 57.3% | 54.8% |
| CHI @ VAN | Puck line | VAN +1.5 | $10.00 | 1.40 | 70.1% | 70.6% |

Total pretend stake $150.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
