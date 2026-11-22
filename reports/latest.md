# NHL model report: 2026-11-22 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-22 13:03 UTC · model `20261121-c3c0b5` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-22T13:02:20+00:00 · API credits left: 172*

## Summary

- 5 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CAR @ NYR (12:30 ET) | P. Kochetkov / D. Garand (probable) | 42.9% | 40.7% | NYR 2.38 | +2.1% | - | - | no bet |
| BUF @ TBL (17:00 ET) | U. Luukkonen / A. Vasilevskiy (probable) | 67.1% | 65.3% | TBL 1.50 | +1.8% | - | - | no bet |
| EDM @ STL (19:00 ET) | D. Levi / J. Hofer (probable) | 55.2% | 49.8% | STL 1.95 | +5.4% | - | - | no bet |
| PIT @ DAL (19:00 ET) | A. Silovs / J. Oettinger (probable) | 64.5% | 67.1% | PIT 2.95 | +2.6% | - | - | no bet |
| CHI @ VAN (20:00 ET) | S. Knight / L. Meriläinen (probable) | 50.0% | 49.5% | VAN 1.96 | +0.5% | - | - | no bet |

## Why

**CAR @ NYR (12:30 ET)**: No bet: edge +2.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: CAR have the clearly better goal differential this season (gap 1.05 goals/game). CAR is on the second night of a back-to-back. CAR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**BUF @ TBL (17:00 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: TBL have the clearly better goal differential this season (gap 0.94 goals/game). BUF is on the second night of a back-to-back. TBL are creating the better share of expected goals recently. Goaltending edge to TBL (goals saved above expected).

**EDM @ STL (19:00 ET)**: No bet: expected value after shrinkage toward the market is only -0.1% (< 1%)
  - Context: EDM have the clearly better goal differential this season (gap 0.60 goals/game). STL and EDM are on the second night of a back-to-back. STL are creating the better share of expected goals recently. Goaltending edge to EDM (goals saved above expected).

**PIT @ DAL (19:00 ET)**: No bet: edge +2.6% below the 4% minimum (starting goalie unconfirmed); model win probability 35.5% below the 40% confidence floor
  - Context: DAL have the clearly better goal differential this season (gap 0.63 goals/game). PIT is on the second night of a back-to-back. DAL are creating the better share of expected goals recently. Goaltending edge to DAL (goals saved above expected).

**CHI @ VAN (20:00 ET)**: No bet: edge +0.5% below the 4% minimum (starting goalie unconfirmed)
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 352 | -1.4% | -11% to +9% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 307 | -4.6% | -15% to +6% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -21% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +24% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -42% to +8% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +2% | 60% | - |  |
| `edge_1pct` | 43 | -30.2% | -60% to +1% | 33% | -1.49% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 367 | -0.3% | -9% to +8% | 59% | -2.01% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 366 | -5.2% | -13% to +2% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 352 | -4.7% | -15% to +5% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 367 | -15.1% | -26% to -4% | 40% | -2.73% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -19% to +22% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -55% to +37% | 14% | - |  |
| `home_ml_control` | 320 | -5.2% | -16% to +6% | 52% | -2.55% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 367 | +1.8% | -6% to +10% | 61% | -2.05% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -74% to +7% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 44 | -25.4% | -53% to +6% | 36% | -2.55% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 74 | -5.4% | -26% to +17% | 50% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 44 | +15.4% | -42% to +74% | 27% | - |  |
| `parlay_players3` | 25 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 44 | -60.2% | -93% to -16% | 9% | - |  |
| `parlay_top3` | 44 | -93.0% | -100% to -79% | 2% | - |  |
| `pts_edge` | 246 | -13.6% | -25% to -2% | 46% | - |  |
| `pts_over_control` | 251 | +0.8% | -10% to +12% | 59% | - |  |
| `pts_over_edge` | 210 | -25.6% | -40% to -10% | 33% | - |  |
| `pts_under_edge` | 244 | -6.3% | -18% to +5% | 52% | - |  |
| `puckline_dog_control` | 319 | -5.0% | -14% to +3% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 107 | -3.1% | -18% to +11% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 156 | -11.8% | -27% to +4% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 78 | -21.5% | -49% to +7% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 301 | -5.0% | -16% to +5% | 51% | - |  |
| `sog_over_control` | 305 | +2.7% | -9% to +14% | 51% | - |  |
| `sog_over_edge` | 219 | -8.9% | -22% to +5% | 46% | - |  |
| `sog_under_edge` | 260 | -5.4% | -17% to +6% | 51% | - |  |
| `totals_edge` | 209 | -6.5% | -20% to +6% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 143 | -6.7% | -23% to +11% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 167 | -25.9% | -43% to -8% | 31% | -2.98% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 320 | -10.5% | -23% to +3% | 38% | -2.93% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CAR @ NYR | Moneyline | CAR moneyline | $10.00 | 1.65 | 57.1% | 59.3% |
| BUF @ TBL | Moneyline | TBL moneyline | $10.00 | 1.50 | 67.1% | 65.3% |
| EDM @ STL | Moneyline | STL moneyline | $10.00 | 1.95 | 55.2% | 49.8% |
| PIT @ DAL | Moneyline | DAL moneyline | $10.00 | 1.47 | 64.5% | 67.1% |
| CHI @ VAN | Moneyline | VAN moneyline | $10.00 | 1.96 | 50.0% | 49.5% |

Total pretend stake $50.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
