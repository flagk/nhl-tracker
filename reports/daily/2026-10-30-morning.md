# NHL model report: 2026-10-30 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-30 12:05 UTC · model `20261029-760ebc` · probability source `online_platt` · bankroll $995.97 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-30T12:04:06+00:00 · API credits left: 64*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| NJD @ VGK (18:00 ET) | J. Allen / C. Hart (probable) | 55.0% | 51.5% | VGK 1.87 | +3.4% | - | - | no bet |
| WPG @ FLA (19:00 ET) | S. Skinner / A. Schmid (probable) | 66.7% | 67.2% | WPG 3.00 | +0.5% | - | - | no bet |
| SEA @ CGY (20:00 ET) | P. Grubauer / D. Cooley (probable) | 53.3% | 48.5% | CGY 2.00 | +4.8% | - | - | no bet |

## Why

**NJD @ VGK (18:00 ET)**: No bet: edge +3.4% below the 4% minimum (starting goalie unconfirmed)
  - Context: VGK have the clearly better goal differential this season (gap 0.39 goals/game). Early in the season: team ratings lean on last year's results and are less reliable.

**WPG @ FLA (19:00 ET)**: No bet: edge +0.5% below the 4% minimum (starting goalie unconfirmed); model win probability 33.3% below the 40% confidence floor
  - Context: FLA have the clearly better goal differential this season (gap 0.44 goals/game). WPG is on the second night of a back-to-back. FLA are creating the better share of expected goals recently. Goaltending edge to WPG (goals saved above expected).

**SEA @ CGY (20:00 ET)**: No bet: expected value after shrinkage toward the market is only -0.2% (< 1%)
  - Context: SEA have the clearly better goal differential this season (gap 0.81 goals/game). Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **208** (bet on 3, passed on 99%); model log loss 0.6587, Brier 0.2332 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (208 games with odds): model log loss 0.6587 vs market 0.6478 (market better)
- Bets: 3 · staked $8.91 · profit $-4.03 · **ROI -45.3%** · win rate 33.3% · avg price 2.47
- Bankroll $995.97 · max drawdown 0.7% · current drawdown 0.4%
- ⚠️ 3 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.0% | 38.5% | 26 |
| 46.6% | 50.0% | 26 |
| 50.3% | 46.2% | 26 |
| 52.8% | 61.5% | 26 |
| 55.7% | 50.0% | 26 |
| 58.9% | 76.9% | 26 |
| 61.9% | 69.2% | 26 |
| 67.4% | 69.2% | 26 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 199 | -4.0% | -17% to +9% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 154 | -1.6% | -16% to +14% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -34% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +33% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -70% to +50% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -23% to +26% | 65% | - |  |
| `edge_1pct` | 9 | -4.6% | - | 44% | -2.63% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 208 | +1.7% | -10% to +13% | 60% | -2.06% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 207 | -6.5% | -17% to +4% | 61% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 199 | -11.6% | -26% to +2% | 46% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 208 | -16.9% | -31% to -3% | 40% | -2.56% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -49% to +30% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +90% | 10% | - |  |
| `home_ml_control` | 161 | +5.3% | -10% to +20% | 58% | -2.54% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 208 | +3.4% | -8% to +15% | 62% | -2.07% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 18 | -5.5% | - | 39% | -1.66% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 11 | +19.4% | - | 55% | -3.16% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 39 | -22.6% | -51% to +6% | 41% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 22 | +42.2% | -37% to +128% | 36% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 22 | -58.0% | -100% to +9% | 9% | - |  |
| `parlay_top3` | 22 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +5% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -25% to +10% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -65% to -9% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -33% to +4% | 47% | - |  |
| `puckline_dog_control` | 160 | -5.4% | -17% to +7% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 68 | -3.2% | -22% to +15% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 102 | -13.2% | -31% to +5% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 33 | -21.3% | -61% to +21% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 142 | +1.7% | -14% to +18% | 56% | - |  |
| `sog_over_control` | 146 | -4.5% | -22% to +12% | 47% | - |  |
| `sog_over_edge` | 72 | -12.6% | -34% to +9% | 46% | - |  |
| `sog_under_edge` | 103 | +3.9% | -14% to +22% | 56% | - |  |
| `totals_edge` | 125 | -18.3% | -35% to -1% | 42% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 72 | -15.5% | -38% to +7% | 43% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 84 | -23.2% | -47% to +1% | 32% | -2.77% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 161 | -15.1% | -32% to +2% | 37% | -2.87% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| NJD @ VGK | Moneyline | VGK moneyline | $10.00 | 1.87 | 55.0% | 51.5% |
| WPG @ FLA | Moneyline | FLA moneyline | $10.00 | 1.46 | 66.7% | 67.2% |
| SEA @ CGY | Moneyline | CGY moneyline | $10.00 | 2.00 | 53.3% | 48.5% |

Total pretend stake $30.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
