# NHL model report: 2026-11-01 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-01 13:05 UTC · model `20261031-0854bc` · probability source `online_platt` · bankroll $990.96 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-01T13:04:29+00:00 · API credits left: 498*

## Summary

- 5 games · **1 recommended bet(s)** · total stake $3.52 (0.36% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| WPG @ CAR (13:00 ET) | S. Skinner / P. Kochetkov (probable) | 72.0% | 71.7% | CAR 1.38 | +0.3% | - | - | no bet |
| MIN @ WSH (15:30 ET) | J. Wallstedt / L. Thompson (probable) | 50.9% | 49.9% | WSH 1.94 | +1.0% | - | - | no bet |
| UTA @ PHI (16:00 ET) | K. Vejmelka / D. Vladar (probable) | 42.5% | 40.5% | PHI 2.39 | +2.1% | - | - | no bet |
| EDM @ NYR (18:00 ET) | T. Jarry / D. Garand (probable) | 59.3% | 53.8% | NYR 1.83 | +5.5% | +8.4% | $3.52 | **BET NYR** |
| OTT @ ANA (20:00 ET) | L. Ullmark / L. Dostal (probable) | 53.8% | 48.1% | ANA 2.02 | +5.7% | - | - | no bet |

## Why

**WPG @ CAR (13:00 ET)**: No bet: edge +0.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: CAR have the clearly better goal differential this season (gap 1.41 goals/game). CAR are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**MIN @ WSH (15:30 ET)**: No bet: edge +1.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: MIN are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**UTA @ PHI (16:00 ET)**: No bet: edge +2.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: UTA have the clearly better goal differential this season (gap 0.96 goals/game). UTA are creating the better share of expected goals recently. Goaltending edge to PHI (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**EDM @ NYR (18:00 ET)**: Bet $3.52 (0.36% of bankroll) on NYR at 1.83. The model gives NYR 59.3% vs the market's no-vig 53.8% (edge +5.5%); after shrinking toward the market the working probability is 55.3%, worth +1.2% per $1 at that price.
  - Context: EDM have the clearly better goal differential this season (gap 0.58 goals/game). EDM is on the second night of a back-to-back. Early in the season: team ratings lean on last year's results and are less reliable.

**OTT @ ANA (20:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.2% (< 1%)
  - Context: OTT have the clearly better goal differential this season (gap 0.51 goals/game). OTT is on the second night of a back-to-back. Goaltending edge to OTT (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **221** (bet on 5, passed on 98%); model log loss 0.6567, Brier 0.2322 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (221 games with odds): model log loss 0.6567 vs market 0.6439 (market better)
- Bets: 5 · staked $13.92 · profit $-9.04 · **ROI -65.0%** · win rate 20.0% · avg price 2.41
- Bankroll $990.96 · max drawdown 0.9% · current drawdown 0.9%
- **CLV**: mean -1.94% per $1 (95% CI -4.33% to +1.45%), beat the close on 20% of 5 bets
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 212 | -2.8% | -16% to +10% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 167 | -3.4% | -17% to +11% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -36% to +22% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -26% to +34% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +27% | 65% | - |  |
| `edge_1pct` | 13 | -33.9% | - | 31% | -0.81% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 221 | +2.3% | -9% to +13% | 61% | -1.97% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 220 | -7.2% | -17% to +3% | 60% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 212 | -10.7% | -24% to +2% | 47% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 221 | -20.1% | -33% to -6% | 39% | -2.40% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -47% to +32% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +91% | 10% | - |  |
| `home_ml_control` | 174 | +4.3% | -10% to +19% | 57% | -2.55% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 221 | +4.7% | -6% to +15% | 63% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 20 | -15.0% | -64% to +38% | 35% | -1.51% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 16 | -5.4% | - | 44% | -2.05% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 42 | -23.8% | -50% to +3% | 40% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 24 | +30.4% | -41% to +111% | 33% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 24 | -61.5% | -100% to -0% | 8% | - |  |
| `parlay_top3` | 24 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +5% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -25% to +11% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -64% to -4% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -33% to +6% | 47% | - |  |
| `puckline_dog_control` | 173 | -6.4% | -18% to +5% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 72 | -4.6% | -23% to +13% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 107 | -14.5% | -32% to +4% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 36 | -27.8% | -65% to +12% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 142 | +1.7% | -14% to +16% | 56% | - |  |
| `sog_over_control` | 146 | -4.5% | -21% to +12% | 47% | - |  |
| `sog_over_edge` | 72 | -12.6% | -35% to +9% | 46% | - |  |
| `sog_under_edge` | 103 | +3.9% | -14% to +22% | 56% | - |  |
| `totals_edge` | 131 | -17.6% | -34% to -2% | 43% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 77 | -15.8% | -38% to +5% | 43% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 92 | -27.7% | -50% to -3% | 30% | -2.44% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 174 | -17.7% | -34% to -1% | 36% | -2.72% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| WPG @ CAR | Moneyline | CAR moneyline | $10.00 | 1.38 | 72.0% | 71.7% |
| MIN @ WSH | Moneyline | WSH moneyline | $10.00 | 1.94 | 50.9% | 49.9% |
| UTA @ PHI | Moneyline | UTA moneyline | $10.00 | 1.65 | 57.5% | 59.5% |
| EDM @ NYR | Moneyline | NYR moneyline | $10.00 | 1.83 | 59.3% | 53.8% |
| OTT @ ANA | Moneyline | ANA moneyline | $10.00 | 2.02 | 53.8% | 48.1% |

Total pretend stake $50.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
