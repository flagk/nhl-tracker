# NHL model report: 2026-11-10 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-10 13:02 UTC · model `20261109-720fd7` · probability source `online_platt` · bankroll $983.82 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-10T13:02:10+00:00 · API credits left: 339*

## Summary

- 7 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| COL @ TOR (19:00 ET) | M. Blackwood / S. Bobrovsky (probable) | 43.9% | 40.9% | TOR 2.38 | +3.0% | - | - | no bet |
| MIN @ MTL (19:00 ET) | J. Wallstedt / J. Fowler (probable) | 49.0% | 46.7% | MTL 2.10 | +2.3% | - | - | no bet |
| WSH @ OTT (19:00 ET) | L. Thompson / L. Ullmark (probable) | 54.6% | 53.4% | OTT 1.82 | +1.1% | - | - | no bet |
| NJD @ WPG (19:00 ET) | J. Allen / S. Skinner (probable) | 43.7% | 41.9% | WPG 2.33 | +1.8% | - | - | no bet |
| FLA @ CBJ (19:30 ET) | J. Markstrom / J. Greaves (probable) | 49.5% | 48.2% | CBJ 2.03 | +1.3% | - | - | no bet |
| DET @ STL (20:00 ET) | J. Gibson / J. Hofer (probable) | 53.3% | 56.6% | DET 2.28 | +3.3% | - | - | no bet |
| NSH @ LAK (22:00 ET) | J. Saros / D. Kuemper (probable) | 57.9% | 56.0% | LAK 1.74 | +1.8% | - | - | no bet |

## Why

**COL @ TOR (19:00 ET)**: No bet: edge +3.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: COL have the clearly better goal differential this season (gap 0.86 goals/game).

**MIN @ MTL (19:00 ET)**: No bet: edge +2.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: MIN have the clearly better goal differential this season (gap 0.36 goals/game). MIN are creating the better share of expected goals recently.

**WSH @ OTT (19:00 ET)**: No bet: edge +1.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: Goaltending edge to OTT (goals saved above expected).

**NJD @ WPG (19:00 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: NJD have the clearly better goal differential this season (gap 0.89 goals/game). NJD are creating the better share of expected goals recently.

**FLA @ CBJ (19:30 ET)**: No bet: edge +1.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: FLA are creating the better share of expected goals recently.

**DET @ STL (20:00 ET)**: No bet: edge +3.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: Goaltending edge to STL (goals saved above expected).

**NSH @ LAK (22:00 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed)
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 267 | -1.7% | -13% to +10% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 222 | -4.5% | -19% to +8% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 112 | -4.6% | -21% to +11% | 60% | - |  |
| `ast_over_control` | 111 | +7.5% | -10% to +25% | 56% | - |  |
| `ast_over_edge` | 94 | -11.6% | -39% to +20% | 29% | - |  |
| `ast_under_edge` | 112 | -0.3% | -14% to +13% | 66% | - |  |
| `edge_1pct` | 22 | -32.4% | -73% to +9% | 32% | -1.30% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 280 | +1.8% | -8% to +12% | 61% | -1.94% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 279 | -4.3% | -13% to +5% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 267 | -6.9% | -18% to +5% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 280 | -19.6% | -32% to -7% | 39% | -2.61% | $10 on every game where the model sees any positive edge |
| `goal_control` | 112 | -7.0% | -32% to +18% | 33% | - |  |
| `goal_edge` | 70 | -21.5% | -71% to +38% | 11% | - |  |
| `home_ml_control` | 233 | +2.9% | -9% to +15% | 57% | -2.52% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 280 | +3.4% | -6% to +13% | 62% | -2.01% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 23 | -26.0% | -68% to +20% | 30% | -1.67% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 31 | -10.4% | -50% to +30% | 42% | -2.43% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 57 | -7.1% | -33% to +16% | 49% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 33 | -5.2% | -61% to +58% | 24% | - |  |
| `parlay_players3` | 15 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 33 | -72.0% | -100% to -27% | 6% | - |  |
| `parlay_top3` | 33 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 159 | -11.5% | -26% to +3% | 47% | - |  |
| `pts_over_control` | 163 | -2.3% | -16% to +12% | 58% | - |  |
| `pts_over_edge` | 133 | -28.5% | -47% to -7% | 31% | - |  |
| `pts_under_edge` | 159 | -3.9% | -18% to +10% | 53% | - |  |
| `puckline_dog_control` | 232 | -6.0% | -16% to +4% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 87 | -3.7% | -20% to +13% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 127 | -17.0% | -34% to -1% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 53 | -28.1% | -58% to +5% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 214 | -1.9% | -14% to +11% | 53% | - |  |
| `sog_over_control` | 218 | +0.1% | -13% to +13% | 50% | - |  |
| `sog_over_edge` | 143 | -9.9% | -26% to +7% | 45% | - |  |
| `sog_under_edge` | 175 | -3.3% | -18% to +11% | 52% | - |  |
| `totals_edge` | 158 | -9.5% | -25% to +5% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 103 | -6.8% | -26% to +12% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 121 | -33.7% | -52% to -14% | 28% | -2.87% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 233 | -16.6% | -30% to -2% | 36% | -3.00% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| COL @ TOR | Moneyline | COL moneyline | $10.00 | 1.65 | 56.1% | 59.1% |
| MIN @ MTL | Moneyline | MIN moneyline | $10.00 | 1.85 | 51.0% | 53.3% |
| WSH @ OTT | Moneyline | OTT moneyline | $10.00 | 1.82 | 54.6% | 53.4% |
| NJD @ WPG | Moneyline | NJD moneyline | $10.00 | 1.68 | 56.3% | 58.1% |
| FLA @ CBJ | Moneyline | FLA moneyline | $10.00 | 1.87 | 50.5% | 51.8% |
| DET @ STL | Moneyline | STL moneyline | $10.00 | 1.74 | 53.3% | 56.6% |
| NSH @ LAK | Moneyline | LAK moneyline | $10.00 | 1.74 | 57.9% | 56.0% |

Total pretend stake $70.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
