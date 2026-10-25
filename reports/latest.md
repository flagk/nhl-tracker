# NHL model report: 2026-10-25 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-25 12:05 UTC · model `20261024-43498a` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-25T12:04:45+00:00 · API credits left: 106*

## Summary

- 5 games · **1 recommended bet(s)** · total stake $3.60 (0.36% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| FLA @ CHI (15:00 ET) | J. Markstrom / S. Knight (probable) | 40.8% | 37.3% | CHI 2.65 | +3.5% | - | - | no bet |
| NYI @ DAL (18:00 ET) | I. Sorokin / J. Oettinger (probable) | 60.5% | 63.8% | NYI 2.66 | +3.3% | - | - | no bet |
| MTL @ WPG (19:00 ET) | J. Dobes / C. Hellebuyck (probable) | 50.3% | 47.4% | WPG 2.07 | +2.9% | - | - | no bet |
| COL @ NSH (20:00 ET) | M. Blackwood / J. Saros (probable) | 36.7% | 32.0% | NSH 3.10 | +4.7% | - | - | no bet |
| MIN @ VAN (22:00 ET) | J. Wallstedt / K. Lankinen (probable) | 41.1% | 37.1% | VAN 2.66 | +4.0% | +9.4% | $3.60 | **BET VAN** |

## Why

**FLA @ CHI (15:00 ET)**: No bet: edge +3.5% below the 4% minimum (starting goalie unconfirmed)
  - Context: FLA have the clearly better goal differential this season (gap 0.74 goals/game). FLA are creating the better share of expected goals recently. Goaltending edge to FLA (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**NYI @ DAL (18:00 ET)**: No bet: edge +3.3% below the 4% minimum (starting goalie unconfirmed); model win probability 39.5% below the 40% confidence floor
  - Context: DAL have the clearly better goal differential this season (gap 0.30 goals/game). DAL are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**MTL @ WPG (19:00 ET)**: No bet: edge +2.9% below the 4% minimum (starting goalie unconfirmed)
  - Context: MTL have the clearly better goal differential this season (gap 0.26 goals/game). Goaltending edge to WPG (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**COL @ NSH (20:00 ET)**: No bet: model win probability 36.7% below the 40% confidence floor
  - Context: COL have the clearly better goal differential this season (gap 1.22 goals/game). NSH is on the second night of a back-to-back. COL are creating the better share of expected goals recently. Division/rival game.

**MIN @ VAN (22:00 ET)**: Bet $3.60 (0.36% of bankroll) on VAN at 2.66. The model gives VAN 41.1% vs the market's no-vig 37.1% (edge +4.0%); after shrinking toward the market the working probability is 38.5%, worth +2.4% per $1 at that price.
  - Context: MIN have the clearly better goal differential this season (gap 1.55 goals/game). MIN is on the second night of a back-to-back. MIN are creating the better share of expected goals recently. Goaltending edge to MIN (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **174** (bet on 0, passed on 100%); model log loss 0.6593, Brier 0.2335 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (174 games with odds): model log loss 0.6593 vs market 0.6455 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 40.1% | 41.4% | 29 |
| 48.3% | 58.6% | 29 |
| 52.4% | 58.6% | 29 |
| 56.3% | 62.1% | 29 |
| 60.1% | 65.5% | 29 |
| 66.4% | 72.4% | 29 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 166 | -5.3% | -20% to +9% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 121 | +0.6% | -17% to +17% | 51% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -36% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -25% to +33% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +54% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -23% to +26% | 65% | - |  |
| `edge_1pct` | 1 | +106.0% | - | 100% | -1.72% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 174 | -0.4% | -13% to +12% | 59% | -2.11% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 173 | -7.7% | -19% to +4% | 60% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 166 | -16.2% | -31% to -1% | 44% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 174 | -18.6% | -34% to -4% | 40% | -2.55% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -49% to +31% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +91% | 10% | - |  |
| `home_ml_control` | 127 | +10.6% | -7% to +26% | 61% | -2.67% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 174 | +2.3% | -10% to +14% | 61% | -2.11% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 15 | -4.5% | - | 40% | -1.36% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 2 | -4.5% | - | 50% | -4.14% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 26 | -41.9% | -71% to -6% | 31% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 17 | -21.2% | - | 24% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 17 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 17 | -100.0% | - | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +5% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -24% to +11% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -63% to -4% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -33% to +7% | 47% | - |  |
| `puckline_dog_control` | 126 | -6.8% | -20% to +7% | 60% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 59 | -4.3% | -24% to +15% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 90 | -13.8% | -31% to +6% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 22 | -14.5% | -64% to +41% | 32% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 134 | +2.6% | -13% to +18% | 56% | - |  |
| `sog_over_control` | 138 | -4.7% | -22% to +12% | 46% | - |  |
| `sog_over_edge` | 67 | -14.4% | -39% to +9% | 45% | - |  |
| `sog_under_edge` | 96 | +4.4% | -14% to +22% | 56% | - |  |
| `totals_edge` | 98 | -23.7% | -42% to -4% | 40% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 57 | -18.2% | -43% to +7% | 42% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 64 | -25.4% | -52% to +3% | 31% | -2.74% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 127 | -14.6% | -34% to +5% | 37% | -2.82% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| FLA @ CHI | Moneyline | FLA moneyline | $10.00 | 1.56 | 59.2% | 62.7% |
| NYI @ DAL | Moneyline | DAL moneyline | $10.00 | 1.54 | 60.5% | 63.8% |
| MTL @ WPG | Moneyline | WPG moneyline | $10.00 | 2.07 | 50.3% | 47.4% |
| COL @ NSH | Moneyline | COL moneyline | $10.00 | 1.44 | 63.3% | 68.0% |
| MIN @ VAN | Moneyline | MIN moneyline | $10.00 | 1.56 | 58.9% | 62.9% |

Total pretend stake $50.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
