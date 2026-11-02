# NHL model report: 2026-11-02 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-02 13:04 UTC · model `20261101-3fd79c` · probability source `online_platt` · bankroll $990.96 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-02T13:03:35+00:00 · API credits left: 481*

## Summary

- 4 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| VGK @ BOS (19:00 ET) | C. Hart / J. Swayman (probable) | 49.7% | 49.1% | BOS 1.98 | +0.6% | - | - | no bet |
| DET @ FLA (19:00 ET) | J. Gibson / A. Schmid (probable) | 57.3% | 62.4% | DET 2.57 | +5.1% | - | - | no bet |
| COL @ CHI (20:00 ET) | M. Blackwood / S. Knight (probable) | 35.8% | 31.4% | CHI 3.11 | +4.4% | - | - | no bet |
| CGY @ SJS (21:00 ET) | D. Wolf / Y. Askarov (probable) | 50.8% | 54.9% | CGY 2.14 | +4.2% | - | - | no bet |

## Why

**VGK @ BOS (19:00 ET)**: No bet: edge +0.6% below the 4% minimum (starting goalie unconfirmed)
  - Context: BOS have the clearly better goal differential this season (gap 0.30 goals/game). VGK travelled 3,819 km across 3 time zone(s) since their last game. VGK are creating the better share of expected goals recently. Goaltending edge to BOS (goals saved above expected).

**DET @ FLA (19:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.5% (< 1%)
  - Context: Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**COL @ CHI (20:00 ET)**: No bet: model win probability 35.8% below the 40% confidence floor
  - Context: COL have the clearly better goal differential this season (gap 1.76 goals/game). COL are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**CGY @ SJS (21:00 ET)**: No bet: expected value after shrinkage toward the market is only -0.5% (< 1%)
  - Context: CGY are creating the better share of expected goals recently. Goaltending edge to SJS (goals saved above expected). Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **226** (bet on 5, passed on 98%); model log loss 0.6573, Brier 0.2326 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (226 games with odds): model log loss 0.6573 vs market 0.6448 (market better)
- Bets: 5 · staked $13.92 · profit $-9.04 · **ROI -65.0%** · win rate 20.0% · avg price 2.41
- Bankroll $990.96 · max drawdown 0.9% · current drawdown 0.9%
- **CLV**: mean -1.94% per $1 (95% CI -4.33% to +1.39%), beat the close on 20% of 5 bets
- ⚠️ 5 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.3% | 44.8% | 29 |
| 46.6% | 42.9% | 28 |
| 50.4% | 46.4% | 28 |
| 52.9% | 64.3% | 28 |
| 55.6% | 46.4% | 28 |
| 58.8% | 71.4% | 28 |
| 62.0% | 71.4% | 28 |
| 67.7% | 72.4% | 29 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 216 | -2.0% | -15% to +11% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 171 | -4.6% | -19% to +10% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 48 | -7.0% | -30% to +20% | 56% | - |  |
| `ast_over_control` | 47 | +9.2% | -19% to +37% | 57% | - |  |
| `ast_over_edge` | 40 | -14.8% | -65% to +49% | 20% | - |  |
| `ast_under_edge` | 48 | +0.6% | -21% to +23% | 65% | - |  |
| `edge_1pct` | 14 | -38.6% | - | 29% | -0.89% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 226 | +1.6% | -10% to +13% | 60% | -1.99% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 225 | -5.8% | -15% to +4% | 61% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 216 | -9.7% | -22% to +3% | 47% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 226 | -19.3% | -33% to -5% | 39% | -2.44% | $10 on every game where the model sees any positive edge |
| `goal_control` | 48 | -15.2% | -51% to +21% | 31% | - |  |
| `goal_edge` | 42 | -1.0% | -75% to +87% | 12% | - |  |
| `home_ml_control` | 179 | +4.6% | -9% to +19% | 58% | -2.61% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 226 | +3.9% | -7% to +15% | 62% | -2.09% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 20 | -15.0% | -63% to +38% | 35% | -1.51% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 18 | -4.7% | - | 44% | -2.16% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 44 | -18.5% | -45% to +10% | 43% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 25 | +25.1% | -45% to +100% | 32% | - |  |
| `parlay_players3` | 7 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 25 | -63.1% | -100% to -4% | 8% | - |  |
| `parlay_top3` | 25 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 95 | -19.7% | -38% to +0% | 42% | - |  |
| `pts_over_control` | 99 | -4.8% | -21% to +12% | 58% | - |  |
| `pts_over_edge` | 70 | -39.6% | -65% to -11% | 23% | - |  |
| `pts_under_edge` | 95 | -14.9% | -35% to +5% | 46% | - |  |
| `puckline_dog_control` | 178 | -5.7% | -17% to +6% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 73 | -3.9% | -22% to +13% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 109 | -14.3% | -31% to +3% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 38 | -26.5% | -63% to +12% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 150 | +2.5% | -12% to +17% | 56% | - |  |
| `sog_over_control` | 154 | -4.5% | -21% to +12% | 47% | - |  |
| `sog_over_edge` | 80 | -11.3% | -31% to +9% | 46% | - |  |
| `sog_under_edge` | 111 | +2.9% | -15% to +21% | 56% | - |  |
| `totals_edge` | 134 | -16.6% | -32% to -1% | 43% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 79 | -15.4% | -38% to +6% | 43% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 94 | -24.6% | -45% to -2% | 32% | -2.51% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 179 | -16.4% | -33% to +1% | 36% | -2.78% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| VGK @ BOS | Moneyline | VGK moneyline | $10.00 | 1.91 | 50.3% | 50.9% |
| DET @ FLA | Moneyline | FLA moneyline | $10.00 | 1.57 | 57.3% | 62.4% |
| COL @ CHI | Moneyline | COL moneyline | $10.00 | 1.43 | 64.2% | 68.6% |
| CGY @ SJS | Moneyline | SJS moneyline | $10.00 | 1.78 | 50.8% | 54.9% |

Total pretend stake $40.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
