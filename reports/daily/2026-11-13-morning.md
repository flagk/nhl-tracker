# NHL model report: 2026-11-13 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-13 13:05 UTC · model `20261112-169812` · probability source `online_platt` · bankroll $978.31 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-13T13:03:58+00:00 · API credits left: 288*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| NYR @ VAN (21:00 ET) | D. Garand / L. Meriläinen (probable) | 39.0% | 33.1% | VAN 2.90 | +5.9% | - | - | no bet |
| NSH @ VGK (22:00 ET) | J. Saros / C. Hart (probable) | 61.9% | 62.6% | NSH 2.58 | +0.7% | - | - | no bet |

## Why

**NYR @ VAN (21:00 ET)**: No bet: model win probability 39.0% below the 40% confidence floor
  - Context: NYR have the clearly better goal differential this season (gap 1.29 goals/game). NYR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**NSH @ VGK (22:00 ET)**: No bet: edge +0.7% below the 4% minimum (starting goalie unconfirmed); model win probability 38.1% below the 40% confidence floor
  - Context: VGK have the clearly better goal differential this season (gap 0.37 goals/game). VGK are creating the better share of expected goals recently.

## Track record (all logged recommendations that have resolved)

- Resolved games: **304** (bet on 10, passed on 97%); model log loss 0.6553, Brier 0.2317 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (304 games with odds): model log loss 0.6553 vs market 0.6386 (market better)
- Bets: 10 · staked $26.57 · profit $-21.69 · **ROI -81.6%** · win rate 10.0% · avg price 2.50
- Bankroll $978.31 · max drawdown 2.2% · current drawdown 2.2%
- **CLV**: mean -2.33% per $1 (95% CI -3.67% to -0.60%), beat the close on 10% of 10 bets
- ⚠️ 10 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.4% | 39.5% | 38 |
| 46.2% | 39.5% | 38 |
| 49.8% | 47.4% | 38 |
| 52.7% | 57.9% | 38 |
| 55.2% | 44.7% | 38 |
| 58.4% | 65.8% | 38 |
| 61.7% | 68.4% | 38 |
| 68.2% | 79.0% | 38 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 291 | -3.2% | -13% to +7% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 246 | -2.5% | -15% to +10% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 136 | -7.9% | -22% to +8% | 57% | - |  |
| `ast_over_control` | 135 | +4.5% | -12% to +21% | 54% | - |  |
| `ast_over_edge` | 109 | -16.4% | -42% to +13% | 28% | - |  |
| `ast_under_edge` | 135 | -6.0% | -19% to +6% | 62% | - |  |
| `edge_1pct` | 28 | -39.5% | -75% to -2% | 29% | -1.59% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 304 | +2.2% | -7% to +12% | 61% | -1.99% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 303 | -3.8% | -12% to +4% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 291 | -5.9% | -17% to +5% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 304 | -22.6% | -34% to -11% | 38% | -2.68% | $10 on every game where the model sees any positive edge |
| `goal_control` | 136 | -5.5% | -28% to +18% | 34% | - |  |
| `goal_edge` | 85 | -17.6% | -64% to +35% | 13% | - |  |
| `home_ml_control` | 257 | -0.9% | -13% to +11% | 55% | -2.54% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 304 | +3.7% | -5% to +13% | 62% | -2.10% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 25 | -32.0% | -72% to +14% | 28% | -1.75% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 35 | -20.6% | -53% to +15% | 37% | -2.56% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 62 | -8.5% | -33% to +16% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 36 | +15.8% | -45% to +80% | 28% | - |  |
| `parlay_players3` | 17 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 36 | -74.3% | -100% to -38% | 6% | - |  |
| `parlay_top3` | 36 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 183 | -10.2% | -24% to +4% | 48% | - |  |
| `pts_over_control` | 187 | -2.2% | -15% to +11% | 58% | - |  |
| `pts_over_edge` | 151 | -30.7% | -48% to -13% | 30% | - |  |
| `pts_under_edge` | 182 | -4.3% | -18% to +10% | 53% | - |  |
| `puckline_dog_control` | 256 | -6.0% | -15% to +3% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 89 | -4.2% | -21% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 137 | -18.8% | -34% to -3% | 46% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 63 | -30.2% | -57% to +0% | 27% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 237 | -4.3% | -17% to +8% | 51% | - |  |
| `sog_over_control` | 242 | -2.0% | -14% to +11% | 49% | - |  |
| `sog_over_edge` | 163 | -5.6% | -21% to +11% | 47% | - |  |
| `sog_under_edge` | 196 | -5.6% | -18% to +8% | 51% | - |  |
| `totals_edge` | 173 | -8.3% | -23% to +6% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 115 | -4.5% | -23% to +14% | 49% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 134 | -36.8% | -55% to -19% | 27% | -2.88% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 257 | -16.7% | -30% to -3% | 37% | -2.90% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| NYR @ VAN | Moneyline | NYR moneyline | $10.00 | 1.47 | 61.0% | 66.9% |
| NSH @ VGK | Moneyline | VGK moneyline | $10.00 | 1.56 | 61.9% | 62.6% |

Total pretend stake $20.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
