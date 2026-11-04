# NHL model report: 2026-11-04 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-04 13:03 UTC · model `20261103-5af87d` · probability source `online_platt` · bankroll $990.96 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-04T13:02:30+00:00 · API credits left: 444*

## Summary

- 4 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| OTT @ BUF (19:30 ET) | L. Ullmark / U. Luukkonen (probable) | 57.2% | 54.0% | BUF 1.80 | +3.2% | - | - | no bet |
| CHI @ FLA (19:30 ET) | S. Knight / J. Markstrom (probable) | 68.3% | 69.7% | CHI 3.20 | +1.4% | - | - | no bet |
| SJS @ LAK (22:00 ET) | Y. Askarov / A. Forsberg (probable) | 64.3% | 59.1% | LAK 1.65 | +5.2% | - | - | no bet |
| CGY @ SEA (22:00 ET) | D. Wolf / P. Grubauer (probable) | 54.3% | 55.4% | CGY 2.17 | +1.1% | - | - | no bet |

## Why

**OTT @ BUF (19:30 ET)**: No bet: edge +3.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: OTT travelled 3,513 km across 3 time zone(s) since their last game. Goaltending edge to OTT (goals saved above expected). Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**CHI @ FLA (19:30 ET)**: No bet: edge +1.4% below the 4% minimum (starting goalie unconfirmed); model win probability 31.7% below the 40% confidence floor
  - Context: FLA have the clearly better goal differential this season (gap 1.05 goals/game). FLA are creating the better share of expected goals recently.

**SJS @ LAK (22:00 ET)**: No bet: expected value after shrinkage toward the market is only -0.0% (< 1%)
  - Context: LAK are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**CGY @ SEA (22:00 ET)**: No bet: edge +1.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: SEA have the clearly better goal differential this season (gap 0.53 goals/game). CGY are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **241** (bet on 5, passed on 98%); model log loss 0.6584, Brier 0.2331 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (241 games with odds): model log loss 0.6584 vs market 0.6462 (market better)
- Bets: 5 · staked $13.92 · profit $-9.04 · **ROI -65.0%** · win rate 20.0% · avg price 2.41
- Bankroll $990.96 · max drawdown 0.9% · current drawdown 0.9%
- **CLV**: mean -1.94% per $1 (95% CI -4.33% to +1.45%), beat the close on 20% of 5 bets
- ⚠️ 5 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 38.9% | 41.9% | 31 |
| 46.3% | 43.3% | 30 |
| 50.1% | 43.3% | 30 |
| 52.8% | 66.7% | 30 |
| 55.5% | 46.7% | 30 |
| 58.6% | 66.7% | 30 |
| 61.9% | 70.0% | 30 |
| 67.7% | 73.3% | 30 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 229 | -1.8% | -14% to +10% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 184 | -4.9% | -19% to +10% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 64 | -12.6% | -35% to +10% | 53% | - |  |
| `ast_over_control` | 63 | +6.8% | -17% to +30% | 57% | - |  |
| `ast_over_edge` | 56 | -11.6% | -54% to +37% | 23% | - |  |
| `ast_under_edge` | 64 | -6.9% | -26% to +12% | 59% | - |  |
| `edge_1pct` | 17 | -34.2% | - | 29% | -1.08% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 241 | +1.6% | -10% to +12% | 60% | -1.96% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 240 | -4.8% | -14% to +5% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 229 | -7.3% | -19% to +6% | 48% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 241 | -19.1% | -32% to -6% | 39% | -2.49% | $10 on every game where the model sees any positive edge |
| `goal_control` | 64 | -8.9% | -40% to +25% | 33% | - |  |
| `goal_edge` | 51 | -6.2% | -71% to +71% | 12% | - |  |
| `home_ml_control` | 194 | +2.5% | -11% to +16% | 56% | -2.59% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 241 | +2.1% | -9% to +12% | 61% | -2.04% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 20 | -15.0% | -64% to +39% | 35% | -1.51% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 20 | -1.3% | -47% to +48% | 45% | -2.15% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 49 | -11.7% | -35% to +15% | 47% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 27 | +15.9% | -50% to +90% | 30% | - |  |
| `parlay_players3` | 9 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 27 | -65.8% | -100% to -11% | 7% | - |  |
| `parlay_top3` | 27 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 111 | -21.7% | -40% to -3% | 41% | - |  |
| `pts_over_control` | 115 | -3.5% | -19% to +11% | 58% | - |  |
| `pts_over_edge` | 86 | -37.4% | -61% to -13% | 24% | - |  |
| `pts_under_edge` | 111 | -20.1% | -36% to -3% | 43% | - |  |
| `puckline_dog_control` | 193 | -4.4% | -15% to +6% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 77 | -3.0% | -21% to +14% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 114 | -15.9% | -33% to +1% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 44 | -24.0% | -59% to +12% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 166 | +3.2% | -11% to +17% | 56% | - |  |
| `sog_over_control` | 170 | -6.2% | -21% to +10% | 46% | - |  |
| `sog_over_edge` | 96 | -14.1% | -34% to +5% | 45% | - |  |
| `sog_under_edge` | 127 | +4.6% | -12% to +20% | 57% | - |  |
| `totals_edge` | 140 | -13.2% | -28% to +3% | 45% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 83 | -12.2% | -34% to +9% | 45% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 100 | -24.4% | -46% to -3% | 32% | -2.62% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 194 | -13.6% | -30% to +2% | 38% | -2.84% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| OTT @ BUF | Moneyline | BUF moneyline | $10.00 | 1.80 | 57.2% | 54.0% |
| CHI @ FLA | Moneyline | FLA moneyline | $10.00 | 1.40 | 68.3% | 69.7% |
| SJS @ LAK | Moneyline | LAK moneyline | $10.00 | 1.65 | 64.3% | 59.1% |
| CGY @ SEA | Moneyline | SEA moneyline | $10.00 | 1.75 | 54.3% | 55.4% |

Total pretend stake $40.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
