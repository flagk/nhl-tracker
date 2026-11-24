# NHL model report: 2026-11-24 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-24 13:03 UTC · model `20261123-65bf35` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-24T13:03:10+00:00 · API credits left: 148*

## Summary

- 1 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| VGK @ EDM (20:00 ET) | C. Hart / T. Jarry (probable) | 50.3% | 48.4% | EDM 2.00 | +1.8% | - | - | no bet |

## Why

**VGK @ EDM (20:00 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: EDM have the clearly better goal differential this season (gap 0.30 goals/game). VGK are creating the better share of expected goals recently. Division/rival game.

## Track record (all logged recommendations that have resolved)

- Resolved games: **379** (bet on 11, passed on 97%); model log loss 0.6662, Brier 0.2369 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (379 games with odds): model log loss 0.6662 vs market 0.6531 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.47% to -0.58%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.6% | 39.6% | 48 |
| 46.7% | 40.4% | 47 |
| 49.9% | 40.4% | 47 |
| 52.7% | 56.2% | 48 |
| 55.3% | 44.7% | 47 |
| 58.2% | 57.5% | 47 |
| 61.5% | 72.3% | 47 |
| 68.3% | 70.8% | 48 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 364 | -1.9% | -12% to +8% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 319 | -3.8% | -15% to +7% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +23% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -41% to +8% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +2% | 60% | - |  |
| `edge_1pct` | 44 | -28.0% | -57% to +3% | 34% | -1.42% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 379 | -1.0% | -9% to +8% | 59% | -1.99% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 378 | -4.4% | -12% to +3% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 364 | -5.2% | -15% to +5% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 379 | -15.7% | -27% to -4% | 40% | -2.73% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -20% to +23% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -56% to +38% | 14% | - |  |
| `home_ml_control` | 332 | -5.6% | -16% to +5% | 52% | -2.53% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 379 | +1.6% | -7% to +10% | 61% | -2.06% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -72% to +7% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 45 | -22.7% | -51% to +6% | 38% | -2.44% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 76 | -7.9% | -30% to +14% | 49% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 46 | +10.3% | -44% to +68% | 26% | - |  |
| `parlay_players3` | 26 | -66.8% | -100% to -1% | 4% | - |  |
| `parlay_top2` | 46 | -61.9% | -93% to -22% | 9% | - |  |
| `parlay_top3` | 46 | -93.3% | -100% to -80% | 2% | - |  |
| `pts_edge` | 262 | -15.9% | -28% to -5% | 45% | - |  |
| `pts_over_control` | 267 | -0.3% | -11% to +10% | 58% | - |  |
| `pts_over_edge` | 221 | -25.9% | -40% to -11% | 33% | - |  |
| `pts_under_edge` | 260 | -6.9% | -18% to +5% | 51% | - |  |
| `puckline_dog_control` | 331 | -4.1% | -12% to +4% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 111 | -0.7% | -15% to +14% | 63% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 162 | -10.0% | -26% to +5% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 82 | -19.2% | -44% to +8% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 317 | -6.4% | -17% to +4% | 50% | - |  |
| `sog_over_control` | 321 | +4.2% | -7% to +15% | 52% | - |  |
| `sog_over_edge` | 227 | -9.8% | -23% to +3% | 46% | - |  |
| `sog_under_edge` | 276 | -7.1% | -18% to +4% | 50% | - |  |
| `totals_edge` | 215 | -6.3% | -19% to +7% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 148 | -5.8% | -21% to +11% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 174 | -26.4% | -43% to -9% | 30% | -2.98% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 332 | -10.3% | -23% to +2% | 39% | -2.94% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| VGK @ EDM | Moneyline | EDM moneyline | $10.00 | 2.00 | 50.3% | 48.4% |

Total pretend stake $10.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
