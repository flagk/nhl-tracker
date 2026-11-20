# NHL model report: 2026-11-20 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-20 20:12 UTC · model `20261119-9a829b` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-20T20:11:54+00:00 · API credits left: 187*

## Summary

No NHL games are scheduled (or none are unplayed) for this date.

## Track record (all logged recommendations that have resolved)

- Resolved games: **353** (bet on 11, passed on 97%); model log loss 0.6655, Brier 0.2366 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (353 games with odds): model log loss 0.6655 vs market 0.6518 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.47% to -0.58%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.8% | 40.0% | 45 |
| 46.6% | 38.6% | 44 |
| 49.9% | 40.9% | 44 |
| 52.7% | 52.3% | 44 |
| 55.2% | 47.7% | 44 |
| 58.2% | 59.1% | 44 |
| 61.6% | 68.2% | 44 |
| 68.5% | 72.7% | 44 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 338 | -3.7% | -14% to +7% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 293 | -2.1% | -13% to +9% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +23% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -42% to +8% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +2% | 60% | - |  |
| `edge_1pct` | 41 | -32.7% | -61% to -0% | 32% | -1.59% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 353 | -0.3% | -10% to +8% | 59% | -2.03% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 352 | -5.6% | -14% to +2% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 338 | -4.8% | -15% to +6% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 353 | -17.3% | -29% to -6% | 39% | -2.71% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -20% to +22% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -56% to +36% | 14% | - |  |
| `home_ml_control` | 306 | -6.6% | -17% to +4% | 52% | -2.53% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 353 | +1.5% | -7% to +10% | 61% | -2.09% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -72% to +10% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 42 | -25.4% | -54% to +6% | 36% | -2.50% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 70 | -8.2% | -31% to +14% | 49% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 43 | +18.0% | -38% to +80% | 28% | - |  |
| `parlay_players3` | 24 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 43 | -59.3% | -93% to -13% | 9% | - |  |
| `parlay_top3` | 43 | -92.8% | -100% to -78% | 2% | - |  |
| `pts_edge` | 238 | -11.6% | -23% to +1% | 47% | - |  |
| `pts_over_control` | 243 | -1.2% | -13% to +10% | 58% | - |  |
| `pts_over_edge` | 204 | -27.7% | -43% to -13% | 32% | - |  |
| `pts_under_edge` | 236 | -4.9% | -17% to +7% | 53% | - |  |
| `puckline_dog_control` | 305 | -6.2% | -15% to +2% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 106 | -2.2% | -17% to +13% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 152 | -13.0% | -29% to +3% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 73 | -23.4% | -52% to +5% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 293 | -5.1% | -16% to +6% | 51% | - |  |
| `sog_over_control` | 297 | +2.7% | -9% to +14% | 51% | - |  |
| `sog_over_edge` | 217 | -10.0% | -23% to +4% | 46% | - |  |
| `sog_under_edge` | 252 | -5.4% | -16% to +7% | 51% | - |  |
| `totals_edge` | 199 | -6.7% | -20% to +6% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 135 | -4.2% | -20% to +13% | 49% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 161 | -26.6% | -44% to -9% | 30% | -2.90% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 306 | -10.5% | -23% to +2% | 39% | -2.89% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
