# NHL model report: 2026-11-26 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-26 13:03 UTC · model `20261125-ee8f99` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-26T13:02:57+00:00 · API credits left: 125*

## Summary

No NHL games are scheduled (or none are unplayed) for this date.

## Track record (all logged recommendations that have resolved)

- Resolved games: **394** (bet on 11, passed on 97%); model log loss 0.6670, Brier 0.2373 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (394 games with odds): model log loss 0.6670 vs market 0.6545 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.48% to -0.52%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.8% | 40.0% | 50 |
| 46.8% | 36.7% | 49 |
| 50.0% | 42.9% | 49 |
| 52.7% | 59.2% | 49 |
| 55.4% | 42.9% | 49 |
| 58.1% | 55.1% | 49 |
| 61.4% | 71.4% | 49 |
| 68.2% | 72.0% | 50 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 379 | -2.7% | -12% to +8% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 334 | -2.9% | -13% to +7% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +22% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -41% to +6% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +3% | 60% | - |  |
| `edge_1pct` | 48 | -28.8% | -56% to +2% | 33% | -1.47% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 394 | -0.7% | -9% to +8% | 59% | -1.98% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 393 | -5.3% | -13% to +2% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 379 | -5.0% | -14% to +5% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 394 | -14.5% | -25% to -4% | 41% | -2.72% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -19% to +23% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -56% to +33% | 14% | - |  |
| `home_ml_control` | 347 | -6.1% | -16% to +4% | 52% | -2.51% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 394 | +2.2% | -6% to +10% | 61% | -2.04% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -73% to +9% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 49 | -21.3% | -48% to +6% | 39% | -2.36% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 78 | -7.8% | -28% to +13% | 49% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 47 | +8.0% | -43% to +63% | 26% | - |  |
| `parlay_players3` | 27 | -68.1% | -100% to -4% | 4% | - |  |
| `parlay_top2` | 47 | -62.7% | -93% to -25% | 9% | - |  |
| `parlay_top3` | 47 | -93.4% | -100% to -80% | 2% | - |  |
| `pts_edge` | 278 | -16.3% | -28% to -5% | 45% | - |  |
| `pts_over_control` | 283 | -1.2% | -12% to +9% | 58% | - |  |
| `pts_over_edge` | 237 | -22.8% | -37% to -9% | 35% | - |  |
| `pts_under_edge` | 273 | -8.7% | -20% to +2% | 50% | - |  |
| `puckline_dog_control` | 346 | -5.1% | -13% to +3% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 118 | -1.4% | -15% to +12% | 63% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 170 | -7.9% | -24% to +8% | 49% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 86 | -15.8% | -41% to +12% | 31% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 333 | -5.8% | -16% to +4% | 50% | - |  |
| `sog_over_control` | 337 | +3.4% | -7% to +14% | 51% | - |  |
| `sog_over_edge` | 241 | -11.3% | -24% to +1% | 45% | - |  |
| `sog_under_edge` | 292 | -5.7% | -16% to +5% | 51% | - |  |
| `totals_edge` | 221 | -6.3% | -20% to +6% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 154 | -4.6% | -20% to +11% | 49% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 182 | -25.5% | -42% to -9% | 31% | -2.97% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 347 | -10.7% | -23% to +2% | 38% | -2.95% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
