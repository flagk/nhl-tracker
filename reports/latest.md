# NHL model report: 2026-10-03 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-04 03:58 UTC · model `20261003-1d6c5e` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: d_elo, d_xg_share, d_cf_pct, d_xga_pg, h_early_w, d_g_sv_l5, d_pen_drawn_pg, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-04T03:58:35+00:00 · API credits left: 455*

## Summary

No NHL games are scheduled (or none are unplayed) for this date.

## Track record (all logged recommendations that have resolved)

- Resolved games: **27** (bet on 0, passed on 100%); model log loss 0.6478, Brier 0.2280 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (27 games with odds): model log loss 0.6478 vs market 0.6509 (model better)

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 25 | +0.6% | -38% to +38% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 27 | +5.9% | -27% to +35% | 56% | -1.87% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 27 | +2.6% | -25% to +27% | 67% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 25 | -41.1% | -75% to -5% | 28% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 27 | +14.4% | -35% to +61% | 52% | -2.67% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `market_favorite` | 27 | -24.0% | -56% to +5% | 48% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 4 | +67.9% | - | 75% | -1.31% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `puckline_edge` | 19 | -31.5% | - | 47% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `sog_edge` | 15 | +17.2% | - | 67% | - |  |
| `sog_over_control` | 16 | -17.8% | - | 38% | - |  |
| `totals_edge` | 13 | -52.9% | - | 23% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
