# NHL model report: 2026-10-11 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-11 12:04 UTC · model `20261010-b2a5f5` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-11T12:03:50+00:00 · API credits left: 272*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| SEA @ WSH (17:00 ET) | J. Daccord / L. Thompson (probable) | 60.6% | 66.0% | SEA 2.83 | +5.3% | - | - | no bet |
| VAN @ NYR (18:00 ET) | K. Lankinen / I. Shesterkin (probable) | 68.8% | 71.2% | VAN 3.36 | +2.3% | - | - | no bet |
| CAR @ PHI (19:00 ET) | F. Andersen / D. Vladar (probable) | 34.3% | 37.5% | CAR 1.56 | +3.3% | - | - | no bet |

## Why

**SEA @ WSH (17:00 ET)**: No bet: early-season sample size: a team has only played 4 games (< 10)
  - Context: Early in the season: team ratings lean on last year's results and are less reliable.

**VAN @ NYR (18:00 ET)**: No bet: early-season sample size: a team has only played 6 games (< 10)
  - Context: NYR have the clearly better goal differential this season (gap 1.00 goals/game). VAN is on the second night of a back-to-back. NYR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**CAR @ PHI (19:00 ET)**: No bet: early-season sample size: a team has only played 6 games (< 10)
  - Context: CAR have the clearly better goal differential this season (gap 1.42 goals/game). PHI and CAR are on the second night of a back-to-back. CAR are creating the better share of expected goals recently. Division/rival game.

## Track record (all logged recommendations that have resolved)

- Resolved games: **78** (bet on 0, passed on 100%); model log loss 0.6512, Brier 0.2296 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (78 games with odds): model log loss 0.6512 vs market 0.6356 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 44.8% | 42.3% | 26 |
| 54.7% | 53.8% | 26 |
| 64.0% | 73.1% | 26 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 76 | -11.8% | -33% to +11% | 46% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 31 | +20.4% | -13% to +53% | 61% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 32 | -9.9% | -44% to +24% | 53% | - |  |
| `ast_over_control` | 32 | +4.4% | -28% to +37% | 56% | - |  |
| `ast_over_edge` | 28 | -71.1% | -100% to -15% | 7% | - |  |
| `ast_under_edge` | 32 | +3.5% | -28% to +37% | 66% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 78 | +7.2% | -12% to +24% | 60% | -2.06% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 78 | -12.9% | -30% to +5% | 56% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 76 | -6.2% | -28% to +16% | 50% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 78 | -12.1% | -39% to +15% | 45% | -2.61% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `goal_control` | 32 | -24.1% | -65% to +18% | 28% | - |  |
| `goal_edge` | 31 | -55.6% | -100% to +20% | 6% | - |  |
| `home_ml_control` | 31 | -2.9% | -35% to +28% | 55% | -2.35% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 78 | +1.2% | -17% to +19% | 62% | -2.14% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 12 | +15.7% | - | 50% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 10 | -22.8% | - | 40% | - | Goals model: $5-$30 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 3 | +30.7% | - | 33% | - |  |
| `parlay_players3` | 3 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 3 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 3 | -100.0% | - | 0% | - |  |
| `pts_edge` | 31 | -15.5% | -53% to +24% | 48% | - |  |
| `pts_over_control` | 35 | -21.1% | -49% to +9% | 46% | - |  |
| `pts_over_edge` | 32 | -70.0% | -95% to -35% | 12% | - |  |
| `pts_under_edge` | 31 | -2.3% | -38% to +35% | 58% | - |  |
| `puckline_dog_control` | 31 | -19.5% | -46% to +8% | 52% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 19 | -34.8% | - | 47% | - | Goals model: $5-$30 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 46 | -25.0% | -49% to +1% | 48% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 2 | +203.8% | - | 100% | - | Goals model: $5-$30 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 72 | -4.5% | -26% to +17% | 56% | - |  |
| `sog_over_control` | 74 | +10.7% | -14% to +34% | 53% | - |  |
| `sog_over_edge` | 31 | +4.5% | -33% to +41% | 52% | - |  |
| `sog_under_edge` | 34 | -14.9% | -45% to +18% | 53% | - |  |
| `totals_edge` | 45 | -11.3% | -40% to +18% | 49% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 14 | +7.6% | - | 64% | - | Goals model: $5-$30 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 16 | -37.9% | - | 31% | -2.40% | $5-$30 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 31 | -24.0% | -62% to +17% | 32% | -2.62% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| SEA @ WSH | Moneyline | WSH moneyline | $16.00 | 1.48 | 60.6% | 66.0% |
| VAN @ NYR | Moneyline | NYR moneyline | $24.00 | 1.38 | 68.8% | 71.2% |
| CAR @ PHI | Moneyline | CAR moneyline | $21.00 | 1.56 | 65.7% | 62.5% |

Total pretend stake $61.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
