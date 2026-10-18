# NHL model report: 2026-10-18 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-18 12:04 UTC · model `20261017-bf741c` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-18T12:03:45+00:00 · API credits left: 165*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CBJ @ MIN (18:00 ET) | J. Greaves / J. Wallstedt (probable) | 59.0% | 62.8% | CBJ 2.60 | +3.8% | - | - | no bet |
| CAR @ CGY (20:00 ET) | F. Andersen / D. Wolf (probable) | 36.6% | 33.4% | CGY 2.87 | +3.2% | - | - | no bet |

## Why

**CBJ @ MIN (18:00 ET)**: No bet: early-season sample size: a team has only played 6 games (< 10)
  - Context: MIN have the clearly better goal differential this season (gap 0.57 goals/game). CBJ is on the second night of a back-to-back. CBJ are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**CAR @ CGY (20:00 ET)**: No bet: early-season sample size: a team has only played 7 games (< 10)
  - Context: CAR have the clearly better goal differential this season (gap 1.95 goals/game). CAR is on the second night of a back-to-back. CAR are creating the better share of expected goals recently. Goaltending edge to CAR (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **129** (bet on 0, passed on 100%); model log loss 0.6502, Brier 0.2292 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (129 games with odds): model log loss 0.6502 vs market 0.6322 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 41.9% | 34.6% | 26 |
| 50.3% | 53.8% | 26 |
| 54.4% | 52.0% | 25 |
| 59.5% | 61.5% | 26 |
| 66.3% | 80.8% | 26 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 123 | -8.3% | -25% to +8% | 48% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 78 | +5.1% | -17% to +27% | 54% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -35% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +33% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -70% to +57% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -23% to +26% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 129 | +0.1% | -15% to +16% | 60% | -2.13% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 128 | -11.5% | -25% to +2% | 58% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 123 | -7.3% | -24% to +8% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 129 | -20.5% | -37% to -3% | 40% | -2.49% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -49% to +32% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -78% to +91% | 10% | - |  |
| `home_ml_control` | 82 | -1.6% | -21% to +18% | 56% | -2.47% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 129 | +2.4% | -11% to +17% | 62% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 13 | +10.2% | - | 46% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 17 | -43.9% | - | 29% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 10 | +2.7% | - | 30% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 10 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 10 | -100.0% | - | 0% | - |  |
| `pts_edge` | 63 | -10.0% | -34% to +15% | 46% | - |  |
| `pts_over_control` | 67 | -14.7% | -35% to +6% | 52% | - |  |
| `pts_over_edge` | 55 | -31.1% | -63% to +5% | 25% | - |  |
| `pts_under_edge` | 63 | -9.0% | -32% to +15% | 49% | - |  |
| `puckline_dog_control` | 81 | -12.1% | -29% to +5% | 57% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 46 | -11.5% | -35% to +11% | 57% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 73 | -12.5% | -33% to +8% | 51% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 10 | +42.5% | - | 50% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 102 | -2.2% | -21% to +16% | 53% | - |  |
| `sog_over_control` | 106 | -0.8% | -21% to +19% | 48% | - |  |
| `sog_over_edge` | 54 | -14.8% | -40% to +10% | 44% | - |  |
| `sog_under_edge` | 64 | -4.7% | -28% to +18% | 50% | - |  |
| `totals_edge` | 66 | -16.3% | -40% to +7% | 44% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 32 | +2.3% | -32% to +33% | 53% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 44 | -42.0% | -70% to -10% | 25% | -2.48% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 82 | -20.1% | -43% to +3% | 35% | -2.71% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CBJ @ MIN | Moneyline | MIN moneyline | $10.00 | 1.56 | 59.0% | 62.8% |
| CAR @ CGY | Moneyline | CAR moneyline | $10.00 | 1.47 | 63.4% | 66.6% |

Total pretend stake $20.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
