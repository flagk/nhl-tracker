# NHL model report: 2026-10-11 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-11 18:03 UTC · model `20261010-b2a5f5` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-11T18:01:48+00:00 · API credits left: 269*

*Player props: 3 game(s) priced*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| SEA @ WSH (17:00 ET) | J. Daccord / L. Thompson (probable) | 60.6% | 66.2% | SEA 2.85 | +5.6% | - | - | no bet |
| VAN @ NYR (18:00 ET) | K. Lankinen / I. Shesterkin (probable) | 68.8% | 71.2% | VAN 3.36 | +2.4% | - | - | no bet |
| CAR @ PHI (19:00 ET) | F. Andersen / D. Vladar (probable) | 34.3% | 33.8% | PHI 2.88 | +0.5% | - | - | no bet |

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

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| SEA @ WSH (17:00 ET) | Puck line | SEA +1.5 | 66.8% | 56.4% | +10.4 pts | +16.3% | 1.74 |
| VAN @ NYR (18:00 ET) | Puck line | VAN +1.5 | 59.6% | 50.8% | +8.8 pts | +13.8% | 1.91 |
| SEA @ WSH (17:00 ET) | Total | Under 6.5 | 57.2% | 51.5% | +5.6 pts | +8.0% | 1.89 |
| CAR @ PHI (19:00 ET) | Total | Over 5.5 | 58.4% | 52.8% | +5.5 pts | +6.8% | 1.83 |
| VAN @ NYR (18:00 ET) | Total | Under 6 | 49.5% | 47.8% | +1.7 pts | -0.0% | 2.02 |
| CAR @ PHI (19:00 ET) | Puck line | PHI +1.5 | 57.3% | 57.2% | +0.0 pts | -2.0% | 1.71 |
| CAR @ PHI (19:00 ET) | Puck line | CAR -1.5 | 42.7% | 42.8% | -0.0 pts | -2.6% | 2.28 |
| VAN @ NYR (18:00 ET) | Total | Over 6 | 50.5% | 52.2% | -1.7 pts | -5.3% | 1.86 |
| CAR @ PHI (19:00 ET) | Total | Under 5.5 | 41.6% | 47.2% | -5.5 pts | -15.1% | 2.04 |
| SEA @ WSH (17:00 ET) | Total | Over 6.5 | 42.8% | 48.5% | -5.6 pts | -13.5% | 2.02 |
| VAN @ NYR (18:00 ET) | Puck line | NYR -1.5 | 40.4% | 49.2% | -8.8 pts | -21.2% | 1.95 |
| SEA @ WSH (17:00 ET) | Puck line | WSH -1.5 | 33.2% | 43.6% | -10.4 pts | -26.4% | 2.22 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| T. Sanheim | CAR @ PHI (19:00 ET) | goal 0.5 | Over | 0.22 | 19.5% | 9.7% | +9.8 pts | +133.5% | 0 0 1 0 0 1 0 0 0 0 | 0.2 |
| F. Gaudreau | SEA @ WSH (17:00 ET) | goal 0.5 | Over | 0.24 | 21.1% | 10.2% | +10.9 pts | +121.8% | 0 1 1 0 0 1 0 0 0 2 | 0.6 |
| N. Acciari | CAR @ PHI (19:00 ET) | goal 0.5 | Over | 0.17 | 15.9% | 8.5% | +7.4 pts | +106.6% | 0 0 1 0 0 0 0 0 0 0 | 0.0 |
| V. Gavrikov | VAN @ NYR (18:00 ET) | goal 0.5 | Over | 0.18 | 16.4% | 9.2% | +7.2 pts | +96.6% | 0 0 0 0 0 0 0 1 0 0 | 0.2 |
| N. Seeler | CAR @ PHI (19:00 ET) | goal 0.5 | Over | 0.10 | 9.6% | 6.2% | +3.4 pts | +92.9% | 0 1 0 0 0 0 0 0 0 0 | 0.0 |
| M. Pettersson | VAN @ NYR (18:00 ET) | goal 0.5 | Over | 0.11 | 10.7% | 6.8% | +3.9 pts | +92.4% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| A. Larsson | SEA @ WSH (17:00 ET) | goal 0.5 | Over | 0.13 | 11.7% | 7.5% | +4.2 pts | +87.9% | 0 0 1 0 0 0 0 0 0 0 | 0.0 |
| T. Liljegren | SEA @ WSH (17:00 ET) | goal 0.5 | Over | 0.13 | 12.5% | 7.8% | +4.7 pts | +87.4% | 1 0 0 0 0 0 0 0 0 1 | - |
| M. Fehérváry | SEA @ WSH (17:00 ET) | goal 0.5 | Over | 0.10 | 9.7% | 6.7% | +3.0 pts | +64.4% | 0 0 1 0 0 0 0 0 0 0 | - |
| J. Oleksiak | VAN @ NYR (18:00 ET) | goal 0.5 | Over | 0.10 | 9.7% | 6.6% | +3.0 pts | +64.2% | 0 0 0 0 0 1 0 0 0 0 | 0.2 |
| K. Miller | CAR @ PHI (19:00 ET) | goal 0.5 | Over | 0.21 | 18.8% | 12.1% | +6.8 pts | +55.4% | 0 1 1 0 0 0 0 0 1 0 | 0.2 |
| M. Jankowski | CAR @ PHI (19:00 ET) | goal 0.5 | Over | 0.24 | 21.2% | 14.6% | +6.5 pts | +48.1% | 0 2 0 1 0 0 0 1 1 1 | 0.6 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 76 | -11.8% | -34% to +11% | 46% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 31 | +20.4% | -13% to +53% | 61% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 32 | -9.9% | -44% to +24% | 53% | - |  |
| `ast_over_control` | 32 | +4.4% | -28% to +37% | 56% | - |  |
| `ast_over_edge` | 28 | -71.1% | -100% to -15% | 7% | - |  |
| `ast_under_edge` | 32 | +3.5% | -29% to +37% | 66% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 78 | +7.2% | -11% to +24% | 60% | -2.06% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 78 | -12.9% | -30% to +5% | 56% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 76 | -6.2% | -28% to +17% | 50% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 78 | -12.1% | -39% to +16% | 45% | -2.61% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `goal_control` | 32 | -24.1% | -66% to +21% | 28% | - |  |
| `goal_edge` | 31 | -55.6% | -100% to +20% | 6% | - |  |
| `home_ml_control` | 31 | -2.9% | -34% to +30% | 55% | -2.35% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 78 | +1.2% | -17% to +19% | 62% | -2.14% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 12 | +15.7% | - | 50% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 10 | -22.8% | - | 40% | - | Goals model: $5-$30 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 3 | +30.7% | - | 33% | - |  |
| `parlay_players3` | 3 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 3 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 3 | -100.0% | - | 0% | - |  |
| `pts_edge` | 31 | -15.5% | -52% to +23% | 48% | - |  |
| `pts_over_control` | 35 | -21.1% | -51% to +9% | 46% | - |  |
| `pts_over_edge` | 32 | -70.0% | -95% to -36% | 12% | - |  |
| `pts_under_edge` | 31 | -2.3% | -38% to +35% | 58% | - |  |
| `puckline_dog_control` | 31 | -19.5% | -46% to +8% | 52% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 19 | -34.8% | - | 47% | - | Goals model: $5-$30 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 46 | -25.0% | -49% to -1% | 48% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 2 | +203.8% | - | 100% | - | Goals model: $5-$30 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 72 | -4.5% | -27% to +17% | 56% | - |  |
| `sog_over_control` | 74 | +10.7% | -14% to +34% | 53% | - |  |
| `sog_over_edge` | 31 | +4.5% | -35% to +40% | 52% | - |  |
| `sog_under_edge` | 34 | -14.9% | -45% to +18% | 53% | - |  |
| `totals_edge` | 45 | -11.3% | -39% to +18% | 49% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 14 | +7.6% | - | 64% | - | Goals model: $5-$30 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 16 | -37.9% | - | 31% | -2.40% | $5-$30 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 31 | -24.0% | -62% to +15% | 32% | -2.62% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| SEA @ WSH | Moneyline | WSH moneyline | $16.00 | 1.48 | 60.6% | 66.2% |
| VAN @ NYR | Moneyline | NYR moneyline | $24.00 | 1.38 | 68.8% | 71.2% |
| CAR @ PHI | Moneyline | CAR moneyline | $21.00 | 1.48 | 65.7% | 66.2% |
| SEA @ WSH | Total | Under 6.5 | $8.00 | 1.89 | 57.2% | 51.5% |
| VAN @ NYR | Total | Over 6 | $7.00 | 1.86 | 50.5% | 52.2% |
| CAR @ PHI | Total | Over 5.5 | $9.00 | 1.83 | 58.4% | 52.8% |
| SEA @ WSH | Puck line | SEA +1.5 | $14.00 | 1.74 | 66.8% | 56.4% |
| VAN @ NYR | Puck line | VAN +1.5 | $8.00 | 1.91 | 59.6% | 50.8% |
| CAR @ PHI | Puck line | PHI +1.5 | $12.00 | 1.71 | 57.3% | 57.2% |

Total pretend stake $119.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
