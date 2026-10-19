# NHL model report: 2026-10-19 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-19 12:04 UTC · model `20261018-fc2668` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-19T12:03:40+00:00 · API credits left: 159*

## Summary

- 4 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| ANA @ NYR (19:00 ET) | L. Dostal / I. Shesterkin (probable) | 53.3% | 56.2% | ANA 2.22 | +2.9% | - | - | no bet |
| COL @ PHI (19:00 ET) | S. Wedgewood / D. Vladar (probable) | 38.2% | 31.8% | PHI 3.06 | +6.4% | - | - | no bet |
| SJS @ TOR (19:30 ET) | Y. Askarov / A. Stolarz (probable) | 54.5% | 56.6% | SJS 2.22 | +2.1% | - | - | no bet |
| PIT @ UTA (21:00 ET) | A. Silovs / K. Vejmelka (probable) | 57.2% | 62.4% | PIT 2.56 | +5.2% | - | - | no bet |

## Why

**ANA @ NYR (19:00 ET)**: No bet: early-season sample size: a team has only played 7 games (< 10)
  - Context: ANA travelled 3,917 km across 3 time zone(s) since their last game. Goaltending edge to NYR (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**COL @ PHI (19:00 ET)**: No bet: early-season sample size: a team has only played 8 games (< 10)
  - Context: COL have the clearly better goal differential this season (gap 1.67 goals/game). COL travelled 2,536 km across 2 time zone(s) since their last game. COL are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**SJS @ TOR (19:30 ET)**: No bet: early-season sample size: a team has only played 8 games (< 10)
  - Context: Early in the season: team ratings lean on last year's results and are less reliable.

**PIT @ UTA (21:00 ET)**: No bet: early-season sample size: a team has only played 9 games (< 10)
  - Context: UTA have the clearly better goal differential this season (gap 0.43 goals/game). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **131** (bet on 0, passed on 100%); model log loss 0.6520, Brier 0.2300 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (131 games with odds): model log loss 0.6520 vs market 0.6345 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 41.7% | 37.0% | 27 |
| 50.3% | 53.8% | 26 |
| 54.5% | 50.0% | 26 |
| 59.6% | 65.4% | 26 |
| 66.3% | 80.8% | 26 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 125 | -9.8% | -26% to +7% | 47% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 80 | +7.3% | -15% to +28% | 55% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -34% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +34% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +54% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +27% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 131 | -0.2% | -14% to +14% | 60% | -2.12% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 130 | -11.6% | -25% to +1% | 58% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 125 | -7.3% | -23% to +10% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 131 | -19.6% | -37% to -2% | 40% | -2.52% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -50% to +29% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +96% | 10% | - |  |
| `home_ml_control` | 84 | +1.3% | -19% to +20% | 57% | -2.49% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 131 | +2.0% | -12% to +15% | 62% | -2.11% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 13 | +10.2% | - | 46% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 17 | -43.9% | - | 29% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 11 | -6.6% | - | 27% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 11 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 11 | -100.0% | - | 0% | - |  |
| `pts_edge` | 63 | -10.0% | -35% to +17% | 46% | - |  |
| `pts_over_control` | 67 | -14.7% | -35% to +4% | 52% | - |  |
| `pts_over_edge` | 55 | -31.1% | -62% to +4% | 25% | - |  |
| `pts_under_edge` | 63 | -9.0% | -33% to +14% | 49% | - |  |
| `puckline_dog_control` | 83 | -10.1% | -27% to +7% | 58% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 47 | -9.9% | -32% to +13% | 57% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 75 | -12.6% | -33% to +8% | 51% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 11 | +29.5% | - | 45% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 102 | -2.2% | -20% to +16% | 53% | - |  |
| `sog_over_control` | 106 | -0.8% | -21% to +19% | 48% | - |  |
| `sog_over_edge` | 54 | -14.8% | -40% to +11% | 44% | - |  |
| `sog_under_edge` | 64 | -4.7% | -29% to +20% | 50% | - |  |
| `totals_edge` | 67 | -14.7% | -35% to +9% | 45% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 33 | +4.8% | -29% to +38% | 55% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 46 | -38.3% | -67% to -6% | 26% | -2.57% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 84 | -18.6% | -41% to +5% | 36% | -2.76% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| ANA @ NYR | Moneyline | NYR moneyline | $10.00 | 1.72 | 53.3% | 56.2% |
| COL @ PHI | Moneyline | COL moneyline | $10.00 | 1.45 | 61.8% | 68.2% |
| SJS @ TOR | Moneyline | TOR moneyline | $10.00 | 1.72 | 54.5% | 56.6% |
| PIT @ UTA | Moneyline | UTA moneyline | $10.00 | 1.57 | 57.2% | 62.4% |

Total pretend stake $40.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
