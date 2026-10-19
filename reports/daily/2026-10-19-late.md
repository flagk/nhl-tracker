# NHL model report: 2026-10-19 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-19 20:04 UTC · model `20261018-fc2668` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-19T20:03:57+00:00 · API credits left: 156*

*Player props: 1 game(s) priced*

## Summary

- 4 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| ANA @ NYR (19:00 ET) | L. Dostal / I. Shesterkin (probable) | 53.3% | 58.2% | ANA 2.33 | +4.9% | - | - | no bet |
| COL @ PHI (19:00 ET) | S. Wedgewood / D. Vladar (probable) | 38.2% | 31.4% | PHI 3.06 | +6.8% | - | - | no bet |
| SJS @ TOR (19:30 ET) | Y. Askarov / A. Stolarz (probable) | 54.5% | 55.2% | SJS 2.20 | +0.8% | - | - | no bet |
| PIT @ UTA (21:00 ET) | A. Silovs / K. Vejmelka (probable) | 57.2% | 63.0% | PIT 2.63 | +5.8% | - | - | no bet |

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

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| PIT @ UTA (21:00 ET) | Total | Under 6.5 | 56.5% | 44.8% | +11.7 pts | +22.5% | 2.17 |
| SJS @ TOR (19:30 ET) | Total | Under 6.5 | 56.4% | 45.6% | +10.9 pts | +21.4% | 2.15 |
| ANA @ NYR (19:00 ET) | Total | Under 6.5 | 56.8% | 50.1% | +6.8 pts | +10.8% | 1.95 |
| ANA @ NYR (19:00 ET) | Puck line | ANA +1.5 | 69.2% | 63.6% | +5.6 pts | +6.5% | 1.54 |
| PIT @ UTA (21:00 ET) | Puck line | PIT +1.5 | 63.9% | 58.9% | +5.0 pts | +6.7% | 1.67 |
| SJS @ TOR (19:30 ET) | Puck line | SJS +1.5 | 69.4% | 66.1% | +3.3 pts | +2.7% | 1.48 |
| COL @ PHI (19:00 ET) | Puck line | COL -1.5 | 49.3% | 46.5% | +2.8 pts | +3.5% | 2.10 |
| COL @ PHI (19:00 ET) | Total | Over 6 | 51.7% | 51.1% | +0.6 pts | -1.2% | 1.91 |
| COL @ PHI (19:00 ET) | Total | Under 6 | 48.3% | 48.9% | -0.6 pts | -3.7% | 1.98 |
| COL @ PHI (19:00 ET) | Puck line | PHI +1.5 | 50.7% | 53.5% | -2.8 pts | -6.2% | 1.85 |
| SJS @ TOR (19:30 ET) | Puck line | TOR -1.5 | 30.6% | 33.9% | -3.3 pts | -13.0% | 2.84 |
| PIT @ UTA (21:00 ET) | Puck line | UTA -1.5 | 36.1% | 41.1% | -5.0 pts | -14.7% | 2.36 |
| ANA @ NYR (19:00 ET) | Puck line | NYR -1.5 | 30.8% | 36.4% | -5.6 pts | -16.7% | 2.70 |
| ANA @ NYR (19:00 ET) | Total | Over 6.5 | 43.2% | 49.9% | -6.8 pts | -14.9% | 1.97 |
| SJS @ TOR (19:30 ET) | Total | Over 6.5 | 43.6% | 54.4% | -10.9 pts | -21.2% | 1.81 |
| PIT @ UTA (21:00 ET) | Total | Over 6.5 | 43.5% | 55.2% | -11.7 pts | -22.9% | 1.77 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| A. Fox | ANA @ NYR (19:00 ET) | pts 0.5 | Under | 0.64 | 47.4% | 61.5% | +14.1 pts | +28.3% | 0 2 1 1 1 0 1 2 1 0 | 1.0 |
| P. Dorofeyev | ANA @ NYR (19:00 ET) | pts 0.5 | Under | 0.55 | 42.6% | 56.9% | +14.3 pts | +25.8% | 0 2 0 0 0 1 0 0 1 0 | 0.4 |
| C. Gauthier | ANA @ NYR (19:00 ET) | sog 3.5 | Under | 2.75 | 30.1% | 44.2% | +14.1 pts | +23.0% | 1 4 4 6 5 4 7 1 0 4 | 3.5 |
| J. Miller | ANA @ NYR (19:00 ET) | sog 1.5 | Under | 1.65 | 48.0% | 58.9% | +11.0 pts | +19.7% | 2 1 2 1 3 0 1 1 0 1 | 1.1 |
| P. Dorofeyev | ANA @ NYR (19:00 ET) | sog 2.5 | Under | 2.55 | 45.6% | 56.1% | +10.5 pts | +19.7% | 5 5 2 3 0 4 2 2 5 2 | 2.8 |
| E. Tolvanen | ANA @ NYR (19:00 ET) | pts 0.5 | Under | 0.31 | 26.5% | 38.1% | +11.7 pts | +18.4% | 0 1 1 1 0 0 1 0 0 0 | 0.4 |
| J. Miller | ANA @ NYR (19:00 ET) | pts 0.5 | Under | 0.72 | 51.2% | 60.8% | +9.6 pts | +17.2% | 0 0 0 1 1 1 2 1 0 0 | 0.7 |
| F. Vatrano | ANA @ NYR (19:00 ET) | pts 0.5 | Under | 0.22 | 19.5% | 33.9% | +14.5 pts | +16.0% | 0 0 0 0 1 0 0 1 0 0 | - |
| M. Zibanejad | ANA @ NYR (19:00 ET) | sog 2.5 | Under | 2.19 | 37.1% | 46.2% | +9.1 pts | +15.1% | 2 3 1 0 3 2 3 3 0 1 | 1.8 |
| A. Fox | ANA @ NYR (19:00 ET) | sog 1.5 | Under | 1.39 | 39.7% | 48.3% | +8.6 pts | +11.6% | 0 0 2 1 0 0 2 1 4 2 | 1.3 |
| M. Granlund | ANA @ NYR (19:00 ET) | sog 1.5 | Over | 2.10 | 60.0% | 52.0% | +8.0 pts | +9.7% | 0 3 3 3 0 2 2 2 4 2 | 2.0 |
| C. Gauthier | ANA @ NYR (19:00 ET) | pts 0.5 | Under | 0.60 | 45.2% | 53.3% | +8.1 pts | +9.6% | 1 2 1 1 1 3 0 0 0 1 | 0.8 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 125 | -9.8% | -26% to +7% | 47% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 80 | +7.3% | -13% to +29% | 55% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -36% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +32% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -71% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -23% to +26% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 131 | -0.2% | -15% to +14% | 60% | -2.12% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 130 | -11.6% | -24% to +2% | 58% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 125 | -7.3% | -24% to +10% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 131 | -19.6% | -36% to -1% | 40% | -2.52% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -49% to +30% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +99% | 10% | - |  |
| `home_ml_control` | 84 | +1.3% | -18% to +21% | 57% | -2.49% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 131 | +2.0% | -13% to +16% | 62% | -2.11% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 13 | +10.2% | - | 46% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 17 | -43.9% | - | 29% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 11 | -6.6% | - | 27% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 11 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 11 | -100.0% | - | 0% | - |  |
| `pts_edge` | 63 | -10.0% | -36% to +16% | 46% | - |  |
| `pts_over_control` | 67 | -14.7% | -35% to +4% | 52% | - |  |
| `pts_over_edge` | 55 | -31.1% | -63% to +4% | 25% | - |  |
| `pts_under_edge` | 63 | -9.0% | -33% to +14% | 49% | - |  |
| `puckline_dog_control` | 83 | -10.1% | -29% to +6% | 58% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 47 | -9.9% | -31% to +13% | 57% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 75 | -12.6% | -33% to +9% | 51% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 11 | +29.5% | - | 45% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 102 | -2.2% | -20% to +15% | 53% | - |  |
| `sog_over_control` | 106 | -0.8% | -21% to +19% | 48% | - |  |
| `sog_over_edge` | 54 | -14.8% | -39% to +11% | 44% | - |  |
| `sog_under_edge` | 64 | -4.7% | -28% to +19% | 50% | - |  |
| `totals_edge` | 67 | -14.7% | -35% to +8% | 45% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 33 | +4.8% | -26% to +38% | 55% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 46 | -38.3% | -66% to -8% | 26% | -2.57% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 84 | -18.6% | -43% to +6% | 36% | -2.76% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| ANA @ NYR | Moneyline | NYR moneyline | $10.00 | 1.70 | 53.3% | 58.2% |
| COL @ PHI | Moneyline | COL moneyline | $10.00 | 1.43 | 61.8% | 68.6% |
| SJS @ TOR | Moneyline | TOR moneyline | $10.00 | 1.77 | 54.5% | 55.2% |
| PIT @ UTA | Moneyline | UTA moneyline | $10.00 | 1.55 | 57.2% | 63.0% |
| ANA @ NYR | Total | Under 6.5 | $10.00 | 1.95 | 56.8% | 50.1% |
| COL @ PHI | Total | Over 6 | $10.00 | 1.91 | 51.7% | 51.1% |
| SJS @ TOR | Total | Under 6.5 | $10.00 | 2.15 | 56.4% | 45.6% |
| PIT @ UTA | Total | Under 6.5 | $10.00 | 2.17 | 56.5% | 44.8% |
| ANA @ NYR | Puck line | ANA +1.5 | $10.00 | 1.54 | 69.2% | 63.6% |
| COL @ PHI | Puck line | PHI +1.5 | $10.00 | 1.85 | 50.7% | 53.5% |
| SJS @ TOR | Puck line | SJS +1.5 | $10.00 | 1.48 | 69.4% | 66.1% |
| PIT @ UTA | Puck line | PIT +1.5 | $10.00 | 1.67 | 63.9% | 58.9% |

Total pretend stake $120.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
