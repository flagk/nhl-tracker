# NHL model report: 2026-10-14 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-14 20:33 UTC · model `20261013-38218a` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-14T20:32:18+00:00 · API credits left: 200*

*Player props: 2 game(s) priced*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| MTL @ WSH (19:30 ET) | J. Dobes / L. Thompson (probable) | 53.5% | 52.3% | WSH 1.88 | +1.3% | - | - | no bet |
| DAL @ COL (20:30 ET) | J. Oettinger / S. Wedgewood (probable) | 57.1% | 55.9% | COL 1.75 | +1.2% | - | - | no bet |

## Why

**MTL @ WSH (19:30 ET)**: No bet: early-season sample size: a team has only played 6 games (< 10)
  - Context: WSH and MTL are on the second night of a back-to-back. Goaltending edge to WSH (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**DAL @ COL (20:30 ET)**: No bet: early-season sample size: a team has only played 6 games (< 10)
  - Context: COL have the clearly better goal differential this season (gap 0.60 goals/game). COL and DAL are on the second night of a back-to-back. Goaltending edge to DAL (goals saved above expected). Division/rival game.

## Track record (all logged recommendations that have resolved)

- Resolved games: **100** (bet on 0, passed on 100%); model log loss 0.6584, Brier 0.2332 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (100 games with odds): model log loss 0.6584 vs market 0.6431 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 43.4% | 36.0% | 25 |
| 51.5% | 52.0% | 25 |
| 57.8% | 56.0% | 25 |
| 65.5% | 72.0% | 25 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| MTL @ WSH (19:30 ET) | Total | Under 6.5 | 56.1% | 48.6% | +7.6 pts | +12.9% | 2.01 |
| DAL @ COL (20:30 ET) | Total | Over 6 | 50.7% | 49.4% | +1.3 pts | -1.0% | 1.95 |
| MTL @ WSH (19:30 ET) | Puck line | MTL +1.5 | 69.4% | 68.7% | +0.8 pts | -0.7% | 1.43 |
| DAL @ COL (20:30 ET) | Puck line | COL -1.5 | 33.5% | 33.4% | +0.0 pts | -3.6% | 2.88 |
| DAL @ COL (20:30 ET) | Puck line | DAL +1.5 | 66.5% | 66.6% | -0.0 pts | -0.8% | 1.49 |
| MTL @ WSH (19:30 ET) | Puck line | WSH -1.5 | 30.6% | 31.3% | -0.8 pts | -5.9% | 3.08 |
| DAL @ COL (20:30 ET) | Total | Under 6 | 49.3% | 50.6% | -1.3 pts | -5.1% | 1.91 |
| MTL @ WSH (19:30 ET) | Total | Over 6.5 | 43.9% | 51.4% | -7.6 pts | -17.6% | 1.88 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| J. Benn | DAL @ COL (20:30 ET) | sog 1.5 | Under | 1.03 | 27.5% | 46.8% | +19.2 pts | +29.7% | 0 0 1 0 0 2 4 0 1 1 | 1.3 |
| C. Kreider | MTL @ WSH (19:30 ET) | sog 1.5 | Under | 1.36 | 38.9% | 53.2% | +14.3 pts | +25.3% | 2 3 1 1 1 1 1 0 2 0 | 0.8 |
| R. Hintz | DAL @ COL (20:30 ET) | pts 0.5 | Under | 0.57 | 43.5% | 56.9% | +13.5 pts | +24.4% | 0 1 1 0 0 0 2 1 0 1 | 0.7 |
| N. Dobson | MTL @ WSH (19:30 ET) | sog 1.5 | Under | 1.62 | 47.1% | 58.8% | +11.7 pts | +21.7% | 1 3 2 0 1 1 2 3 1 2 | 1.8 |
| M. Rantanen | DAL @ COL (20:30 ET) | pts 0.5 | Under | 0.82 | 55.8% | 66.4% | +10.6 pts | +19.3% | 2 0 1 0 0 1 1 4 0 0 | 1.0 |
| M. Heiskanen | DAL @ COL (20:30 ET) | sog 1.5 | Under | 1.54 | 44.4% | 54.2% | +9.8 pts | +19.0% | 1 1 1 0 1 5 2 1 0 1 | 1.7 |
| N. Kadri | DAL @ COL (20:30 ET) | pts 0.5 | Under | 0.50 | 39.0% | 52.1% | +13.0 pts | +18.9% | 0 0 0 0 2 0 0 1 0 0 | 0.5 |
| T. Hughes | DAL @ COL (20:30 ET) | pts 0.5 | Under | 0.34 | 28.8% | 42.4% | +13.6 pts | +18.8% | 0 0 0 1 0 0 | 0.2 |
| E. Frank | MTL @ WSH (19:30 ET) | pts 0.5 | Under | 0.16 | 14.8% | 31.3% | +16.5 pts | +18.5% | 0 0 0 0 0 0 0 0 0 0 | - |
| S. Malinski | DAL @ COL (20:30 ET) | sog 1.5 | Under | 1.59 | 46.0% | 57.5% | +11.5 pts | +18.3% | 2 0 4 2 4 2 1 1 0 0 | 0.8 |
| D. Strome | MTL @ WSH (19:30 ET) | pts 0.5 | Under | 0.62 | 45.9% | 57.2% | +11.2 pts | +17.9% | 2 1 0 0 2 0 1 1 0 1 | 0.8 |
| T. Seguin | DAL @ COL (20:30 ET) | sog 1.5 | Under | 1.68 | 48.7% | 58.6% | +10.0 pts | +16.5% | 1 0 1 1 4 3 2 0 2 7 | 3.0 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 96 | -8.2% | -27% to +10% | 48% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 51 | +7.0% | -20% to +33% | 55% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -35% to +21% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +34% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +26% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 100 | -2.0% | -18% to +13% | 58% | -2.14% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 100 | -14.5% | -30% to +1% | 56% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 96 | -7.1% | -27% to +12% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 100 | -11.8% | -31% to +9% | 44% | -2.63% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -48% to +32% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +92% | 10% | - |  |
| `home_ml_control` | 53 | -8.4% | -33% to +17% | 51% | -2.69% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 100 | -0.5% | -17% to +16% | 60% | -2.11% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 12 | +19.4% | - | 50% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 13 | -26.7% | - | 38% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 6 | -34.7% | - | 17% | - |  |
| `parlay_players3` | 5 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 6 | -100.0% | - | 0% | - |  |
| `pts_edge` | 56 | -12.7% | -38% to +14% | 45% | - |  |
| `pts_over_control` | 59 | -8.2% | -29% to +14% | 56% | - |  |
| `pts_over_edge` | 47 | -24.0% | -61% to +15% | 28% | - |  |
| `pts_under_edge` | 56 | -11.6% | -34% to +12% | 48% | - |  |
| `puckline_dog_control` | 53 | -18.0% | -39% to +3% | 53% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 28 | -11.1% | -40% to +17% | 57% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 54 | -5.4% | -30% to +20% | 54% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 4 | +200.0% | - | 100% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 95 | +1.1% | -17% to +20% | 55% | - |  |
| `sog_over_control` | 99 | +2.4% | -19% to +23% | 49% | - |  |
| `sog_over_edge` | 50 | -16.4% | -41% to +10% | 44% | - |  |
| `sog_under_edge` | 57 | +0.5% | -26% to +25% | 53% | - |  |
| `totals_edge` | 51 | -10.4% | -36% to +15% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 18 | +7.3% | - | 56% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 25 | -24.4% | -64% to +21% | 32% | -2.73% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 53 | -13.2% | -44% to +19% | 38% | -2.73% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| MTL @ WSH | Moneyline | WSH moneyline | $10.00 | 1.88 | 53.5% | 52.3% |
| DAL @ COL | Moneyline | COL moneyline | $10.00 | 1.75 | 57.1% | 55.9% |
| MTL @ WSH | Total | Under 6.5 | $10.00 | 2.01 | 56.1% | 48.6% |
| DAL @ COL | Total | Over 6 | $10.00 | 1.95 | 50.7% | 49.4% |
| MTL @ WSH | Puck line | MTL +1.5 | $10.00 | 1.43 | 69.4% | 68.7% |
| DAL @ COL | Puck line | DAL +1.5 | $10.00 | 1.49 | 66.5% | 66.6% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
