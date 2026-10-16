# NHL model report: 2026-10-16 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-16 22:02 UTC · model `20261015-77bd53` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-16T22:01:33+00:00 · API credits left: 180*

*Player props: 0 game(s) priced*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PIT @ COL (21:00 ET) | A. Silovs / S. Wedgewood (probable) | 60.8% | 65.6% | PIT 2.85 | +4.9% | - | - | no bet |
| BOS @ ANA (22:00 ET) | J. Swayman / L. Dostal (probable) | 55.3% | 49.9% | ANA 1.96 | +5.4% | - | - | no bet |

## Why

**PIT @ COL (21:00 ET)**: No bet: early-season sample size: a team has only played 7 games (< 10)
  - Context: COL have the clearly better goal differential this season (gap 0.58 goals/game). Early in the season: team ratings lean on last year's results and are less reliable.

**BOS @ ANA (22:00 ET)**: No bet: early-season sample size: a team has only played 6 games (< 10)
  - Context: BOS have the clearly better goal differential this season (gap 0.43 goals/game). ANA are creating the better share of expected goals recently. Goaltending edge to BOS (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **113** (bet on 0, passed on 100%); model log loss 0.6533, Brier 0.2307 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (113 games with odds): model log loss 0.6533 vs market 0.6376 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 43.7% | 37.9% | 29 |
| 51.9% | 57.1% | 28 |
| 57.8% | 53.6% | 28 |
| 65.5% | 75.0% | 28 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| PIT @ COL (21:00 ET) | Total | Under 6.5 | 56.1% | 46.5% | +9.5 pts | +17.2% | 2.09 |
| PIT @ COL (21:00 ET) | Puck line | PIT +1.5 | 63.8% | 56.2% | +7.6 pts | +11.0% | 1.74 |
| BOS @ ANA (22:00 ET) | Total | Under 6.5 | 55.7% | 51.0% | +4.7 pts | +7.5% | 1.93 |
| BOS @ ANA (22:00 ET) | Puck line | ANA +1.5 | 73.8% | 71.3% | +2.5 pts | +3.3% | 1.40 |
| BOS @ ANA (22:00 ET) | Puck line | BOS -1.5 | 26.2% | 28.7% | -2.5 pts | -11.8% | 3.37 |
| BOS @ ANA (22:00 ET) | Total | Over 6.5 | 44.3% | 49.0% | -4.7 pts | -10.5% | 2.02 |
| PIT @ COL (21:00 ET) | Puck line | COL -1.5 | 36.2% | 43.8% | -7.6 pts | -19.6% | 2.22 |
| PIT @ COL (21:00 ET) | Total | Over 6.5 | 43.9% | 53.5% | -9.5 pts | -20.1% | 1.82 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 108 | -6.0% | -24% to +13% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 63 | +2.3% | -22% to +27% | 52% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -34% to +22% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +35% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +26% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 113 | -0.2% | -16% to +16% | 59% | -2.18% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 112 | -11.0% | -26% to +3% | 58% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 108 | -6.8% | -24% to +11% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 113 | -14.8% | -33% to +5% | 42% | -2.60% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -48% to +31% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +95% | 10% | - |  |
| `home_ml_control` | 66 | -3.8% | -25% to +19% | 55% | -2.55% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 113 | +1.1% | -14% to +16% | 61% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 12 | +19.4% | - | 50% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 15 | -36.5% | - | 33% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 8 | -51.0% | - | 12% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 8 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 8 | -100.0% | - | 0% | - |  |
| `pts_edge` | 63 | -10.0% | -34% to +15% | 46% | - |  |
| `pts_over_control` | 67 | -14.7% | -35% to +5% | 52% | - |  |
| `pts_over_edge` | 55 | -31.1% | -63% to +4% | 25% | - |  |
| `pts_under_edge` | 63 | -9.0% | -32% to +15% | 49% | - |  |
| `puckline_dog_control` | 65 | -11.3% | -30% to +8% | 57% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 36 | -3.4% | -30% to +21% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 63 | -5.6% | -28% to +18% | 54% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 6 | +100.0% | - | 67% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 102 | -2.2% | -20% to +16% | 53% | - |  |
| `sog_over_control` | 106 | -0.8% | -22% to +20% | 48% | - |  |
| `sog_over_edge` | 54 | -14.8% | -42% to +10% | 44% | - |  |
| `sog_under_edge` | 64 | -4.7% | -28% to +20% | 50% | - |  |
| `totals_edge` | 57 | -13.0% | -37% to +13% | 46% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 23 | +0.8% | -41% to +37% | 52% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 34 | -31.3% | -66% to +5% | 29% | -2.66% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 66 | -17.1% | -42% to +9% | 36% | -2.69% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PIT @ COL | Moneyline | COL moneyline | $10.00 | 1.49 | 60.8% | 65.6% |
| BOS @ ANA | Moneyline | ANA moneyline | $10.00 | 1.96 | 55.3% | 49.9% |
| PIT @ COL | Total | Under 6.5 | $10.00 | 2.09 | 56.1% | 46.5% |
| BOS @ ANA | Total | Under 6.5 | $10.00 | 1.93 | 55.7% | 51.0% |
| PIT @ COL | Puck line | PIT +1.5 | $10.00 | 1.74 | 63.8% | 56.2% |
| BOS @ ANA | Puck line | ANA +1.5 | $10.00 | 1.40 | 73.8% | 71.3% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
