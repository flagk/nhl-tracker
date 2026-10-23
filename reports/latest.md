# NHL model report: 2026-10-23 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-23 21:02 UTC · model `20261022-2f1b49` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-23T21:02:21+00:00 · API credits left: 120*

*Player props: 0 game(s) priced*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| MTL @ CHI (20:00 ET) | J. Dobes / S. Knight (probable) | 42.3% | 40.2% | CHI 2.42 | +2.1% | - | - | no bet |
| TBL @ COL (21:00 ET) | A. Vasilevskiy / S. Wedgewood (probable) | 52.5% | 52.7% | TBL 2.05 | +0.2% | - | - | no bet |

## Why

**MTL @ CHI (20:00 ET)**: No bet: early-season sample size: a team has only played 9 games (< 10)
  - Context: MTL have the clearly better goal differential this season (gap 0.77 goals/game). MTL are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**TBL @ COL (21:00 ET)**: No bet: early-season sample size: a team has only played 9 games (< 10)
  - Context: Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **160** (bet on 0, passed on 100%); model log loss 0.6617, Brier 0.2347 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (160 games with odds): model log loss 0.6617 vs market 0.6475 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 40.0% | 44.4% | 27 |
| 48.7% | 55.6% | 27 |
| 52.7% | 61.5% | 26 |
| 56.5% | 51.8% | 27 |
| 60.4% | 73.1% | 26 |
| 66.7% | 70.4% | 27 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| TBL @ COL (21:00 ET) | Total | Under 6.5 | 55.7% | 52.4% | +3.3 pts | +2.5% | 1.84 |
| MTL @ CHI (20:00 ET) | Total | Under 6.5 | 54.9% | 51.6% | +3.3 pts | +4.8% | 1.91 |
| MTL @ CHI (20:00 ET) | Puck line | MTL -1.5 | 39.1% | 37.4% | +1.7 pts | +3.5% | 2.65 |
| TBL @ COL (21:00 ET) | Puck line | COL -1.5 | 31.6% | 31.5% | +0.1 pts | -3.6% | 3.05 |
| TBL @ COL (21:00 ET) | Puck line | TBL +1.5 | 68.4% | 68.5% | -0.1 pts | -1.5% | 1.44 |
| MTL @ CHI (20:00 ET) | Puck line | CHI +1.5 | 60.9% | 62.6% | -1.7 pts | -4.3% | 1.57 |
| MTL @ CHI (20:00 ET) | Total | Over 6.5 | 45.1% | 48.4% | -3.3 pts | -8.0% | 2.04 |
| TBL @ COL (21:00 ET) | Total | Over 6.5 | 44.3% | 47.6% | -3.3 pts | -10.1% | 2.03 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 153 | -6.1% | -21% to +9% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 108 | +1.7% | -17% to +20% | 52% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -35% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -25% to +35% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -68% to +53% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -21% to +26% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 160 | -2.5% | -16% to +11% | 58% | -2.07% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 159 | -6.2% | -18% to +5% | 61% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 153 | -14.0% | -28% to +1% | 45% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 160 | -19.4% | -35% to -4% | 39% | -2.57% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -52% to +31% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +97% | 10% | - |  |
| `home_ml_control` | 113 | +9.6% | -8% to +27% | 60% | -2.61% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 160 | +0.5% | -13% to +13% | 61% | -2.05% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 14 | +2.4% | - | 43% | -1.26% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 2 | -4.5% | - | 50% | -4.14% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 24 | -44.6% | -77% to -12% | 29% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 15 | -31.5% | - | 20% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 15 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 15 | -100.0% | - | 0% | - |  |
| `pts_edge` | 79 | -12.2% | -34% to +9% | 46% | - |  |
| `pts_over_control` | 83 | -12.5% | -30% to +6% | 53% | - |  |
| `pts_over_edge` | 62 | -38.9% | -66% to -8% | 23% | - |  |
| `pts_under_edge` | 79 | -9.2% | -30% to +11% | 49% | - |  |
| `puckline_dog_control` | 112 | -4.5% | -19% to +10% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 55 | -8.3% | -29% to +12% | 58% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 86 | -16.4% | -34% to +4% | 49% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 18 | -10.6% | - | 33% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 126 | +1.7% | -15% to +17% | 56% | - |  |
| `sog_over_control` | 130 | -5.4% | -23% to +12% | 46% | - |  |
| `sog_over_edge` | 63 | -18.4% | -42% to +5% | 43% | - |  |
| `sog_under_edge` | 88 | +3.8% | -17% to +23% | 56% | - |  |
| `totals_edge` | 89 | -22.3% | -42% to -2% | 40% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 50 | -14.3% | -43% to +13% | 44% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 57 | -27.7% | -55% to +2% | 30% | -2.74% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 113 | -12.0% | -33% to +9% | 38% | -2.97% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| MTL @ CHI | Moneyline | MTL moneyline | $10.00 | 1.67 | 57.7% | 59.8% |
| TBL @ COL | Moneyline | COL moneyline | $10.00 | 1.87 | 52.5% | 52.7% |
| MTL @ CHI | Total | Under 6.5 | $10.00 | 1.91 | 54.9% | 51.6% |
| TBL @ COL | Total | Under 6.5 | $10.00 | 1.84 | 55.7% | 52.4% |
| MTL @ CHI | Puck line | CHI +1.5 | $10.00 | 1.57 | 60.9% | 62.6% |
| TBL @ COL | Puck line | TBL +1.5 | $10.00 | 1.44 | 68.4% | 68.5% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
