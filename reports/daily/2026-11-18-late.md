# NHL model report: 2026-11-18 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-18 21:32 UTC · model `20261117-9b04ec` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-18T21:31:43+00:00 · API credits left: 212*

*Player props: 3 game(s) priced*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| BOS @ DET (19:30 ET) | J. Swayman / J. Gibson (probable) | 56.4% | 49.7% | DET 1.96 | +6.7% | - | - | no bet |
| MTL @ NJD (19:30 ET) | J. Fowler / J. Allen (probable) | 57.2% | 60.6% | MTL 2.50 | +3.4% | - | - | no bet |
| COL @ UTA (22:00 ET) | M. Blackwood / K. Vejmelka (probable) | 48.2% | 49.9% | COL 1.93 | +1.7% | - | - | no bet |

## Why

**BOS @ DET (19:30 ET)**: No bet: expected value after shrinkage toward the market is only +0.2% (< 1%)
  - Context: BOS have the clearly better goal differential this season (gap 0.74 goals/game). BOS is on the second night of a back-to-back. Goaltending edge to BOS (goals saved above expected). Division/rival game.

**MTL @ NJD (19:30 ET)**: No bet: edge +3.4% below the 4% minimum (starting goalie unconfirmed)
  - Context: NJD have the clearly better goal differential this season (gap 0.28 goals/game). NJD are creating the better share of expected goals recently.

**COL @ UTA (22:00 ET)**: No bet: edge +1.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: COL have the clearly better goal differential this season (gap 0.90 goals/game). COL travelled 3,124 km across 2 time zone(s) since their last game. UTA are creating the better share of expected goals recently. Goaltending edge to COL (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **338** (bet on 11, passed on 97%); model log loss 0.6633, Brier 0.2355 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (338 games with odds): model log loss 0.6633 vs market 0.6486 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.47% to -0.58%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.7% | 41.9% | 43 |
| 46.6% | 38.1% | 42 |
| 49.9% | 40.5% | 42 |
| 52.6% | 54.8% | 42 |
| 55.1% | 47.6% | 42 |
| 58.2% | 61.9% | 42 |
| 61.6% | 66.7% | 42 |
| 68.5% | 74.4% | 43 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| COL @ UTA (22:00 ET) | Puck line | COL +1.5 | 74.8% | 70.6% | +4.2 pts | +4.7% | 1.40 |
| COL @ UTA (22:00 ET) | Total | Under 6.5 | 55.9% | 52.6% | +3.3 pts | +3.4% | 1.85 |
| BOS @ DET (19:30 ET) | Total | Over 6 | 51.7% | 50.1% | +1.7 pts | -0.2% | 1.93 |
| MTL @ NJD (19:30 ET) | Puck line | MTL +1.5 | 62.8% | 61.8% | +1.0 pts | -1.4% | 1.57 |
| BOS @ DET (19:30 ET) | Puck line | BOS -1.5 | 30.0% | 29.7% | +0.3 pts | -4.0% | 3.20 |
| MTL @ NJD (19:30 ET) | Total | Under 6 | 48.2% | 48.1% | +0.1 pts | -3.1% | 2.00 |
| MTL @ NJD (19:30 ET) | Total | Over 6 | 51.8% | 51.9% | -0.1 pts | -3.2% | 1.86 |
| BOS @ DET (19:30 ET) | Puck line | DET +1.5 | 70.0% | 70.3% | -0.3 pts | -2.0% | 1.40 |
| MTL @ NJD (19:30 ET) | Puck line | NJD -1.5 | 37.2% | 38.2% | -1.0 pts | -5.2% | 2.55 |
| BOS @ DET (19:30 ET) | Total | Under 6 | 48.3% | 49.9% | -1.7 pts | -6.0% | 1.93 |
| COL @ UTA (22:00 ET) | Total | Over 6.5 | 44.1% | 47.4% | -3.3 pts | -8.2% | 2.08 |
| COL @ UTA (22:00 ET) | Puck line | UTA -1.5 | 25.2% | 29.4% | -4.2 pts | -18.2% | 3.24 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| P. Danault | MTL @ NJD (19:30 ET) | pts 0.5 | Over | 0.49 | 38.5% | 27.2% | +11.3 pts | +28.3% | 0 2 1 0 0 0 0 1 0 1 | 0.4 |
| A. Peeke | COL @ UTA (22:00 ET) | pts 0.5 | Over | 0.34 | 28.5% | 19.7% | +8.8 pts | +25.4% | 0 1 0 1 0 0 2 0 0 0 | 0.4 |
| B. Pesce | MTL @ NJD (19:30 ET) | pts 0.5 | Over | 0.43 | 34.6% | 24.5% | +10.1 pts | +24.6% | 1 1 1 0 0 0 2 1 0 0 | 0.4 |
| S. Malinski | COL @ UTA (22:00 ET) | sog 1.5 | Under | 1.55 | 44.7% | 58.0% | +13.3 pts | +24.4% | 0 3 3 2 1 3 0 2 4 1 | 1.5 |
| B. Burns | COL @ UTA (22:00 ET) | sog 1.5 | Under | 1.23 | 34.4% | 47.7% | +13.3 pts | +20.1% | 1 4 1 3 0 2 2 1 2 0 | 1.5 |
| E. Rodrigues | MTL @ NJD (19:30 ET) | pts 0.5 | Over | 0.59 | 44.8% | 35.1% | +9.7 pts | +18.3% | 0 0 0 0 1 1 0 0 2 2 | 0.5 |
| N. Kadri | COL @ UTA (22:00 ET) | sog 2.5 | Under | 2.57 | 46.2% | 54.2% | +8.1 pts | +15.8% | 3 2 2 4 1 3 3 5 5 4 | 2.9 |
| C. McAvoy | BOS @ DET (19:30 ET) | pts 0.5 | Under | 0.52 | 40.7% | 50.9% | +10.1 pts | +15.6% | 0 0 0 2 0 0 1 0 0 0 | 0.6 |
| A. DeBrincat | BOS @ DET (19:30 ET) | pts 0.5 | Under | 0.74 | 52.1% | 61.0% | +8.9 pts | +15.3% | 1 2 0 1 0 0 0 0 0 0 | 0.8 |
| M. Geekie | BOS @ DET (19:30 ET) | pts 0.5 | Under | 0.53 | 41.0% | 51.7% | +10.7 pts | +15.1% | 1 0 0 1 1 2 1 0 1 0 | 0.8 |
| C. Caufield | MTL @ NJD (19:30 ET) | pts 0.5 | Under | 0.74 | 52.5% | 61.1% | +8.6 pts | +15.0% | 0 2 0 2 1 1 1 1 1 0 | 0.9 |
| L. Hutson | MTL @ NJD (19:30 ET) | pts 0.5 | Under | 0.65 | 47.9% | 56.1% | +8.2 pts | +14.6% | 0 1 2 1 2 1 1 0 0 1 | 0.9 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 325 | -2.8% | -13% to +8% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 280 | -3.2% | -15% to +8% | 49% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -23% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +23% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -41% to +8% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +2% | 60% | - |  |
| `edge_1pct` | 37 | -42.7% | -73% to -11% | 27% | -1.50% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 338 | +0.6% | -8% to +9% | 60% | -2.01% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 337 | -4.5% | -13% to +4% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 325 | -4.6% | -14% to +6% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 338 | -19.1% | -30% to -8% | 39% | -2.71% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -20% to +22% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -54% to +38% | 14% | - |  |
| `home_ml_control` | 291 | -4.9% | -15% to +7% | 53% | -2.57% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 338 | +1.8% | -7% to +11% | 61% | -2.09% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -75% to +10% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 39 | -28.7% | -60% to +5% | 33% | -2.55% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 69 | -9.6% | -32% to +14% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 41 | +23.8% | -33% to +85% | 29% | - |  |
| `parlay_players3` | 22 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 41 | -57.3% | -92% to -13% | 10% | - |  |
| `parlay_top3` | 41 | -92.4% | -100% to -77% | 2% | - |  |
| `pts_edge` | 222 | -12.0% | -24% to +1% | 47% | - |  |
| `pts_over_control` | 227 | -0.0% | -12% to +11% | 59% | - |  |
| `pts_over_edge` | 188 | -30.8% | -45% to -15% | 31% | - |  |
| `pts_under_edge` | 220 | -7.4% | -20% to +5% | 51% | - |  |
| `puckline_dog_control` | 290 | -5.6% | -14% to +3% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 99 | -3.3% | -19% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 148 | -13.5% | -29% to +3% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 72 | -26.2% | -53% to +2% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 277 | -4.3% | -15% to +7% | 51% | - |  |
| `sog_over_control` | 281 | +2.9% | -9% to +15% | 51% | - |  |
| `sog_over_edge` | 201 | -8.3% | -22% to +6% | 46% | - |  |
| `sog_under_edge` | 236 | -6.1% | -18% to +5% | 50% | - |  |
| `totals_edge` | 191 | -7.9% | -22% to +5% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 128 | -5.1% | -23% to +11% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 151 | -30.1% | -47% to -12% | 29% | -2.88% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 291 | -11.7% | -25% to +2% | 38% | -2.88% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| BOS @ DET | Moneyline | DET moneyline | $10.00 | 1.96 | 56.4% | 49.7% |
| MTL @ NJD | Moneyline | NJD moneyline | $10.00 | 1.62 | 57.2% | 60.6% |
| COL @ UTA | Moneyline | COL moneyline | $10.00 | 1.93 | 51.8% | 50.1% |
| BOS @ DET | Total | Over 6 | $10.00 | 1.93 | 51.7% | 50.1% |
| MTL @ NJD | Total | Over 6 | $10.00 | 1.86 | 51.8% | 51.9% |
| COL @ UTA | Total | Under 6.5 | $10.00 | 1.85 | 55.9% | 52.6% |
| BOS @ DET | Puck line | DET +1.5 | $10.00 | 1.40 | 70.0% | 70.3% |
| MTL @ NJD | Puck line | MTL +1.5 | $10.00 | 1.57 | 62.8% | 61.8% |
| COL @ UTA | Puck line | COL +1.5 | $10.00 | 1.40 | 74.8% | 70.6% |

Total pretend stake $90.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
