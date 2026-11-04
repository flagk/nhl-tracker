# NHL model report: 2026-11-04 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-04 21:35 UTC · model `20261103-5af87d` · probability source `online_platt` · bankroll $990.96 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-04T21:33:30+00:00 · API credits left: 441*

*Player props: 3 game(s) priced*

## Summary

- 4 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| OTT @ BUF (19:30 ET) | L. Ullmark / U. Luukkonen (probable) | 57.2% | 53.6% | BUF 1.81 | +3.6% | - | - | no bet |
| CHI @ FLA (19:30 ET) | S. Knight / J. Markstrom (probable) | 68.3% | 70.3% | CHI 3.25 | +2.0% | - | - | no bet |
| SJS @ LAK (22:00 ET) | Y. Askarov / A. Forsberg (probable) | 64.3% | 59.7% | LAK 1.64 | +4.5% | - | - | no bet |
| CGY @ SEA (22:00 ET) | D. Wolf / P. Grubauer (probable) | 54.3% | 55.0% | CGY 2.15 | +0.7% | - | - | no bet |

## Why

**OTT @ BUF (19:30 ET)**: No bet: edge +3.6% below the 4% minimum (starting goalie unconfirmed)
  - Context: OTT travelled 3,513 km across 3 time zone(s) since their last game. Goaltending edge to OTT (goals saved above expected). Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**CHI @ FLA (19:30 ET)**: No bet: edge +2.0% below the 4% minimum (starting goalie unconfirmed); model win probability 31.7% below the 40% confidence floor
  - Context: FLA have the clearly better goal differential this season (gap 1.05 goals/game). FLA are creating the better share of expected goals recently.

**SJS @ LAK (22:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.3% (< 1%)
  - Context: LAK are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**CGY @ SEA (22:00 ET)**: No bet: edge +0.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: SEA have the clearly better goal differential this season (gap 0.53 goals/game). CGY are creating the better share of expected goals recently. Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **241** (bet on 5, passed on 98%); model log loss 0.6584, Brier 0.2331 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (241 games with odds): model log loss 0.6584 vs market 0.6462 (market better)
- Bets: 5 · staked $13.92 · profit $-9.04 · **ROI -65.0%** · win rate 20.0% · avg price 2.41
- Bankroll $990.96 · max drawdown 0.9% · current drawdown 0.9%
- **CLV**: mean -1.94% per $1 (95% CI -4.33% to +1.45%), beat the close on 20% of 5 bets
- ⚠️ 5 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 38.9% | 41.9% | 31 |
| 46.3% | 43.3% | 30 |
| 50.1% | 43.3% | 30 |
| 52.8% | 66.7% | 30 |
| 55.5% | 46.7% | 30 |
| 58.6% | 66.7% | 30 |
| 61.9% | 70.0% | 30 |
| 67.7% | 73.3% | 30 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| OTT @ BUF (19:30 ET) | Total | Under 6.5 | 54.5% | 47.4% | +7.2 pts | +12.9% | 2.07 |
| OTT @ BUF (19:30 ET) | Puck line | BUF -1.5 | 36.7% | 32.8% | +3.9 pts | +6.0% | 2.89 |
| CHI @ FLA (19:30 ET) | Puck line | FLA -1.5 | 50.6% | 47.0% | +3.6 pts | +4.8% | 2.07 |
| CHI @ FLA (19:30 ET) | Total | Over 6 | 53.5% | 49.9% | +3.6 pts | +2.8% | 1.93 |
| SJS @ LAK (22:00 ET) | Puck line | LAK -1.5 | 40.2% | 37.5% | +2.7 pts | +4.5% | 2.60 |
| SJS @ LAK (22:00 ET) | Total | Over 6 | 53.2% | 51.7% | +1.5 pts | -0.5% | 1.87 |
| CGY @ SEA (22:00 ET) | Puck line | SEA -1.5 | 33.7% | 33.2% | +0.5 pts | -2.9% | 2.88 |
| CGY @ SEA (22:00 ET) | Total | Over 6 | 52.9% | 52.6% | +0.3 pts | -2.3% | 1.84 |
| CGY @ SEA (22:00 ET) | Total | Under 6 | 47.1% | 47.4% | -0.3 pts | -3.0% | 2.05 |
| CGY @ SEA (22:00 ET) | Puck line | CGY +1.5 | 66.3% | 66.8% | -0.5 pts | -3.2% | 1.46 |
| SJS @ LAK (22:00 ET) | Total | Under 6 | 46.8% | 48.3% | -1.5 pts | -5.6% | 2.00 |
| SJS @ LAK (22:00 ET) | Puck line | SJS +1.5 | 59.8% | 62.5% | -2.7 pts | -6.7% | 1.56 |
| CHI @ FLA (19:30 ET) | Total | Under 6 | 46.5% | 50.1% | -3.6 pts | -8.9% | 1.93 |
| CHI @ FLA (19:30 ET) | Puck line | CHI +1.5 | 49.4% | 53.0% | -3.6 pts | -8.7% | 1.85 |
| OTT @ BUF (19:30 ET) | Puck line | OTT +1.5 | 63.3% | 67.2% | -3.9 pts | -7.6% | 1.46 |
| OTT @ BUF (19:30 ET) | Total | Over 6.5 | 45.5% | 52.6% | -7.2 pts | -15.9% | 1.85 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| N. Mikkola | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.10 | 9.8% | 7.8% | +2.0 pts | +77.0% | 0 0 0 1 0 0 0 1 0 0 | 0.1 |
| W. Kaiser | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.09 | 8.5% | 6.8% | +1.7 pts | +62.1% | 0 0 0 0 0 0 1 0 0 1 | 0.1 |
| C. Smith | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.12 | 11.5% | 10.2% | +1.3 pts | +49.9% | 0 0 0 0 0 0 0 0 0 0 | 0.1 |
| J. Greenway | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.12 | 11.4% | 9.0% | +2.4 pts | +48.1% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| R. Greene | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.17 | 16.0% | 12.1% | +3.8 pts | +43.8% | 0 0 0 0 0 1 0 0 1 0 | 0.1 |
| S. Bennett | CHI @ FLA (19:30 ET) | pts 0.5 | Under | 0.58 | 43.7% | 61.0% | +17.3 pts | +35.0% | 0 0 0 3 0 0 0 0 0 1 | 0.5 |
| M. Samuelsson | OTT @ BUF (19:30 ET) | ast 0.5 | Over | 0.48 | 37.9% | 25.0% | +12.9 pts | +34.5% | 0 0 2 1 1 0 0 0 1 1 | 0.5 |
| I. Cole | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.05 | 4.7% | 5.9% | -1.2 pts | +32.5% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| M. Samuelsson | OTT @ BUF (19:30 ET) | pts 0.5 | Over | 0.60 | 45.2% | 32.9% | +12.2 pts | +27.3% | 0 0 2 2 1 0 0 0 1 1 | 0.6 |
| S. Bennett | CHI @ FLA (19:30 ET) | ast 0.5 | Under | 0.30 | 25.6% | 43.0% | +17.3 pts | +25.7% | 0 0 0 0 0 0 0 0 0 0 | 0.2 |
| Z. Parekh | CGY @ SEA (22:00 ET) | sog 1.5 | Under | 1.38 | 39.5% | 53.8% | +14.3 pts | +24.1% | 0 4 1 2 2 3 1 3 0 0 | 1.7 |
| L. Eller | CHI @ FLA (19:30 ET) | goal 0.5 | Over | 0.21 | 18.8% | 15.7% | +3.1 pts | +21.9% | 0 0 0 0 0 0 1 0 0 0 | 0.1 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 229 | -1.8% | -14% to +11% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 184 | -4.9% | -19% to +10% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 64 | -12.6% | -35% to +10% | 53% | - |  |
| `ast_over_control` | 63 | +6.8% | -17% to +30% | 57% | - |  |
| `ast_over_edge` | 56 | -11.6% | -54% to +37% | 23% | - |  |
| `ast_under_edge` | 64 | -6.9% | -26% to +12% | 59% | - |  |
| `edge_1pct` | 17 | -34.2% | - | 29% | -1.08% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 241 | +1.6% | -9% to +12% | 60% | -1.96% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 240 | -4.8% | -14% to +5% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 229 | -7.3% | -20% to +6% | 48% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 241 | -19.1% | -33% to -6% | 39% | -2.49% | $10 on every game where the model sees any positive edge |
| `goal_control` | 64 | -8.9% | -40% to +25% | 33% | - |  |
| `goal_edge` | 51 | -6.2% | -71% to +71% | 12% | - |  |
| `home_ml_control` | 194 | +2.5% | -11% to +16% | 56% | -2.59% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 241 | +2.1% | -8% to +13% | 61% | -2.04% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 20 | -15.0% | -64% to +39% | 35% | -1.51% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 20 | -1.3% | -47% to +48% | 45% | -2.15% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 49 | -11.7% | -35% to +15% | 47% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 27 | +15.9% | -50% to +90% | 30% | - |  |
| `parlay_players3` | 9 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 27 | -65.8% | -100% to -11% | 7% | - |  |
| `parlay_top3` | 27 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 111 | -21.7% | -39% to -4% | 41% | - |  |
| `pts_over_control` | 115 | -3.5% | -19% to +11% | 58% | - |  |
| `pts_over_edge` | 86 | -37.4% | -61% to -13% | 24% | - |  |
| `pts_under_edge` | 111 | -20.1% | -37% to -3% | 43% | - |  |
| `puckline_dog_control` | 193 | -4.4% | -15% to +6% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 77 | -3.0% | -20% to +13% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 114 | -15.9% | -33% to -0% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 44 | -24.0% | -58% to +12% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 166 | +3.2% | -11% to +17% | 56% | - |  |
| `sog_over_control` | 170 | -6.2% | -21% to +9% | 46% | - |  |
| `sog_over_edge` | 96 | -14.1% | -34% to +5% | 45% | - |  |
| `sog_under_edge` | 127 | +4.6% | -12% to +20% | 57% | - |  |
| `totals_edge` | 140 | -13.2% | -28% to +3% | 45% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 83 | -12.2% | -34% to +9% | 45% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 100 | -24.4% | -45% to -2% | 32% | -2.62% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 194 | -13.6% | -30% to +3% | 38% | -2.84% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| OTT @ BUF | Moneyline | BUF moneyline | $10.00 | 1.81 | 57.2% | 53.6% |
| CHI @ FLA | Moneyline | FLA moneyline | $10.00 | 1.39 | 68.3% | 70.3% |
| SJS @ LAK | Moneyline | LAK moneyline | $10.00 | 1.64 | 64.3% | 59.7% |
| CGY @ SEA | Moneyline | SEA moneyline | $10.00 | 1.77 | 54.3% | 55.0% |
| OTT @ BUF | Total | Under 6.5 | $10.00 | 2.07 | 54.5% | 47.4% |
| CHI @ FLA | Total | Over 6 | $10.00 | 1.93 | 53.5% | 49.9% |
| SJS @ LAK | Total | Over 6 | $10.00 | 1.87 | 53.2% | 51.7% |
| CGY @ SEA | Total | Over 6 | $10.00 | 1.84 | 52.9% | 52.6% |
| OTT @ BUF | Puck line | OTT +1.5 | $10.00 | 1.46 | 63.3% | 67.2% |
| CHI @ FLA | Puck line | FLA -1.5 | $10.00 | 2.07 | 50.6% | 47.0% |
| SJS @ LAK | Puck line | SJS +1.5 | $10.00 | 1.56 | 59.8% | 62.5% |
| CGY @ SEA | Puck line | CGY +1.5 | $10.00 | 1.46 | 66.3% | 66.8% |

Total pretend stake $120.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
