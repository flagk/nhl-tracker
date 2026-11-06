# NHL model report: 2026-11-06 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-06 21:06 UTC · model `20261105-54203c` · probability source `online_platt` · bankroll $989.34 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-06T21:03:53+00:00 · API credits left: 406*

*Player props: 3 game(s) priced*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CBJ @ BUF (19:00 ET) | J. Greaves / U. Luukkonen (probable) | 62.2% | 61.2% | BUF 1.61 | +1.0% | - | - | no bet |
| SJS @ MIN (20:00 ET) | Y. Askarov / J. Wallstedt (probable) | 69.3% | 71.7% | SJS 3.36 | +2.4% | - | - | no bet |
| ANA @ CGY (20:00 ET) | L. Dostal / D. Wolf (probable) | 56.4% | 57.3% | ANA 2.27 | +0.8% | - | - | no bet |

## Why

**CBJ @ BUF (19:00 ET)**: No bet: edge +1.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: CBJ is on the second night of a back-to-back. Early in the season: team ratings lean on last year's results and are less reliable.

**SJS @ MIN (20:00 ET)**: No bet: edge +2.4% below the 4% minimum (starting goalie unconfirmed); model win probability 30.7% below the 40% confidence floor
  - Context: MIN have the clearly better goal differential this season (gap 1.30 goals/game). MIN are creating the better share of expected goals recently. Goaltending edge to MIN (goals saved above expected).

**ANA @ CGY (20:00 ET)**: No bet: edge +0.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: ANA have the clearly better goal differential this season (gap 0.54 goals/game). ANA is on the second night of a back-to-back. Division/rival game.

## Track record (all logged recommendations that have resolved)

- Resolved games: **256** (bet on 6, passed on 98%); model log loss 0.6565, Brier 0.2322 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (256 games with odds): model log loss 0.6565 vs market 0.6441 (market better)
- Bets: 6 · staked $15.54 · profit $-10.66 · **ROI -68.6%** · win rate 16.7% · avg price 2.46
- Bankroll $989.34 · max drawdown 1.1% · current drawdown 1.1%
- **CLV**: mean -2.36% per $1 (95% CI -4.36% to +0.48%), beat the close on 17% of 6 bets
- ⚠️ 6 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.1% | 40.6% | 32 |
| 46.4% | 46.9% | 32 |
| 50.1% | 40.6% | 32 |
| 52.9% | 65.6% | 32 |
| 55.8% | 43.8% | 32 |
| 58.8% | 68.8% | 32 |
| 62.1% | 71.9% | 32 |
| 68.0% | 75.0% | 32 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| SJS @ MIN (20:00 ET) | Total | Under 6.5 | 53.7% | 50.1% | +3.6 pts | +5.7% | 1.97 |
| ANA @ CGY (20:00 ET) | Total | Under 6.5 | 54.4% | 51.4% | +3.0 pts | +2.2% | 1.88 |
| ANA @ CGY (20:00 ET) | Puck line | ANA +1.5 | 67.5% | 64.7% | +2.8 pts | +3.3% | 1.53 |
| CBJ @ BUF (19:00 ET) | Total | Under 6.5 | 54.6% | 52.4% | +2.2 pts | +1.0% | 1.85 |
| CBJ @ BUF (19:00 ET) | Puck line | CBJ +1.5 | 62.9% | 60.8% | +2.1 pts | +1.9% | 1.62 |
| SJS @ MIN (20:00 ET) | Puck line | SJS +1.5 | 50.0% | 49.6% | +0.4 pts | -2.5% | 1.95 |
| SJS @ MIN (20:00 ET) | Puck line | MIN -1.5 | 50.0% | 50.4% | -0.4 pts | -3.5% | 1.93 |
| CBJ @ BUF (19:00 ET) | Puck line | BUF -1.5 | 37.1% | 39.2% | -2.1 pts | -7.3% | 2.50 |
| CBJ @ BUF (19:00 ET) | Total | Over 6.5 | 45.4% | 47.6% | -2.2 pts | -6.4% | 2.06 |
| ANA @ CGY (20:00 ET) | Puck line | CGY -1.5 | 32.5% | 35.3% | -2.8 pts | -9.6% | 2.78 |
| ANA @ CGY (20:00 ET) | Total | Over 6.5 | 45.6% | 48.6% | -3.0 pts | -7.8% | 2.02 |
| SJS @ MIN (20:00 ET) | Total | Over 6.5 | 46.3% | 49.9% | -3.6 pts | -7.3% | 2.00 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| T. Washe | ANA @ CGY (20:00 ET) | goal 0.5 | Over | 0.18 | 16.8% | 13.1% | +3.7 pts | +51.1% | 1 0 0 1 0 0 0 0 0 0 | 0.2 |
| P. Mintyukov | ANA @ CGY (20:00 ET) | goal 0.5 | Over | 0.09 | 8.9% | 8.3% | +0.6 pts | +42.2% | 0 0 0 0 0 0 0 0 0 2 | 0.2 |
| D. Severson | CBJ @ BUF (19:00 ET) | goal 0.5 | Over | 0.13 | 12.0% | 10.5% | +1.5 pts | +32.0% | 0 0 0 0 0 0 0 1 0 0 | 0.2 |
| Z. Parekh | ANA @ CGY (20:00 ET) | pts 0.5 | Under | 0.29 | 25.3% | 43.9% | +18.6 pts | +26.3% | 0 1 0 1 1 0 0 1 0 0 | 0.3 |
| R. Dahlin | CBJ @ BUF (19:00 ET) | sog 2.5 | Under | 2.34 | 40.7% | 51.7% | +11.0 pts | +24.5% | 3 1 4 4 4 3 1 4 2 1 | 2.8 |
| D. Mateychuk | CBJ @ BUF (19:00 ET) | goal 0.5 | Over | 0.14 | 13.0% | 11.7% | +1.3 pts | +23.8% | 1 0 0 0 1 1 1 0 0 0 | 0.3 |
| M. Olivier | CBJ @ BUF (19:00 ET) | goal 0.5 | Over | 0.22 | 19.7% | 17.1% | +2.6 pts | +23.2% | 0 0 0 0 0 0 0 0 1 0 | 0.2 |
| C. Sillinger | CBJ @ BUF (19:00 ET) | goal 0.5 | Over | 0.21 | 18.9% | 15.8% | +3.2 pts | +23.0% | 1 0 0 0 0 0 0 0 1 0 | 0.3 |
| I. Provorov | CBJ @ BUF (19:00 ET) | goal 0.5 | Over | 0.12 | 11.2% | 10.3% | +0.9 pts | +22.7% | 0 1 0 0 0 0 0 0 1 0 | - |
| Z. Parekh | ANA @ CGY (20:00 ET) | sog 1.5 | Under | 1.36 | 38.7% | 52.4% | +13.7 pts | +22.6% | 4 1 2 2 3 1 3 0 0 1 | 1.6 |
| A. Suniev | ANA @ CGY (20:00 ET) | sog 1.5 | Under | 1.54 | 44.6% | 52.9% | +8.3 pts | +21.9% | 3 0 3 0 6 0 2 1 0 2 | 1.6 |
| M. Samuelsson | CBJ @ BUF (19:00 ET) | pts 0.5 | Over | 0.59 | 44.5% | 33.7% | +10.8 pts | +21.8% | 0 2 2 1 0 0 0 1 1 0 | 0.6 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 244 | -1.5% | -13% to +10% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 199 | -5.1% | -18% to +9% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 80 | -8.7% | -27% to +10% | 56% | - |  |
| `ast_over_control` | 79 | +7.7% | -13% to +28% | 57% | - |  |
| `ast_over_edge` | 67 | -19.5% | -57% to +21% | 22% | - |  |
| `ast_under_edge` | 80 | -5.0% | -22% to +12% | 61% | - |  |
| `edge_1pct` | 19 | -32.5% | - | 32% | -1.22% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 256 | +1.1% | -10% to +11% | 60% | -1.92% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 255 | -2.9% | -12% to +6% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 244 | -7.4% | -20% to +5% | 48% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 256 | -18.8% | -31% to -6% | 39% | -2.56% | $10 on every game where the model sees any positive edge |
| `goal_control` | 80 | -8.6% | -37% to +21% | 32% | - |  |
| `goal_edge` | 54 | -4.4% | -67% to +74% | 13% | - |  |
| `home_ml_control` | 209 | +2.3% | -10% to +15% | 56% | -2.54% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 256 | +2.3% | -8% to +12% | 62% | -2.02% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 21 | -19.0% | -65% to +33% | 33% | -1.65% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 26 | -1.0% | -41% to +43% | 46% | -2.44% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 54 | -9.0% | -34% to +16% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 29 | +7.9% | -53% to +80% | 28% | - |  |
| `parlay_players3` | 11 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 29 | -68.1% | -100% to -17% | 7% | - |  |
| `parlay_top3` | 29 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 127 | -13.3% | -30% to +4% | 46% | - |  |
| `pts_over_control` | 131 | -6.5% | -22% to +8% | 56% | - |  |
| `pts_over_edge` | 101 | -34.6% | -56% to -12% | 27% | - |  |
| `pts_under_edge` | 127 | -10.1% | -26% to +6% | 49% | - |  |
| `puckline_dog_control` | 208 | -3.9% | -14% to +6% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 80 | -2.6% | -19% to +14% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 120 | -15.8% | -32% to +1% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 50 | -23.8% | -56% to +9% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 182 | +0.4% | -14% to +14% | 54% | - |  |
| `sog_over_control` | 186 | -3.8% | -18% to +11% | 48% | - |  |
| `sog_over_edge` | 112 | -11.8% | -30% to +5% | 46% | - |  |
| `sog_under_edge` | 143 | +0.9% | -15% to +16% | 55% | - |  |
| `totals_edge` | 148 | -13.8% | -29% to +2% | 45% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 90 | -12.4% | -32% to +8% | 44% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 108 | -27.6% | -48% to -7% | 31% | -2.76% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 209 | -14.1% | -29% to +1% | 37% | -2.94% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CBJ @ BUF | Moneyline | BUF moneyline | $10.00 | 1.61 | 62.2% | 61.2% |
| SJS @ MIN | Moneyline | MIN moneyline | $10.00 | 1.36 | 69.3% | 71.7% |
| ANA @ CGY | Moneyline | CGY moneyline | $10.00 | 1.71 | 56.4% | 57.3% |
| CBJ @ BUF | Total | Under 6.5 | $10.00 | 1.85 | 54.6% | 52.4% |
| SJS @ MIN | Total | Under 6.5 | $10.00 | 1.97 | 53.7% | 50.1% |
| ANA @ CGY | Total | Under 6.5 | $10.00 | 1.88 | 54.4% | 51.4% |
| CBJ @ BUF | Puck line | CBJ +1.5 | $10.00 | 1.62 | 62.9% | 60.8% |
| SJS @ MIN | Puck line | SJS +1.5 | $10.00 | 1.95 | 50.0% | 49.6% |
| ANA @ CGY | Puck line | ANA +1.5 | $10.00 | 1.53 | 67.5% | 64.7% |

Total pretend stake $90.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
