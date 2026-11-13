# NHL model report: 2026-11-13 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-13 23:04 UTC · model `20261112-169812` · probability source `online_platt` · bankroll $978.31 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-13T23:03:16+00:00 · API credits left: 285*

*Player props: 2 game(s) priced*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| NYR @ VAN (21:00 ET) | D. Garand / L. Meriläinen (probable) | 39.0% | 33.8% | VAN 2.85 | +5.2% | - | - | no bet |
| NSH @ VGK (22:00 ET) | J. Saros / C. Hart (probable) | 61.9% | 59.4% | VGK 1.66 | +2.5% | - | - | no bet |

## Why

**NYR @ VAN (21:00 ET)**: No bet: model win probability 39.0% below the 40% confidence floor
  - Context: NYR have the clearly better goal differential this season (gap 1.29 goals/game). NYR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**NSH @ VGK (22:00 ET)**: No bet: edge +2.5% below the 4% minimum (starting goalie unconfirmed)
  - Context: VGK have the clearly better goal differential this season (gap 0.37 goals/game). VGK are creating the better share of expected goals recently.

## Track record (all logged recommendations that have resolved)

- Resolved games: **304** (bet on 10, passed on 97%); model log loss 0.6553, Brier 0.2317 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (304 games with odds): model log loss 0.6553 vs market 0.6386 (market better)
- Bets: 10 · staked $26.57 · profit $-21.69 · **ROI -81.6%** · win rate 10.0% · avg price 2.50
- Bankroll $978.31 · max drawdown 2.2% · current drawdown 2.2%
- **CLV**: mean -2.33% per $1 (95% CI -3.67% to -0.60%), beat the close on 10% of 10 bets
- ⚠️ 10 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.4% | 39.5% | 38 |
| 46.2% | 39.5% | 38 |
| 49.8% | 47.4% | 38 |
| 52.7% | 57.9% | 38 |
| 55.2% | 44.7% | 38 |
| 58.4% | 65.8% | 38 |
| 61.7% | 68.4% | 38 |
| 68.2% | 79.0% | 38 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| NYR @ VAN (21:00 ET) | Total | Under 6.5 | 55.1% | 51.3% | +3.8 pts | +3.6% | 1.88 |
| NSH @ VGK (22:00 ET) | Puck line | VGK -1.5 | 39.1% | 37.3% | +1.8 pts | +1.7% | 2.60 |
| NSH @ VGK (22:00 ET) | Total | Over 6 | 51.6% | 50.7% | +0.9 pts | -1.2% | 1.91 |
| NYR @ VAN (21:00 ET) | Puck line | VAN +1.5 | 55.6% | 55.5% | +0.2 pts | -3.2% | 1.74 |
| NYR @ VAN (21:00 ET) | Puck line | NYR -1.5 | 44.4% | 44.5% | -0.2 pts | -3.7% | 2.17 |
| NSH @ VGK (22:00 ET) | Total | Under 6 | 48.4% | 49.3% | -0.9 pts | -3.7% | 1.98 |
| NSH @ VGK (22:00 ET) | Puck line | NSH +1.5 | 60.9% | 62.7% | -1.8 pts | -5.0% | 1.56 |
| NYR @ VAN (21:00 ET) | Total | Over 6.5 | 44.9% | 48.7% | -3.8 pts | -11.1% | 1.98 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| J. Drury | NSH @ VGK (22:00 ET) | goal 0.5 | Over | 0.20 | 18.3% | 11.9% | +6.4 pts | +55.3% | 0 1 0 0 0 0 0 0 0 0 | 0.1 |
| J. Struble | NSH @ VGK (22:00 ET) | goal 0.5 | Over | 0.07 | 6.4% | 5.2% | +1.2 pts | +48.0% | 1 0 0 0 1 0 0 0 0 1 | 0.2 |
| M. Sasson | NYR @ VAN (21:00 ET) | goal 0.5 | Over | 0.18 | 16.1% | 11.8% | +4.3 pts | +45.0% | 1 0 1 0 0 1 0 0 1 0 | 0.3 |
| A. Wilsby | NSH @ VGK (22:00 ET) | goal 0.5 | Over | 0.08 | 7.2% | 6.5% | +0.7 pts | +44.4% | 0 0 0 0 0 0 0 0 1 0 | 0.1 |
| D. O'Connor | NYR @ VAN (21:00 ET) | goal 0.5 | Over | 0.19 | 17.1% | 13.3% | +3.7 pts | +36.6% | 0 0 0 0 1 0 0 0 0 0 | 0.1 |
| N. Perbix | NSH @ VGK (22:00 ET) | goal 0.5 | Over | 0.08 | 7.5% | 7.0% | +0.5 pts | +35.1% | 0 0 0 0 0 1 0 1 0 0 | 0.1 |
| P. Wotherspoon | NSH @ VGK (22:00 ET) | goal 0.5 | Over | 0.05 | 5.1% | 6.0% | -0.9 pts | +33.4% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| A. Fox | NYR @ VAN (21:00 ET) | ast 0.5 | Under | 0.65 | 47.6% | 62.3% | +14.7 pts | +31.1% | 1 4 2 0 0 1 2 1 0 0 | 1.1 |
| P. Dorofeyev | NYR @ VAN (21:00 ET) | pts 0.5 | Under | 0.59 | 44.7% | 59.3% | +14.6 pts | +27.1% | 1 1 1 1 1 0 1 1 0 1 | 0.6 |
| L. Ohgren | NYR @ VAN (21:00 ET) | goal 0.5 | Over | 0.19 | 17.5% | 14.6% | +2.9 pts | +26.8% | 0 0 0 1 1 0 1 1 0 0 | 0.3 |
| T. Willander | NYR @ VAN (21:00 ET) | goal 0.5 | Over | 0.06 | 6.1% | 6.1% | +0.0 pts | +22.4% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| P. Dorofeyev | NYR @ VAN (21:00 ET) | sog 2.5 | Under | 2.65 | 47.8% | 58.4% | +10.6 pts | +22.2% | 5 2 4 1 1 1 4 3 6 3 | 2.9 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 291 | -3.2% | -14% to +7% | 51% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 246 | -2.5% | -15% to +10% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 136 | -7.9% | -21% to +8% | 57% | - |  |
| `ast_over_control` | 135 | +4.5% | -12% to +20% | 54% | - |  |
| `ast_over_edge` | 109 | -16.4% | -43% to +12% | 28% | - |  |
| `ast_under_edge` | 135 | -6.0% | -19% to +7% | 62% | - |  |
| `edge_1pct` | 28 | -39.5% | -75% to -2% | 29% | -1.59% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 304 | +2.2% | -7% to +12% | 61% | -1.99% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 303 | -3.8% | -12% to +4% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 291 | -5.9% | -17% to +5% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 304 | -22.6% | -34% to -12% | 38% | -2.68% | $10 on every game where the model sees any positive edge |
| `goal_control` | 136 | -5.5% | -28% to +17% | 34% | - |  |
| `goal_edge` | 85 | -17.6% | -62% to +35% | 13% | - |  |
| `home_ml_control` | 257 | -0.9% | -13% to +11% | 55% | -2.54% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 304 | +3.7% | -5% to +13% | 62% | -2.10% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 25 | -32.0% | -72% to +14% | 28% | -1.75% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 35 | -20.6% | -54% to +14% | 37% | -2.56% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 62 | -8.5% | -33% to +16% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 36 | +15.8% | -45% to +80% | 28% | - |  |
| `parlay_players3` | 17 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 36 | -74.3% | -100% to -38% | 6% | - |  |
| `parlay_top3` | 36 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 183 | -10.2% | -24% to +4% | 48% | - |  |
| `pts_over_control` | 187 | -2.2% | -14% to +11% | 58% | - |  |
| `pts_over_edge` | 151 | -30.7% | -49% to -13% | 30% | - |  |
| `pts_under_edge` | 182 | -4.3% | -18% to +10% | 53% | - |  |
| `puckline_dog_control` | 256 | -6.0% | -15% to +3% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 89 | -4.2% | -20% to +11% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 137 | -18.8% | -34% to -2% | 46% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 63 | -30.2% | -57% to +0% | 27% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 237 | -4.3% | -16% to +8% | 51% | - |  |
| `sog_over_control` | 242 | -2.0% | -15% to +11% | 49% | - |  |
| `sog_over_edge` | 163 | -5.6% | -21% to +10% | 47% | - |  |
| `sog_under_edge` | 196 | -5.6% | -19% to +7% | 51% | - |  |
| `totals_edge` | 173 | -8.3% | -23% to +6% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 115 | -4.5% | -22% to +14% | 49% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 134 | -36.8% | -55% to -19% | 27% | -2.88% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 257 | -16.7% | -30% to -3% | 37% | -2.90% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| NYR @ VAN | Moneyline | NYR moneyline | $10.00 | 1.49 | 61.0% | 66.2% |
| NSH @ VGK | Moneyline | VGK moneyline | $10.00 | 1.66 | 61.9% | 59.4% |
| NYR @ VAN | Total | Under 6.5 | $10.00 | 1.88 | 55.1% | 51.3% |
| NSH @ VGK | Total | Over 6 | $10.00 | 1.91 | 51.6% | 50.7% |
| NYR @ VAN | Puck line | VAN +1.5 | $10.00 | 1.74 | 55.6% | 55.5% |
| NSH @ VGK | Puck line | NSH +1.5 | $10.00 | 1.56 | 60.9% | 62.7% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
