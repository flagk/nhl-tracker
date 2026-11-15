# NHL model report: 2026-11-15 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-15 21:35 UTC · model `20261114-0255ca` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-15T21:33:22+00:00 · API credits left: 253*

*Player props: 3 game(s) priced*

## Summary

- 5 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| EDM @ PHI (19:30 ET) | D. Levi / J. Woll (probable) | 49.4% | 42.8% | PHI 2.27 | +6.6% | - | - | no bet |
| FLA @ DAL (20:00 ET) | J. Markstrom / J. Oettinger (probable) | 55.3% | 54.4% | DAL 1.80 | +1.0% | - | - | no bet |
| CGY @ ANA (20:00 ET) | D. Wolf / L. Dostal (probable) | 54.3% | 58.3% | CGY 2.33 | +4.0% | - | - | no bet |
| VAN @ VGK (21:00 ET) | L. Meriläinen / A. Hill (probable) | 69.4% | 78.8% | VAN 4.40 | +9.5% | - | - | no bet |
| NSH @ SJS (21:00 ET) | J. Saros / Y. Askarov (probable) | 40.7% | 38.9% | SJS 2.49 | +1.8% | - | - | no bet |

## Why

**EDM @ PHI (19:30 ET)**: No bet: expected value after shrinkage toward the market is only +0.5% (< 1%)
  - Context: EDM have the clearly better goal differential this season (gap 1.24 goals/game). PHI and EDM are on the second night of a back-to-back. Goaltending edge to EDM (goals saved above expected).

**FLA @ DAL (20:00 ET)**: No bet: edge +1.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: DAL and FLA are on the second night of a back-to-back. Goaltending edge to DAL (goals saved above expected).

**CGY @ ANA (20:00 ET)**: No bet: edge +4.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: ANA have the clearly better goal differential this season (gap 0.51 goals/game). ANA and CGY are on the second night of a back-to-back. Division/rival game.

**VAN @ VGK (21:00 ET)**: No bet: model win probability 30.6% below the 40% confidence floor; price 4.40 above the 4.0 long-shot cap
  - Context: VGK have the clearly better goal differential this season (gap 1.76 goals/game). VGK are creating the better share of expected goals recently. Goaltending edge to VGK (goals saved above expected). Division/rival game.

**NSH @ SJS (21:00 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: NSH have the clearly better goal differential this season (gap 0.78 goals/game). SJS is on the second night of a back-to-back. NSH are creating the better share of expected goals recently.

## Track record (all logged recommendations that have resolved)

- Resolved games: **320** (bet on 11, passed on 97%); model log loss 0.6589, Brier 0.2335 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (320 games with odds): model log loss 0.6589 vs market 0.6413 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.48% to -0.52%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.5% | 37.5% | 40 |
| 46.3% | 37.5% | 40 |
| 49.8% | 45.0% | 40 |
| 52.6% | 55.0% | 40 |
| 55.2% | 42.5% | 40 |
| 58.4% | 65.0% | 40 |
| 61.8% | 65.0% | 40 |
| 68.4% | 77.5% | 40 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| CGY @ ANA (20:00 ET) | Total | Under 6.5 | 56.3% | 47.4% | +8.9 pts | +15.4% | 2.05 |
| VAN @ VGK (21:00 ET) | Puck line | VAN +1.5 | 48.7% | 40.9% | +7.8 pts | +15.4% | 2.37 |
| CGY @ ANA (20:00 ET) | Puck line | CGY +1.5 | 67.7% | 62.9% | +4.8 pts | +4.9% | 1.55 |
| VAN @ VGK (21:00 ET) | Total | Under 6.5 | 54.5% | 50.2% | +4.3 pts | +5.2% | 1.93 |
| EDM @ PHI (19:30 ET) | Puck line | EDM -1.5 | 39.5% | 35.2% | +4.3 pts | +8.6% | 2.75 |
| NSH @ SJS (21:00 ET) | Total | Under 6.5 | 55.0% | 51.5% | +3.5 pts | +2.9% | 1.87 |
| FLA @ DAL (20:00 ET) | Total | Over 6 | 50.6% | 47.5% | +3.1 pts | +2.9% | 2.04 |
| FLA @ DAL (20:00 ET) | Puck line | FLA +1.5 | 69.2% | 67.7% | +1.5 pts | +0.3% | 1.45 |
| NSH @ SJS (21:00 ET) | Puck line | NSH -1.5 | 40.5% | 39.2% | +1.3 pts | +0.5% | 2.48 |
| EDM @ PHI (19:30 ET) | Total | Over 6 | 51.3% | 50.2% | +1.1 pts | -0.9% | 1.93 |
| EDM @ PHI (19:30 ET) | Total | Under 6 | 48.7% | 49.8% | -1.1 pts | -4.3% | 1.95 |
| NSH @ SJS (21:00 ET) | Puck line | SJS +1.5 | 59.5% | 60.8% | -1.3 pts | -3.7% | 1.62 |
| FLA @ DAL (20:00 ET) | Puck line | DAL -1.5 | 30.8% | 32.3% | -1.5 pts | -8.8% | 2.96 |
| FLA @ DAL (20:00 ET) | Total | Under 6 | 49.4% | 52.5% | -3.1 pts | -7.6% | 1.85 |
| NSH @ SJS (21:00 ET) | Total | Over 6.5 | 45.0% | 48.5% | -3.5 pts | -10.5% | 1.99 |
| EDM @ PHI (19:30 ET) | Puck line | PHI +1.5 | 60.5% | 64.8% | -4.3 pts | -7.4% | 1.53 |
| VAN @ VGK (21:00 ET) | Total | Over 6.5 | 45.5% | 49.8% | -4.3 pts | -10.8% | 1.96 |
| CGY @ ANA (20:00 ET) | Puck line | ANA -1.5 | 32.3% | 37.1% | -4.8 pts | -15.9% | 2.60 |
| VAN @ VGK (21:00 ET) | Puck line | VGK -1.5 | 51.3% | 59.1% | -7.8 pts | -15.4% | 1.65 |
| CGY @ ANA (20:00 ET) | Total | Over 6.5 | 43.7% | 52.6% | -8.9 pts | -19.6% | 1.84 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| N. Acciari | EDM @ PHI (19:30 ET) | goal 0.5 | Over | 0.16 | 14.4% | 12.1% | +2.4 pts | +37.1% | 0 0 0 0 0 1 1 0 0 0 | 0.1 |
| T. Washe | CGY @ ANA (20:00 ET) | goal 0.5 | Over | 0.18 | 16.1% | 13.1% | +3.0 pts | +37.0% | 0 0 0 0 0 0 0 0 0 0 | 0.1 |
| M. Joseph | EDM @ PHI (19:30 ET) | goal 0.5 | Over | 0.11 | 10.4% | 9.7% | +0.6 pts | +34.7% | 0 1 0 0 0 0 0 0 0 0 | 0.1 |
| N. Seeler | EDM @ PHI (19:30 ET) | goal 0.5 | Over | 0.05 | 5.0% | 5.4% | -0.4 pts | +30.2% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| T. Sanheim | EDM @ PHI (19:30 ET) | goal 0.5 | Over | 0.15 | 13.5% | 11.2% | +2.3 pts | +28.4% | 0 0 0 0 0 0 0 1 1 0 | 0.1 |
| K. Kapanen | EDM @ PHI (19:30 ET) | goal 0.5 | Over | 0.23 | 20.8% | 16.9% | +3.9 pts | +25.1% | 0 1 0 0 0 1 0 0 0 0 | 0.2 |
| Z. Whitecloud | CGY @ ANA (20:00 ET) | goal 0.5 | Over | 0.08 | 7.3% | 6.5% | +0.8 pts | +23.9% | 0 0 0 0 0 0 0 1 0 0 | 0.1 |
| J. Castagna | CGY @ ANA (20:00 ET) | goal 0.5 | Over | 0.12 | 11.2% | 11.1% | +0.1 pts | +23.6% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| S. Couturier | EDM @ PHI (19:30 ET) | goal 0.5 | Over | 0.18 | 16.2% | 13.3% | +2.9 pts | +21.4% | 0 0 1 0 0 0 0 0 0 0 | 0.0 |
| E. Bouchard | EDM @ PHI (19:30 ET) | sog 2.5 | Under | 2.03 | 33.0% | 45.2% | +12.3 pts | +20.7% | 2 4 3 3 1 3 1 1 2 2 | 2.5 |
| M. Coronato | CGY @ ANA (20:00 ET) | pts 0.5 | Under | 0.63 | 46.8% | 57.2% | +10.3 pts | +19.6% | 2 0 4 0 0 0 0 1 0 0 | 0.8 |
| Z. Parekh | CGY @ ANA (20:00 ET) | pts 0.5 | Under | 0.34 | 28.5% | 42.2% | +13.7 pts | +19.3% | 1 0 0 1 0 0 2 1 0 0 | 0.4 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 307 | -4.5% | -15% to +6% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 262 | -1.0% | -12% to +11% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 152 | -9.3% | -23% to +6% | 56% | - |  |
| `ast_over_control` | 151 | +7.3% | -8% to +23% | 56% | - |  |
| `ast_over_edge` | 122 | -17.3% | -42% to +11% | 28% | - |  |
| `ast_under_edge` | 151 | -8.9% | -21% to +3% | 60% | - |  |
| `edge_1pct` | 33 | -48.7% | -77% to -17% | 24% | -1.71% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 320 | +1.3% | -8% to +10% | 60% | -1.98% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 319 | -3.7% | -12% to +4% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 307 | -5.2% | -16% to +6% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 320 | -23.6% | -35% to -12% | 37% | -2.71% | $10 on every game where the model sees any positive edge |
| `goal_control` | 152 | -2.0% | -23% to +20% | 35% | - |  |
| `goal_edge` | 95 | -21.5% | -62% to +28% | 13% | - |  |
| `home_ml_control` | 273 | -5.6% | -17% to +6% | 52% | -2.56% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 320 | +3.3% | -5% to +12% | 62% | -2.07% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -73% to +8% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 38 | -26.9% | -59% to +7% | 34% | -2.56% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 65 | -9.7% | -34% to +11% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 38 | +20.0% | -38% to +86% | 29% | - |  |
| `parlay_players3` | 19 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 38 | -75.7% | -100% to -37% | 5% | - |  |
| `parlay_top3` | 38 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 199 | -13.8% | -28% to -1% | 46% | - |  |
| `pts_over_control` | 203 | +0.3% | -12% to +13% | 59% | - |  |
| `pts_over_edge` | 166 | -30.5% | -47% to -15% | 31% | - |  |
| `pts_under_edge` | 198 | -8.6% | -21% to +4% | 51% | - |  |
| `puckline_dog_control` | 272 | -6.4% | -15% to +2% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 92 | -4.0% | -20% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 141 | -15.4% | -31% to +1% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 68 | -25.9% | -53% to +5% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 253 | -3.8% | -16% to +8% | 51% | - |  |
| `sog_over_control` | 258 | -0.3% | -12% to +12% | 50% | - |  |
| `sog_over_edge` | 177 | -5.3% | -19% to +10% | 47% | - |  |
| `sog_under_edge` | 212 | -5.9% | -19% to +7% | 50% | - |  |
| `totals_edge` | 180 | -6.5% | -20% to +7% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 121 | -2.8% | -21% to +14% | 50% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 141 | -36.1% | -54% to -18% | 27% | -2.85% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 273 | -15.4% | -28% to -2% | 37% | -2.91% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| EDM @ PHI | Moneyline | EDM moneyline | $10.00 | 1.71 | 50.6% | 57.2% |
| FLA @ DAL | Moneyline | DAL moneyline | $10.00 | 1.80 | 55.3% | 54.4% |
| CGY @ ANA | Moneyline | ANA moneyline | $10.00 | 1.68 | 54.3% | 58.3% |
| VAN @ VGK | Moneyline | VGK moneyline | $10.00 | 1.25 | 69.4% | 78.8% |
| NSH @ SJS | Moneyline | NSH moneyline | $10.00 | 1.61 | 59.3% | 61.1% |
| EDM @ PHI | Total | Over 6 | $10.00 | 1.93 | 51.3% | 50.2% |
| FLA @ DAL | Total | Over 6 | $10.00 | 2.04 | 50.6% | 47.5% |
| CGY @ ANA | Total | Under 6.5 | $10.00 | 2.05 | 56.3% | 47.4% |
| VAN @ VGK | Total | Under 6.5 | $10.00 | 1.93 | 54.5% | 50.2% |
| NSH @ SJS | Total | Under 6.5 | $10.00 | 1.87 | 55.0% | 51.5% |
| EDM @ PHI | Puck line | PHI +1.5 | $10.00 | 1.53 | 60.5% | 64.8% |
| FLA @ DAL | Puck line | FLA +1.5 | $10.00 | 1.45 | 69.2% | 67.7% |
| CGY @ ANA | Puck line | CGY +1.5 | $10.00 | 1.55 | 67.7% | 62.9% |
| VAN @ VGK | Puck line | VGK -1.5 | $10.00 | 1.65 | 51.3% | 59.1% |
| NSH @ SJS | Puck line | SJS +1.5 | $10.00 | 1.62 | 59.5% | 60.8% |

Total pretend stake $150.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
