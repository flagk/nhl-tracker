# NHL model report: 2026-11-11 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-11 21:06 UTC · model `20261110-9b4d65` · probability source `online_platt` · bankroll $981.21 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-11T21:04:17+00:00 · API credits left: 318*

*Player props: 3 game(s) priced*

## Summary

- 6 games · **1 recommended bet(s)** · total stake $2.90 (0.30% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| TBL @ BUF (19:00 ET) | A. Vasilevskiy / U. Luukkonen (probable) | 47.6% | 45.6% | BUF 2.13 | +2.0% | - | - | no bet |
| NYR @ CGY (19:30 ET) | D. Garand / D. Wolf (probable) | 47.2% | 42.9% | CGY 2.29 | +4.3% | +8.1% | $2.90 | **BET CGY** |
| PHI @ EDM (21:00 ET) | J. Woll / D. Levi (probable) | 60.2% | 63.0% | PHI 2.63 | +2.8% | - | - | no bet |
| UTA @ VGK (21:30 ET) | K. Vejmelka / C. Hart (probable) | 53.4% | 53.7% | UTA 2.10 | +0.3% | - | - | no bet |
| NSH @ ANA (22:00 ET) | J. Saros / L. Dostal (probable) | 58.3% | 56.5% | ANA 1.73 | +1.8% | - | - | no bet |
| PIT @ SJS (22:00 ET) | A. Silovs / Y. Askarov (probable) | 42.6% | 43.3% | PIT 1.73 | +0.7% | - | - | no bet |

## Why

**TBL @ BUF (19:00 ET)**: No bet: edge +2.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: TBL have the clearly better goal differential this season (gap 0.56 goals/game). TBL are creating the better share of expected goals recently. Division/rival game.

**NYR @ CGY (19:30 ET)**: Bet $2.90 (0.30% of bankroll) on CGY at 2.29. The model gives CGY 47.2% vs the market's no-vig 42.9% (edge +4.3%); after shrinking toward the market the working probability is 44.3%, worth +1.5% per $1 at that price.
  - Context: NYR have the clearly better goal differential this season (gap 0.64 goals/game). NYR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**PHI @ EDM (21:00 ET)**: No bet: edge +2.8% below the 4% minimum (starting goalie unconfirmed); model win probability 39.8% below the 40% confidence floor
  - Context: EDM have the clearly better goal differential this season (gap 1.26 goals/game). EDM are creating the better share of expected goals recently.

**UTA @ VGK (21:30 ET)**: No bet: edge +0.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: VGK have the clearly better goal differential this season (gap 0.38 goals/game). UTA travelled 3,425 km across 3 time zone(s) since their last game. UTA are creating the better share of expected goals recently. Goaltending edge to VGK (goals saved above expected).

**NSH @ ANA (22:00 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: NSH is on the second night of a back-to-back. Goaltending edge to NSH (goals saved above expected).

**PIT @ SJS (22:00 ET)**: No bet: edge +0.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: PIT have the clearly better goal differential this season (gap 0.52 goals/game). PIT are creating the better share of expected goals recently.

## Track record (all logged recommendations that have resolved)

- Resolved games: **287** (bet on 9, passed on 97%); model log loss 0.6545, Brier 0.2313 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (287 games with odds): model log loss 0.6545 vs market 0.6392 (market better)
- Bets: 9 · staked $23.67 · profit $-18.79 · **ROI -79.4%** · win rate 11.1% · avg price 2.52
- Bankroll $981.21 · max drawdown 1.9% · current drawdown 1.9%
- **CLV**: mean -2.39% per $1 (95% CI -3.87% to -0.37%), beat the close on 11% of 9 bets
- ⚠️ 9 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.5% | 41.7% | 36 |
| 46.4% | 44.4% | 36 |
| 49.9% | 41.7% | 36 |
| 52.8% | 63.9% | 36 |
| 55.4% | 45.7% | 35 |
| 58.5% | 66.7% | 36 |
| 62.0% | 69.4% | 36 |
| 68.4% | 77.8% | 36 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| PIT @ SJS (22:00 ET) | Total | Under 6.5 | 55.0% | 45.1% | +9.9 pts | +18.2% | 2.15 |
| NSH @ ANA (22:00 ET) | Total | Under 6.5 | 55.5% | 46.0% | +9.5 pts | +16.6% | 2.10 |
| TBL @ BUF (19:00 ET) | Total | Under 6.5 | 55.4% | 48.7% | +6.7 pts | +10.7% | 2.00 |
| NYR @ CGY (19:30 ET) | Total | Over 5.5 | 58.3% | 53.9% | +4.4 pts | +5.0% | 1.80 |
| PHI @ EDM (21:00 ET) | Puck line | EDM -1.5 | 44.3% | 40.7% | +3.6 pts | +6.4% | 2.40 |
| UTA @ VGK (21:30 ET) | Total | Under 6.5 | 54.9% | 51.4% | +3.4 pts | +2.6% | 1.87 |
| PIT @ SJS (22:00 ET) | Puck line | PIT -1.5 | 38.3% | 35.5% | +2.8 pts | +5.3% | 2.75 |
| NSH @ ANA (22:00 ET) | Puck line | NSH +1.5 | 66.4% | 64.7% | +1.7 pts | +0.3% | 1.51 |
| TBL @ BUF (19:00 ET) | Puck line | TBL -1.5 | 34.4% | 33.1% | +1.3 pts | -0.2% | 2.90 |
| UTA @ VGK (21:30 ET) | Puck line | UTA +1.5 | 69.1% | 67.9% | +1.2 pts | -0.5% | 1.44 |
| NYR @ CGY (19:30 ET) | Puck line | CGY +1.5 | 66.5% | 65.5% | +1.0 pts | -0.2% | 1.50 |
| PHI @ EDM (21:00 ET) | Total | Under 6 | 48.3% | 48.1% | +0.2 pts | -2.6% | 2.01 |
| PHI @ EDM (21:00 ET) | Total | Over 6 | 51.7% | 51.9% | -0.2 pts | -2.9% | 1.87 |
| NYR @ CGY (19:30 ET) | Puck line | NYR -1.5 | 33.5% | 34.5% | -1.0 pts | -6.3% | 2.80 |
| UTA @ VGK (21:30 ET) | Puck line | VGK -1.5 | 30.9% | 32.1% | -1.2 pts | -7.3% | 3.00 |
| TBL @ BUF (19:00 ET) | Puck line | BUF +1.5 | 65.6% | 66.9% | -1.3 pts | -3.6% | 1.47 |
| NSH @ ANA (22:00 ET) | Puck line | ANA -1.5 | 33.6% | 35.3% | -1.7 pts | -8.3% | 2.73 |
| PIT @ SJS (22:00 ET) | Puck line | SJS +1.5 | 61.7% | 64.5% | -2.8 pts | -6.2% | 1.52 |
| UTA @ VGK (21:30 ET) | Total | Over 6.5 | 45.1% | 48.6% | -3.4 pts | -10.2% | 1.99 |
| PHI @ EDM (21:00 ET) | Puck line | PHI +1.5 | 55.7% | 59.3% | -3.6 pts | -7.0% | 1.67 |
| NYR @ CGY (19:30 ET) | Total | Under 5.5 | 41.7% | 46.1% | -4.4 pts | -11.6% | 2.12 |
| TBL @ BUF (19:00 ET) | Total | Over 6.5 | 44.6% | 51.3% | -6.7 pts | -16.5% | 1.87 |
| NSH @ ANA (22:00 ET) | Total | Over 6.5 | 44.5% | 54.0% | -9.5 pts | -19.9% | 1.80 |
| PIT @ SJS (22:00 ET) | Total | Over 6.5 | 45.0% | 54.9% | -9.9 pts | -19.9% | 1.78 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Z. Whitecloud | NYR @ CGY (19:30 ET) | goal 0.5 | Over | 0.09 | 8.2% | 6.2% | +2.0 pts | +64.0% | 0 0 0 0 0 0 0 0 0 1 | 0.1 |
| J. Castagna | NYR @ CGY (19:30 ET) | goal 0.5 | Over | 0.16 | 14.4% | 10.7% | +3.7 pts | +51.0% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| N. Laba | NYR @ CGY (19:30 ET) | goal 0.5 | Over | 0.14 | 12.7% | 10.8% | +1.9 pts | +33.3% | 0 0 0 0 0 0 0 0 0 0 | 0.0 |
| Z. Girgensons | TBL @ BUF (19:00 ET) | goal 0.5 | Over | 0.16 | 14.5% | 12.9% | +1.6 pts | +30.6% | 0 1 0 0 0 0 0 0 0 0 | 0.1 |
| M. Pospisil | NYR @ CGY (19:30 ET) | goal 0.5 | Over | 0.18 | 16.2% | 13.2% | +3.0 pts | +29.6% | 0 0 1 0 0 1 0 0 0 0 | 0.2 |
| O. Power | TBL @ BUF (19:00 ET) | ast 0.5 | Over | 0.56 | 42.9% | 31.0% | +11.8 pts | +28.6% | 0 0 0 1 2 0 0 0 3 0 | 0.7 |
| I. Mikheyev | TBL @ BUF (19:00 ET) | goal 0.5 | Over | 0.21 | 19.3% | 15.8% | +3.5 pts | +25.1% | 0 0 0 0 0 0 0 2 0 0 | 0.2 |
| O. Power | TBL @ BUF (19:00 ET) | pts 0.5 | Over | 0.68 | 49.1% | 37.3% | +11.9 pts | +23.8% | 1 0 0 1 2 0 0 0 3 0 | 0.8 |
| N. Ostlund | TBL @ BUF (19:00 ET) | sog 1.5 | Over | 2.29 | 64.1% | 52.3% | +11.8 pts | +22.4% | 1 1 3 1 1 5 3 4 3 3 | 2.5 |
| J. Middleton | NYR @ CGY (19:30 ET) | goal 0.5 | Over | 0.07 | 6.4% | 6.7% | -0.3 pts | +21.5% | 0 0 0 0 0 0 0 0 0 0 | 0.1 |
| J. Carlson | TBL @ BUF (19:00 ET) | pts 0.5 | Under | 0.44 | 35.3% | 49.6% | +14.3 pts | +20.9% | 0 1 0 0 1 2 0 0 2 0 | 0.6 |
| M. Ekholm | PHI @ EDM (21:00 ET) | sog 1.5 | Under | 1.22 | 34.0% | 47.4% | +13.4 pts | +20.8% | 1 1 0 3 0 1 3 1 1 2 | 1.1 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 274 | -3.6% | -15% to +8% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 229 | -2.3% | -15% to +10% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 120 | -4.6% | -19% to +11% | 59% | - |  |
| `ast_over_control` | 119 | +6.2% | -11% to +23% | 55% | - |  |
| `ast_over_edge` | 99 | -10.6% | -38% to +20% | 29% | - |  |
| `ast_under_edge` | 120 | -3.2% | -17% to +10% | 64% | - |  |
| `edge_1pct` | 24 | -38.0% | -75% to +4% | 29% | -1.52% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 287 | +1.7% | -8% to +11% | 61% | -1.97% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 286 | -4.0% | -13% to +4% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 274 | -7.2% | -18% to +4% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 287 | -20.8% | -32% to -9% | 39% | -2.64% | $10 on every game where the model sees any positive edge |
| `goal_control` | 120 | -6.8% | -31% to +18% | 33% | - |  |
| `goal_edge` | 78 | -25.5% | -72% to +30% | 12% | - |  |
| `home_ml_control` | 240 | +1.5% | -10% to +13% | 56% | -2.56% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 287 | +4.0% | -6% to +13% | 63% | -2.03% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 24 | -29.1% | -70% to +19% | 29% | -1.75% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 33 | -15.8% | -51% to +21% | 39% | -2.52% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 59 | -10.3% | -35% to +14% | 47% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 34 | +8.7% | -51% to +73% | 26% | - |  |
| `parlay_players3` | 16 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 34 | -72.8% | -100% to -30% | 6% | - |  |
| `parlay_top3` | 34 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 167 | -9.7% | -25% to +5% | 49% | - |  |
| `pts_over_control` | 171 | -4.3% | -17% to +9% | 57% | - |  |
| `pts_over_edge` | 141 | -29.1% | -47% to -11% | 31% | - |  |
| `pts_under_edge` | 167 | -4.6% | -19% to +10% | 53% | - |  |
| `puckline_dog_control` | 239 | -5.7% | -15% to +4% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 89 | -4.2% | -20% to +10% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 130 | -18.9% | -34% to -3% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 55 | -30.7% | -60% to +2% | 27% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 222 | -0.8% | -13% to +12% | 53% | - |  |
| `sog_over_control` | 226 | -0.9% | -14% to +12% | 50% | - |  |
| `sog_over_edge` | 151 | -6.6% | -23% to +9% | 47% | - |  |
| `sog_under_edge` | 183 | -2.0% | -15% to +12% | 52% | - |  |
| `totals_edge` | 162 | -9.4% | -24% to +6% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 106 | -5.8% | -24% to +13% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 126 | -36.3% | -54% to -17% | 27% | -2.90% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 240 | -17.2% | -32% to -3% | 36% | -2.99% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| TBL @ BUF | Moneyline | TBL moneyline | $10.00 | 1.78 | 52.4% | 54.4% |
| NYR @ CGY | Moneyline | NYR moneyline | $10.00 | 1.71 | 52.8% | 57.1% |
| PHI @ EDM | Moneyline | EDM moneyline | $10.00 | 1.56 | 60.2% | 63.0% |
| UTA @ VGK | Moneyline | VGK moneyline | $10.00 | 1.80 | 53.4% | 53.7% |
| NSH @ ANA | Moneyline | ANA moneyline | $10.00 | 1.73 | 58.3% | 56.5% |
| PIT @ SJS | Moneyline | PIT moneyline | $10.00 | 1.73 | 57.4% | 56.7% |
| TBL @ BUF | Total | Under 6.5 | $10.00 | 2.00 | 55.4% | 48.7% |
| NYR @ CGY | Total | Over 5.5 | $10.00 | 1.80 | 58.3% | 53.9% |
| PHI @ EDM | Total | Over 6 | $10.00 | 1.87 | 51.7% | 51.9% |
| UTA @ VGK | Total | Under 6.5 | $10.00 | 1.87 | 54.9% | 51.4% |
| NSH @ ANA | Total | Under 6.5 | $10.00 | 2.10 | 55.5% | 46.0% |
| PIT @ SJS | Total | Under 6.5 | $10.00 | 2.15 | 55.0% | 45.1% |
| TBL @ BUF | Puck line | BUF +1.5 | $10.00 | 1.47 | 65.6% | 66.9% |
| NYR @ CGY | Puck line | CGY +1.5 | $10.00 | 1.50 | 66.5% | 65.5% |
| PHI @ EDM | Puck line | PHI +1.5 | $10.00 | 1.67 | 55.7% | 59.3% |
| UTA @ VGK | Puck line | UTA +1.5 | $10.00 | 1.44 | 69.1% | 67.9% |
| NSH @ ANA | Puck line | NSH +1.5 | $10.00 | 1.51 | 66.4% | 64.7% |
| PIT @ SJS | Puck line | SJS +1.5 | $10.00 | 1.52 | 61.7% | 64.5% |

Total pretend stake $180.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
