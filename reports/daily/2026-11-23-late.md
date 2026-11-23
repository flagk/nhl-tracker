# NHL model report: 2026-11-23 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-23 21:04 UTC · model `20261122-54dfe0` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-23T21:02:11+00:00 · API credits left: 159*

*Player props: 3 game(s) priced*

## Summary

- 7 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CBJ @ NJD (19:00 ET) | C. Talbot / C. Hellebuyck (probable) | 62.1% | 65.0% | CBJ 2.75 | +2.9% | - | - | no bet |
| LAK @ MTL (19:30 ET) | D. Kuemper / J. Fowler (probable) | 56.3% | 58.2% | LAK 2.32 | +1.9% | - | - | no bet |
| CGY @ OTT (19:30 ET) | D. Wolf / L. Ullmark (probable) | 63.9% | 69.6% | CGY 3.20 | +5.8% | - | - | no bet |
| TOR @ NYI (19:30 ET) | S. Bobrovsky / I. Sorokin (probable) | 56.3% | 59.5% | TOR 2.38 | +3.2% | - | - | no bet |
| UTA @ WPG (19:30 ET) | K. Vejmelka / S. Skinner (probable) | 46.1% | 49.9% | UTA 1.93 | +3.8% | - | - | no bet |
| ANA @ SEA (21:40 ET) | L. Dostal / P. Grubauer (probable) | 49.1% | 51.9% | ANA 2.01 | +2.8% | - | - | no bet |
| MIN @ SJS (22:00 ET) | J. Wallstedt / Y. Askarov (probable) | 37.0% | 38.7% | MIN 1.63 | +1.7% | - | - | no bet |

## Why

**CBJ @ NJD (19:00 ET)**: No bet: edge +2.9% below the 4% minimum (starting goalie unconfirmed); model win probability 37.9% below the 40% confidence floor
  - Context: NJD have the clearly better goal differential this season (gap 0.73 goals/game). NJD are creating the better share of expected goals recently. Division/rival game.

**LAK @ MTL (19:30 ET)**: No bet: edge +1.9% below the 4% minimum (starting goalie unconfirmed)
  - Context: MTL have the clearly better goal differential this season (gap 0.61 goals/game). LAK are creating the better share of expected goals recently. Goaltending edge to MTL (goals saved above expected).

**CGY @ OTT (19:30 ET)**: No bet: model win probability 36.1% below the 40% confidence floor
  - Context: OTT have the clearly better goal differential this season (gap 1.01 goals/game). CGY travelled 2,864 km across 2 time zone(s) since their last game. OTT are creating the better share of expected goals recently. Goaltending edge to OTT (goals saved above expected).

**TOR @ NYI (19:30 ET)**: No bet: edge +3.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: NYI have the clearly better goal differential this season (gap 0.60 goals/game). Goaltending edge to NYI (goals saved above expected).

**UTA @ WPG (19:30 ET)**: No bet: edge +3.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: UTA have the clearly better goal differential this season (gap 0.27 goals/game). UTA are creating the better share of expected goals recently. Goaltending edge to WPG (goals saved above expected). Division/rival game.

**ANA @ SEA (21:40 ET)**: No bet: edge +2.8% below the 4% minimum (starting goalie unconfirmed)
  - Context: Division/rival game.

**MIN @ SJS (22:00 ET)**: No bet: edge +1.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: MIN have the clearly better goal differential this season (gap 1.51 goals/game). MIN are creating the better share of expected goals recently. Goaltending edge to MIN (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **372** (bet on 11, passed on 97%); model log loss 0.6666, Brier 0.2371 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (372 games with odds): model log loss 0.6666 vs market 0.6536 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.47% to -0.58%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.7% | 40.4% | 47 |
| 46.7% | 39.1% | 46 |
| 50.0% | 38.3% | 47 |
| 52.7% | 56.5% | 46 |
| 55.2% | 47.8% | 46 |
| 58.2% | 57.5% | 47 |
| 61.5% | 71.7% | 46 |
| 68.4% | 70.2% | 47 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| ANA @ SEA (21:40 ET) | Total | Under 6.5 | 53.8% | 45.1% | +8.7 pts | +16.8% | 2.17 |
| CGY @ OTT (19:30 ET) | Total | Over 5.5 | 58.8% | 51.6% | +7.1 pts | +9.9% | 1.87 |
| MIN @ SJS (22:00 ET) | Puck line | MIN -1.5 | 46.9% | 39.9% | +7.0 pts | +16.3% | 2.48 |
| ANA @ SEA (21:40 ET) | Puck line | ANA +1.5 | 74.1% | 68.9% | +5.2 pts | +6.0% | 1.43 |
| UTA @ WPG (19:30 ET) | Puck line | UTA -1.5 | 34.7% | 29.8% | +4.9 pts | +10.9% | 3.20 |
| CGY @ OTT (19:30 ET) | Puck line | CGY +1.5 | 58.3% | 53.7% | +4.6 pts | +6.0% | 1.82 |
| UTA @ WPG (19:30 ET) | Total | Under 6.5 | 54.6% | 51.1% | +3.5 pts | +3.8% | 1.90 |
| TOR @ NYI (19:30 ET) | Total | Under 6.5 | 54.2% | 50.8% | +3.3 pts | +3.5% | 1.91 |
| CBJ @ NJD (19:00 ET) | Total | Over 6 | 52.6% | 50.3% | +2.3 pts | +2.3% | 1.95 |
| TOR @ NYI (19:30 ET) | Puck line | TOR +1.5 | 64.8% | 62.7% | +2.1 pts | +1.1% | 1.56 |
| LAK @ MTL (19:30 ET) | Puck line | LAK +1.5 | 65.6% | 63.6% | +1.9 pts | +1.0% | 1.54 |
| MIN @ SJS (22:00 ET) | Total | Under 6.5 | 53.8% | 52.5% | +1.3 pts | -0.5% | 1.85 |
| CBJ @ NJD (19:00 ET) | Puck line | NJD -1.5 | 43.3% | 42.5% | +0.7 pts | -1.4% | 2.28 |
| LAK @ MTL (19:30 ET) | Total | Under 6 | 47.9% | 47.5% | +0.4 pts | -1.6% | 2.05 |
| LAK @ MTL (19:30 ET) | Total | Over 6 | 52.1% | 52.5% | -0.4 pts | -3.6% | 1.84 |
| CBJ @ NJD (19:00 ET) | Puck line | CBJ +1.5 | 56.7% | 57.5% | -0.7 pts | -4.1% | 1.69 |
| MIN @ SJS (22:00 ET) | Total | Over 6.5 | 46.2% | 47.5% | -1.3 pts | -5.7% | 2.04 |
| LAK @ MTL (19:30 ET) | Puck line | MTL -1.5 | 34.4% | 36.4% | -1.9 pts | -8.8% | 2.65 |
| TOR @ NYI (19:30 ET) | Puck line | NYI -1.5 | 35.2% | 37.3% | -2.1 pts | -8.2% | 2.61 |
| CBJ @ NJD (19:00 ET) | Total | Under 6 | 47.4% | 49.7% | -2.3 pts | -6.7% | 1.95 |
| TOR @ NYI (19:30 ET) | Total | Over 6.5 | 45.8% | 49.2% | -3.3 pts | -10.2% | 1.96 |
| UTA @ WPG (19:30 ET) | Total | Over 6.5 | 45.4% | 48.9% | -3.5 pts | -10.2% | 1.98 |
| CGY @ OTT (19:30 ET) | Puck line | OTT -1.5 | 41.7% | 46.3% | -4.6 pts | -12.4% | 2.10 |
| UTA @ WPG (19:30 ET) | Puck line | WPG +1.5 | 65.3% | 70.2% | -4.9 pts | -8.5% | 1.40 |
| ANA @ SEA (21:40 ET) | Puck line | SEA -1.5 | 25.9% | 31.1% | -5.2 pts | -19.2% | 3.12 |
| MIN @ SJS (22:00 ET) | Puck line | SJS +1.5 | 53.1% | 60.1% | -7.0 pts | -14.0% | 1.62 |
| CGY @ OTT (19:30 ET) | Total | Under 5.5 | 41.2% | 48.4% | -7.1 pts | -17.6% | 2.00 |
| ANA @ SEA (21:40 ET) | Total | Over 6.5 | 46.2% | 54.9% | -8.7 pts | -18.3% | 1.77 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| C. Caufield | LAK @ MTL (19:30 ET) | pts 0.5 | Under | 0.76 | 53.3% | 65.0% | +11.7 pts | +23.7% | 2 1 1 1 1 1 0 0 1 0 | 0.9 |
| V. Nichushkin | CBJ @ NJD (19:00 ET) | pts 0.5 | Over | 0.61 | 45.8% | 36.4% | +9.4 pts | +23.7% | 0 2 0 0 1 0 1 0 1 0 | 0.5 |
| K. Johnson | CBJ @ NJD (19:00 ET) | pts 0.5 | Over | 0.88 | 58.7% | 48.5% | +10.2 pts | +20.4% | 1 1 1 2 0 1 1 1 2 1 | 1.0 |
| S. Halliday | CGY @ OTT (19:30 ET) | sog 1.5 | Under | 1.27 | 35.7% | 48.6% | +12.9 pts | +20.3% | 3 2 0 1 1 1 1 2 2 2 | 1.5 |
| C. Coyle | CBJ @ NJD (19:00 ET) | sog 1.5 | Over | 2.53 | 69.1% | 54.6% | +14.4 pts | +20.2% | 2 3 1 2 2 3 5 2 6 1 | 2.2 |
| M. Backlund | CGY @ OTT (19:30 ET) | pts 0.5 | Over | 0.55 | 42.3% | 34.2% | +8.1 pts | +18.5% | 0 0 0 0 0 0 1 1 0 1 | 0.4 |
| B. Clarke | LAK @ MTL (19:30 ET) | sog 1.5 | Under | 1.32 | 37.6% | 46.5% | +8.9 pts | +15.5% | 0 1 0 1 3 2 1 0 2 1 | 1.6 |
| N. Dobson | LAK @ MTL (19:30 ET) | sog 1.5 | Under | 1.64 | 47.5% | 55.3% | +7.8 pts | +15.4% | 2 1 3 3 1 0 1 2 1 2 | 2.0 |
| M. Knies | CBJ @ NJD (19:00 ET) | sog 1.5 | Over | 2.14 | 60.8% | 51.1% | +9.7 pts | +14.2% | 0 0 1 1 2 3 3 2 4 5 | 1.9 |
| T. Chabot | CGY @ OTT (19:30 ET) | sog 1.5 | Under | 1.56 | 45.0% | 53.8% | +8.8 pts | +13.8% | 2 0 1 2 1 1 3 1 1 1 | 1.7 |
| J. Slafkovský | LAK @ MTL (19:30 ET) | pts 0.5 | Under | 0.66 | 48.5% | 56.6% | +8.0 pts | +13.2% | 2 1 0 0 0 1 0 1 0 0 | 0.6 |
| A. Fantilli | CBJ @ NJD (19:00 ET) | sog 2.5 | Over | 3.06 | 56.5% | 47.3% | +9.2 pts | +13.1% | 3 4 5 0 3 2 2 0 5 8 | 3.1 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 357 | -0.6% | -11% to +9% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 312 | -5.4% | -17% to +6% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +4% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -8% to +23% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -42% to +7% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -22% to +2% | 60% | - |  |
| `edge_1pct` | 43 | -30.2% | -60% to +2% | 33% | -1.49% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 372 | -0.3% | -9% to +8% | 59% | -2.00% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 371 | -5.1% | -13% to +2% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 357 | -5.0% | -15% to +5% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 372 | -15.8% | -27% to -5% | 40% | -2.73% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -20% to +23% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -58% to +33% | 14% | - |  |
| `home_ml_control` | 325 | -5.6% | -16% to +5% | 52% | -2.53% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 372 | +1.8% | -7% to +10% | 61% | -2.07% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -74% to +9% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 45 | -22.7% | -51% to +6% | 38% | -2.44% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 74 | -5.4% | -28% to +16% | 50% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 45 | +12.8% | -41% to +71% | 27% | - |  |
| `parlay_players3` | 25 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 45 | -61.1% | -93% to -17% | 9% | - |  |
| `parlay_top3` | 45 | -93.1% | -100% to -79% | 2% | - |  |
| `pts_edge` | 254 | -15.0% | -26% to -4% | 46% | - |  |
| `pts_over_control` | 259 | +1.6% | -9% to +13% | 59% | - |  |
| `pts_over_edge` | 213 | -24.9% | -40% to -10% | 34% | - |  |
| `pts_under_edge` | 252 | -7.7% | -19% to +4% | 51% | - |  |
| `puckline_dog_control` | 324 | -4.9% | -13% to +4% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 108 | -2.4% | -17% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 158 | -11.3% | -27% to +5% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 80 | -20.3% | -46% to +7% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 309 | -6.8% | -17% to +4% | 50% | - |  |
| `sog_over_control` | 313 | +4.5% | -7% to +16% | 52% | - |  |
| `sog_over_edge` | 219 | -8.9% | -22% to +4% | 46% | - |  |
| `sog_under_edge` | 268 | -7.5% | -19% to +4% | 50% | - |  |
| `totals_edge` | 211 | -7.4% | -20% to +6% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 145 | -8.0% | -24% to +8% | 47% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 170 | -26.0% | -43% to -8% | 31% | -2.96% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 325 | -10.4% | -22% to +3% | 38% | -2.93% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CBJ @ NJD | Moneyline | NJD moneyline | $10.00 | 1.51 | 62.1% | 65.0% |
| LAK @ MTL | Moneyline | MTL moneyline | $10.00 | 1.67 | 56.3% | 58.2% |
| CGY @ OTT | Moneyline | OTT moneyline | $10.00 | 1.42 | 63.9% | 69.6% |
| TOR @ NYI | Moneyline | NYI moneyline | $10.00 | 1.65 | 56.3% | 59.5% |
| UTA @ WPG | Moneyline | UTA moneyline | $10.00 | 1.93 | 53.9% | 50.1% |
| ANA @ SEA | Moneyline | ANA moneyline | $10.00 | 2.01 | 50.9% | 48.1% |
| MIN @ SJS | Moneyline | MIN moneyline | $10.00 | 1.63 | 63.0% | 61.3% |
| CBJ @ NJD | Total | Over 6 | $10.00 | 1.95 | 52.6% | 50.3% |
| LAK @ MTL | Total | Over 6 | $10.00 | 1.84 | 52.1% | 52.5% |
| CGY @ OTT | Total | Over 5.5 | $10.00 | 1.87 | 58.8% | 51.6% |
| TOR @ NYI | Total | Under 6.5 | $10.00 | 1.91 | 54.2% | 50.8% |
| UTA @ WPG | Total | Under 6.5 | $10.00 | 1.90 | 54.6% | 51.1% |
| ANA @ SEA | Total | Under 6.5 | $10.00 | 2.17 | 53.8% | 45.1% |
| MIN @ SJS | Total | Under 6.5 | $10.00 | 1.85 | 53.8% | 52.5% |
| CBJ @ NJD | Puck line | CBJ +1.5 | $10.00 | 1.69 | 56.7% | 57.5% |
| LAK @ MTL | Puck line | LAK +1.5 | $10.00 | 1.54 | 65.6% | 63.6% |
| CGY @ OTT | Puck line | CGY +1.5 | $10.00 | 1.82 | 58.3% | 53.7% |
| TOR @ NYI | Puck line | TOR +1.5 | $10.00 | 1.56 | 64.8% | 62.7% |
| UTA @ WPG | Puck line | WPG +1.5 | $10.00 | 1.40 | 65.3% | 70.2% |
| ANA @ SEA | Puck line | ANA +1.5 | $10.00 | 1.43 | 74.1% | 68.9% |
| MIN @ SJS | Puck line | SJS +1.5 | $10.00 | 1.62 | 53.1% | 60.1% |

Total pretend stake $210.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
