# NHL model report: 2026-10-25 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-25 16:05 UTC · model `20261024-43498a` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-25T16:04:00+00:00 · API credits left: 103*

*Player props: 1 game(s) priced*

## Summary

- 5 games · **1 recommended bet(s)** · total stake $2.34 (0.23% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| FLA @ CHI (15:00 ET) | J. Markstrom / S. Knight (probable) | 40.8% | 36.9% | CHI 2.65 | +3.9% | - | - | no bet |
| NYI @ DAL (18:00 ET) | I. Sorokin / J. Oettinger (probable) | 60.5% | 63.6% | NYI 2.66 | +3.1% | - | - | no bet |
| MTL @ WPG (19:00 ET) | J. Dobes / C. Hellebuyck (probable) | 50.3% | 48.1% | WPG 2.04 | +2.2% | - | - | no bet |
| COL @ NSH (20:00 ET) | M. Blackwood / J. Saros (probable) | 36.7% | 32.1% | NSH 3.10 | +4.6% | - | - | no bet |
| MIN @ VAN (22:00 ET) | J. Wallstedt / K. Lankinen (probable) | 41.1% | 36.0% | VAN 2.71 | +5.1% | +11.4% | $2.34 | **BET VAN** |

## Why

**FLA @ CHI (15:00 ET)**: No bet: edge +3.9% below the 4% minimum (starting goalie unconfirmed)
  - Context: FLA have the clearly better goal differential this season (gap 0.74 goals/game). FLA are creating the better share of expected goals recently. Goaltending edge to FLA (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**NYI @ DAL (18:00 ET)**: No bet: edge +3.1% below the 4% minimum (starting goalie unconfirmed); model win probability 39.5% below the 40% confidence floor
  - Context: DAL have the clearly better goal differential this season (gap 0.30 goals/game). DAL are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**MTL @ WPG (19:00 ET)**: No bet: edge +2.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: MTL have the clearly better goal differential this season (gap 0.26 goals/game). Goaltending edge to WPG (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**COL @ NSH (20:00 ET)**: No bet: model win probability 36.7% below the 40% confidence floor
  - Context: COL have the clearly better goal differential this season (gap 1.22 goals/game). NSH is on the second night of a back-to-back. COL are creating the better share of expected goals recently. Division/rival game.

**MIN @ VAN (22:00 ET)**: Bet $2.34 (0.23% of bankroll) on VAN at 2.71. The model gives VAN 41.1% vs the market's no-vig 36.0% (edge +5.1%); after shrinking toward the market the working probability is 37.5%, worth +1.6% per $1 at that price.
  - Context: MIN have the clearly better goal differential this season (gap 1.55 goals/game). MIN is on the second night of a back-to-back. MIN are creating the better share of expected goals recently. Goaltending edge to MIN (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **174** (bet on 0, passed on 100%); model log loss 0.6593, Brier 0.2335 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (174 games with odds): model log loss 0.6593 vs market 0.6455 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 40.1% | 41.4% | 29 |
| 48.3% | 58.6% | 29 |
| 52.4% | 58.6% | 29 |
| 56.3% | 62.1% | 29 |
| 60.1% | 65.5% | 29 |
| 66.4% | 72.4% | 29 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| NYI @ DAL (18:00 ET) | Total | Over 5.5 | 58.0% | 48.2% | +9.8 pts | +17.8% | 2.03 |
| COL @ NSH (20:00 ET) | Total | Under 6.5 | 54.7% | 48.7% | +6.0 pts | +9.3% | 2.00 |
| MIN @ VAN (22:00 ET) | Total | Under 6.5 | 55.3% | 50.3% | +5.0 pts | +6.2% | 1.92 |
| MTL @ WPG (19:00 ET) | Total | Over 6 | 52.4% | 48.2% | +4.2 pts | +4.7% | 2.01 |
| MIN @ VAN (22:00 ET) | Puck line | MIN -1.5 | 45.5% | 42.4% | +3.1 pts | +4.7% | 2.30 |
| NYI @ DAL (18:00 ET) | Puck line | NYI +1.5 | 63.3% | 60.3% | +3.1 pts | +4.5% | 1.65 |
| COL @ NSH (20:00 ET) | Puck line | NSH +1.5 | 56.4% | 53.7% | +2.7 pts | +3.1% | 1.83 |
| FLA @ CHI (15:00 ET) | Puck line | FLA -1.5 | 42.7% | 40.7% | +2.0 pts | +2.5% | 2.40 |
| MTL @ WPG (19:00 ET) | Puck line | MTL -1.5 | 30.7% | 30.0% | +0.8 pts | -2.9% | 3.16 |
| FLA @ CHI (15:00 ET) | Total | Under 6 | 48.1% | 47.6% | +0.5 pts | -1.2% | 2.05 |
| FLA @ CHI (15:00 ET) | Total | Over 6 | 51.9% | 52.4% | -0.5 pts | -3.5% | 1.85 |
| MTL @ WPG (19:00 ET) | Puck line | WPG +1.5 | 69.3% | 70.0% | -0.8 pts | -3.0% | 1.40 |
| FLA @ CHI (15:00 ET) | Puck line | CHI +1.5 | 57.3% | 59.3% | -2.0 pts | -5.5% | 1.65 |
| COL @ NSH (20:00 ET) | Puck line | COL -1.5 | 43.6% | 46.3% | -2.7 pts | -8.3% | 2.10 |
| NYI @ DAL (18:00 ET) | Puck line | DAL -1.5 | 36.7% | 39.7% | -3.1 pts | -9.1% | 2.48 |
| MIN @ VAN (22:00 ET) | Puck line | VAN +1.5 | 54.5% | 57.6% | -3.1 pts | -7.9% | 1.69 |
| MTL @ WPG (19:00 ET) | Total | Under 6 | 47.6% | 51.8% | -4.2 pts | -9.6% | 1.87 |
| MIN @ VAN (22:00 ET) | Total | Over 6.5 | 44.7% | 49.7% | -5.0 pts | -12.8% | 1.95 |
| COL @ NSH (20:00 ET) | Total | Over 6.5 | 45.3% | 51.3% | -6.0 pts | -14.8% | 1.88 |
| NYI @ DAL (18:00 ET) | Total | Under 5.5 | 42.0% | 51.8% | -9.8 pts | -21.5% | 1.87 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| C. Verhaeghe | FLA @ CHI (15:00 ET) | sog 2.5 | Under | 2.08 | 34.2% | 47.7% | +13.5 pts | +20.5% | 0 3 2 0 0 3 1 5 1 1 | 1.6 |
| M. Tkachuk | FLA @ CHI (15:00 ET) | sog 2.5 | Under | 2.32 | 40.2% | 51.6% | +11.4 pts | +16.6% | 2 1 2 4 2 1 1 4 2 1 | 2.0 |
| B. Tkachuk | FLA @ CHI (15:00 ET) | sog 3.5 | Under | 2.96 | 34.3% | 43.8% | +9.4 pts | +16.2% | 4 4 2 1 3 4 6 7 3 3 | 3.7 |
| S. Reinhart | FLA @ CHI (15:00 ET) | sog 2.5 | Over | 3.00 | 55.3% | 46.6% | +8.6 pts | +14.9% | 2 3 3 2 2 7 5 3 0 6 | 3.3 |
| F. Nazar | FLA @ CHI (15:00 ET) | sog 1.5 | Over | 1.86 | 53.7% | 48.6% | +5.2 pts | +4.8% | 0 0 3 0 2 3 1 3 2 0 | 1.6 |
| P. Kane | FLA @ CHI (15:00 ET) | sog 2.5 | Under | 2.20 | 37.3% | 41.4% | +4.1 pts | +4.8% | 4 0 0 2 0 5 0 2 3 3 | 1.7 |
| S. Bennett | FLA @ CHI (15:00 ET) | sog 2.5 | Over | 2.60 | 46.7% | 43.4% | +3.2 pts | +2.7% | 1 4 1 2 1 1 6 4 2 4 | 2.6 |
| T. Teravainen | FLA @ CHI (15:00 ET) | sog 1.5 | Under | 1.27 | 35.7% | 40.3% | +4.6 pts | +2.2% | 1 1 1 1 0 0 3 2 0 2 | 1.1 |
| A. Lundell | FLA @ CHI (15:00 ET) | sog 1.5 | Over | 2.17 | 61.5% | 57.9% | +3.6 pts | +1.5% | 3 1 3 6 3 1 4 1 1 2 | 2.5 |
| T. Bertuzzi | FLA @ CHI (15:00 ET) | sog 1.5 | Under | 1.49 | 43.1% | 46.0% | +3.0 pts | +0.8% | 0 2 1 0 1 1 3 1 1 0 | 1.1 |
| E. Luostarinen | FLA @ CHI (15:00 ET) | sog 1.5 | - | 1.54 | 44.7% | 49.0% | +4.3 pts | -0.4% | 3 0 2 2 2 1 0 2 2 1 | 1.5 |
| S. Jones | FLA @ CHI (15:00 ET) | sog 1.5 | - | 2.04 | 58.3% | 55.6% | +2.7 pts | -1.4% | 2 4 2 3 4 3 5 1 3 1 | 2.8 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 166 | -5.3% | -20% to +9% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 121 | +0.6% | -17% to +18% | 51% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -35% to +23% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -25% to +33% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +55% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -23% to +26% | 65% | - |  |
| `edge_1pct` | 1 | +106.0% | - | 100% | -1.72% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 174 | -0.4% | -13% to +11% | 59% | -2.11% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 173 | -7.7% | -19% to +4% | 60% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 166 | -16.2% | -30% to -2% | 44% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 174 | -18.6% | -34% to -3% | 40% | -2.55% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -49% to +31% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +92% | 10% | - |  |
| `home_ml_control` | 127 | +10.6% | -6% to +27% | 61% | -2.67% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 174 | +2.3% | -10% to +14% | 61% | -2.11% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 15 | -4.5% | - | 40% | -1.36% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 2 | -4.5% | - | 50% | -4.14% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 26 | -41.9% | -71% to -6% | 31% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 17 | -21.2% | - | 24% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 17 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 17 | -100.0% | - | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +4% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -24% to +11% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -64% to -6% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -34% to +5% | 47% | - |  |
| `puckline_dog_control` | 126 | -6.8% | -20% to +7% | 60% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 59 | -4.3% | -25% to +14% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 90 | -13.8% | -33% to +6% | 50% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 22 | -14.5% | -66% to +38% | 32% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 134 | +2.6% | -12% to +18% | 56% | - |  |
| `sog_over_control` | 138 | -4.7% | -22% to +13% | 46% | - |  |
| `sog_over_edge` | 67 | -14.4% | -37% to +9% | 45% | - |  |
| `sog_under_edge` | 96 | +4.4% | -14% to +24% | 56% | - |  |
| `totals_edge` | 98 | -23.7% | -42% to -5% | 40% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 57 | -18.2% | -43% to +6% | 42% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 64 | -25.4% | -53% to +4% | 31% | -2.74% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 127 | -14.6% | -35% to +5% | 37% | -2.82% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| FLA @ CHI | Moneyline | FLA moneyline | $10.00 | 1.55 | 59.2% | 63.1% |
| NYI @ DAL | Moneyline | DAL moneyline | $10.00 | 1.55 | 60.5% | 63.6% |
| MTL @ WPG | Moneyline | WPG moneyline | $10.00 | 2.04 | 50.3% | 48.1% |
| COL @ NSH | Moneyline | COL moneyline | $10.00 | 1.44 | 63.3% | 67.9% |
| MIN @ VAN | Moneyline | MIN moneyline | $10.00 | 1.54 | 58.9% | 64.0% |
| FLA @ CHI | Total | Over 6 | $10.00 | 1.85 | 51.9% | 52.4% |
| NYI @ DAL | Total | Over 5.5 | $10.00 | 2.03 | 58.0% | 48.2% |
| MTL @ WPG | Total | Over 6 | $10.00 | 2.01 | 52.4% | 48.2% |
| COL @ NSH | Total | Under 6.5 | $10.00 | 2.00 | 54.7% | 48.7% |
| MIN @ VAN | Total | Under 6.5 | $10.00 | 1.92 | 55.3% | 50.3% |
| FLA @ CHI | Puck line | CHI +1.5 | $10.00 | 1.65 | 57.3% | 59.3% |
| NYI @ DAL | Puck line | NYI +1.5 | $10.00 | 1.65 | 63.3% | 60.3% |
| MTL @ WPG | Puck line | WPG +1.5 | $10.00 | 1.40 | 69.3% | 70.0% |
| COL @ NSH | Puck line | NSH +1.5 | $10.00 | 1.83 | 56.4% | 53.7% |
| MIN @ VAN | Puck line | VAN +1.5 | $10.00 | 1.69 | 54.5% | 57.6% |

Total pretend stake $150.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
