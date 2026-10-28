# NHL model report: 2026-10-28 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-28 12:04 UTC · model `20261027-c41a5a` · probability source `online_platt` · bankroll $992.91 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-28T12:03:56+00:00 · API credits left: 80*

## Summary

- 4 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PHI @ WSH (19:30 ET) | D. Vladar / L. Thompson (probable) | 65.7% | 67.5% | PHI 2.95 | +1.8% | - | - | no bet |
| CGY @ EDM (20:30 ET) | D. Wolf / D. Levi (probable) | 59.9% | 66.3% | CGY 2.85 | +6.3% | - | - | no bet |
| NJD @ COL (21:00 ET) | J. Allen / M. Blackwood (probable) | 60.7% | 60.8% | NJD 2.45 | +0.2% | - | - | no bet |
| TOR @ SEA (22:00 ET) | S. Bobrovsky / J. Daccord (probable) | 52.2% | 47.8% | SEA 2.04 | +4.5% | - | - | no bet |

## Why

**PHI @ WSH (19:30 ET)**: No bet: edge +1.8% below the 4% minimum (starting goalie unconfirmed); model win probability 34.3% below the 40% confidence floor
  - Context: WSH have the clearly better goal differential this season (gap 0.79 goals/game). PHI is on the second night of a back-to-back. WSH are creating the better share of expected goals recently. Division/rival game.

**CGY @ EDM (20:30 ET)**: No bet: expected value after shrinkage toward the market is only +0.4% (< 1%)
  - Context: EDM have the clearly better goal differential this season (gap 1.02 goals/game). EDM are creating the better share of expected goals recently. Goaltending edge to EDM (goals saved above expected). Division/rival game.

**NJD @ COL (21:00 ET)**: No bet: edge +0.2% below the 4% minimum (starting goalie unconfirmed); model win probability 39.3% below the 40% confidence floor
  - Context: COL have the clearly better goal differential this season (gap 0.43 goals/game). NJD is on the second night of a back-to-back. NJD are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**TOR @ SEA (22:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.4% (< 1%)
  - Context: TOR are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **193** (bet on 2, passed on 99%); model log loss 0.6630, Brier 0.2352 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (193 games with odds): model log loss 0.6630 vs market 0.6504 (market better)
- Bets: 2 · staked $7.09 · profit $-7.09 · **ROI -100.0%** · win rate 0.0% · avg price 2.37
- Bankroll $992.91 · max drawdown 0.7% · current drawdown 0.7%
- ⚠️ 2 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.3% | 39.3% | 28 |
| 47.2% | 51.8% | 27 |
| 50.9% | 50.0% | 28 |
| 53.9% | 55.6% | 27 |
| 57.3% | 60.7% | 28 |
| 60.8% | 74.1% | 27 |
| 66.8% | 67.9% | 28 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 184 | -6.4% | -20% to +8% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 139 | +1.8% | -14% to +18% | 52% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -36% to +22% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -25% to +32% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -69% to +54% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +26% | 65% | - |  |
| `edge_1pct` | 5 | -58.8% | - | 20% | -2.71% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 193 | -0.2% | -12% to +11% | 59% | -2.10% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 192 | -6.5% | -17% to +4% | 61% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 184 | -11.8% | -26% to +2% | 46% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 193 | -18.8% | -34% to -5% | 39% | -2.55% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -47% to +32% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +94% | 10% | - |  |
| `home_ml_control` | 146 | +4.4% | -11% to +20% | 57% | -2.62% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 193 | +2.0% | -9% to +14% | 61% | -2.10% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 17 | -15.7% | - | 35% | -1.57% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 7 | -5.7% | - | 43% | -3.51% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 33 | -25.6% | -55% to +9% | 39% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 20 | +17.9% | -60% to +102% | 30% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 20 | -72.6% | -100% to -18% | 5% | - |  |
| `parlay_top3` | 20 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 87 | -16.5% | -37% to +3% | 44% | - |  |
| `pts_over_control` | 91 | -6.5% | -24% to +11% | 56% | - |  |
| `pts_over_edge` | 64 | -36.7% | -65% to -7% | 23% | - |  |
| `pts_under_edge` | 87 | -13.9% | -34% to +6% | 47% | - |  |
| `puckline_dog_control` | 145 | -5.3% | -18% to +7% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 64 | -4.2% | -23% to +14% | 61% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 98 | -15.4% | -34% to +3% | 49% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 29 | -27.2% | -69% to +16% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 142 | +1.7% | -14% to +16% | 56% | - |  |
| `sog_over_control` | 146 | -4.5% | -21% to +12% | 47% | - |  |
| `sog_over_edge` | 72 | -12.6% | -35% to +10% | 46% | - |  |
| `sog_under_edge` | 103 | +3.9% | -14% to +21% | 56% | - |  |
| `totals_edge` | 113 | -16.6% | -34% to +2% | 43% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 65 | -12.8% | -37% to +11% | 45% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 75 | -23.2% | -47% to +2% | 32% | -2.74% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 146 | -12.8% | -31% to +6% | 38% | -2.84% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PHI @ WSH | Moneyline | WSH moneyline | $10.00 | 1.45 | 65.7% | 67.5% |
| CGY @ EDM | Moneyline | EDM moneyline | $10.00 | 1.48 | 59.9% | 66.3% |
| NJD @ COL | Moneyline | COL moneyline | $10.00 | 1.61 | 60.7% | 60.8% |
| TOR @ SEA | Moneyline | SEA moneyline | $10.00 | 2.04 | 52.2% | 47.8% |

Total pretend stake $40.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
