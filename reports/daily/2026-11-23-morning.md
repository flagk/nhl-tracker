# NHL model report: 2026-11-23 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-23 13:04 UTC · model `20261122-54dfe0` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-23T13:03:50+00:00 · API credits left: 162*

## Summary

- 7 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CBJ @ NJD (19:00 ET) | C. Talbot / C. Hellebuyck (probable) | 62.1% | 65.1% | CBJ 2.75 | +3.0% | - | - | no bet |
| LAK @ MTL (19:30 ET) | D. Kuemper / J. Fowler (probable) | 56.3% | 58.4% | LAK 2.36 | +2.1% | - | - | no bet |
| CGY @ OTT (19:30 ET) | D. Wolf / L. Ullmark (probable) | 63.9% | 67.6% | CGY 2.98 | +3.7% | - | - | no bet |
| TOR @ NYI (19:30 ET) | S. Bobrovsky / I. Sorokin (probable) | 56.3% | 59.5% | TOR 2.39 | +3.1% | - | - | no bet |
| UTA @ WPG (19:30 ET) | K. Vejmelka / S. Skinner (probable) | 46.1% | 48.3% | UTA 1.88 | +2.2% | - | - | no bet |
| ANA @ SEA (21:40 ET) | L. Dostal / P. Grubauer (probable) | 49.1% | 50.3% | ANA 1.96 | +1.2% | - | - | no bet |
| MIN @ SJS (22:00 ET) | J. Wallstedt / Y. Askarov (probable) | 37.0% | 35.4% | SJS 2.75 | +1.7% | - | - | no bet |

## Why

**CBJ @ NJD (19:00 ET)**: No bet: edge +3.0% below the 4% minimum (starting goalie unconfirmed); model win probability 37.9% below the 40% confidence floor
  - Context: NJD have the clearly better goal differential this season (gap 0.73 goals/game). NJD are creating the better share of expected goals recently. Division/rival game.

**LAK @ MTL (19:30 ET)**: No bet: edge +2.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: MTL have the clearly better goal differential this season (gap 0.61 goals/game). LAK are creating the better share of expected goals recently. Goaltending edge to MTL (goals saved above expected).

**CGY @ OTT (19:30 ET)**: No bet: edge +3.7% below the 4% minimum (starting goalie unconfirmed); model win probability 36.1% below the 40% confidence floor
  - Context: OTT have the clearly better goal differential this season (gap 1.01 goals/game). CGY travelled 2,864 km across 2 time zone(s) since their last game. OTT are creating the better share of expected goals recently. Goaltending edge to OTT (goals saved above expected).

**TOR @ NYI (19:30 ET)**: No bet: edge +3.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: NYI have the clearly better goal differential this season (gap 0.60 goals/game). Goaltending edge to NYI (goals saved above expected).

**UTA @ WPG (19:30 ET)**: No bet: edge +2.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: UTA have the clearly better goal differential this season (gap 0.27 goals/game). UTA are creating the better share of expected goals recently. Goaltending edge to WPG (goals saved above expected). Division/rival game.

**ANA @ SEA (21:40 ET)**: No bet: edge +1.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: Division/rival game.

**MIN @ SJS (22:00 ET)**: No bet: edge +1.7% below the 4% minimum (starting goalie unconfirmed); model win probability 37.0% below the 40% confidence floor
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 357 | -0.6% | -10% to +10% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 312 | -5.4% | -17% to +5% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +22% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -41% to +7% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -21% to +2% | 60% | - |  |
| `edge_1pct` | 43 | -30.2% | -60% to +1% | 33% | -1.49% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 372 | -0.3% | -8% to +8% | 59% | -2.00% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 371 | -5.1% | -12% to +3% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 357 | -5.0% | -15% to +5% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 372 | -15.8% | -27% to -5% | 40% | -2.73% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -19% to +23% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -56% to +34% | 14% | - |  |
| `home_ml_control` | 325 | -5.6% | -16% to +5% | 52% | -2.53% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 372 | +1.8% | -6% to +10% | 61% | -2.07% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -74% to +9% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 45 | -22.7% | -51% to +6% | 38% | -2.44% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 74 | -5.4% | -27% to +17% | 50% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 45 | +12.8% | -41% to +71% | 27% | - |  |
| `parlay_players3` | 25 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 45 | -61.1% | -93% to -17% | 9% | - |  |
| `parlay_top3` | 45 | -93.1% | -100% to -79% | 2% | - |  |
| `pts_edge` | 254 | -15.0% | -26% to -3% | 46% | - |  |
| `pts_over_control` | 259 | +1.6% | -10% to +13% | 59% | - |  |
| `pts_over_edge` | 213 | -24.9% | -39% to -10% | 34% | - |  |
| `pts_under_edge` | 252 | -7.7% | -19% to +3% | 51% | - |  |
| `puckline_dog_control` | 324 | -4.9% | -13% to +3% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 108 | -2.4% | -17% to +11% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 158 | -11.3% | -27% to +5% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 80 | -20.3% | -47% to +7% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 309 | -6.8% | -17% to +4% | 50% | - |  |
| `sog_over_control` | 313 | +4.5% | -7% to +16% | 52% | - |  |
| `sog_over_edge` | 219 | -8.9% | -22% to +5% | 46% | - |  |
| `sog_under_edge` | 268 | -7.5% | -18% to +4% | 50% | - |  |
| `totals_edge` | 211 | -7.4% | -20% to +6% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 145 | -8.0% | -24% to +8% | 47% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 170 | -26.0% | -43% to -8% | 31% | -2.96% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 325 | -10.4% | -24% to +2% | 38% | -2.93% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CBJ @ NJD | Moneyline | NJD moneyline | $10.00 | 1.50 | 62.1% | 65.1% |
| LAK @ MTL | Moneyline | MTL moneyline | $10.00 | 1.67 | 56.3% | 58.4% |
| CGY @ OTT | Moneyline | OTT moneyline | $10.00 | 1.45 | 63.9% | 67.6% |
| TOR @ NYI | Moneyline | NYI moneyline | $10.00 | 1.65 | 56.3% | 59.5% |
| UTA @ WPG | Moneyline | UTA moneyline | $10.00 | 1.88 | 53.9% | 51.7% |
| ANA @ SEA | Moneyline | ANA moneyline | $10.00 | 1.96 | 50.9% | 49.7% |
| MIN @ SJS | Moneyline | MIN moneyline | $10.00 | 1.51 | 63.0% | 64.6% |

Total pretend stake $70.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
