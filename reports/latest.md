# NHL model report: 2026-11-16 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-16 13:04 UTC · model `20261115-b5ee33` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-16T13:03:27+00:00 · API credits left: 239*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| MTL @ NYR (19:30 ET) | J. Fowler / D. Garand (probable) | 50.7% | 48.8% | NYR 2.03 | +2.0% | - | - | no bet |
| VAN @ UTA (21:30 ET) | L. Meriläinen / K. Vejmelka (probable) | 73.8% | 75.3% | VAN 3.87 | +1.5% | - | - | no bet |

## Why

**MTL @ NYR (19:30 ET)**: No bet: edge +2.0% below the 4% minimum (starting goalie unconfirmed)

**VAN @ UTA (21:30 ET)**: No bet: edge +1.5% below the 4% minimum (starting goalie unconfirmed); model win probability 26.2% below the 40% confidence floor
  - Context: UTA have the clearly better goal differential this season (gap 1.14 goals/game). VAN is on the second night of a back-to-back. UTA are creating the better share of expected goals recently. Goaltending edge to UTA (goals saved above expected).

## Track record (all logged recommendations that have resolved)

- Resolved games: **325** (bet on 11, passed on 97%); model log loss 0.6609, Brier 0.2344 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (325 games with odds): model log loss 0.6609 vs market 0.6445 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.47% to -0.58%), beat the close on 9% of 11 bets
- ⚠️ 11 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.5% | 39.0% | 41 |
| 46.4% | 39.0% | 41 |
| 49.8% | 42.5% | 40 |
| 52.6% | 53.7% | 41 |
| 55.2% | 47.5% | 40 |
| 58.3% | 63.4% | 41 |
| 61.8% | 65.0% | 40 |
| 68.4% | 75.6% | 41 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 312 | -4.2% | -15% to +6% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 267 | -1.4% | -14% to +10% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 158 | -8.6% | -22% to +5% | 56% | - |  |
| `ast_over_control` | 159 | +7.8% | -7% to +23% | 56% | - |  |
| `ast_over_edge` | 130 | -17.9% | -41% to +7% | 28% | - |  |
| `ast_under_edge` | 157 | -9.4% | -22% to +3% | 60% | - |  |
| `edge_1pct` | 34 | -50.2% | -78% to -17% | 24% | -1.72% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 325 | +1.4% | -7% to +10% | 60% | -1.98% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 324 | -4.2% | -13% to +4% | 62% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 312 | -5.5% | -16% to +5% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 325 | -22.1% | -33% to -11% | 38% | -2.72% | $10 on every game where the model sees any positive edge |
| `goal_control` | 160 | +0.9% | -20% to +23% | 36% | - |  |
| `goal_edge` | 99 | -12.7% | -55% to +39% | 14% | - |  |
| `home_ml_control` | 278 | -5.1% | -16% to +7% | 53% | -2.55% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 325 | +3.3% | -5% to +12% | 62% | -2.08% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -73% to +10% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 39 | -28.7% | -59% to +5% | 33% | -2.55% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 66 | -11.1% | -34% to +13% | 47% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 39 | +30.1% | -28% to +95% | 31% | - |  |
| `parlay_players3` | 20 | -100.0% | -100% to -100% | 0% | - |  |
| `parlay_top2` | 39 | -76.3% | -100% to -39% | 5% | - |  |
| `parlay_top3` | 39 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 206 | -13.1% | -27% to -1% | 47% | - |  |
| `pts_over_control` | 211 | +1.5% | -10% to +13% | 59% | - |  |
| `pts_over_edge` | 174 | -30.9% | -48% to -15% | 31% | - |  |
| `pts_under_edge` | 204 | -7.7% | -21% to +5% | 51% | - |  |
| `puckline_dog_control` | 277 | -6.1% | -15% to +3% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 94 | -3.5% | -19% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 144 | -13.6% | -29% to +2% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 69 | -23.0% | -52% to +7% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 261 | -3.9% | -15% to +8% | 51% | - |  |
| `sog_over_control` | 266 | +1.8% | -11% to +14% | 51% | - |  |
| `sog_over_edge` | 185 | -6.3% | -20% to +8% | 47% | - |  |
| `sog_under_edge` | 220 | -6.9% | -19% to +6% | 50% | - |  |
| `totals_edge` | 184 | -7.5% | -21% to +6% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 124 | -3.6% | -21% to +14% | 49% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 144 | -34.4% | -52% to -16% | 27% | -2.88% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 278 | -14.4% | -28% to -1% | 37% | -2.92% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| MTL @ NYR | Moneyline | NYR moneyline | $10.00 | 2.03 | 50.7% | 48.8% |
| VAN @ UTA | Moneyline | UTA moneyline | $10.00 | 1.30 | 73.8% | 75.3% |

Total pretend stake $20.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
