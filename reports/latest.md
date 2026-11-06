# NHL model report: 2026-11-06 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-06 13:03 UTC · model `20261105-54203c` · probability source `online_platt` · bankroll $989.34 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-06T13:02:06+00:00 · API credits left: 409*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| CBJ @ BUF (19:00 ET) | J. Greaves / U. Luukkonen (probable) | 62.2% | 61.3% | BUF 1.59 | +0.9% | - | - | no bet |
| SJS @ MIN (20:00 ET) | Y. Askarov / J. Wallstedt (probable) | 69.3% | 70.6% | SJS 3.30 | +1.3% | - | - | no bet |
| ANA @ CGY (20:00 ET) | L. Dostal / D. Wolf (probable) | 56.4% | 53.7% | CGY 1.82 | +2.7% | - | - | no bet |

## Why

**CBJ @ BUF (19:00 ET)**: No bet: edge +0.9% below the 4% minimum (starting goalie unconfirmed)
  - Context: CBJ is on the second night of a back-to-back. Early in the season: team ratings lean on last year's results and are less reliable.

**SJS @ MIN (20:00 ET)**: No bet: edge +1.3% below the 4% minimum (starting goalie unconfirmed); model win probability 30.7% below the 40% confidence floor
  - Context: MIN have the clearly better goal differential this season (gap 1.30 goals/game). MIN are creating the better share of expected goals recently. Goaltending edge to MIN (goals saved above expected).

**ANA @ CGY (20:00 ET)**: No bet: edge +2.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: ANA have the clearly better goal differential this season (gap 0.54 goals/game). ANA is on the second night of a back-to-back. Division/rival game.

## Track record (all logged recommendations that have resolved)

- Resolved games: **256** (bet on 6, passed on 98%); model log loss 0.6565, Brier 0.2322 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (256 games with odds): model log loss 0.6565 vs market 0.6441 (market better)
- Bets: 6 · staked $15.54 · profit $-10.66 · **ROI -68.6%** · win rate 16.7% · avg price 2.46
- Bankroll $989.34 · max drawdown 1.1% · current drawdown 1.1%
- **CLV**: mean -2.36% per $1 (95% CI -4.35% to +0.63%), beat the close on 17% of 6 bets
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 244 | -1.5% | -13% to +11% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 199 | -5.1% | -19% to +8% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 80 | -8.7% | -27% to +9% | 56% | - |  |
| `ast_over_control` | 79 | +7.7% | -14% to +28% | 57% | - |  |
| `ast_over_edge` | 67 | -19.5% | -56% to +22% | 22% | - |  |
| `ast_under_edge` | 80 | -5.0% | -22% to +12% | 61% | - |  |
| `edge_1pct` | 19 | -32.5% | - | 32% | -1.22% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 256 | +1.1% | -9% to +11% | 60% | -1.92% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 255 | -2.9% | -12% to +6% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 244 | -7.4% | -19% to +5% | 48% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 256 | -18.8% | -31% to -6% | 39% | -2.56% | $10 on every game where the model sees any positive edge |
| `goal_control` | 80 | -8.6% | -37% to +21% | 32% | - |  |
| `goal_edge` | 54 | -4.4% | -69% to +71% | 13% | - |  |
| `home_ml_control` | 209 | +2.3% | -11% to +15% | 56% | -2.54% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 256 | +2.3% | -8% to +12% | 62% | -2.02% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 21 | -19.0% | -66% to +32% | 33% | -1.65% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 26 | -1.0% | -41% to +43% | 46% | -2.44% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 54 | -9.0% | -34% to +16% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 29 | +7.9% | -53% to +80% | 28% | - |  |
| `parlay_players3` | 11 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 29 | -68.1% | -100% to -17% | 7% | - |  |
| `parlay_top3` | 29 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 127 | -13.3% | -31% to +4% | 46% | - |  |
| `pts_over_control` | 131 | -6.5% | -22% to +8% | 56% | - |  |
| `pts_over_edge` | 101 | -34.6% | -56% to -12% | 27% | - |  |
| `pts_under_edge` | 127 | -10.1% | -26% to +6% | 49% | - |  |
| `puckline_dog_control` | 208 | -3.9% | -14% to +7% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 80 | -2.6% | -20% to +14% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 120 | -15.8% | -32% to +1% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 50 | -23.8% | -57% to +9% | 30% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 182 | +0.4% | -14% to +14% | 54% | - |  |
| `sog_over_control` | 186 | -3.8% | -18% to +11% | 48% | - |  |
| `sog_over_edge` | 112 | -11.8% | -30% to +6% | 46% | - |  |
| `sog_under_edge` | 143 | +0.9% | -14% to +16% | 55% | - |  |
| `totals_edge` | 148 | -13.8% | -28% to +2% | 45% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 90 | -12.4% | -31% to +7% | 44% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 108 | -27.6% | -48% to -7% | 31% | -2.76% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 209 | -14.1% | -29% to +1% | 37% | -2.94% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| CBJ @ BUF | Moneyline | BUF moneyline | $10.00 | 1.59 | 62.2% | 61.3% |
| SJS @ MIN | Moneyline | MIN moneyline | $10.00 | 1.39 | 69.3% | 70.6% |
| ANA @ CGY | Moneyline | CGY moneyline | $10.00 | 1.82 | 56.4% | 53.7% |

Total pretend stake $30.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
