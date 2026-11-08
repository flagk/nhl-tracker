# NHL model report: 2026-11-08 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-08 13:03 UTC · model `20261107-e20502` · probability source `online_platt` · bankroll $986.76 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-08T13:02:24+00:00 · API credits left: 373*

## Summary

- 6 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| FLA @ BOS (13:00 ET) | A. Schmid / J. Swayman (probable) | 57.2% | 56.0% | BOS 1.75 | +1.2% | - | - | no bet |
| VGK @ BUF (15:30 ET) | C. Hart / U. Luukkonen (probable) | 53.0% | 50.1% | BUF 1.93 | +2.9% | - | - | no bet |
| TBL @ WPG (18:00 ET) | A. Vasilevskiy / S. Skinner (probable) | 43.6% | 42.9% | WPG 2.28 | +0.7% | - | - | no bet |
| NJD @ TOR (19:00 ET) | J. Allen / S. Bobrovsky (probable) | 43.8% | 41.0% | TOR 2.36 | +2.7% | - | - | no bet |
| PHI @ CGY (19:00 ET) | D. Vladar / D. Wolf (probable) | 53.7% | 48.5% | CGY 2.02 | +5.2% | - | - | no bet |
| PIT @ ANA (20:00 ET) | A. Silovs / L. Dostal (probable) | 55.0% | 52.6% | ANA 1.84 | +2.4% | - | - | no bet |

## Why

**FLA @ BOS (13:00 ET)**: No bet: edge +1.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: BOS have the clearly better goal differential this season (gap 0.73 goals/game). FLA is on the second night of a back-to-back. FLA are creating the better share of expected goals recently. Goaltending edge to BOS (goals saved above expected).

**VGK @ BUF (15:30 ET)**: No bet: edge +2.9% below the 4% minimum (starting goalie unconfirmed)
  - Context: VGK have the clearly better goal differential this season (gap 0.37 goals/game).

**TBL @ WPG (18:00 ET)**: No bet: edge +0.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: TBL have the clearly better goal differential this season (gap 1.06 goals/game). TBL is on the second night of a back-to-back. TBL are creating the better share of expected goals recently.

**NJD @ TOR (19:00 ET)**: No bet: edge +2.7% below the 4% minimum (starting goalie unconfirmed)
  - Context: NJD have the clearly better goal differential this season (gap 0.28 goals/game). TOR and NJD are on the second night of a back-to-back. NJD are creating the better share of expected goals recently.

**PHI @ CGY (19:00 ET)**: No bet: expected value after shrinkage toward the market is only +1.0% (< 1%)
  - Context: PHI travelled 3,257 km across 2 time zone(s) since their last game. CGY are creating the better share of expected goals recently.

**PIT @ ANA (20:00 ET)**: No bet: edge +2.4% below the 4% minimum (starting goalie unconfirmed)
  - Context: PIT is on the second night of a back-to-back. PIT are creating the better share of expected goals recently.

## Track record (all logged recommendations that have resolved)

- Resolved games: **271** (bet on 7, passed on 97%); model log loss 0.6520, Brier 0.2301 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (271 games with odds): model log loss 0.6520 vs market 0.6382 (market better)
- Bets: 7 · staked $18.12 · profit $-13.24 · **ROI -73.1%** · win rate 14.3% · avg price 2.53
- Bankroll $986.76 · max drawdown 1.3% · current drawdown 1.3%
- **CLV**: mean -2.38% per $1 (95% CI -4.18% to +0.14%), beat the close on 14% of 7 bets
- ⚠️ 7 bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV.

| Predicted | Observed | Games |
|---|---|---|
| 39.3% | 41.2% | 34 |
| 46.7% | 41.2% | 34 |
| 50.1% | 47.1% | 34 |
| 52.9% | 61.8% | 34 |
| 55.8% | 42.4% | 33 |
| 58.9% | 67.7% | 34 |
| 62.4% | 73.5% | 34 |
| 68.6% | 79.4% | 34 |

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 259 | -0.2% | -12% to +11% | 52% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 214 | -6.3% | -19% to +7% | 48% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 96 | -9.0% | -25% to +8% | 57% | - |  |
| `ast_over_control` | 95 | +13.0% | -7% to +33% | 59% | - |  |
| `ast_over_edge` | 79 | -11.8% | -44% to +26% | 28% | - |  |
| `ast_under_edge` | 96 | -4.6% | -20% to +10% | 62% | - |  |
| `edge_1pct` | 20 | -35.8% | -75% to +13% | 30% | -1.28% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 271 | +1.9% | -8% to +12% | 61% | -1.93% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 270 | -2.7% | -11% to +6% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 259 | -7.6% | -19% to +4% | 48% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 271 | -19.9% | -32% to -8% | 39% | -2.59% | $10 on every game where the model sees any positive edge |
| `goal_control` | 96 | -2.4% | -30% to +26% | 34% | - |  |
| `goal_edge` | 66 | -16.7% | -70% to +47% | 12% | - |  |
| `home_ml_control` | 224 | +1.8% | -10% to +14% | 57% | -2.49% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 271 | +3.0% | -6% to +12% | 62% | -2.03% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 22 | -22.7% | -68% to +25% | 32% | -1.69% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 28 | -8.1% | -48% to +32% | 43% | -2.41% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 56 | -8.8% | -36% to +15% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 31 | +0.9% | -58% to +67% | 26% | - |  |
| `parlay_players3` | 13 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 31 | -70.2% | -100% to -23% | 6% | - |  |
| `parlay_top3` | 31 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 143 | -13.0% | -29% to +4% | 46% | - |  |
| `pts_over_control` | 147 | -2.1% | -16% to +11% | 59% | - |  |
| `pts_over_edge` | 117 | -27.5% | -47% to -7% | 32% | - |  |
| `pts_under_edge` | 143 | -7.7% | -23% to +8% | 50% | - |  |
| `puckline_dog_control` | 223 | -4.2% | -14% to +5% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 86 | -4.2% | -21% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 124 | -16.1% | -31% to -0% | 48% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 51 | -25.3% | -57% to +7% | 29% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 198 | -1.1% | -13% to +12% | 54% | - |  |
| `sog_over_control` | 202 | -3.2% | -17% to +10% | 49% | - |  |
| `sog_over_edge` | 127 | -12.3% | -29% to +5% | 45% | - |  |
| `sog_under_edge` | 159 | +0.5% | -14% to +15% | 54% | - |  |
| `totals_edge` | 155 | -11.4% | -26% to +4% | 46% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 98 | -9.7% | -29% to +9% | 46% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 116 | -32.6% | -52% to -13% | 28% | -2.85% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 224 | -16.1% | -31% to -1% | 37% | -2.97% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| FLA @ BOS | Moneyline | BOS moneyline | $10.00 | 1.75 | 57.2% | 56.0% |
| VGK @ BUF | Moneyline | BUF moneyline | $10.00 | 1.93 | 53.0% | 50.1% |
| TBL @ WPG | Moneyline | TBL moneyline | $10.00 | 1.71 | 56.4% | 57.1% |
| NJD @ TOR | Moneyline | NJD moneyline | $10.00 | 1.65 | 56.2% | 59.0% |
| PHI @ CGY | Moneyline | CGY moneyline | $10.00 | 2.02 | 53.7% | 48.5% |
| PIT @ ANA | Moneyline | ANA moneyline | $10.00 | 1.84 | 55.0% | 52.6% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
