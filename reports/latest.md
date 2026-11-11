# NHL model report: 2026-11-11 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-11 13:03 UTC · model `20261110-9b4d65` · probability source `online_platt` · bankroll $981.21 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-11T13:02:52+00:00 · API credits left: 321*

## Summary

- 6 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| TBL @ BUF (19:00 ET) | A. Vasilevskiy / U. Luukkonen (probable) | 47.6% | 46.1% | BUF 2.10 | +1.5% | - | - | no bet |
| NYR @ CGY (19:30 ET) | D. Garand / D. Wolf (probable) | 47.2% | 43.2% | CGY 2.27 | +4.0% | - | - | no bet |
| PHI @ EDM (21:00 ET) | J. Woll / D. Levi (probable) | 60.2% | 65.2% | PHI 2.78 | +4.9% | - | - | no bet |
| UTA @ VGK (21:30 ET) | K. Vejmelka / C. Hart (probable) | 53.4% | 53.5% | UTA 2.10 | +0.2% | - | - | no bet |
| NSH @ ANA (22:00 ET) | J. Saros / L. Dostal (probable) | 58.3% | 56.9% | ANA 1.73 | +1.4% | - | - | no bet |
| PIT @ SJS (22:00 ET) | A. Silovs / Y. Askarov (probable) | 42.6% | 44.4% | PIT 1.75 | +1.7% | - | - | no bet |

## Why

**TBL @ BUF (19:00 ET)**: No bet: edge +1.5% below the 4% minimum (starting goalie unconfirmed)
  - Context: TBL have the clearly better goal differential this season (gap 0.56 goals/game). TBL are creating the better share of expected goals recently. Division/rival game.

**NYR @ CGY (19:30 ET)**: No bet: edge +4.0% below the 4% minimum (starting goalie unconfirmed)
  - Context: NYR have the clearly better goal differential this season (gap 0.64 goals/game). NYR are creating the better share of expected goals recently. Goaltending edge to NYR (goals saved above expected).

**PHI @ EDM (21:00 ET)**: No bet: model win probability 39.8% below the 40% confidence floor
  - Context: EDM have the clearly better goal differential this season (gap 1.26 goals/game). EDM are creating the better share of expected goals recently.

**UTA @ VGK (21:30 ET)**: No bet: edge +0.2% below the 4% minimum (starting goalie unconfirmed)
  - Context: VGK have the clearly better goal differential this season (gap 0.38 goals/game). UTA travelled 3,425 km across 3 time zone(s) since their last game. UTA are creating the better share of expected goals recently. Goaltending edge to VGK (goals saved above expected).

**NSH @ ANA (22:00 ET)**: No bet: edge +1.4% below the 4% minimum (starting goalie unconfirmed)
  - Context: NSH is on the second night of a back-to-back. Goaltending edge to NSH (goals saved above expected).

**PIT @ SJS (22:00 ET)**: No bet: edge +1.7% below the 4% minimum (starting goalie unconfirmed)
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 274 | -3.6% | -15% to +8% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 229 | -2.3% | -15% to +10% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 120 | -4.6% | -19% to +11% | 59% | - |  |
| `ast_over_control` | 119 | +6.2% | -11% to +23% | 55% | - |  |
| `ast_over_edge` | 99 | -10.6% | -39% to +20% | 29% | - |  |
| `ast_under_edge` | 120 | -3.2% | -17% to +10% | 64% | - |  |
| `edge_1pct` | 24 | -38.0% | -75% to +4% | 29% | -1.52% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 287 | +1.7% | -8% to +11% | 61% | -1.97% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 286 | -4.0% | -12% to +4% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 274 | -7.2% | -19% to +4% | 49% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 287 | -20.8% | -32% to -9% | 39% | -2.64% | $10 on every game where the model sees any positive edge |
| `goal_control` | 120 | -6.8% | -30% to +18% | 33% | - |  |
| `goal_edge` | 78 | -25.5% | -71% to +29% | 12% | - |  |
| `home_ml_control` | 240 | +1.5% | -10% to +13% | 56% | -2.56% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 287 | +4.0% | -6% to +13% | 63% | -2.03% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 24 | -29.1% | -70% to +19% | 29% | -1.75% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 33 | -15.8% | -52% to +21% | 39% | -2.52% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 59 | -10.3% | -35% to +13% | 47% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 34 | +8.7% | -51% to +73% | 26% | - |  |
| `parlay_players3` | 16 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 34 | -72.8% | -100% to -30% | 6% | - |  |
| `parlay_top3` | 34 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 167 | -9.7% | -25% to +4% | 49% | - |  |
| `pts_over_control` | 171 | -4.3% | -17% to +9% | 57% | - |  |
| `pts_over_edge` | 141 | -29.1% | -47% to -10% | 31% | - |  |
| `pts_under_edge` | 167 | -4.6% | -19% to +9% | 53% | - |  |
| `puckline_dog_control` | 239 | -5.7% | -15% to +4% | 62% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 89 | -4.2% | -21% to +11% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 130 | -18.9% | -35% to -3% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 55 | -30.7% | -60% to -0% | 27% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 222 | -0.8% | -13% to +11% | 53% | - |  |
| `sog_over_control` | 226 | -0.9% | -14% to +13% | 50% | - |  |
| `sog_over_edge` | 151 | -6.6% | -23% to +10% | 47% | - |  |
| `sog_under_edge` | 183 | -2.0% | -15% to +12% | 52% | - |  |
| `totals_edge` | 162 | -9.4% | -24% to +6% | 47% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 106 | -5.8% | -24% to +13% | 48% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 126 | -36.3% | -54% to -16% | 27% | -2.90% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 240 | -17.2% | -31% to -3% | 36% | -2.99% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| TBL @ BUF | Moneyline | TBL moneyline | $10.00 | 1.81 | 52.4% | 53.9% |
| NYR @ CGY | Moneyline | NYR moneyline | $10.00 | 1.71 | 52.8% | 56.8% |
| PHI @ EDM | Moneyline | EDM moneyline | $10.00 | 1.51 | 60.2% | 65.2% |
| UTA @ VGK | Moneyline | VGK moneyline | $10.00 | 1.81 | 53.4% | 53.5% |
| NSH @ ANA | Moneyline | ANA moneyline | $10.00 | 1.73 | 58.3% | 56.9% |
| PIT @ SJS | Moneyline | PIT moneyline | $10.00 | 1.75 | 57.4% | 55.6% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
