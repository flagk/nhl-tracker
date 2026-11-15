# NHL model report: 2026-11-15 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-11-15 13:06 UTC · model `20261114-0255ca` · probability source `online_platt` · bankroll $974.65 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w. Treat picks with extra scepticism.

*Odds snapshot: 2026-11-15T13:05:06+00:00 · API credits left: 256*

## Summary

- 5 games · **1 recommended bet(s)** · total stake $2.18 (0.22% of bankroll)

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| EDM @ PHI (19:30 ET) | D. Levi / J. Woll (probable) | 49.4% | 42.1% | PHI 2.32 | +7.3% | +14.6% | $2.18 | **BET PHI** |
| FLA @ DAL (20:00 ET) | J. Markstrom / J. Oettinger (probable) | 55.3% | 55.4% | FLA 2.20 | +0.1% | - | - | no bet |
| CGY @ ANA (20:00 ET) | D. Wolf / L. Dostal (probable) | 54.3% | 58.3% | CGY 2.33 | +4.0% | - | - | no bet |
| VAN @ VGK (21:00 ET) | L. Meriläinen / A. Hill (probable) | 69.4% | 78.5% | VAN 4.35 | +9.1% | - | - | no bet |
| NSH @ SJS (21:00 ET) | J. Saros / Y. Askarov (probable) | 40.7% | 39.4% | SJS 2.46 | +1.3% | - | - | no bet |

## Why

**EDM @ PHI (19:30 ET)**: Bet $2.18 (0.22% of bankroll) on PHI at 2.32. The model gives PHI 49.4% vs the market's no-vig 42.1% (edge +7.3%); after shrinking toward the market the working probability is 43.6%, worth +1.2% per $1 at that price.
  - Context: EDM have the clearly better goal differential this season (gap 1.24 goals/game). PHI and EDM are on the second night of a back-to-back. Goaltending edge to EDM (goals saved above expected).

**FLA @ DAL (20:00 ET)**: No bet: edge +0.1% below the 4% minimum (starting goalie unconfirmed)
  - Context: DAL and FLA are on the second night of a back-to-back. Goaltending edge to DAL (goals saved above expected).

**CGY @ ANA (20:00 ET)**: No bet: expected value after shrinkage toward the market is only +0.3% (< 1%)
  - Context: ANA have the clearly better goal differential this season (gap 0.51 goals/game). ANA and CGY are on the second night of a back-to-back. Division/rival game.

**VAN @ VGK (21:00 ET)**: No bet: model win probability 30.6% below the 40% confidence floor; price 4.35 above the 4.0 long-shot cap
  - Context: VGK have the clearly better goal differential this season (gap 1.76 goals/game). VGK are creating the better share of expected goals recently. Goaltending edge to VGK (goals saved above expected). Division/rival game.

**NSH @ SJS (21:00 ET)**: No bet: edge +1.3% below the 4% minimum (starting goalie unconfirmed)
  - Context: NSH have the clearly better goal differential this season (gap 0.78 goals/game). SJS is on the second night of a back-to-back. NSH are creating the better share of expected goals recently.

## Track record (all logged recommendations that have resolved)

- Resolved games: **320** (bet on 11, passed on 97%); model log loss 0.6589, Brier 0.2335 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (320 games with odds): model log loss 0.6589 vs market 0.6413 (market better)
- Bets: 11 · staked $30.23 · profit $-25.35 · **ROI -83.9%** · win rate 9.1% · avg price 2.49
- Bankroll $974.65 · max drawdown 2.5% · current drawdown 2.5%
- **CLV**: mean -2.15% per $1 (95% CI -3.47% to -0.58%), beat the close on 9% of 11 bets
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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 307 | -4.5% | -15% to +7% | 50% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 262 | -1.0% | -13% to +11% | 50% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 152 | -9.3% | -23% to +4% | 56% | - |  |
| `ast_over_control` | 151 | +7.3% | -8% to +23% | 56% | - |  |
| `ast_over_edge` | 122 | -17.3% | -43% to +11% | 28% | - |  |
| `ast_under_edge` | 151 | -8.9% | -22% to +3% | 60% | - |  |
| `edge_1pct` | 33 | -48.7% | -77% to -17% | 24% | -1.71% | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 320 | +1.3% | -8% to +11% | 60% | -1.98% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 319 | -3.7% | -12% to +4% | 63% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 307 | -5.2% | -16% to +6% | 50% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 320 | -23.6% | -34% to -12% | 37% | -2.71% | $10 on every game where the model sees any positive edge |
| `goal_control` | 152 | -2.0% | -24% to +20% | 35% | - |  |
| `goal_edge` | 95 | -21.5% | -63% to +31% | 13% | - |  |
| `home_ml_control` | 273 | -5.6% | -16% to +5% | 52% | -2.56% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 320 | +3.3% | -6% to +12% | 62% | -2.07% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 26 | -34.6% | -73% to +8% | 27% | -1.70% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 38 | -26.9% | -60% to +6% | 34% | -2.56% | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 65 | -9.7% | -34% to +12% | 48% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 38 | +20.0% | -38% to +86% | 29% | - |  |
| `parlay_players3` | 19 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 38 | -75.7% | -100% to -37% | 5% | - |  |
| `parlay_top3` | 38 | -100.0% | -100% to -100% | 0% | - |  |
| `pts_edge` | 199 | -13.8% | -28% to -0% | 46% | - |  |
| `pts_over_control` | 203 | +0.3% | -12% to +13% | 59% | - |  |
| `pts_over_edge` | 166 | -30.5% | -47% to -13% | 31% | - |  |
| `pts_under_edge` | 198 | -8.6% | -21% to +4% | 51% | - |  |
| `puckline_dog_control` | 272 | -6.4% | -16% to +2% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 92 | -4.0% | -20% to +12% | 62% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 141 | -15.4% | -32% to +1% | 47% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 68 | -25.9% | -53% to +4% | 28% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 253 | -3.8% | -16% to +8% | 51% | - |  |
| `sog_over_control` | 258 | -0.3% | -13% to +12% | 50% | - |  |
| `sog_over_edge` | 177 | -5.3% | -20% to +10% | 47% | - |  |
| `sog_under_edge` | 212 | -5.9% | -19% to +7% | 50% | - |  |
| `totals_edge` | 180 | -6.5% | -20% to +7% | 48% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 121 | -2.8% | -20% to +14% | 50% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 141 | -36.1% | -54% to -19% | 27% | -2.85% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 273 | -15.4% | -29% to -2% | 37% | -2.91% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| EDM @ PHI | Moneyline | EDM moneyline | $10.00 | 1.69 | 50.6% | 57.9% |
| FLA @ DAL | Moneyline | DAL moneyline | $10.00 | 1.77 | 55.3% | 55.4% |
| CGY @ ANA | Moneyline | ANA moneyline | $10.00 | 1.67 | 54.3% | 58.3% |
| VAN @ VGK | Moneyline | VGK moneyline | $10.00 | 1.25 | 69.4% | 78.5% |
| NSH @ SJS | Moneyline | NSH moneyline | $10.00 | 1.61 | 59.3% | 60.6% |

Total pretend stake $50.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
