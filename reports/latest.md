# NHL model report: 2026-10-21 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-21 20:33 UTC · model `20261020-4f2b24` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-21T20:33:08+00:00 · API credits left: 136*

*Player props: 0 game(s) priced*

## Summary

- 2 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| PHI @ BUF (19:30 ET) | D. Vladar / U. Luukkonen (probable) | 66.4% | 64.3% | BUF 1.53 | +2.1% | - | - | no bet |
| FLA @ OTT (19:30 ET) | D. Tarasov / L. Ullmark (probable) | 54.1% | 50.0% | OTT 1.94 | +4.1% | - | - | no bet |

## Why

**PHI @ BUF (19:30 ET)**: No bet: early-season sample size: a team has only played 9 games (< 10)
  - Context: BUF have the clearly better goal differential this season (gap 0.59 goals/game). BUF are creating the better share of expected goals recently. Early in the season: team ratings lean on last year's results and are less reliable.

**FLA @ OTT (19:30 ET)**: No bet: early-season sample size: a team has only played 9 games (< 10)
  - Context: Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **146** (bet on 0, passed on 100%); model log loss 0.6557, Brier 0.2318 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (146 games with odds): model log loss 0.6557 vs market 0.6402 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 41.3% | 40.0% | 30 |
| 50.4% | 51.7% | 29 |
| 54.5% | 58.6% | 29 |
| 59.4% | 65.5% | 29 |
| 66.2% | 75.9% | 29 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| FLA @ OTT (19:30 ET) | Total | Over 5.5 | 57.8% | 50.5% | +7.3 pts | +11.6% | 1.93 |
| PHI @ BUF (19:30 ET) | Puck line | BUF -1.5 | 44.6% | 41.2% | +3.3 pts | +4.7% | 2.35 |
| PHI @ BUF (19:30 ET) | Total | Over 6 | 51.7% | 49.9% | +1.8 pts | -0.1% | 1.93 |
| FLA @ OTT (19:30 ET) | Puck line | OTT -1.5 | 29.5% | 28.3% | +1.2 pts | -0.9% | 3.36 |
| FLA @ OTT (19:30 ET) | Puck line | FLA +1.5 | 70.5% | 71.7% | -1.2 pts | -4.1% | 1.36 |
| PHI @ BUF (19:30 ET) | Total | Under 6 | 48.3% | 50.1% | -1.8 pts | -6.0% | 1.93 |
| PHI @ BUF (19:30 ET) | Puck line | PHI +1.5 | 55.4% | 58.8% | -3.3 pts | -8.5% | 1.65 |
| FLA @ OTT (19:30 ET) | Total | Under 5.5 | 42.2% | 49.5% | -7.3 pts | -15.6% | 2.00 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 140 | -7.2% | -23% to +9% | 49% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 95 | +3.0% | -18% to +21% | 53% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -6.5% | -34% to +22% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -24% to +34% | 56% | - |  |
| `ast_over_edge` | 36 | -12.6% | -71% to +54% | 19% | - |  |
| `ast_under_edge` | 40 | +2.7% | -22% to +27% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 146 | -0.2% | -13% to +13% | 60% | -2.10% | $10 on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 145 | -6.6% | -19% to +6% | 61% | - | Goals model: $10 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 140 | -11.7% | -27% to +4% | 46% | - | Goals model: $10 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 146 | -19.4% | -35% to -3% | 40% | -2.50% | $10 on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -51% to +30% | 32% | - |  |
| `goal_edge` | 39 | -5.1% | -82% to +95% | 10% | - |  |
| `home_ml_control` | 99 | +4.9% | -13% to +23% | 59% | -2.50% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 146 | +1.9% | -12% to +15% | 62% | -2.09% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 14 | +2.4% | - | 43% | -1.26% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 21 | -45.9% | -82% to -9% | 29% | - | Goals model: $10 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 13 | -21.0% | - | 23% | - |  |
| `parlay_players3` | 6 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 13 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 13 | -100.0% | - | 0% | - |  |
| `pts_edge` | 71 | -13.3% | -36% to +9% | 45% | - |  |
| `pts_over_control` | 75 | -13.3% | -32% to +5% | 53% | - |  |
| `pts_over_edge` | 59 | -35.8% | -64% to -2% | 24% | - |  |
| `pts_under_edge` | 71 | -12.4% | -34% to +9% | 48% | - |  |
| `puckline_dog_control` | 98 | -4.8% | -21% to +10% | 61% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 52 | -8.9% | -30% to +12% | 58% | - | Goals model: $10 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 80 | -11.8% | -32% to +9% | 51% | - | Goals model: $10 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 14 | +15.0% | - | 43% | - | Goals model: $10 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 118 | -0.2% | -16% to +16% | 54% | - |  |
| `sog_over_control` | 122 | -0.6% | -18% to +18% | 48% | - |  |
| `sog_over_edge` | 58 | -14.2% | -39% to +11% | 45% | - |  |
| `sog_under_edge` | 80 | -1.3% | -22% to +19% | 52% | - |  |
| `totals_edge` | 79 | -20.2% | -42% to +1% | 42% | - | Goals model: $10 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 43 | -10.1% | -38% to +18% | 47% | - | Goals model: $10 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 52 | -35.0% | -62% to -4% | 27% | -2.62% | $10 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 99 | -16.1% | -37% to +6% | 36% | -2.80% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| PHI @ BUF | Moneyline | BUF moneyline | $10.00 | 1.53 | 66.4% | 64.3% |
| FLA @ OTT | Moneyline | OTT moneyline | $10.00 | 1.94 | 54.1% | 50.0% |
| PHI @ BUF | Total | Over 6 | $10.00 | 1.93 | 51.7% | 49.9% |
| FLA @ OTT | Total | Over 5.5 | $10.00 | 1.93 | 57.8% | 50.5% |
| PHI @ BUF | Puck line | PHI +1.5 | $10.00 | 1.65 | 55.4% | 58.8% |
| FLA @ OTT | Puck line | FLA +1.5 | $10.00 | 1.36 | 70.5% | 71.7% |

Total pretend stake $60.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
