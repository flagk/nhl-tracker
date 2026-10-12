# NHL model report: 2026-10-12 (late run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-12 14:03 UTC · model `20261011-d85d2b` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-12T14:03:03+00:00 · API credits left: 247*

*Player props: 1 game(s) priced*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| FLA @ BUF (13:00 ET) | D. Tarasov / U. Luukkonen (probable) | 60.6% | 50.3% | BUF 1.93 | +10.3% | - | - | no bet |
| OTT @ NJD (19:00 ET) | L. Ullmark / J. Allen (probable) | 54.4% | 54.8% | OTT 2.17 | +0.4% | - | - | no bet |
| VGK @ MIN (20:00 ET) | C. Hart / J. Wallstedt (probable) | 50.9% | 50.8% | MIN 1.93 | +0.1% | - | - | no bet |

## Why

**FLA @ BUF (13:00 ET)**: No bet: early-season sample size: a team has only played 5 games (< 10)
  - Context: BUF are creating the better share of expected goals recently. Goaltending edge to FLA (goals saved above expected). Division/rival game. Early in the season: team ratings lean on last year's results and are less reliable.

**OTT @ NJD (19:00 ET)**: No bet: early-season sample size: a team has only played 4 games (< 10)
  - Context: OTT have the clearly better goal differential this season (gap 0.64 goals/game). NJD are creating the better share of expected goals recently. Goaltending edge to OTT (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

**VGK @ MIN (20:00 ET)**: No bet: early-season sample size: a team has only played 5 games (< 10)
  - Context: VGK are creating the better share of expected goals recently. Goaltending edge to MIN (goals saved above expected). Early in the season: team ratings lean on last year's results and are less reliable.

## Track record (all logged recommendations that have resolved)

- Resolved games: **81** (bet on 0, passed on 100%); model log loss 0.6484, Brier 0.2283 (coin flip: 0.6931 / 0.2500)
- **Model vs market close** (81 games with odds): model log loss 0.6484 vs market 0.6346 (market better)

| Predicted | Observed | Games |
|---|---|---|
| 44.4% | 40.7% | 27 |
| 54.8% | 55.6% | 27 |
| 64.2% | 70.4% | 27 |

## Other markets: totals and puck line (experimental, paper-trading only)

The goals model prices the over/under and the puck line. **No real stakes are suggested here**: this model has no track record against the market yet, so it is only paper-traded (see below) until results, not backtests, say otherwise.

| Game | Market | Side | Model | Market (no-vig) | Edge | EV per $1 | Best price |
|---|---|---|---|---|---|---|---|
| FLA @ BUF (13:00 ET) | Puck line | BUF -1.5 | 33.4% | 28.8% | +4.6 pts | +10.3% | 3.30 |
| OTT @ NJD (19:00 ET) | Puck line | OTT +1.5 | 70.0% | 67.4% | +2.6 pts | +1.5% | 1.45 |
| OTT @ NJD (19:00 ET) | Total | Over 6 | 50.6% | 48.7% | +1.9 pts | +0.2% | 1.98 |
| VGK @ MIN (20:00 ET) | Puck line | VGK +1.5 | 72.3% | 70.7% | +1.6 pts | +1.3% | 1.40 |
| FLA @ BUF (13:00 ET) | Total | Over 6 | 49.8% | 48.6% | +1.2 pts | -0.7% | 1.99 |
| VGK @ MIN (20:00 ET) | Total | Over 6 | 50.0% | 49.5% | +0.5 pts | -1.8% | 1.96 |
| VGK @ MIN (20:00 ET) | Total | Under 6 | 50.0% | 50.5% | -0.5 pts | -3.9% | 1.91 |
| FLA @ BUF (13:00 ET) | Total | Under 6 | 50.2% | 51.4% | -1.2 pts | -5.4% | 1.87 |
| VGK @ MIN (20:00 ET) | Puck line | MIN -1.5 | 27.7% | 29.3% | -1.6 pts | -8.1% | 3.32 |
| OTT @ NJD (19:00 ET) | Total | Under 6 | 49.4% | 51.3% | -1.9 pts | -6.2% | 1.88 |
| OTT @ NJD (19:00 ET) | Puck line | NJD -1.5 | 30.0% | 32.6% | -2.6 pts | -11.1% | 2.96 |
| FLA @ BUF (13:00 ET) | Puck line | FLA +1.5 | 66.6% | 71.2% | -4.6 pts | -8.1% | 1.38 |


## Player props: shots, points, assists, anytime goals (experimental, paper-trading only)

A player-level model (own recent shot rate, opponent, home ice, ice time, rest) against the market's over/under. **Experimental, no real stakes suggested.** "Take" is a lean to check against your own app's price.

| Player | Game | Stat / line | Take | Model expects | Model P(over) | Market P(over) | Edge on take | EV per $1 | Last 10 | Season avg |
|---|---|---|---|---|---|---|---|---|---|---|
| B. Tkachuk | FLA @ BUF (13:00 ET) | sog 2.5 | Under | 2.29 | 39.5% | 58.5% | +19.0 pts | +36.1% | 7 5 3 1 1 1 4 4 2 1 | - |
| B. Tkachuk | FLA @ BUF (13:00 ET) | pts 0.5 | Under | 0.52 | 40.2% | 58.0% | +17.8 pts | +34.6% | 0 2 4 1 0 0 0 0 0 0 | - |
| M. Tkachuk | FLA @ BUF (13:00 ET) | sog 2.5 | Under | 2.22 | 37.7% | 53.9% | +16.2 pts | +30.8% | 5 2 0 3 0 2 2 1 2 4 | - |
| J. Doan | FLA @ BUF (13:00 ET) | sog 1.5 | Under | 1.49 | 42.9% | 58.2% | +15.3 pts | +28.4% | 0 1 4 1 1 3 1 0 1 2 | 1.4 |
| S. Reinhart | FLA @ BUF (13:00 ET) | pts 0.5 | Under | 0.63 | 46.3% | 58.8% | +12.5 pts | +21.4% | 1 0 1 1 2 0 0 0 0 0 | - |
| J. Quinn | FLA @ BUF (13:00 ET) | pts 0.5 | Under | 0.54 | 41.4% | 52.7% | +11.3 pts | +20.0% | 1 0 1 0 0 1 1 0 0 0 | 0.4 |
| J. Doan | FLA @ BUF (13:00 ET) | pts 0.5 | Under | 0.39 | 32.0% | 43.9% | +11.9 pts | +18.2% | 1 1 2 1 0 0 0 0 0 0 | 0.0 |
| C. Verhaeghe | FLA @ BUF (13:00 ET) | sog 2.5 | Under | 1.95 | 31.1% | 43.5% | +12.4 pts | +16.5% | 1 2 5 3 1 1 0 3 2 0 | - |
| M. Tkachuk | FLA @ BUF (13:00 ET) | pts 0.5 | Under | 0.78 | 54.0% | 63.3% | +9.3 pts | +15.4% | 4 0 1 0 2 0 1 1 1 1 | - |
| K. Helenius | FLA @ BUF (13:00 ET) | sog 1.5 | Under | 1.39 | 39.8% | 50.0% | +10.2 pts | +13.2% | 2 2 2 3 1 0 2 3 0 1 | - |
| T. Thompson | FLA @ BUF (13:00 ET) | pts 0.5 | Under | 0.81 | 55.2% | 63.2% | +8.0 pts | +12.0% | 0 0 1 0 2 0 2 2 0 1 | 1.0 |
| R. Dahlin | FLA @ BUF (13:00 ET) | sog 2.5 | Under | 2.29 | 39.5% | 47.7% | +8.3 pts | +10.2% | 3 1 2 3 0 4 3 3 1 4 | 3.0 |


## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 79 | -10.2% | -32% to +12% | 47% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 34 | +15.8% | -19% to +48% | 59% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -5.2% | -36% to +28% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -26% to +35% | 56% | - |  |
| `ast_over_edge` | 36 | -25.3% | -76% to +37% | 19% | - |  |
| `ast_under_edge` | 40 | +1.4% | -26% to +30% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 81 | +7.1% | -11% to +23% | 60% | -2.04% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 81 | -13.2% | -30% to +3% | 56% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 79 | -7.4% | -28% to +14% | 49% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 81 | -10.3% | -36% to +18% | 44% | -2.66% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -49% to +34% | 32% | - |  |
| `goal_edge` | 39 | -24.4% | -86% to +62% | 10% | - |  |
| `home_ml_control` | 34 | -7.4% | -38% to +25% | 53% | -2.37% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 81 | +1.0% | -16% to +18% | 62% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 12 | +15.7% | - | 50% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 11 | -29.7% | - | 36% | - | Goals model: $5-$30 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 4 | -2.0% | - | 25% | - |  |
| `parlay_players3` | 4 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 4 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 4 | -100.0% | - | 0% | - |  |
| `pts_edge` | 39 | -4.2% | -38% to +31% | 51% | - |  |
| `pts_over_control` | 43 | -14.1% | -40% to +12% | 51% | - |  |
| `pts_over_edge` | 40 | -37.9% | -74% to +3% | 22% | - |  |
| `pts_under_edge` | 39 | -1.6% | -33% to +33% | 56% | - |  |
| `puckline_dog_control` | 34 | -21.5% | -48% to +5% | 50% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 21 | -32.5% | -67% to +6% | 48% | - | Goals model: $5-$30 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 48 | -24.5% | -49% to +0% | 48% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 2 | +203.8% | - | 100% | - | Goals model: $5-$30 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 79 | -5.0% | -25% to +16% | 54% | - |  |
| `sog_over_control` | 82 | +9.2% | -13% to +33% | 52% | - |  |
| `sog_over_edge` | 39 | -6.1% | -41% to +24% | 44% | - |  |
| `sog_under_edge` | 41 | -14.7% | -45% to +17% | 51% | - |  |
| `totals_edge` | 47 | -14.0% | -42% to +15% | 47% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 15 | +1.4% | - | 60% | - | Goals model: $5-$30 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 18 | -28.1% | - | 33% | -2.55% | $5-$30 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 34 | -22.3% | -59% to +18% | 32% | -2.73% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| FLA @ BUF | Moneyline | BUF moneyline | $16.00 | 1.93 | 60.6% | 50.3% |
| OTT @ NJD | Moneyline | NJD moneyline | $9.00 | 1.78 | 54.4% | 54.8% |
| VGK @ MIN | Moneyline | MIN moneyline | $6.00 | 1.93 | 50.9% | 50.8% |
| FLA @ BUF | Total | Under 6 | $6.00 | 1.87 | 50.2% | 51.4% |
| OTT @ NJD | Total | Over 6 | $5.00 | 1.98 | 50.6% | 48.7% |
| VGK @ MIN | Total | Under 6 | $5.00 | 1.91 | 50.0% | 50.5% |
| FLA @ BUF | Puck line | FLA +1.5 | $25.00 | 1.38 | 66.6% | 71.2% |
| OTT @ NJD | Puck line | OTT +1.5 | $23.00 | 1.45 | 70.0% | 67.4% |
| VGK @ MIN | Puck line | VGK +1.5 | $26.00 | 1.40 | 72.3% | 70.7% |

Total pretend stake $121.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
