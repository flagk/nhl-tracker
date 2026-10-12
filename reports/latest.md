# NHL model report: 2026-10-12 (morning run)

> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.

*Generated 2026-10-12 12:02 UTC · model `20261011-d85d2b` · probability source `online_platt` · bankroll $1,000.00 · quarter-Kelly x1, per-bet cap 2%, daily cap 5%*

> **Model health: WARN.** feature drift: h_early_w, d_ga_season_shrunk, d_g_sv_season. Treat picks with extra scepticism.

*Odds snapshot: 2026-10-12T12:02:01+00:00 · API credits left: 250*

## Summary

- 3 games · **0 recommended bet(s)** · total stake $0.00 (0.00% of bankroll)
- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, not-too-good-to-be-true edge over the market's no-vig price.

## Games

| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |
|---|---|---|---|---|---|---|---|---|
| FLA @ BUF (13:00 ET) | D. Tarasov / U. Luukkonen (probable) | 60.6% | 50.0% | BUF 1.93 | +10.6% | - | - | no bet |
| OTT @ NJD (19:00 ET) | L. Ullmark / J. Allen (probable) | 54.4% | 54.8% | OTT 2.15 | +0.4% | - | - | no bet |
| VGK @ MIN (20:00 ET) | C. Hart / J. Wallstedt (probable) | 50.9% | 50.7% | MIN 1.93 | +0.1% | - | - | no bet |

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

## Paper trading (fake money, for measurement)

Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy only means something if it beats it by more than the noise.

| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |
|---|---|---|---|---|---|---|
| `always_over` | 79 | -10.2% | -32% to +12% | 47% | - | CONTROL: flat $10 on the Over of every game (no skill; shows what totals vig plus base rate cost) |
| `always_under` | 34 | +15.8% | -18% to +45% | 59% | - | CONTROL: flat $10 on the Under of every game (no skill; the mirror of always_over) |
| `ast_edge` | 40 | -5.2% | -35% to +27% | 55% | - |  |
| `ast_over_control` | 39 | +4.8% | -25% to +36% | 56% | - |  |
| `ast_over_edge` | 36 | -25.3% | -77% to +38% | 19% | - |  |
| `ast_under_edge` | 40 | +1.4% | -26% to +31% | 65% | - |  |
| `edge_1pct` | 0 | - | - | - | - | Live policy at a 1% raw-edge threshold instead of 3% |
| `every_game` | 81 | +7.1% | -10% to +24% | 60% | -2.04% | $5-$30 (more when the model is surer) on the model's preferred side of EVERY game with fresh odds, no edge filter, even at a negative edge |
| `every_puckline` | 81 | -13.2% | -30% to +4% | 56% | - | Goals model: $5-$30 on its puck-line side of EVERY game with fresh spread odds, no edge filter |
| `every_total` | 79 | -7.4% | -28% to +14% | 49% | - | Goals model: $5-$30 on its over/under side of EVERY game with fresh totals odds, no edge filter |
| `flat_model_side` | 81 | -10.3% | -38% to +17% | 44% | -2.66% | $5-$30 (more for a bigger edge) on every game where the model sees any positive edge |
| `goal_control` | 40 | -10.3% | -47% to +33% | 32% | - |  |
| `goal_edge` | 39 | -24.4% | -87% to +62% | 10% | - |  |
| `home_ml_control` | 34 | -7.4% | -39% to +24% | 53% | -2.37% | CONTROL: flat $10 on the HOME team's moneyline in every game (no skill; is there a home-ice price bias?) |
| `market_favorite` | 81 | +1.0% | -16% to +18% | 62% | -2.12% | CONTROL: flat $10 on the market favourite (no skill; shows what the bookmaker margin costs) |
| `no_guard` | 12 | +15.7% | - | 50% | -1.12% | Live policy without the early-season guard (does the guard help?) |
| `no_shrink` | 0 | - | - | - | - | Live policy trusting the raw model fully (no shrinkage toward the market) |
| `over_edge` | 11 | -29.7% | - | 36% | - | Goals model: $5-$30 on the Over only, when it has a 2%+ edge |
| `parlay_favorites_control` | 4 | -2.0% | - | 25% | - |  |
| `parlay_players3` | 4 | -100.0% | - | 0% | - |  |
| `parlay_top2` | 4 | -100.0% | - | 0% | - |  |
| `parlay_top3` | 4 | -100.0% | - | 0% | - |  |
| `pts_edge` | 39 | -4.2% | -39% to +32% | 51% | - |  |
| `pts_over_control` | 43 | -14.1% | -40% to +10% | 51% | - |  |
| `pts_over_edge` | 40 | -37.9% | -74% to +6% | 22% | - |  |
| `pts_under_edge` | 39 | -1.6% | -36% to +32% | 56% | - |  |
| `puckline_dog_control` | 34 | -21.5% | -48% to +5% | 50% | - | CONTROL: flat $10 on the +1.5 underdog of every game (no skill; what the puck-line vig costs) |
| `puckline_dog_edge` | 21 | -32.5% | -67% to +8% | 48% | - | Goals model: $5-$30 on the +1.5 underdog side of the puck line, when it has a 2%+ edge |
| `puckline_edge` | 48 | -24.5% | -48% to +0% | 48% | - | Goals model: $5-$30 on the puck-line side with a 3%+ raw edge vs the market (experimental market) |
| `puckline_fav_edge` | 2 | +203.8% | - | 100% | - | Goals model: $5-$30 on the -1.5 favourite side of the puck line, when it has a 2%+ edge |
| `sog_edge` | 79 | -5.0% | -26% to +16% | 54% | - |  |
| `sog_over_control` | 82 | +9.2% | -12% to +32% | 52% | - |  |
| `sog_over_edge` | 39 | -6.1% | -40% to +25% | 44% | - |  |
| `sog_under_edge` | 41 | -14.7% | -45% to +17% | 51% | - |  |
| `totals_edge` | 47 | -14.0% | -42% to +16% | 47% | - | Goals model: $5-$30 on the over/under side with a 3%+ raw edge vs the market (experimental market) |
| `under_edge` | 15 | +1.4% | - | 60% | - | Goals model: $5-$30 on the Under only, when it has a 2%+ edge (are we better at low-scoring games?) |
| `underdog_ml` | 18 | -28.1% | - | 33% | -2.55% | $5-$30 on the moneyline UNDERDOG, only when the model rates it 2%+ better than the market does |
| `underdog_ml_control` | 34 | -22.3% | -60% to +17% | 32% | -2.73% | CONTROL: flat $10 on the moneyline underdog of every game (no skill) |

*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*

### Today's fake bets on every game (pretend money, NOT recommendations)

| Game | Type | Pretend pick | Pretend stake | Price | Model | Market |
|---|---|---|---|---|---|---|
| FLA @ BUF | Moneyline | BUF moneyline | $16.00 | 1.93 | 60.6% | 50.0% |
| OTT @ NJD | Moneyline | NJD moneyline | $9.00 | 1.78 | 54.4% | 54.8% |
| VGK @ MIN | Moneyline | MIN moneyline | $6.00 | 1.93 | 50.9% | 50.7% |

Total pretend stake $31.00. Settled results feed the `every_*` rows above; real bets follow the normal policy only.

---
> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.
