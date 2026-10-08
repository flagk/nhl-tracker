# Performance diagnostic, 2026-09-30 to 2026-10-08

> Research and education only. Everything below is paper (fake) money over about 8 days; the samples are small and several bets share a game, so read it as a health check, not as proof of anything.

## Verdict in one paragraph
The plumbing is healthy (every scheduled run succeeded, odds credits are fine, data is fresh). The **moneyline model is roughly as good as the market but not better**, and every **experimental market (totals, puck line, player props) has been worse than the market on the very picks it chose**. That is the expected shape for new models: where a model disagrees most with the market is mostly where the model is wrong. The fix applied is to trust those models much less in the ranking and parlays, with the trust now learned from their own settled bets.

## 1. The core moneyline model
| | model | market close |
|---|---|---|
| Log loss, 50 settled games this season | 0.658 | 0.645 |
| Brier | 0.233 | 0.227 |
| Accuracy | 56% | |
| Walk-forward 2024-26 (2,629 games) log loss | 0.674 | (coin flip 0.693) |

- Slightly behind the market on this small sample, which is normal and fits the live policy placing **0 real bets** (it only bets a clear edge).
- Model health is **WARN** from input drift only (early-season stats on 3-8 games per team); predictions themselves are fine (recent vs historical log loss 0.6735 vs 0.6748). It should clear by mid-November.
- Closing-line value is about -2% for every strategy including the no-skill favourite control, so it is a baseline offset, not a model signal.

## 2. Paper trading by market (distinct picks, a pick shared by several strategies counted once)
| market | picks | games | model chance | market chance | actual | log loss model / market | ROI |
|---|---|---|---|---|---|---|---|
| moneyline | 84 | 50 | 51.9% | 51.3% | 50.0% | 0.651 / 0.645 | +1.0% |
| totals | 68 | 48 | 51.4% | 50.6% | 48.5% | 0.707 / 0.690 | -6.6% |
| puck line | 52 | 50 | 67.7% | 63.0% | 55.8% | 0.682 / 0.651 | -14.9% |
| player shots | 106 | 15 | 53.2% | 49.1% | 47.2% | 0.725 / 0.692 | -10.8% |
| player points | 22 | 3 | 46.6% | 44.2% | 36.4% | 0.701 / 0.660 | -23.8% |
| player assists | 24 | 3 | 46.0% | 43.3% | 41.7% | 0.540 / 0.520 | -19.2% |
| anytime goals | 16 | 3 | 25.6% | 25.0% | 18.8% | 0.407 / 0.404 | -67.4% |

Reading it:
- In every market the model's chance sits **above** both the market and what actually happened, i.e. its edges were mostly noise or bias, and the market's probabilities scored better (lower log loss) in all seven.
- The best blend of model and market on these picks puts almost **zero weight on the model** for every experimental market (about 0.2 for moneylines).
- The puck line is the clearest case: the model said the +1.5 side covers 68% of the time, it covered 56%. Player points: the model said 57-69% on its biggest-edge unders/overs, the realised rate was about 31-33%.
- Controls behave as expected: flat bets on the market favourite and "always over" lose about the bookmaker margin (-4% and 0%), and random-looking single-day swings (+$74 on Oct 4, -$1,014 on Oct 7) are mostly many strategies betting the same legs together.

## 3. Things found and what was done
1. **Trust weights were guesses.** Ranking and parlays gave the experimental models 0.15-0.5 of their edge. Lowered the priors (totals 0.3, puck line 0.25, shots 0.2, points/assists 0.15, anytime goals 0.1) and added `learn_weights`: each market's weight is re-estimated from its own settled paper bets every run (lowest log loss of `market + w * (model - market)`), blended with the prior by sample size, floor 0.05. A market that proves itself earns trust back automatically.
2. **Totals and puck-line prices could be hours old** (the morning run reused a 13-hour-old capture). Fixed earlier: captures older than 2 hours are not used.
3. **A deploy stuck for a day** kept the site on old data. The heartbeat now cancels and redeploys stuck deploys.
4. **The morning run ran at midnight ET**, before late games were final. It now waits until 8 am ET.
5. **Duplicate strategies inflate the totals.** `pts_edge` and `pts_under_edge` (and the `ast_` pair) currently place identical bets, and the headline fake-money total counts each strategy separately. Compare strategies by ROI against their control, not by the grand total.

## 4. System health
- 90 of the last 100 workflow runs succeeded, 8 were cancelled by design (superseded hourly heartbeat triggers), 0 failed.
- Heartbeat is running, morning run fires at 8 am ET, late run before first puck drop, closing-odds snapshots near puck drop.
- Odds credits: about 343 remaining; player markets cost about 12 a day and are switched off automatically below 250 (assists/goals) and 100 (all player prices).

## 5. What to watch next
- Whether learned weights for totals/puck line/player props stay near the floor as the sample grows (about 100+ distinct picks per market and 3+ weeks of games). If one climbs, that model is earning trust; if all stay at the floor the experimental markets are not beating the market.
- Moneyline model vs market log loss once there are 150+ settled games; the live policy only bets when the model clearly disagrees.
- The Oct 7 style loss days: stakes on experimental markets are still sized from raw edge ($5-$30), so they overstate conviction. Not changed yet because fake money is cheap and the variety is useful for learning; a candidate change is to size those stakes by the learned weight.
