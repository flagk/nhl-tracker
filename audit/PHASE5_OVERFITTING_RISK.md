# Phase 5: overfitting and risk accounting

Source: `audit/scripts/p5_trials.py` (trial register), `audit/scripts/p5_overfit.py` run in GitHub Actions on the real database; statistics in `nhlbet/stats/overfit.py` and `nhlbet/risk/uncertainty.py`, both unit-tested with positive and negative controls (noise gives PBO near 0.5 and no reality-check rejection; a planted edge is found). Live record only, decision-time prices only.

## 1. Verdict

- **There is no evidence that any strategy ever tried has an edge once the number of tries is accounted for.** White's reality check p = 0.80 and Hansen's SPA p = 0.46 across 29 paper strategies: the best one is what the best of 29 strategies with no edge would look like.
- The best in-sample strategy is **home_ml_control**, a control that bets the home team in every game. Strategies with a positive mean: home_ml_control, market_favorite, puckline_fav_edge, sog_edge, sog_under_edge (mostly controls and prop variants). Deflated for 29 trials its deflated Sharpe ratio is 0.001, and for the 324 trials in the register 2.8e-07: nowhere near the 0.95 that would count as evidence.
- Probability of backtest overfitting: **0.57** (29 variants, 25 days, 924 train/test splits). Above 0.5 means the strategy that looked best in one half tended to rank below the median in the other half. It is computed from only 25 days, so it is a weak statement, but it points the wrong way for any claim of edge.
- **Bankroll simulation:** if the policy were followed with stakes sized on the model's claims, the result depends almost entirely on how much of the model's disagreement with the market is real (theta), and the live record cannot pin that down (section 4). Under no skill the bankroll bleeds the margin and the 20% drawdown kill switch trips; under the estimate from the live record it is worse.

## 2. Trial register: every variant that could have shaped a result

| Family | Variants | Source |
|---|---|---|
| base models in the zoo | 7 | nhlbet/models/base.py make_zoo |
| hyper-parameter grid points (tuned by time-series CV) | 61 | nhlbet/models/tune.py GRIDS |
| ensemble heads and calibration maps | 6 | nhlbet/models/walkforward.py, calibration.py |
| candidate features screened for the 14 kept | 149 | data/models/feature_selection.json |
| builder settings (half-life, shrinkage, window, Elo K, home advantage, regression) | 6 | nhlbet/features/builder.py BuilderConfig |
| goals-model regularisation grid | 9 | scripts/goals_backtest.py |
| props shrinkage grid and dispersions | 16 | nhlbet/models/props.py, scripts/props_backtest.py |
| paper-trading game strategies | 20 | nhlbet/risk/shadow.py STRATEGIES |
| paper-trading prop strategies | 14 | nhlbet/risk/shadow.py PROP_STRATEGIES |
| paper-trading parlay strategies | 4 | nhlbet/risk/parlays.py |
| edge thresholds examined in reports | 8 | config/policy.yaml evaluation.edge_thresholds |
| de-vig methods | 3 | nhlbet/odds/math.py |
| settlement rule sets | 2 | config/policy.yaml settlement.rulesets |
| model-selection candidates compared in Phase 4 (hardening) | 9 | reports/model_comparison_phase4.md |
| walk-forward step lengths and minimum training sizes tried | 2 | scripts/backtest.py defaults |
| markets reported (h2h, totals, spreads, 4 player props, parlays) | 8 | reports/evaluation.md |
| UNKNOWN manual experiments | 0 | not recoverable |
| TOTAL (sum, lower bound) | 324 |  |

The total is a **lower bound**: manual experiments that left no trace in the repository are not counted. It is the number of trials used in the deflated Sharpe ratio. Grid points, feature candidates and settings are not independent bets, so the true effective number of trials is smaller than the sum, but the deflated Sharpe ratio is far below 0.95 for any count above a few dozen.

## 3. Overfitting statistics on the paper-strategy returns

Matrix: 29 strategies by 25 days (daily return per unit staked; days without a bet count as zero); 9 strategies dropped for fewer than 10 betting days.

| Statistic | Value | Reading |
|---|---|---|
| Probability of backtest overfitting (CSCV, 12 blocks) | 0.57 | above 0.5: the in-sample winner tends to disappoint |
| Best strategy's Sharpe (per day) | 0.28 | skew 1.5, kurtosis 5.6 over 25 days |
| Probabilistic Sharpe vs zero | 0.956 | what a naive reading would claim |
| Deflated Sharpe, 29 strategies | 0.0012 | expected maximum Sharpe under no skill: 0.78 |
| Deflated Sharpe, 324 register trials | 2.8e-07 | expected maximum Sharpe under no skill: 1.11 |
| White's reality check p | 0.80 | H0: no strategy has a positive expected return |
| Hansen's SPA p | 0.46 | studentised, consistent centring |

The headline Sharpe of the best strategy looks passable on its own (probabilistic Sharpe 0.96); deflation for the number of tries removes all of it. 25 days is a short window, and these results will be rerun as the forward record grows.

## 4. Bankroll simulation under parameter uncertainty

True win probability of a bet: p_true = p_market + theta x (p_model - p_market). theta is estimated by a date-clustered regression on all 161 live games with a market price: **theta = -0.60, standard error 0.59, 95% interval -1.61 to 0.67**. The interval contains 0 (no skill) and 1 (the model is exactly right), so the data cannot say which; each simulated path draws its own theta from the sampling distribution. Bets: the 260 paper moneyline picks that cleared the policy's edge floor (mean decimal 2.02, mean claimed edge 1.5 points), at the logged decision-time best price, with an optional 2% price haircut for the best price not being obtainable at your own book. Stakes: Kelly on the model's claim x Kelly fraction, capped at the policy's 2% per bet, 10,000 paths, $1,000 start, ruin = bankroll at or below $500, policy kill switch at 20% drawdown.

### 200 bets

| theta scenario | Kelly | price haircut | median final | 5th to 95th pct | P(end below start) | P(ruin) | P(drawdown at least 20%) | median / 95th pct max drawdown |
|---|---|---|---|---|---|---|---|---|
| estimated from the live record | 0.25 | 0% | $835 | $591 to $1,195 | 80% | 1% | 65% | 24% / 43% |
| estimated from the live record | 0.25 | 2% | $835 | $605 to $1,170 | 81% | 0% | 63% | 23% / 42% |
| estimated from the live record | 0.5 | 0% | $812 | $547 to $1,221 | 80% | 3% | 75% | 28% / 48% |
| estimated from the live record | 0.5 | 2% | $810 | $557 to $1,191 | 82% | 2% | 74% | 27% / 47% |
| no skill (theta = 0, known) | 0.25 | 0% | $947 | $713 to $1,273 | 62% | 0% | 44% | 19% / 34% |
| no skill (theta = 0, known) | 0.25 | 2% | $940 | $721 to $1,240 | 65% | 0% | 40% | 18% / 32% |
| no skill (theta = 0, known) | 0.5 | 0% | $933 | $675 to $1,314 | 63% | 0% | 57% | 22% / 38% |
| no skill (theta = 0, known) | 0.5 | 2% | $925 | $679 to $1,276 | 66% | 0% | 55% | 21% / 37% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 0% | $1,052 | $778 to $1,444 | 40% | 0% | 27% | 15% / 29% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 2% | $1,038 | $782 to $1,393 | 42% | 0% | 24% | 15% / 28% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 0% | $1,051 | $740 to $1,509 | 41% | 0% | 39% | 18% / 34% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 2% | $1,035 | $742 to $1,459 | 44% | 0% | 37% | 17% / 33% |

### 500 bets

| theta scenario | Kelly | price haircut | median final | 5th to 95th pct | P(end below start) | P(ruin) | P(drawdown at least 20%) | median / 95th pct max drawdown |
|---|---|---|---|---|---|---|---|---|
| estimated from the live record | 0.25 | 0% | $643 | $332 to $1,249 | 86% | 31% | 93% | 43% / 68% |
| estimated from the live record | 0.25 | 2% | $644 | $345 to $1,196 | 88% | 29% | 93% | 42% / 67% |
| estimated from the live record | 0.5 | 0% | $598 | $281 to $1,286 | 87% | 40% | 97% | 48% / 74% |
| estimated from the live record | 0.5 | 2% | $593 | $290 to $1,218 | 88% | 40% | 96% | 48% / 73% |
| no skill (theta = 0, known) | 0.25 | 0% | $874 | $556 to $1,379 | 69% | 3% | 86% | 30% / 50% |
| no skill (theta = 0, known) | 0.25 | 2% | $858 | $560 to $1,315 | 73% | 3% | 84% | 30% / 49% |
| no skill (theta = 0, known) | 0.5 | 0% | $846 | $499 to $1,430 | 70% | 8% | 93% | 35% / 56% |
| no skill (theta = 0, known) | 0.5 | 2% | $824 | $500 to $1,352 | 74% | 8% | 92% | 34% / 55% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 0% | $1,141 | $676 to $1,930 | 34% | 1% | 62% | 23% / 42% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 2% | $1,099 | $674 to $1,799 | 38% | 1% | 59% | 22% / 41% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 0% | $1,134 | $623 to $2,077 | 36% | 2% | 77% | 26% / 48% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 2% | $1,090 | $619 to $1,934 | 40% | 2% | 75% | 26% / 47% |

Reading:
- With no skill the policy loses roughly the margin, and the 20% drawdown kill switch is the mechanism that would stop it; a drawdown of that size is the likely outcome over 500 bets, not a tail event.
- The "estimated" scenario is worse because the point estimate of theta is negative (the model's disagreements with the market lost money on average), with a standard error so large that the interval spans everything from harmful to perfect.
- Only the optimistic prior, which the live record neither supports nor excludes, produces profitable paths.
- Quarter Kelly (the policy) has shallower drawdowns than half Kelly in every row; nothing here argues for more aggressive staking.

## 5. Limits of this phase

- The strategy matrix covers 25 days of paper trading (since 2026-09-30). PBO and the reality check need far more periods to mean much; they are reported because the instructions require them and because their direction is consistent with every other test, not because they are conclusive.
- theta is estimated on all games with a price, but bets are chosen where the model disagrees most; if skill differs between large and small disagreements, the simulation is off in either direction.
- Prices are the best across books at the decision-time capture. Per-book obtainability is not recorded (raw quotes live only in the cache, and the haircut stands in for it). The official policy has placed no bets, so there are no fills to check prices against.
- The simulation resamples bets independently; correlated same-night exposure is limited by the policy's caps but not modelled here.