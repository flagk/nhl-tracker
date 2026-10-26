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

True win probability of a bet: p_true = p_market + theta x (p_model - p_market), where theta is how much of the model's disagreement with the market is real. theta is estimated by a date-clustered regression on the live games with a market price; each simulated path draws its own theta from the estimate's sampling distribution. Bets come from the paper record at the logged decision-time best price, with a 2% price haircut (the best price across books is not always obtainable at your own book; 0% is in the JSON). Stakes: Kelly on the model's claim x Kelly fraction, capped at the policy's 2% per bet; 10,000 paths; $1,000 start; ruin = at or below $500; the policy kill switch trips at a 20% drawdown.

### Template: policy-like: the live policy without its early-season guard (3% edge floor, shrunk claims)

**Caution: only 15 settled bets, resampled with replacement, so these paths are indicative of stake size and drawdown depth, not of outcomes. Theta here is measured on the shrunk scale (the policy shrinks claims toward the market, so p_adj - p_market is small and theta is large in magnitude and very uncertain); it is not comparable with the other template's theta.**

15 bets, mean decimal 2.41, mean claimed edge 1.4 points, 100% with a positive claimed edge. **theta = -5.06, standard error 2.67, 95% interval -10.12 to 0.45** over 161 games: the interval contains 0 (no skill) and 1 (exactly right).

| theta scenario | Kelly | bets | median final | 5th to 95th pct | P(end below start) | P(ruin) | P(drawdown at least 20%) | median / 95th pct max drawdown |
|---|---|---|---|---|---|---|---|---|
| estimated from the live record | 0.25 | 200 | $971 | $932 to $1,013 | 88% | 0% | 0% | 4% / 7% |
| estimated from the live record | 0.25 | 500 | $931 | $860 to $1,007 | 93% | 0% | 0% | 8% / 15% |
| estimated from the live record | 0.5 | 200 | $943 | $868 to $1,025 | 88% | 0% | 0% | 8% / 14% |
| estimated from the live record | 0.5 | 500 | $865 | $739 to $1,014 | 93% | 0% | 26% | 15% / 27% |
| no skill (theta = 0, known) | 0.25 | 200 | $996 | $961 to $1,032 | 57% | 0% | 0% | 2% / 5% |
| no skill (theta = 0, known) | 0.25 | 500 | $992 | $938 to $1,049 | 59% | 0% | 0% | 4% / 8% |
| no skill (theta = 0, known) | 0.5 | 200 | $992 | $924 to $1,065 | 57% | 0% | 0% | 5% / 9% |
| no skill (theta = 0, known) | 0.5 | 500 | $983 | $878 to $1,098 | 60% | 0% | 0% | 8% / 15% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 200 | $999 | $964 to $1,035 | 52% | 0% | 0% | 2% / 5% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 500 | $998 | $943 to $1,056 | 52% | 0% | 0% | 4% / 7% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 200 | $997 | $929 to $1,071 | 52% | 0% | 0% | 4% / 9% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 500 | $995 | $888 to $1,115 | 53% | 0% | 0% | 7% / 14% |

### Template: aggressive: any model pick, claim = unshrunk p_model

260 bets, mean decimal 2.02, mean claimed edge 1.5 points, 67% with a positive claimed edge. **theta = -0.60, standard error 0.59, 95% interval -1.61 to 0.67** over 161 games: the interval contains 0 (no skill) and 1 (exactly right).

| theta scenario | Kelly | bets | median final | 5th to 95th pct | P(end below start) | P(ruin) | P(drawdown at least 20%) | median / 95th pct max drawdown |
|---|---|---|---|---|---|---|---|---|
| estimated from the live record | 0.25 | 200 | $835 | $609 to $1,161 | 82% | 0% | 63% | 23% / 41% |
| estimated from the live record | 0.25 | 500 | $640 | $339 to $1,195 | 88% | 29% | 92% | 42% / 68% |
| estimated from the live record | 0.5 | 200 | $807 | $562 to $1,184 | 82% | 2% | 73% | 27% / 46% |
| estimated from the live record | 0.5 | 500 | $592 | $283 to $1,224 | 88% | 40% | 96% | 48% / 73% |
| no skill (theta = 0, known) | 0.25 | 200 | $943 | $722 to $1,227 | 65% | 0% | 40% | 18% / 32% |
| no skill (theta = 0, known) | 0.25 | 500 | $856 | $564 to $1,312 | 72% | 3% | 84% | 30% / 49% |
| no skill (theta = 0, known) | 0.5 | 200 | $926 | $682 to $1,263 | 66% | 0% | 55% | 21% / 37% |
| no skill (theta = 0, known) | 0.5 | 500 | $824 | $503 to $1,354 | 74% | 7% | 92% | 35% / 55% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 200 | $1,037 | $784 to $1,387 | 42% | 0% | 23% | 15% / 28% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.25 | 500 | $1,096 | $679 to $1,784 | 38% | 1% | 59% | 22% / 41% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 200 | $1,034 | $746 to $1,448 | 44% | 0% | 37% | 17% / 32% |
| optimistic prior (theta ~ N(0.5, 0.3)) | 0.5 | 500 | $1,085 | $622 to $1,917 | 40% | 2% | 75% | 26% / 47% |

Reading:
- The aggressive template (any model pick, unshrunk claims) overstates what the actual policy would do: the policy stakes on shrunk probabilities with a 3% edge floor, and in fact has placed no bets in the live record (at a 1% floor it would have placed 1, and with no shrinkage none). The policy-like template (the policy's own rules with only the early-season guard removed; the guard blocks nearly everything in the first weeks of a season) is the relevant one; compare it with `reports/risk.md` from the hardening branch, which used a different, smaller template and reported a 0.7% kill-switch probability under no skill.
- With no skill every template loses roughly the margin; whether the 20% drawdown kill switch trips depends on stake size, which is why the two templates differ so much.
- The "estimated" scenario is worse than no skill when the point estimate of theta is negative (the model's disagreements with the market lost money on average), with a standard error so large that the interval spans everything from harmful to perfect.
- Only the optimistic prior, which the live record neither supports nor excludes, produces profitable paths.
- The policy-like template shows the practical consequence of the policy's caution: because stakes are sized on claims shrunk toward the market, drawdowns stay shallow even under a negative theta; the 20% kill switch is a remote event at quarter Kelly and the risk that matters is the slow loss of the margin, not ruin.
- Quarter Kelly (the policy) has shallower drawdowns than half Kelly in every row; nothing here argues for more aggressive staking.

## 5. Limits of this phase

- The strategy matrix covers 25 days of paper trading (since 2026-09-30). PBO and the reality check need far more periods to mean much; they are reported because the instructions require them and because their direction is consistent with every other test, not because they are conclusive.
- theta is estimated on all games with a price, but bets are chosen where the model disagrees most; if skill differs between large and small disagreements, the simulation is off in either direction.
- Prices are the best across books at the decision-time capture. Per-book obtainability is not recorded (raw quotes live only in the cache, and the haircut stands in for it). The official policy has placed no bets, so there are no fills to check prices against.
- The simulation resamples bets independently; correlated same-night exposure is limited by the policy's caps but not modelled here.