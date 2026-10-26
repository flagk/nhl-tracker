# Phase 4: honest evaluation

Source: `audit/scripts/p4_ladder.py` run in GitHub Actions on the real database (10,694 completed games), results `audit/results/phase4_real.json`. Part A uses games up to 2026-01-04 only; part B is the live record, which is the only data nobody has tuned on.

## 1. Verdict

- **Against the betting market: no model beats it, and none is close.** On 161 live games (2026-09-30 to 2026-10-24) every model scores worse than the devigged market at the decision time and at the close, for all three devig methods; every Holm-adjusted Diebold-Mariano p-value is 1.0. The least bad, logistic regression, is 0.0111 log-loss worse than the decision-time market (95% interval -0.0254 to 0.0058).
- **Among the models themselves, the simplest one wins.** Over 1,973 research-window games with a 1-day embargo, logistic regression has the lowest log loss (0.6714). Every model beats the home rate, and logistic and the production stack beat Elo after Holm correction. The production stack (the ensemble) is not better than its own logistic component.
- **The hierarchical Bayesian challenger fails.** It beats the home rate but not Elo (its log loss is above Elo's), so team strength with partial pooling and no engineered features adds nothing over a plain Elo rating.
- **Recalibration does not help.** Raw probabilities are overconfident (calibration slope 0.74 to 0.86), yet online Platt, isotonic and beta calibration leave log loss unchanged or worse; beta calibration improves the reliability measure (ECE) but not log loss.
- **Purging and embargo change nothing** (log loss differs by at most 0.0001 between 0, 1 and 7 days), which is expected because labels resolve on the day of the game. It is now in place and tested anyway.
- **Closing-line value is small, positive, and not distinguishable from zero** under every definition of the close (section 4).

## 2. The ablation ladder (research window, walk-forward, embargo 1 day)

| Model | n | Log loss | Brier | ECE | Calibration slope |
|---|---|---|---|---|---|
| home rate | 1,973 | **0.6891** | 0.2480 | 0.027 | 0.03 |
| Elo | 1,973 | **0.6782** | 0.2427 | 0.043 | 0.82 |
| logistic regression | 1,973 | **0.6714** | 0.2396 | 0.041 | 0.86 |
| random forest | 1,973 | **0.6764** | 0.2419 | 0.040 | 0.74 |
| production stack | 1,973 | **0.6722** | 0.2400 | 0.039 | 0.79 |
| hierarchical Bayes | 1,973 | **0.6807** | 0.2439 | 0.037 | 0.75 |

Log-loss gain over a reference (positive = better than the reference), date-clustered 95% interval, one-sided Diebold-Mariano p with Bartlett HAC variance, Holm-adjusted within each reference:

| Reference | Model | Gain | 95% interval | DM p (Holm) |
|---|---|---|---|---|
| home rate | Elo | +0.0109 | +0.0030 to +0.0187 | 0.009 |
| home rate | logistic regression | +0.0177 | +0.0083 to +0.0269 | 0.000 |
| home rate | random forest | +0.0127 | +0.0030 to +0.0222 | 0.009 |
| home rate | production stack | +0.0169 | +0.0068 to +0.0268 | 0.001 |
| home rate | hierarchical Bayes | +0.0085 | +0.0007 to +0.0160 | 0.015 |
| Elo | home rate | -0.0109 | -0.0187 to -0.0030 | 1.000 |
| Elo | logistic regression | +0.0067 | +0.0025 to +0.0110 | 0.007 |
| Elo | random forest | +0.0018 | -0.0027 to +0.0063 | 0.692 |
| Elo | production stack | +0.0060 | +0.0013 to +0.0105 | 0.024 |
| Elo | hierarchical Bayes | -0.0025 | -0.0065 to +0.0016 | 1.000 |

Reliability of the production stack (8 equal-count bins; the pattern is the usual overconfidence at the extremes: predictions of 34% win 44%, predictions of 64% win 61%):

| Mean predicted | Observed | n |
|---|---|---|
| 0.343 | 0.441 | 247 |
| 0.433 | 0.429 | 247 |
| 0.482 | 0.541 | 246 |
| 0.523 | 0.522 | 247 |
| 0.560 | 0.537 | 246 |
| 0.597 | 0.542 | 247 |
| 0.643 | 0.610 | 246 |
| 0.725 | 0.749 | 247 |

Plot: `audit/results/p4_reliability.png`.

### Calibration variants (online: each block calibrated on earlier blocks only; same games, so compare with the raw column)

| Model | raw (same games) | Platt | isotonic | beta | ECE raw / beta |
|---|---|---|---|---|---|
| home rate | 0.6888 | 0.6913 | 0.6904 | 0.6890 | 0.027 / 0.029 |
| Elo | 0.6795 | 0.6802 | 0.6782 | 0.6806 | 0.043 / 0.047 |
| logistic regression | 0.6729 | 0.6732 | 0.6771 | 0.6722 | 0.041 / 0.027 |
| random forest | 0.6784 | 0.6770 | 0.6803 | 0.6761 | 0.040 / 0.035 |
| production stack | 0.6738 | 0.6733 | 0.6763 | 0.6722 | 0.039 / 0.026 |
| hierarchical Bayes | 0.6818 | 0.6817 | 0.6900 | 0.6820 | 0.037 / 0.048 |

### Embargo sensitivity (log loss)

| Embargo (days) | Elo | logistic | hierarchical Bayes |
|---|---|---|---|
| 0 | 0.6782 | 0.6715 | 0.6806 |
| 1 | 0.6782 | 0.6714 | 0.6807 |
| 7 | 0.6782 | 0.6715 | 0.6807 |

## 3. Against the devigged market (live record)

161 finished games with both a decision-time and a closing price; prices devigged by Shin, power and proportional (multiplicative). Gain = market log loss minus model log loss, so negative means the market is better.

| Model | log loss | Gain vs market at decision (Shin) | 95% interval | DM p (Holm) | Gain vs close (Shin) |
|---|---|---|---|---|---|
| production logged | 0.6518 | -0.0120 | -0.0229 to +0.0006 | 1.000 | -0.0135 |
| home rate | 0.6799 | -0.0400 | -0.0609 to -0.0166 | 1.000 | -0.0416 |
| elo | 0.6517 | -0.0118 | -0.0213 to -0.0008 | 1.000 | -0.0134 |
| logistic | 0.6509 | -0.0111 | -0.0254 to +0.0058 | 1.000 | -0.0127 |
| random forest | 0.6584 | -0.0185 | -0.0322 to -0.0037 | 1.000 | -0.0201 |
| hierarchical bayes | 0.6659 | -0.0261 | -0.0451 to -0.0074 | 1.000 | -0.0277 |

Market log loss by devig method (decision / close): shin: 0.6398 / 0.6383; power: 0.6392 / 0.6376; proportional: 0.6412 / 0.6397. The choice of method moves the market's log loss by about 0.002, ten times less than the gap to any model.

**Power.** The per-game standard deviation of the log-loss difference is about 0.101. Detecting a true advantage of 0.005 over the market (one-sided 5%, 80% power) would take about 2,518 games, roughly 1.9 full regular seasons of live record. With 161 games, a model with a small real edge could not be told apart from one with none; what can be said is that the point estimates are all on the wrong side and the data give no support for an edge.

## 4. Closing-line value per bet

Moneyline picks of the model group (paper picks; the official policy has placed no bets). CLV in probability points: the no-vig closing probability of the picked side minus the no-vig probability at the decision. "Close within X min" keeps only picks whose last pre-game capture was at most X minutes before puck drop, because a stale "close" is not a close (Phase 1: 49% of games have no capture in their last 15 minutes).

| Closing capture within | Picks | Picks with a usable close | Mean CLV (points) | 95% interval | Adjusted p (CLV > 0) | ROI | Verdict |
|---|---|---|---|---|---|---|---|
| 60 min (default) | 260 | 242 | +0.043 | -0.024 to +0.111 | 0.10 | -9.1% | no demonstrable edge on CLV (adjusted p=0.10) |
| 15 min | 260 | 108 | +0.059 | -0.046 to +0.155 | 0.11 | -9.1% | no demonstrable edge on CLV (adjusted p=0.11) |
| 5 min | 260 | 9 | +0.183 | -0.678 to +1.374 | 0.14 | -9.1% | insufficient evidence (n=9 < 30) |

Reading: the mean CLV rises a little as the close gets tighter (0.043, 0.059, 0.183 points) but the 5-minute row rests on 9 picks and every interval includes zero. On the 15-minute definition, the best-defined one with a useful sample (108 picks), the interval is -0.05 to +0.16 points: any true CLV is within a fraction of a percentage point either side of zero. Meanwhile ROI is about -9% and its interval excludes zero on the negative side.

## 5. What this phase does and does not establish

- It establishes that on this data no model in the ladder, including the production stack and a Bayesian challenger, beats the market, and that the simplest competent model (logistic regression) is as good as anything.
- It does not establish that no edge exists anywhere: the live record is 161 games, the detectable effect at that size is large, and totals, puck line and props have no closing prices at all (not measurable).
- The market comparison cannot be run on the long backtest because there are no historical prices: that comparison only gets stronger with time, via the forward ledger (Phase 6).
