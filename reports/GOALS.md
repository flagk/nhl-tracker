# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,700 out-of-sample games from 2021-10-12 to 2026-10-19; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen on games up to 2024-06-30: **3.0** (of [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |   rps_model |   rps_base |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|------------:|-----------:|
| home   | 6700 |      3.1096 |        3.1146 |      1.9412 |     1.9597 | -0.0185 |  -0.0233 |   -0.0138 |      0.0644 |     0.0658 |
| away   | 6700 |      2.8929 |        2.9043 |      1.9045 |     1.9259 | -0.0214 |  -0.0268 |   -0.0156 |      0.0622 |     0.0637 |

## Which joint score distribution? (out-of-sample log-likelihood of the observed regulation score; higher is better)

| joint distribution                              |    n |   mean_log_likelihood |   diff_vs_production |   ci_low |   ci_high |
|:------------------------------------------------|-----:|----------------------:|---------------------:|---------:|----------:|
| production (negative binomial + tie correction) | 6700 |               -3.8333 |               0      |   0      |    0      |
| independent negative binomial                   | 6700 |               -3.8457 |              -0.0124 |  -0.0159 |   -0.0088 |
| independent Poisson                             | 6700 |               -3.8457 |              -0.0124 |  -0.0159 |   -0.0088 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 6700 |     0.5712 |   0.5533 |     0.6836 |    0.684  | -0.0004 |  -0.0018 |    0.0009 |      0.5193 |         0.2059 |
| total over 6.0  | 5964 |     0.5183 |   0.5016 |     0.6928 |    0.6938 | -0.0009 |  -0.0026 |    0.0008 |      0.5473 |         0.1963 |
| total over 6.5  | 6700 |     0.4613 |   0.4493 |     0.6902 |    0.6912 | -0.0011 |  -0.0025 |    0.0005 |      0.5719 |         0.2061 |
| home -1.5 cover | 6700 |     0.3279 |   0.2787 |     0.6161 |    0.6328 | -0.0167 |  -0.0223 |   -0.011  |      1.0654 |         0.0663 |
| home +1.5 cover | 6700 |     0.7269 |   0.7734 |     0.5709 |    0.5869 | -0.016  |  -0.0213 |   -0.0109 |      1.0257 |         0.0662 |

**Scored only after the selection date (2024-06-30), when the regularisation was already fixed:**

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 2764 |     0.555  |   0.5428 |     0.6875 |    0.6873 |  0.0002 |  -0.0019 |    0.0022 |      0.4667 |         0.4154 |
| total over 6.0  | 2462 |     0.5004 |   0.4895 |     0.6938 |    0.6933 |  0.0004 |  -0.0021 |    0.0028 |      0.3568 |         0.3951 |
| total over 6.5  | 2764 |     0.4457 |   0.4386 |     0.6879 |    0.6874 |  0.0005 |  -0.0013 |    0.0024 |      0.2385 |         0.4128 |
| home -1.5 cover | 2764 |     0.3271 |   0.2791 |     0.6245 |    0.6322 | -0.0076 |  -0.0155 |   -0.0003 |      0.9413 |         0.1187 |
| home +1.5 cover | 2764 |     0.7337 |   0.7772 |     0.5734 |    0.5796 | -0.0062 |  -0.0124 |    0.0002 |      0.8739 |         0.1193 |

## Totals as a whole distribution (shootout rule `so_adds_goal`: a shootout adds one goal to the total = True)

Each forecast is scored on the observed game total under the CURRENT rule. `log_score` is higher-is-better, `rps` (ranked probability score) lower-is-better; the interval is for the paired RPS difference to the training base rate, resampling whole days. 'previous model' is the same goal model with the shootout goal excluded, i.e. what the pipeline published before the rule was made explicit.

| forecast                                |    n |   log_score |     rps |   rps_minus_base |   rps_diff_ci_low |   rps_diff_ci_high |
|:----------------------------------------|-----:|------------:|--------:|-----------------:|------------------:|-------------------:|
| production (current shootout rule)      | 6700 |    -2.17903 | 0.06778 |         -1e-05   |          -0.00015 |            0.00013 |
| previous model (shootout goal excluded) | 6700 |    -2.19304 | 0.06793 |          0.00014 |          -1e-05   |            0.00028 |
| training base rate                      | 6700 |    -2.17779 | 0.06779 |          0       |           0       |            0       |

Randomised probability-integral-transform of the totals forecast (a calibrated forecast fills the bins equally):

| pit_bin   |   count |   expected |   chi2_p_uniform |
|:----------|--------:|-----------:|-----------------:|
| 0.0-0.1   |     503 |        670 |                0 |
| 0.1-0.2   |     650 |        670 |                0 |
| 0.2-0.3   |     649 |        670 |                0 |
| 0.3-0.4   |     664 |        670 |                0 |
| 0.4-0.5   |     749 |        670 |                0 |
| 0.5-0.6   |     707 |        670 |                0 |
| 0.6-0.7   |     678 |        670 |                0 |
| 0.7-0.8   |     767 |        670 |                0 |
| 0.8-0.9   |     689 |        670 |                0 |
| 0.9-1.0   |     644 |        670 |                0 |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 6190 |     0.574  |   0.5786 |     0.6822 |    0.6834 | -0.0012 |  -0.0028 |    0.0002 |      0.505  |         0.2597 |
| total over 6.0  | 5514 |     0.5218 |   0.5249 |     0.6922 |    0.6939 | -0.0017 |  -0.0037 |    0.0001 |      0.504  |         0.2402 |
| total over 6.5  | 6190 |     0.4648 |   0.4698 |     0.6909 |    0.6921 | -0.0013 |  -0.0031 |    0.0005 |      0.4443 |         0.2342 |
| home -1.5 cover | 6190 |     0.3278 |   0.3242 |     0.6108 |    0.6327 | -0.0219 |  -0.0279 |   -0.0159 |      0.9219 |         0.0594 |
| home +1.5 cover | 6190 |     0.7255 |   0.7202 |     0.5667 |    0.5883 | -0.0216 |  -0.0279 |   -0.015  |      0.8972 |         0.058  |

`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 1.347, "b": 0.544, "c": -0.214}, "spreads": {"a": -0.014, "b": 1.045, "c": -0.199}}. `cal_slope_se` is its standard error: totals probabilities barely vary, so a slope of 0.5 versus 0.8 is within noise.

Mean predicted total 6.18 vs actual 6.24; predicted tie-after-60 rate 0.221 vs actual 0.222.

## Alpha comparison (mean Poisson NLL diff)

- alpha 0.3: -0.02501
- alpha 1.0: -0.02621
- alpha 3.0: -0.02678
- alpha 10.0: -0.02546
- alpha 30.0: -0.02101
- alpha 100.0: -0.01279
- alpha 300.0: -0.00652
