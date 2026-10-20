# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,700 out-of-sample games from 2021-10-12 to 2026-10-19; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |   rps_model |   rps_base |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|------------:|-----------:|
| home   | 6700 |      3.1101 |        3.1146 |      1.9421 |     1.9597 | -0.0176 |  -0.0216 |   -0.0137 |      0.0645 |     0.0658 |
| away   | 6700 |      2.8941 |        2.9043 |      1.9052 |     1.9259 | -0.0207 |  -0.0255 |   -0.0157 |      0.0623 |     0.0637 |

## Which joint score distribution? (out-of-sample log-likelihood of the observed regulation score; higher is better)

| joint distribution                              |    n |   mean_log_likelihood |   diff_vs_production |   ci_low |   ci_high |
|:------------------------------------------------|-----:|----------------------:|---------------------:|---------:|----------:|
| production (negative binomial + tie correction) | 6700 |               -3.8355 |               0      |   0      |    0      |
| independent negative binomial                   | 6700 |               -3.8473 |              -0.0118 |  -0.0153 |   -0.0083 |
| independent Poisson                             | 6700 |               -3.8473 |              -0.0118 |  -0.0153 |   -0.0083 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 6700 |     0.5712 |   0.5539 |     0.6832 |    0.684  | -0.0008 |  -0.002  |    0.0005 |      0.6455 |         0.2261 |
| total over 6.0  | 5964 |     0.5183 |   0.5023 |     0.6924 |    0.6938 | -0.0014 |  -0.0029 |    0.0002 |      0.678  |         0.2152 |
| total over 6.5  | 6700 |     0.4613 |   0.4501 |     0.6898 |    0.6912 | -0.0014 |  -0.0029 |    0      |      0.7028 |         0.2255 |
| home -1.5 cover | 6700 |     0.3279 |   0.2775 |     0.6181 |    0.6328 | -0.0147 |  -0.0197 |   -0.0096 |      1.2481 |         0.078  |
| home +1.5 cover | 6700 |     0.7269 |   0.775  |     0.5727 |    0.5869 | -0.0142 |  -0.0187 |   -0.0099 |      1.2045 |         0.078  |

## Totals as a whole distribution (shootout rule `so_adds_goal`: a shootout adds one goal to the total = True)

Each forecast is scored on the observed game total under the CURRENT rule. `log_score` is higher-is-better, `rps` (ranked probability score) lower-is-better; the interval is for the paired RPS difference to the training base rate, resampling whole days. 'previous model' is the same goal model with the shootout goal excluded, i.e. what the pipeline published before the rule was made explicit.

| forecast                                |    n |   log_score |     rps |   rps_minus_base |   rps_diff_ci_low |   rps_diff_ci_high |
|:----------------------------------------|-----:|------------:|--------:|-----------------:|------------------:|-------------------:|
| production (current shootout rule)      | 6700 |    -2.1784  | 0.06773 |           -6e-05 |          -0.00019 |            8e-05   |
| previous model (shootout goal excluded) | 6700 |    -2.19209 | 0.06788 |            9e-05 |          -5e-05   |            0.00022 |
| training base rate                      | 6700 |    -2.17779 | 0.06779 |            0     |           0       |            0       |

Randomised probability-integral-transform of the totals forecast (a calibrated forecast fills the bins equally):

| pit_bin   |   count |   expected |   chi2_p_uniform |
|:----------|--------:|-----------:|-----------------:|
| 0.0-0.1   |     511 |        670 |                0 |
| 0.1-0.2   |     645 |        670 |                0 |
| 0.2-0.3   |     652 |        670 |                0 |
| 0.3-0.4   |     663 |        670 |                0 |
| 0.4-0.5   |     751 |        670 |                0 |
| 0.5-0.6   |     717 |        670 |                0 |
| 0.6-0.7   |     671 |        670 |                0 |
| 0.7-0.8   |     755 |        670 |                0 |
| 0.8-0.9   |     699 |        670 |                0 |
| 0.9-1.0   |     636 |        670 |                0 |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 6190 |     0.574  |   0.5776 |     0.682  |    0.6834 | -0.0014 |  -0.003  |    0      |      0.5516 |         0.23   |
| total over 6.0  | 5514 |     0.5218 |   0.5241 |     0.6921 |    0.6939 | -0.0018 |  -0.0039 |    0      |      0.5293 |         0.2137 |
| total over 6.5  | 6190 |     0.4648 |   0.4689 |     0.6909 |    0.6921 | -0.0013 |  -0.0033 |    0.0006 |      0.4602 |         0.2078 |
| home -1.5 cover | 6190 |     0.3278 |   0.3237 |     0.6119 |    0.6327 | -0.0208 |  -0.0266 |   -0.0148 |      0.9058 |         0.0598 |
| home +1.5 cover | 6190 |     0.7255 |   0.7192 |     0.5677 |    0.5883 | -0.0206 |  -0.0271 |   -0.0139 |      0.8808 |         0.0581 |

`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 1.026, "b": 0.674, "c": -0.161}, "spreads": {"a": -0.04, "b": 1.225, "c": -0.331}}. `cal_slope_se` is its standard error: totals probabilities barely vary, so a slope of 0.5 versus 0.8 is within noise.

Mean predicted total 6.11 vs actual 6.24; predicted tie-after-60 rate 0.221 vs actual 0.222.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01915
- alpha 50.0: -0.01333
- alpha 200.0: -0.00605
