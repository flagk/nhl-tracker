# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,665 out-of-sample games from 2021-10-12 to 2026-10-13; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |   rps_model |   rps_base |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|------------:|-----------:|
| home   | 6665 |      3.11   |        3.1139 |      1.9418 |     1.9594 | -0.0176 |  -0.0217 |   -0.0137 |      0.0645 |     0.0658 |
| away   | 6665 |      2.8941 |        2.9061 |      1.9061 |     1.927  | -0.0209 |  -0.0259 |   -0.0161 |      0.0623 |     0.0638 |

## Which joint score distribution? (out-of-sample log-likelihood of the observed regulation score; higher is better)

| joint distribution                              |    n |   mean_log_likelihood |   diff_vs_production |   ci_low |   ci_high |
|:------------------------------------------------|-----:|----------------------:|---------------------:|---------:|----------:|
| production (negative binomial + tie correction) | 6665 |               -3.8362 |               0      |   0      |    0      |
| independent negative binomial                   | 6665 |               -3.848  |              -0.0117 |  -0.0153 |   -0.0081 |
| independent Poisson                             | 6665 |               -3.848  |              -0.0117 |  -0.0153 |   -0.0081 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 6665 |     0.5712 |   0.5539 |     0.6831 |    0.684  | -0.0009 |  -0.0022 |    0.0004 |      0.6662 |         0.2264 |
| total over 6.0  | 5767 |     0.5044 |   0.4899 |     0.6927 |    0.6944 | -0.0017 |  -0.0035 |   -0      |      0.7112 |         0.2155 |
| total over 6.5  | 6665 |     0.4365 |   0.4282 |     0.6845 |    0.686  | -0.0016 |  -0.0032 |   -0.0001 |      0.7553 |         0.2295 |
| home -1.5 cover | 6665 |     0.328  |   0.2775 |     0.6182 |    0.6329 | -0.0147 |  -0.0197 |   -0.0098 |      1.2464 |         0.0781 |
| home +1.5 cover | 6665 |     0.7263 |   0.775  |     0.5732 |    0.5875 | -0.0143 |  -0.0186 |   -0.0101 |      1.2078 |         0.0781 |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |   cal_slope_se |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|---------------:|
| total over 5.5  | 6155 |     0.574  |   0.5781 |     0.682  |    0.6834 | -0.0014 |  -0.0031 |   -0      |      0.5574 |         0.2279 |
| total over 6.0  | 5333 |     0.5083 |   0.5111 |     0.6929 |    0.6948 | -0.0019 |  -0.004  |   -0      |      0.5223 |         0.2149 |
| total over 6.5  | 6155 |     0.4405 |   0.4445 |     0.6864 |    0.6875 | -0.0011 |  -0.0031 |    0.0007 |      0.4172 |         0.2121 |
| home -1.5 cover | 6155 |     0.3279 |   0.3236 |     0.612  |    0.6328 | -0.0208 |  -0.0266 |   -0.0151 |      0.9045 |         0.0599 |
| home +1.5 cover | 6155 |     0.7249 |   0.7191 |     0.5681 |    0.5889 | -0.0208 |  -0.0271 |   -0.014  |      0.8831 |         0.0583 |

`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 1.148, "b": 0.707, "c": -0.184}, "spreads": {"a": -0.041, "b": 1.226, "c": -0.332}}. `cal_slope_se` is its standard error: totals probabilities barely vary, so a slope of 0.5 versus 0.8 is within noise.

Mean predicted total 6.11 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.222.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01925
- alpha 50.0: -0.01340
- alpha 200.0: -0.00609
