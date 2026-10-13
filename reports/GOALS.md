# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,649 out-of-sample games from 2021-10-12 to 2026-10-12; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|
| home   | 6649 |      3.1113 |        3.1139 |      1.9423 |     1.96   | -0.0177 |  -0.0218 |   -0.0138 |
| away   | 6649 |      2.8938 |        2.9042 |      1.9058 |     1.9267 | -0.0209 |  -0.0259 |   -0.0161 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6649 |     0.5709 |   0.5541 |     0.6832 |    0.684  | -0.0009 |  -0.0022 |    0.0004 |      0.6652 |
| total over 6.0  | 5754 |     0.5042 |   0.4901 |     0.6927 |    0.6944 | -0.0017 |  -0.0035 |   -0      |      0.7118 |
| total over 6.5  | 6649 |     0.4363 |   0.4284 |     0.6844 |    0.686  | -0.0016 |  -0.0032 |   -0.0001 |      0.7564 |
| home -1.5 cover | 6649 |     0.3282 |   0.2777 |     0.6182 |    0.633  | -0.0148 |  -0.0198 |   -0.01   |      1.2533 |
| home +1.5 cover | 6649 |     0.7266 |   0.7752 |     0.5729 |    0.5872 | -0.0144 |  -0.0187 |   -0.0102 |      1.211  |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6139 |     0.5737 |   0.576  |     0.6821 |    0.6835 | -0.0014 |  -0.0031 |    0      |      0.5367 |
| total over 6.0  | 5320 |     0.5081 |   0.5113 |     0.6928 |    0.6948 | -0.002  |  -0.0043 |   -0      |      0.5636 |
| total over 6.5  | 6139 |     0.4403 |   0.4475 |     0.6859 |    0.6875 | -0.0015 |  -0.0038 |    0.0005 |      0.5632 |
| home -1.5 cover | 6139 |     0.3281 |   0.3122 |     0.6144 |    0.6329 | -0.0185 |  -0.0223 |   -0.0149 |      1.4524 |
| home +1.5 cover | 6139 |     0.7252 |   0.7347 |     0.5695 |    0.5887 | -0.0192 |  -0.0233 |   -0.0151 |      1.4291 |

`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 0.051, "b": 1.018}, "spreads": {"a": 0.018, "b": 0.811}}.

Mean predicted total 6.11 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.222.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01931
- alpha 50.0: -0.01343
- alpha 200.0: -0.00610
