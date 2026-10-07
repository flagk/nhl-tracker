# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,612 out-of-sample games from 2021-10-12 to 2026-10-06; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|
| home   | 6612 |      3.1112 |        3.1158 |      1.9426 |     1.9602 | -0.0176 |  -0.0218 |   -0.0136 |
| away   | 6612 |      2.892  |        2.9035 |      1.9058 |     1.9266 | -0.0207 |  -0.0255 |   -0.0158 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6612 |     0.5712 |   0.5538 |     0.6831 |    0.684  | -0.0009 |  -0.0021 |    0.0004 |      0.6661 |
| total over 6.0  | 5722 |     0.5045 |   0.4897 |     0.6928 |    0.6944 | -0.0017 |  -0.0033 |    0      |      0.7102 |
| total over 6.5  | 6612 |     0.4366 |   0.4281 |     0.6845 |    0.6861 | -0.0016 |  -0.0031 |   -0.0001 |      0.7531 |
| home -1.5 cover | 6612 |     0.3283 |   0.2779 |     0.6184 |    0.6331 | -0.0147 |  -0.0198 |   -0.0095 |      1.2494 |
| home +1.5 cover | 6612 |     0.7275 |   0.7754 |     0.5721 |    0.5863 | -0.0143 |  -0.0187 |   -0.0099 |      1.2016 |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6102 |     0.5741 |   0.5761 |     0.682  |    0.6834 | -0.0014 |  -0.0031 |    0.0001 |      0.5384 |
| total over 6.0  | 5288 |     0.5085 |   0.5114 |     0.6928 |    0.6948 | -0.0021 |  -0.0043 |    0      |      0.5624 |
| total over 6.5  | 6102 |     0.4407 |   0.4477 |     0.686  |    0.6876 | -0.0016 |  -0.0038 |    0.0005 |      0.5591 |
| home -1.5 cover | 6102 |     0.3283 |   0.3122 |     0.6146 |    0.6331 | -0.0184 |  -0.0225 |   -0.0146 |      1.4474 |
| home +1.5 cover | 6102 |     0.7262 |   0.7347 |     0.5687 |    0.5877 | -0.019  |  -0.0232 |   -0.0148 |      1.4205 |

`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 0.054, "b": 1.019}, "spreads": {"a": 0.02, "b": 0.812}}.

Mean predicted total 6.11 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.222.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01916
- alpha 50.0: -0.01333
- alpha 200.0: -0.00605
