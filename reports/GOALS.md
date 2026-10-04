# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,592 out-of-sample games from 2021-10-12 to 2026-10-03; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|
| home   | 6592 |      3.1109 |        3.1153 |      1.943  |     1.9606 | -0.0176 |  -0.0218 |   -0.0134 |
| away   | 6592 |      2.8915 |        2.9038 |      1.9061 |     1.9268 | -0.0207 |  -0.0255 |   -0.0157 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6592 |     0.5715 |   0.5536 |     0.6831 |    0.6839 | -0.0008 |  -0.0022 |    0.0004 |      0.6649 |
| total over 6.0  | 5704 |     0.5047 |   0.4896 |     0.6928 |    0.6944 | -0.0017 |  -0.0034 |    0      |      0.7131 |
| total over 6.5  | 6592 |     0.4367 |   0.428  |     0.6845 |    0.6861 | -0.0016 |  -0.0031 |   -0.0001 |      0.7592 |
| home -1.5 cover | 6592 |     0.3281 |   0.278  |     0.6182 |    0.633  | -0.0148 |  -0.0197 |   -0.0097 |      1.2496 |
| home +1.5 cover | 6592 |     0.7272 |   0.7754 |     0.5724 |    0.5866 | -0.0142 |  -0.0185 |   -0.0096 |      1.1989 |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   cal_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6082 |     0.5743 |   0.5761 |     0.682  |    0.6834 | -0.0014 |  -0.0031 |   -0      |      0.5354 |
| total over 6.0  | 5270 |     0.5087 |   0.5114 |     0.6927 |    0.6948 | -0.0021 |  -0.0044 |   -0.0001 |      0.5632 |
| total over 6.5  | 6082 |     0.4408 |   0.4476 |     0.686  |    0.6876 | -0.0016 |  -0.0038 |    0.0004 |      0.5632 |
| home -1.5 cover | 6082 |     0.328  |   0.3123 |     0.6144 |    0.6329 | -0.0185 |  -0.0224 |   -0.0145 |      1.4472 |
| home +1.5 cover | 6082 |     0.7259 |   0.7347 |     0.569  |    0.5879 | -0.019  |  -0.023  |   -0.0149 |      1.4167 |

`cal_slope` is the logistic calibration slope (outcome on the model's logit): 1 = calibrated, below 1 = overconfident, near 0 = no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 0.055, "b": 1.02}, "spreads": {"a": 0.018, "b": 0.811}}.

Mean predicted total 6.11 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.222.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01915
- alpha 50.0: -0.01332
- alpha 200.0: -0.00605
