# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,565 out-of-sample games from 2021-10-12 to 2026-09-29; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|
| home   | 6565 |      3.1121 |        3.1149 |      1.942  |     1.9596 | -0.0176 |  -0.0218 |   -0.0133 |
| away   | 6565 |      2.8924 |        2.9037 |      1.9052 |     1.926  | -0.0208 |  -0.0256 |   -0.0157 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   lin_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6565 |     0.5715 |   0.554  |     0.6831 |    0.6839 | -0.0008 |  -0.0022 |    0.0005 |      0.1584 |
| total over 6.0  | 5680 |     0.5048 |   0.49   |     0.6928 |    0.6944 | -0.0016 |  -0.0034 |    0.0001 |      0.1747 |
| total over 6.5  | 6565 |     0.4367 |   0.4283 |     0.6845 |    0.6861 | -0.0016 |  -0.0031 |   -0      |      0.183  |
| home -1.5 cover | 6565 |     0.3281 |   0.278  |     0.6181 |    0.6329 | -0.0149 |  -0.0198 |   -0.0097 |      0.2579 |
| home +1.5 cover | 6565 |     0.7273 |   0.7754 |     0.5721 |    0.5865 | -0.0143 |  -0.0187 |   -0.0098 |      0.2256 |

## After walk-forward recalibration (each block mapped by a Platt fit on strictly earlier out-of-sample predictions)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   lin_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6055 |     0.5744 |   0.576  |     0.682  |    0.6833 | -0.0014 |  -0.003  |    0      |      0.1256 |
| total over 6.0  | 5246 |     0.5088 |   0.5114 |     0.6928 |    0.6948 | -0.002  |  -0.0043 |   -0      |      0.1368 |
| total over 6.5  | 6055 |     0.4408 |   0.4477 |     0.6861 |    0.6876 | -0.0016 |  -0.0038 |    0.0004 |      0.1355 |
| home -1.5 cover | 6055 |     0.328  |   0.3123 |     0.6144 |    0.6329 | -0.0185 |  -0.0225 |   -0.0145 |      0.2979 |
| home +1.5 cover | 6055 |     0.726  |   0.7349 |     0.5687 |    0.5878 | -0.0191 |  -0.0231 |   -0.015  |      0.2674 |

`lin_slope` is the coefficient of outcome on the model's logit: about 1 means calibrated, well below 1 means overconfident, near 0 means no signal. The maps used in production (logit p' = a + b*logit p): {"totals": {"a": 0.054, "b": 1.019}, "spreads": {"a": 0.018, "b": 0.812}}.

Mean predicted total 6.11 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.223.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01919
- alpha 50.0: -0.01336
- alpha 200.0: -0.00609
