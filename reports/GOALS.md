# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,565 out-of-sample games from 2021-10-12 to 2026-09-29; retrained every 14 days on strictly earlier games; 69 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|
| home   | 6565 |      3.0882 |        3.1149 |      1.6491 |     1.9596 | -0.3105 |  -0.3214 |   -0.299  |
| away   | 6565 |      2.8522 |        2.9037 |      1.6278 |     1.926  | -0.2982 |  -0.3076 |   -0.2886 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   lin_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6565 |     0.5715 |   0.5325 |     0.5053 |    0.6839 | -0.1786 |  -0.1817 |   -0.1753 |      0.6599 |
| total over 6.0  | 5680 |     0.5048 |   0.4698 |     0.4647 |    0.6944 | -0.2298 |  -0.2328 |   -0.2264 |      0.5955 |
| total over 6.5  | 6565 |     0.4367 |   0.4125 |     0.5021 |    0.6861 | -0.184  |  -0.1869 |   -0.1808 |      0.6716 |
| home -1.5 cover | 6565 |     0.3281 |   0.2834 |     0.4427 |    0.6329 | -0.1902 |  -0.1935 |   -0.187  |      0.4856 |
| home +1.5 cover | 6565 |     0.7273 |   0.7692 |     0.4058 |    0.5865 | -0.1807 |  -0.1851 |   -0.1765 |      0.4112 |

Mean predicted total 6.04 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.223.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.30431
- alpha 50.0: -0.11123
- alpha 200.0: -0.03412
