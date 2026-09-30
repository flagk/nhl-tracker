# Goals model: walk-forward backtest (totals and puck line)

> Research/education only. There are no historical sportsbook lines here, so this tests the model against base rates and calibration; whether it beats the *market* is measured by the paper-trading bets going forward.

6,565 out-of-sample games from 2021-10-12 to 2026-09-29; retrained every 14 days on strictly earlier games; 63 candidate features; regularisation alpha chosen: **10.0** (of [10.0, 50.0, 200.0]).

## Goal rates (Poisson negative log-likelihood per team-game; negative diff = model better than the training average)

| side   |    n |   mean_pred |   mean_actual |   nll_model |   nll_base |    diff |   ci_low |   ci_high |
|:-------|-----:|------------:|--------------:|------------:|-----------:|--------:|---------:|----------:|
| home   | 6565 |      3.0738 |        3.1149 |      1.9412 |     1.9596 | -0.0183 |  -0.0222 |   -0.0144 |
| away   | 6565 |      2.8237 |        2.9037 |      1.9069 |     1.926  | -0.0191 |  -0.0234 |   -0.0149 |

## Totals and puck line vs base rates (log loss; negative diff = model better)

| market          |    n |   hit_rate |   mean_p |   ll_model |   ll_base |    diff |   ci_low |   ci_high |   lin_slope |
|:----------------|-----:|-----------:|---------:|-----------:|----------:|--------:|---------:|----------:|------------:|
| total over 5.5  | 6565 |     0.5715 |   0.5369 |     0.6858 |    0.6839 |  0.0019 |   0.001  |    0.0028 |     -0.0713 |
| total over 6.0  | 5680 |     0.5048 |   0.4703 |     0.6962 |    0.6944 |  0.0018 |   0.0009 |    0.0027 |     -0.0881 |
| total over 6.5  | 6565 |     0.4367 |   0.4112 |     0.6871 |    0.6861 |  0.0009 |   0.0004 |    0.0015 |     -0.1034 |
| home -1.5 cover | 6565 |     0.3281 |   0.2805 |     0.6168 |    0.6329 | -0.0161 |  -0.0207 |   -0.0113 |      0.279  |
| home +1.5 cover | 6565 |     0.7273 |   0.7808 |     0.5732 |    0.5865 | -0.0133 |  -0.0175 |   -0.0093 |      0.2402 |

Mean predicted total 6.01 vs actual 6.17; predicted tie-after-60 rate 0.221 vs actual 0.223.

## Alpha comparison (mean Poisson NLL diff)

- alpha 10.0: -0.01873
- alpha 50.0: -0.01287
- alpha 200.0: -0.00560
