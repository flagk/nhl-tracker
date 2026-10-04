# Player shots on goal: walk-forward backtest

> Research/education only. There are no historical sportsbook prop lines here, so this tests the model against each player's own shrunk shot rate and checks calibration at typical lines; whether it beats the *market* is measured by the paper-trading bets going forward.

187,954 out-of-sample player-games (1,226 players) from 2022-10-07 to 2026-10-03; retrained every 28 days on strictly earlier games; players need 5+ prior games.

## Model vs the player's own rate (negative diff = model better)

| what                                |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |
|:------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|
| shots per player-game (Poisson NLL) | 187954 |  1.5741 |     1.5799 | -0.0057 |  -0.0063 |   -0.0052 |   nan      | nan      |
| over 1.5 (log loss)                 | 167559 |  0.6321 |     0.6358 | -0.0037 |  -0.004  |   -0.0033 |     0.4243 |   0.4308 |
| over 2.5 (log loss)                 | 140214 |  0.5512 |     0.5538 | -0.0025 |  -0.0029 |   -0.0021 |     0.2931 |   0.2923 |
| over 3.5 (log loss)                 |  86307 |  0.4719 |     0.4742 | -0.0023 |  -0.0028 |   -0.0018 |     0.2042 |   0.201  |

## Calibration of P(over 2.5 shots)

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 17527 |      0.1237 |   0.1212 |
| 17527 |      0.1547 |   0.1552 |
| 17526 |      0.1905 |   0.1851 |
| 17527 |      0.2331 |   0.2326 |
| 17527 |      0.2849 |   0.2828 |
| 17526 |      0.3481 |   0.3454 |
| 17527 |      0.4299 |   0.4352 |
| 17527 |      0.5733 |   0.5872 |

Average shots: predicted 1.642, actual 1.639.
