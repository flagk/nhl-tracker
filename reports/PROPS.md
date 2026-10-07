# Player shots on goal: walk-forward backtest

> Research/education only. There are no historical sportsbook prop lines here, so this tests the model against each player's own shrunk shot rate and checks calibration at typical lines; whether it beats the *market* is measured by the paper-trading bets going forward.

188,651 out-of-sample player-games (1,229 players) from 2022-10-07 to 2026-10-06; retrained every 28 days on strictly earlier games; players need 5+ prior games.

## Model vs the player's own rate (negative diff = model better)

| what                                |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |
|:------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|
| shots per player-game (Poisson NLL) | 188651 |  1.5739 |     1.5796 | -0.0057 |  -0.0062 |   -0.0052 |   nan      | nan      |
| over 1.5 (log loss)                 | 168202 |  0.632  |     0.6357 | -0.0037 |  -0.004  |   -0.0033 |     0.4242 |   0.4307 |
| over 2.5 (log loss)                 | 140706 |  0.5511 |     0.5537 | -0.0026 |  -0.0029 |   -0.0022 |     0.293  |   0.2922 |
| over 3.5 (log loss)                 |  86584 |  0.4718 |     0.4741 | -0.0023 |  -0.0028 |   -0.0018 |     0.2041 |   0.2009 |

## Calibration of P(over 2.5 shots)

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 17589 |      0.1237 |   0.1212 |
| 17588 |      0.1546 |   0.155  |
| 17588 |      0.1904 |   0.1847 |
| 17588 |      0.233  |   0.2328 |
| 17588 |      0.2848 |   0.2829 |
| 17588 |      0.3479 |   0.3451 |
| 17588 |      0.4297 |   0.4351 |
| 17589 |      0.5731 |   0.5872 |

Average shots: predicted 1.642, actual 1.638.
