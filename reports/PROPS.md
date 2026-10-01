# Player shots on goal: walk-forward backtest

> Research/education only. There are no historical sportsbook prop lines here, so this tests the model against each player's own shrunk shot rate and checks calibration at typical lines; whether it beats the *market* is measured by the paper-trading bets going forward.

187,115 out-of-sample player-games (1,225 players) from 2022-10-07 to 2026-09-30; retrained every 28 days on strictly earlier games; players need 5+ prior games.

## Model vs the player's own rate (negative diff = model better)

| what                                |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |
|:------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|
| shots per player-game (Poisson NLL) | 187115 |  1.5744 |     1.5801 | -0.0057 |  -0.0063 |   -0.0052 |   nan      | nan      |
| over 1.5 (log loss)                 | 166783 |  0.6322 |     0.6358 | -0.0037 |  -0.004  |   -0.0033 |     0.4245 |   0.4309 |
| over 2.5 (log loss)                 | 139618 |  0.5512 |     0.5537 | -0.0026 |  -0.003  |   -0.0022 |     0.2931 |   0.2924 |
| over 3.5 (log loss)                 |  85974 |  0.4719 |     0.4742 | -0.0023 |  -0.0028 |   -0.0018 |     0.2042 |   0.2011 |

## Calibration of P(over 2.5 shots)

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 17453 |      0.1237 |   0.1211 |
| 17452 |      0.1547 |   0.1548 |
| 17452 |      0.1906 |   0.1851 |
| 17452 |      0.2332 |   0.233  |
| 17452 |      0.2851 |   0.2831 |
| 17452 |      0.3482 |   0.3448 |
| 17452 |      0.43   |   0.4355 |
| 17453 |      0.5734 |   0.5873 |

Average shots: predicted 1.643, actual 1.639.
