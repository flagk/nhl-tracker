# Player shots on goal: walk-forward backtest

> Research/education only. There are no historical sportsbook prop lines here, so this tests the model against each player's own shrunk shot rate and checks calibration at typical lines; whether it beats the *market* is measured by the paper-trading bets going forward.

189,937 out-of-sample player-games (1,234 players) from 2022-10-07 to 2026-10-12; retrained every 28 days on strictly earlier games; players need 5+ prior games.

## Model vs the player's own rate (negative diff = model better)

| what                                |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |
|:------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|
| shots per player-game (Poisson NLL) | 189937 |  1.5737 |     1.5794 | -0.0057 |  -0.0062 |   -0.0052 |   nan      | nan      |
| over 1.5 (log loss)                 | 169377 |  0.632  |     0.6357 | -0.0037 |  -0.004  |   -0.0033 |     0.424  |   0.4306 |
| over 2.5 (log loss)                 | 141601 |  0.551  |     0.5536 | -0.0026 |  -0.0029 |   -0.0021 |     0.2929 |   0.292  |
| over 3.5 (log loss)                 |  87092 |  0.4718 |     0.4741 | -0.0023 |  -0.0028 |   -0.0018 |     0.204  |   0.2009 |

## Calibration of P(over 2.5 shots)

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 17701 |      0.1237 |   0.121  |
| 17700 |      0.1546 |   0.1551 |
| 17700 |      0.1904 |   0.1849 |
| 17700 |      0.2329 |   0.2325 |
| 17700 |      0.2847 |   0.2825 |
| 17700 |      0.3478 |   0.3447 |
| 17700 |      0.4295 |   0.4351 |
| 17700 |      0.5729 |   0.5869 |

Average shots: predicted 1.641, actual 1.637.
