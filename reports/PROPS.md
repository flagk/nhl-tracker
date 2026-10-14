# Player count models: walk-forward backtest

> Research/education only. There are no historical sportsbook prop lines here, so this tests each model against the player's own shrunk rate and checks calibration at typical lines; whether it beats the *market* is measured by the paper-trading bets going forward. `n_to_confirm` is the number of player-games a significance test would need at the observed effect size; where `n` is smaller the verdict reads "insufficient evidence".

## Player shots on goal

190,498 out-of-sample player-games (1,243 players) from 2022-10-07 to 2026-10-13; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **25**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                1.59772 |     1.54576 |
|          6 |                1.59725 |     1.54539 |
|         12 |                1.59684 |     1.54512 |
|         25 |                1.59682 |     1.54537 |
|         50 |                1.59817 |     1.54717 |

**Model vs the player's own rate** (negative diff = model better):

| what                                             |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:-------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| shots per player-game (PMF log-likelihood, NLL)  | 190498 |  1.5706 |     1.588  | -0.0174 |  -0.0184 |   -0.0163 |   nan      | nan      |            649 | skill vs own rate |
| shots per player-game (ranked probability score) | 190498 |  0.0506 |     0.0515 | -0.001  |  -0.001  |   -0.0009 |   nan      | nan      |            698 | skill vs own rate |
| over 1.5 (log loss)                              | 171105 |  0.6301 |     0.6409 | -0.0107 |  -0.0114 |   -0.0101 |     0.4202 |   0.427  |            919 | skill vs own rate |
| over 2.5 (log loss)                              | 142607 |  0.5501 |     0.5562 | -0.0061 |  -0.0067 |   -0.0054 |     0.2912 |   0.2901 |           2156 | skill vs own rate |
| over 3.5 (log loss)                              |  85948 |  0.4745 |     0.4805 | -0.006  |  -0.007  |   -0.005  |     0.2057 |   0.2042 |           2475 | skill vs own rate |

**Reliability of P(over 1.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 21389 |      0.2069 |   0.2009 |
| 21388 |      0.2761 |   0.2627 |
| 21388 |      0.3305 |   0.3174 |
| 21388 |      0.3844 |   0.3748 |
| 21388 |      0.4438 |   0.434  |
| 21388 |      0.5109 |   0.5091 |
| 21388 |      0.5864 |   0.5836 |
| 21388 |      0.6767 |   0.679  |

**Reliability of P(over 2.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 17826 |      0.1229 |   0.1204 |
| 17826 |      0.1519 |   0.1515 |
| 17826 |      0.1861 |   0.1844 |
| 17826 |      0.2282 |   0.2321 |
| 17825 |      0.2796 |   0.2817 |
| 17826 |      0.3435 |   0.3462 |
| 17826 |      0.4285 |   0.4324 |
| 17826 |      0.5802 |   0.5811 |

**Reliability of P(over 3.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 10744 |      0.0905 |   0.092  |
| 10743 |      0.1083 |   0.1115 |
| 10744 |      0.1295 |   0.1307 |
| 10743 |      0.1557 |   0.1579 |
| 10743 |      0.1885 |   0.1999 |
| 10744 |      0.2317 |   0.2286 |
| 10743 |      0.2955 |   0.2982 |
| 10744 |      0.4335 |   0.4271 |

Average sog: predicted 1.648, actual 1.637.

## Player points (goals + assists)

190,498 out-of-sample player-games (1,243 players) from 2022-10-07 to 2026-10-13; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **50**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                0.84644 |     0.83742 |
|          6 |                0.84591 |     0.83684 |
|         12 |                0.84528 |     0.83615 |
|         25 |                0.84465 |     0.83544 |
|         50 |                0.84445 |     0.8351  |

**Model vs the player's own rate** (negative diff = model better):

| what                                              |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:--------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| points per player-game (PMF log-likelihood, NLL)  | 190498 |  0.8397 |     0.8571 | -0.0174 |  -0.0183 |   -0.0165 |   nan      | nan      |            618 | skill vs own rate |
| points per player-game (ranked probability score) | 190498 |  0.0216 |     0.0221 | -0.0006 |  -0.0006 |   -0.0005 |   nan      | nan      |            623 | skill vs own rate |
| over 0.5 (log loss)                               | 190498 |  0.5951 |     0.6094 | -0.0143 |  -0.0151 |   -0.0135 |     0.3482 |   0.3443 |            685 | skill vs own rate |
| over 1.5 (log loss)                               | 190498 |  0.2684 |     0.2766 | -0.0081 |  -0.0087 |   -0.0075 |     0.0902 |   0.0849 |           1395 | skill vs own rate |

**Reliability of P(over 0.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 23813 |      0.1531 |   0.1567 |
| 23812 |      0.2078 |   0.205  |
| 23812 |      0.2494 |   0.2424 |
| 23812 |      0.2935 |   0.2876 |
| 23812 |      0.345  |   0.3477 |
| 23812 |      0.4078 |   0.4123 |
| 23812 |      0.4864 |   0.508  |
| 23813 |      0.6117 |   0.6258 |

**Reliability of P(over 1.5), by predicted-probability bin:**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 4272 |      0.1529 |   0.1711 |
| 4272 |      0.1652 |   0.1802 |
| 4271 |      0.1797 |   0.1953 |
| 4272 |      0.1958 |   0.2022 |
| 4272 |      0.2147 |   0.2257 |
| 4271 |      0.2388 |   0.2489 |
| 4272 |      0.2739 |   0.2711 |
| 4272 |      0.3639 |   0.3694 |

Average points: predicted 0.451, actual 0.462.

## Player assists

190,498 out-of-sample player-games (1,243 players) from 2022-10-07 to 2026-10-13; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **50**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                0.64935 |     0.64228 |
|          6 |                0.64913 |     0.64203 |
|         12 |                0.6489  |     0.64175 |
|         25 |                0.64869 |     0.64145 |
|         50 |                0.64861 |     0.64127 |

**Model vs the player's own rate** (negative diff = model better):

| what                                               |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:---------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| assists per player-game (PMF log-likelihood, NLL)  | 190498 |  0.6449 |     0.6567 | -0.0118 |  -0.0124 |   -0.0111 |   nan      | nan      |            838 | skill vs own rate |
| assists per player-game (ranked probability score) | 190498 |  0.0154 |     0.0157 | -0.0003 |  -0.0003 |   -0.0003 |   nan      | nan      |            872 | skill vs own rate |
| over 0.5 (log loss)                                | 190388 |  0.5196 |     0.5298 | -0.0103 |  -0.0109 |   -0.0097 |     0.2414 |   0.238  |            912 | skill vs own rate |
| over 1.5 (log loss)                                | 190388 |  0.1577 |     0.1621 | -0.0044 |  -0.0048 |   -0.004  |     0.0419 |   0.0392 |           2379 | skill vs own rate |

**Reliability of P(over 0.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 23799 |      0.1023 |   0.1042 |
| 23798 |      0.141  |   0.1426 |
| 23799 |      0.1713 |   0.1669 |
| 23798 |      0.2019 |   0.198  |
| 23798 |      0.2362 |   0.2349 |
| 23799 |      0.2782 |   0.28   |
| 23798 |      0.3345 |   0.3481 |
| 23799 |      0.4383 |   0.4565 |

**Reliability of P(over 1.5), by predicted-probability bin:**

|   n |   predicted |   actual |
|----:|------------:|---------:|
| 507 |      0.1507 |   0.144  |
| 506 |      0.1559 |   0.1621 |
| 506 |      0.1623 |   0.1877 |
| 506 |      0.1706 |   0.1957 |
| 506 |      0.1808 |   0.1779 |
| 506 |      0.1956 |   0.2095 |
| 506 |      0.2193 |   0.253  |
| 506 |      0.2704 |   0.2806 |

Average assists: predicted 0.283, actual 0.290.

## Player goals (anytime scorer = over 0.5)

190,498 out-of-sample player-games (1,243 players) from 2022-10-07 to 2026-10-13; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **50**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                0.46071 |     0.45498 |
|          6 |                0.45997 |     0.45425 |
|         12 |                0.45894 |     0.45323 |
|         25 |                0.45767 |     0.452   |
|         50 |                0.45662 |     0.45094 |

**Model vs the player's own rate** (negative diff = model better):

| what                                             |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:-------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| goals per player-game (PMF log-likelihood, NLL)  | 190498 |  0.4537 |     0.4596 | -0.0058 |  -0.0062 |   -0.0055 |   nan      | nan      |           1491 | skill vs own rate |
| goals per player-game (ranked probability score) | 190498 |  0.01   |     0.0101 | -0.0001 |  -0.0001 |   -0.0001 |   nan      | nan      |           1685 | skill vs own rate |
| over 0.5 (log loss)                              | 173416 |  0.4177 |     0.4234 | -0.0057 |  -0.0061 |   -0.0054 |     0.1637 |   0.1606 |           1489 | skill vs own rate |

**Reliability of P(over 0.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 21677 |      0.0592 |   0.0578 |
| 21677 |      0.0819 |   0.077  |
| 21677 |      0.105  |   0.1015 |
| 21677 |      0.1298 |   0.1298 |
| 21677 |      0.1582 |   0.1599 |
| 21677 |      0.1927 |   0.2004 |
| 21677 |      0.2384 |   0.2514 |
| 21677 |      0.3198 |   0.3315 |

Average goals: predicted 0.169, actual 0.172.

## Goalie saves (research model, not priced live)

10,577 goalie starts (127 goalies) from 2022-10-07 to 2026-10-13. Saves = shots against thinned by the goalie's shrunk save percentage; a goalie who is pulled has fewer shots against in the history, which is how the risk of a short night enters the distribution. Settlement of a goalie who starts but is replaced differs by book, so nothing is priced or staked from this model until a book's rule and live prices are verified.

**Save-percentage prior sensitivity** (pseudo-shots of league-average goaltending). Chosen on starts up to 2024-06-30: **1600**; scored after:

|   sv_prior |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|        200 |                3.47604 |     3.38224 |
|        400 |                3.47593 |     3.38224 |
|        800 |                3.4758  |     3.38224 |
|       1600 |                3.47568 |     3.38225 |

**Model vs the goalie's own average:**

| what                                             |     n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:-------------------------------------------------|------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| saves per player-game (PMF log-likelihood, NLL)  | 10577 |  3.4278 |     3.4918 | -0.064  |  -0.0712 |   -0.0568 |   nan      | nan      |            179 | skill vs own rate |
| saves per player-game (ranked probability score) | 10577 |  0.0659 |     0.07   | -0.0041 |  -0.0046 |   -0.0037 |   nan      | nan      |            196 | skill vs own rate |
| over 22.5 (log loss)                             | 10577 |  0.6054 |     0.6332 | -0.0278 |  -0.0337 |   -0.0221 |     0.668  |   0.6376 |            609 | skill vs own rate |
| over 24.5 (log loss)                             | 10577 |  0.6489 |     0.6851 | -0.0362 |  -0.0415 |   -0.0311 |     0.5536 |   0.5299 |            369 | skill vs own rate |
| over 26.5 (log loss)                             | 10577 |  0.6476 |     0.6858 | -0.0381 |  -0.0434 |   -0.0326 |     0.4429 |   0.4247 |            318 | skill vs own rate |
| over 28.5 (log loss)                             | 10577 |  0.6008 |     0.64   | -0.0392 |  -0.0447 |   -0.034  |     0.3416 |   0.3285 |            268 | skill vs own rate |

**Reliability of P(over 22.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1323 |      0.4507 |   0.4528 |
| 1322 |      0.5323 |   0.5772 |
| 1322 |      0.5803 |   0.615  |
| 1322 |      0.6221 |   0.6339 |
| 1322 |      0.6624 |   0.6952 |
| 1322 |      0.7025 |   0.7489 |
| 1322 |      0.7438 |   0.7769 |
| 1322 |      0.8066 |   0.8442 |

**Reliability of P(over 24.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1323 |      0.3366 |   0.3144 |
| 1322 |      0.4151 |   0.4357 |
| 1322 |      0.464  |   0.4818 |
| 1322 |      0.5082 |   0.5265 |
| 1322 |      0.552  |   0.5696 |
| 1322 |      0.5969 |   0.6445 |
| 1322 |      0.645  |   0.6884 |
| 1322 |      0.7217 |   0.7678 |

**Reliability of P(over 26.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1323 |      0.2395 |   0.22   |
| 1322 |      0.3092 |   0.326  |
| 1322 |      0.3551 |   0.3759 |
| 1322 |      0.398  |   0.3896 |
| 1322 |      0.4419 |   0.4629 |
| 1322 |      0.4884 |   0.5136 |
| 1322 |      0.5397 |   0.5711 |
| 1322 |      0.6261 |   0.6846 |

**Reliability of P(over 28.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1323 |      0.1627 |   0.1421 |
| 1322 |      0.2203 |   0.2163 |
| 1322 |      0.2603 |   0.2723 |
| 1322 |      0.299  |   0.2837 |
| 1322 |      0.3399 |   0.3495 |
| 1322 |      0.3846 |   0.4085 |
| 1322 |      0.4356 |   0.4758 |
| 1322 |      0.526  |   0.5847 |

Average saves: predicted 25.67, actual 25.60.

