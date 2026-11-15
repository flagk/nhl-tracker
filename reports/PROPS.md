# Player count models: walk-forward backtest

> Research/education only. There are no historical sportsbook prop lines here, so this tests each model against the player's own shrunk rate and checks calibration at typical lines; whether it beats the *market* is measured by the paper-trading bets going forward. `n_to_confirm` is the number of player-games a significance test would need at the observed effect size; where `n` is smaller the verdict reads "insufficient evidence".

## Player shots on goal

191,737 out-of-sample player-games (1,247 players) from 2022-10-07 to 2026-10-19; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **25**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                1.59772 |     1.54552 |
|          6 |                1.59725 |     1.54516 |
|         12 |                1.59684 |     1.54489 |
|         25 |                1.59682 |     1.54515 |
|         50 |                1.59817 |     1.54697 |
|        100 |                1.60271 |     1.55176 |
|        200 |                1.61089 |     1.55984 |
|        400 |                1.61968 |     1.56748 |

**Model vs the player's own rate** (negative diff = model better):

| what                                             |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:-------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| shots per player-game (PMF log-likelihood, NLL)  | 191737 |  1.5703 |     1.5877 | -0.0174 |  -0.0183 |   -0.0164 |   nan      | nan      |            648 | skill vs own rate |
| shots per player-game (ranked probability score) | 191737 |  0.0505 |     0.0515 | -0.001  |  -0.001  |   -0.0009 |   nan      | nan      |            697 | skill vs own rate |
| over 1.5 (log loss)                              | 172242 |  0.6301 |     0.6409 | -0.0107 |  -0.0114 |   -0.0101 |     0.4201 |   0.4268 |            921 | skill vs own rate |
| over 2.5 (log loss)                              | 143471 |  0.5501 |     0.5562 | -0.0061 |  -0.0067 |   -0.0054 |     0.2911 |   0.29   |           2172 | skill vs own rate |
| over 3.5 (log loss)                              |  86399 |  0.4743 |     0.4804 | -0.0061 |  -0.007  |   -0.0051 |     0.2056 |   0.2041 |           2442 | skill vs own rate |

**Reliability of P(over 1.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 21531 |      0.2068 |   0.2009 |
| 21530 |      0.2759 |   0.2629 |
| 21530 |      0.3303 |   0.3172 |
| 21530 |      0.3843 |   0.3748 |
| 21530 |      0.4436 |   0.4336 |
| 21530 |      0.5106 |   0.5087 |
| 21530 |      0.5862 |   0.5836 |
| 21531 |      0.6766 |   0.6789 |

**Reliability of P(over 2.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 17934 |      0.1229 |   0.1208 |
| 17934 |      0.1518 |   0.1519 |
| 17934 |      0.186  |   0.1842 |
| 17934 |      0.228  |   0.232  |
| 17933 |      0.2793 |   0.2805 |
| 17934 |      0.3433 |   0.3469 |
| 17934 |      0.4283 |   0.4319 |
| 17934 |      0.58   |   0.5809 |

**Reliability of P(over 3.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 10800 |      0.0905 |   0.0919 |
| 10800 |      0.1082 |   0.1108 |
| 10800 |      0.1295 |   0.1306 |
| 10800 |      0.1556 |   0.1576 |
| 10799 |      0.1885 |   0.2007 |
| 10800 |      0.2316 |   0.2282 |
| 10800 |      0.2954 |   0.2981 |
| 10800 |      0.4334 |   0.4272 |

Average sog: predicted 1.647, actual 1.636.

**Why this family** (out-of-sample count NLL; the simplest family that is calibrated wins ties):

| family                                |   mean_count_NLL |   dispersion |   nll_vs_poisson |
|:--------------------------------------|-----------------:|-------------:|-----------------:|
| Poisson (dispersion 0)                |          1.57303 |         0    |           0      |
| negative binomial (fitted dispersion) |          1.57033 |         0.05 |          -0.0027 |

## Player points (goals + assists)

191,737 out-of-sample player-games (1,247 players) from 2022-10-07 to 2026-10-19; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **50**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                0.84644 |     0.83707 |
|          6 |                0.84591 |     0.83648 |
|         12 |                0.84528 |     0.83579 |
|         25 |                0.84465 |     0.83507 |
|         50 |                0.84445 |     0.83472 |
|        100 |                0.84523 |     0.83526 |
|        200 |                0.84715 |     0.83689 |
|        400 |                0.8493  |     0.83876 |

**Model vs the player's own rate** (negative diff = model better):

| what                                              |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:--------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| points per player-game (PMF log-likelihood, NLL)  | 191737 |  0.8395 |     0.8569 | -0.0175 |  -0.0184 |   -0.0166 |   nan      | nan      |            614 | skill vs own rate |
| points per player-game (ranked probability score) | 191737 |  0.0216 |     0.0221 | -0.0006 |  -0.0006 |   -0.0005 |   nan      | nan      |            620 | skill vs own rate |
| over 0.5 (log loss)                               | 191737 |  0.595  |     0.6094 | -0.0144 |  -0.0151 |   -0.0136 |     0.3482 |   0.3443 |            681 | skill vs own rate |
| over 1.5 (log loss)                               | 191737 |  0.2683 |     0.2764 | -0.0081 |  -0.0087 |   -0.0076 |     0.0902 |   0.0848 |           1384 | skill vs own rate |

**Reliability of P(over 0.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 23968 |      0.1531 |   0.1562 |
| 23967 |      0.2078 |   0.2049 |
| 23967 |      0.2494 |   0.2423 |
| 23967 |      0.2935 |   0.2879 |
| 23967 |      0.345  |   0.3479 |
| 23967 |      0.4077 |   0.4122 |
| 23967 |      0.4863 |   0.5082 |
| 23967 |      0.6117 |   0.6256 |

**Reliability of P(over 1.5), by predicted-probability bin:**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 4297 |      0.1529 |   0.171  |
| 4296 |      0.1652 |   0.1802 |
| 4297 |      0.1797 |   0.1953 |
| 4296 |      0.1958 |   0.2018 |
| 4296 |      0.2147 |   0.227  |
| 4297 |      0.2389 |   0.2495 |
| 4296 |      0.274  |   0.2712 |
| 4297 |      0.364  |   0.3691 |

Average points: predicted 0.451, actual 0.462.

**Why this family** (out-of-sample count NLL; the simplest family that is calibrated wins ties):

| family                                |   mean_count_NLL |   dispersion |   nll_vs_poisson |
|:--------------------------------------|-----------------:|-------------:|-----------------:|
| Poisson (dispersion 0)                |          0.83946 |            0 |                0 |
| negative binomial (fitted dispersion) |          0.83946 |            0 |                0 |

## Player assists

191,737 out-of-sample player-games (1,247 players) from 2022-10-07 to 2026-10-19; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **50**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                0.64935 |     0.64205 |
|          6 |                0.64913 |     0.64179 |
|         12 |                0.6489  |     0.64152 |
|         25 |                0.64869 |     0.64122 |
|         50 |                0.64861 |     0.64103 |
|        100 |                0.6487  |     0.64099 |
|        200 |                0.64931 |     0.64139 |
|        400 |                0.65083 |     0.64266 |

**Model vs the player's own rate** (negative diff = model better):

| what                                               |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:---------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| assists per player-game (PMF log-likelihood, NLL)  | 191737 |  0.6447 |     0.6566 | -0.0118 |  -0.0125 |   -0.0112 |   nan      | nan      |            832 | skill vs own rate |
| assists per player-game (ranked probability score) | 191737 |  0.0154 |     0.0157 | -0.0003 |  -0.0003 |   -0.0003 |   nan      | nan      |            867 | skill vs own rate |
| over 0.5 (log loss)                                | 191625 |  0.5195 |     0.5298 | -0.0103 |  -0.0109 |   -0.0097 |     0.2413 |   0.2379 |            905 | skill vs own rate |
| over 1.5 (log loss)                                | 191625 |  0.1576 |     0.1621 | -0.0044 |  -0.0048 |   -0.004  |     0.0419 |   0.0392 |           2360 | skill vs own rate |

**Reliability of P(over 0.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 23954 |      0.1023 |   0.104  |
| 23953 |      0.141  |   0.1427 |
| 23953 |      0.1713 |   0.1671 |
| 23953 |      0.2018 |   0.1977 |
| 23953 |      0.2361 |   0.2352 |
| 23953 |      0.278  |   0.2795 |
| 23953 |      0.3344 |   0.348  |
| 23953 |      0.4383 |   0.4566 |

**Reliability of P(over 1.5), by predicted-probability bin:**

|   n |   predicted |   actual |
|----:|------------:|---------:|
| 510 |      0.1507 |   0.1431 |
| 510 |      0.156  |   0.1608 |
| 509 |      0.1624 |   0.1886 |
| 510 |      0.1706 |   0.1961 |
| 510 |      0.1808 |   0.1804 |
| 509 |      0.1955 |   0.2102 |
| 510 |      0.2191 |   0.251  |
| 510 |      0.2704 |   0.2824 |

Average assists: predicted 0.283, actual 0.290.

**Why this family** (out-of-sample count NLL; the simplest family that is calibrated wins ties):

| family                                |   mean_count_NLL |   dispersion |   nll_vs_poisson |
|:--------------------------------------|-----------------:|-------------:|-----------------:|
| Poisson (dispersion 0)                |          0.64474 |         0    |            0     |
| negative binomial (fitted dispersion) |          0.64473 |         0.02 |           -1e-05 |

## Player goals (anytime scorer = over 0.5)

191,737 out-of-sample player-games (1,247 players) from 2022-10-07 to 2026-10-19; retrained every 28 days on strictly earlier games; players need 5+ prior games.

**Shrinkage sensitivity** (pseudo-games of position-average rate blended into the player's own; mean count NLL, lower is better). Chosen on games up to 2024-06-30: **100**; scored on later games:

|   shrink_k |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|          3 |                0.46071 |     0.45496 |
|          6 |                0.45997 |     0.45422 |
|         12 |                0.45894 |     0.45321 |
|         25 |                0.45767 |     0.45198 |
|         50 |                0.45662 |     0.45093 |
|        100 |                0.45625 |     0.45048 |
|        200 |                0.45666 |     0.45074 |
|        400 |                0.45743 |     0.45134 |

**Model vs the player's own rate** (negative diff = model better):

| what                                             |      n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:-------------------------------------------------|-------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| goals per player-game (PMF log-likelihood, NLL)  | 191737 |  0.4533 |     0.4622 | -0.009  |  -0.0094 |   -0.0085 |   nan      | nan      |            975 | skill vs own rate |
| goals per player-game (ranked probability score) | 191737 |  0.01   |     0.0101 | -0.0002 |  -0.0002 |   -0.0002 |   nan      | nan      |           1134 | skill vs own rate |
| over 0.5 (log loss)                              | 174828 |  0.4173 |     0.4258 | -0.0085 |  -0.009  |   -0.0081 |     0.1636 |   0.1604 |            987 | skill vs own rate |

**Reliability of P(over 0.5), by predicted-probability bin:**

|     n |   predicted |   actual |
|------:|------------:|---------:|
| 21854 |      0.058  |   0.0567 |
| 21853 |      0.0794 |   0.0778 |
| 21854 |      0.1044 |   0.1036 |
| 21853 |      0.1313 |   0.1269 |
| 21853 |      0.1612 |   0.1581 |
| 21854 |      0.1961 |   0.1987 |
| 21853 |      0.2397 |   0.2569 |
| 21854 |      0.3132 |   0.33   |

Average goals: predicted 0.169, actual 0.172.

**Why this family** (out-of-sample count NLL; the simplest family that is calibrated wins ties):

| family                                |   mean_count_NLL |   dispersion |   nll_vs_poisson |
|:--------------------------------------|-----------------:|-------------:|-----------------:|
| Poisson (dispersion 0)                |          0.45329 |         0    |                0 |
| negative binomial (fitted dispersion) |          0.45329 |         0.05 |                0 |

## Goalie saves (research model, not priced live)

10,646 goalie starts (128 goalies) from 2022-10-07 to 2026-10-19. Saves = shots against thinned by the goalie's shrunk save percentage; a goalie who is pulled has fewer shots against in the history, which is how the risk of a short night enters the distribution. Settlement of a goalie who starts but is replaced differs by book, so nothing is priced or staked from this model until a book's rule and live prices are verified.

**Save-percentage prior sensitivity** (pseudo-shots of league-average goaltending). Chosen on starts up to 2024-06-30: **12800**; scored after:

|   sv_prior |   NLL up to 2024-06-30 |   NLL after |
|-----------:|-----------------------:|------------:|
|        200 |                3.47604 |     3.3801  |
|        400 |                3.47593 |     3.38009 |
|        800 |                3.4758  |     3.38009 |
|       1600 |                3.47568 |     3.38008 |
|       3200 |                3.47557 |     3.38008 |
|       6400 |                3.47551 |     3.38009 |
|      12800 |                3.47549 |     3.3801  |

**Model vs the goalie's own average:**

| what                                             |     n |   model |   baseline |    diff |   ci_low |   ci_high |   hit_rate |   mean_p |   n_to_confirm | verdict           |
|:-------------------------------------------------|------:|--------:|-----------:|--------:|---------:|----------:|-----------:|---------:|---------------:|:------------------|
| saves per player-game (PMF log-likelihood, NLL)  | 10646 |  3.4263 |     3.4907 | -0.0643 |  -0.0717 |   -0.0569 |   nan      | nan      |            180 | skill vs own rate |
| saves per player-game (ranked probability score) | 10646 |  0.0658 |     0.0699 | -0.0041 |  -0.0046 |   -0.0037 |   nan      | nan      |            199 | skill vs own rate |
| over 22.5 (log loss)                             | 10646 |  0.6056 |     0.6334 | -0.0278 |  -0.0342 |   -0.022  |     0.6679 |   0.6354 |            623 | skill vs own rate |
| over 24.5 (log loss)                             | 10646 |  0.6492 |     0.6853 | -0.0361 |  -0.0415 |   -0.0308 |     0.5534 |   0.5275 |            379 | skill vs own rate |
| over 26.5 (log loss)                             | 10646 |  0.6475 |     0.6858 | -0.0383 |  -0.0438 |   -0.0329 |     0.4426 |   0.4223 |            324 | skill vs own rate |
| over 28.5 (log loss)                             | 10646 |  0.6008 |     0.64   | -0.0392 |  -0.0447 |   -0.0337 |     0.3413 |   0.3263 |            275 | skill vs own rate |

**Reliability of P(over 22.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1331 |      0.449  |   0.4568 |
| 1331 |      0.5304 |   0.568  |
| 1330 |      0.5782 |   0.6158 |
| 1331 |      0.6197 |   0.6349 |
| 1331 |      0.66   |   0.6935 |
| 1330 |      0.6998 |   0.7504 |
| 1331 |      0.7413 |   0.7769 |
| 1331 |      0.8049 |   0.8467 |

**Reliability of P(over 24.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1331 |      0.3349 |   0.3193 |
| 1331 |      0.4132 |   0.429  |
| 1330 |      0.4618 |   0.4842 |
| 1331 |      0.5056 |   0.5214 |
| 1331 |      0.5493 |   0.5687 |
| 1330 |      0.5939 |   0.6466 |
| 1331 |      0.642  |   0.6897 |
| 1331 |      0.7196 |   0.7678 |

**Reliability of P(over 26.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1331 |      0.238  |   0.2269 |
| 1331 |      0.3074 |   0.3186 |
| 1330 |      0.353  |   0.3714 |
| 1331 |      0.3954 |   0.3914 |
| 1331 |      0.4392 |   0.4508 |
| 1330 |      0.4852 |   0.5271 |
| 1331 |      0.5365 |   0.5702 |
| 1331 |      0.6237 |   0.6844 |

**Reliability of P(over 28.5 saves):**

|    n |   predicted |   actual |
|-----:|------------:|---------:|
| 1331 |      0.1615 |   0.1458 |
| 1331 |      0.2188 |   0.2119 |
| 1330 |      0.2585 |   0.2722 |
| 1331 |      0.2966 |   0.2825 |
| 1331 |      0.3373 |   0.3426 |
| 1330 |      0.3815 |   0.4173 |
| 1331 |      0.4323 |   0.4733 |
| 1331 |      0.5235 |   0.5853 |

Average saves: predicted 25.62, actual 25.59.

