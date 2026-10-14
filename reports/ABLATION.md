# Ablations and parameter sensitivity

> Research/education only. Same walk-forward protocol as the headline backtest (retrained every 28 days on strictly earlier games; the features and C were chosen on games before 2024-07-01). `cost_vs_full` is the out-of-sample log-loss increase from removing that group (positive = the group helps); the interval is a week-clustered paired bootstrap.

2,729 evaluation games from 2024-10-01.

## Remove one feature group at a time

| variant                               |   n_features |   log_loss |   cost_vs_full |   ci_low |   ci_high | verdict                        |
|:--------------------------------------|-------------:|-----------:|---------------:|---------:|----------:|:-------------------------------|
| all selected features                 |           14 |     0.6733 |         0      |   0      |    0      | reference                      |
| without advanced (3 features)         |           11 |     0.6763 |         0.003  |   0.0004 |    0.0054 | group earns its place          |
| without goaltending (2 features)      |           12 |     0.6732 |        -0.0001 |  -0.0008 |    0.0007 | not distinguishable from noise |
| without rest_schedule (4 features)    |           10 |     0.6761 |         0.0028 |   0.0004 |    0.0051 | group earns its place          |
| without situational_team (2 features) |           12 |     0.6726 |        -0.0007 |  -0.0017 |    0.0005 | not distinguishable from noise |
| without special_teams (1 features)    |           13 |     0.6733 |        -0      |  -0.0005 |    0.0005 | not distinguishable from noise |
| without strength (2 features)         |           12 |     0.6749 |         0.0016 |  -0.0016 |    0.0048 | not distinguishable from noise |
| elo only                              |            0 |     0.6792 |         0.0059 |   0.0016 |    0.0103 | group earns its place          |
| home_rate only                        |            0 |     0.6898 |         0.0165 |   0.0084 |    0.0245 | group earns its place          |

## Regularisation strength (logistic C; tuned value 0.003)

|      C |   log_loss |   vs_elo |   vs_home_rate |
|-------:|-----------:|---------:|---------------:|
| 0.0003 |     0.6741 |  -0.0051 |        -0.0156 |
| 0.001  |     0.673  |  -0.0063 |        -0.0168 |
| 0.003  |     0.6733 |  -0.0059 |        -0.0165 |
| 0.01   |     0.6739 |  -0.0054 |        -0.0159 |
| 0.03   |     0.6742 |  -0.0051 |        -0.0156 |
| 0.1    |     0.6743 |  -0.0049 |        -0.0155 |
| 0.3    |     0.6743 |  -0.0049 |        -0.0154 |

## Stability by season (logistic minus Elo, negative = better than Elo)

| season   |    n |   logistic |    elo |   home_rate |   logistic_minus_elo |   ci_low |   ci_high |
|:---------|-----:|-----------:|-------:|------------:|---------------------:|---------:|----------:|
| 2024-25  | 1312 |     0.6609 | 0.669  |      0.6868 |              -0.0082 |  -0.0133 |   -0.0026 |
| 2025-26  | 1312 |     0.6868 | 0.6908 |      0.6927 |              -0.004  |  -0.0105 |    0.0034 |

