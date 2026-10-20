# Phase 4: model comparison (research window only)

Protocol: `reports/model_comparison_protocol.md` (committed before this result). 1,613 games, 2024-11-29 to 2026-01-04; the holdout (2026-01-05 to 2026-09-29) was not read here.

Erratum to the protocol: it counted 1,973 research games; the production output `stack__online` only exists after the online calibrator's burn-in, so every candidate is compared on the 1,613 games where it exists.

Gain = reference log loss minus candidate log loss (positive = candidate better). Two one-sided tests per candidate (date-clustered bootstrap, Diebold-Mariano with HAC variance), Holm-corrected across candidates; a candidate qualifies only if BOTH adjusted p-values are below 0.05.

| model            | role      |    n |   log_loss |   brier |   cal_slope |     gain |   gain_lo |   gain_hi |    p_boot |      p_dm |   p_boot_adj |   p_dm_adj | qualifies   |
|:-----------------|:----------|-----:|-----------:|--------:|------------:|---------:|----------:|----------:|----------:|----------:|-------------:|-----------:|:------------|
| stack__online    | reference | 1613 |    0.67313 | 0.24042 |     0.82463 |  0       |   0       |   0       | nan       | nan       |          nan |        nan | False       |
| home_rate        | candidate | 1613 |    0.6888  | 0.24783 |   -10.604   | -0.01568 |  -0.0252  |  -0.00624 |   1       |   0.9995  |            1 |          1 | False       |
| elo              | candidate | 1613 |    0.68228 | 0.24472 |     0.63665 | -0.00916 |  -0.01445 |  -0.00388 |   0.9995  |   0.99979 |            1 |          1 | False       |
| elo__online      | candidate | 1613 |    0.68024 | 0.2437  |     0.7537  | -0.00712 |  -0.01164 |  -0.00264 |   0.99875 |   0.99925 |            1 |          1 | False       |
| logistic__online | candidate | 1613 |    0.67331 | 0.24049 |     0.81458 | -0.00018 |  -0.00105 |   0.00069 |   0.65809 |   0.65013 |            1 |          1 | False       |
| rf__online       | candidate | 1613 |    0.67674 | 0.24208 |     0.77577 | -0.00361 |  -0.00759 |   0.00012 |   0.97251 |   0.96278 |            1 |          1 | False       |
| lgbm__online     | candidate | 1613 |    0.6762  | 0.2418  |     0.79573 | -0.00307 |  -0.00634 |   0.0001  |   0.97001 |   0.96746 |            1 |          1 | False       |
| xgb__online      | candidate | 1613 |    0.6758  | 0.24167 |     0.78878 | -0.00267 |  -0.00553 |   0.00015 |   0.96776 |   0.96337 |            1 |          1 | False       |
| wavg__online     | candidate | 1613 |    0.67344 | 0.24056 |     0.81355 | -0.00031 |  -0.00112 |   0.00048 |   0.78205 |   0.77407 |            1 |          1 | False       |
| stack__platt     | candidate | 1613 |    0.67366 | 0.24072 |     0.7659  | -0.00053 |  -0.00244 |   0.0014  |   0.70732 |   0.71105 |            1 |          1 | False       |

**Decision: keep the reference stack__online.**

## The single holdout evaluation

656 games, 2026-01-05 to 2026-09-29; evaluated once at 2026-10-20T20:47:29+00:00 (`reports/holdout_eval.json`). These games were examined in earlier reports, so this is a discipline check, not a pristine sample (see `reports/phase1_leakage.md`). Disclosure: before this recorded run, a local dry run of this same script (writing to a scratch path, not to `reports/holdout_eval.json`) printed these numbers. The selection above depends only on the research window, so the dry run could not have changed it, but the holdout was in fact seen once before it was recorded once.

| model         |   n |   log_loss |   brier |   cal_slope |   gain_over_home_rate |   ci_low |   ci_high |
|:--------------|----:|-----------:|--------:|------------:|----------------------:|---------:|----------:|
| stack__online | 656 |    0.68264 | 0.24478 |     0.8549  |               0.00898 | -0.00405 |   0.02129 |
| elo           | 656 |    0.67849 | 0.24266 |     0.89364 |               0.01312 | -0.00097 |   0.02678 |
| home_rate     | 656 |    0.69161 | 0.24923 |     0.01655 |               0       |  0       |   0       |

Research-window log loss of the same models, for comparison (a large gap would indicate overfitting to the research window):

| model         |   research_log_loss |
|:--------------|--------------------:|
| stack__online |             0.67313 |
| elo           |             0.68228 |
| home_rate     |             0.6888  |

## Shrinking the model toward the market (live games, informational: no historical prices exist, so nothing here is a selection)

122 finished live games. p = (1-w) * market at decision + w * model; the market is the proportional no-vig probability.

|   model_weight |   n |   log_loss |   gain_over_market |   ci_low |   ci_high |   p_not_better_than_market |
|---------------:|----:|-----------:|-------------------:|---------:|----------:|---------------------------:|
|           0    | 122 |     0.6297 |             0      |   0      |    0      |                     1      |
|           0.25 | 122 |     0.632  |            -0.0023 |  -0.0055 |    0.0017 |                     0.8888 |
|           0.5  | 122 |     0.6349 |            -0.0053 |  -0.0117 |    0.0028 |                     0.9135 |
|           0.75 | 122 |     0.6384 |            -0.0088 |  -0.0185 |    0.0032 |                     0.9315 |
|           1    | 122 |     0.6426 |            -0.0129 |  -0.026  |    0.0031 |                     0.9478 |

