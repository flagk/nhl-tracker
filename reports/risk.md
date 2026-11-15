# Risk: Monte Carlo bankroll simulation

Template: 49 real moneyline picks that clear the policy's edge floor, staked as the live policy would. 10,000 paths per scenario, bankroll $1,000, quarter Kelly with a 2% per-bet cap (policy `staking`). The kill switch trips at a 20% drawdown from the high-water mark (policy `risk.kill_switch`); ruin = the bankroll ever at or below 50% of its start.

Scenarios: *model true* (the model's probabilities are the real ones), *shrunk halfway* (the real probability is midway between the model's and the market's no-vig one), *market true* (the model has no skill: the only thing that happens to the bankroll is the bookmaker's margin). The live evidence points at the last one (`reports/evaluation.md`), so the third row is the one to plan around. This answers "what could happen if the policy were followed", not "what will happen". Research and education only.

| scenario                           |   bets |   median_final |   p05_final |   p95_final |   p_loss |   p_ruin_50pct |   median_maxdd |   p90_maxdd |   p99_maxdd |   p_kill_switch |
|:-----------------------------------|-------:|---------------:|------------:|------------:|---------:|---------------:|---------------:|------------:|------------:|----------------:|
| model true                         |    200 |       1067.11  |     986.411 |     1154.29 |    0.086 |              0 |          0.032 |       0.053 |       0.079 |           0     |
| model shrunk halfway to the market |    200 |       1028.62  |     954.292 |     1112.41 |    0.26  |              0 |          0.04  |       0.068 |       0.098 |           0     |
| market true (no skill)             |    200 |        991.987 |     920.04  |     1072.62 |    0.567 |              0 |          0.052 |       0.088 |       0.125 |           0     |
| model true                         |    500 |       1177.28  |    1037.4   |     1334.66 |    0.017 |              0 |          0.044 |       0.069 |       0.101 |           0     |
| model shrunk halfway to the market |    500 |       1073.83  |     949.754 |     1216.83 |    0.172 |              0 |          0.058 |       0.096 |       0.142 |           0     |
| market true (no skill)             |    500 |        980.044 |     870.455 |     1112.5  |    0.607 |              0 |          0.086 |       0.142 |       0.194 |           0.007 |

