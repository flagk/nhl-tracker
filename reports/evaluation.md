# Evaluation (decision-time prices; closing-line value is the primary evidence)

Generated 2026-10-20T20:34:07+00:00 at commit `1913e9c`, policy sha256 `85224cdbe92e`, data through 2026-10-19. Reproduce with `python scripts/evaluate.py`.

Primary no-vig method: **proportional**; sensitivity: power, shin. Production edges use Shin. Intervals resample whole game dates (4000 resamples). Significance is one-sided against zero and corrected with **holm** across markets, and across all split cells with at least 30 bets. Below 100 bets a market's ROI is labelled insufficient evidence; below 30 CLV observations, likewise. Research and education only; nothing here is betting advice.

## Per market: model picks

| market         |   picks |   pushes_or_void | win_rate   | roi     | roi_lo   | roi_hi   |   p_roi_adj | verdict_roi                                   |
|:---------------|--------:|-----------------:|:-----------|:--------|:---------|:---------|------------:|:----------------------------------------------|
| h2h            |     201 |                0 | 49.8%      | -9.4%   | -16.6%   | -1.2%    |           1 | no demonstrable edge on ROI (adjusted p=1.00) |
| parlay         |      12 |                0 | 0.0%       | -100.0% | -100.0%  | -100.0%  |           1 | insufficient evidence (n=12 < 100)            |
| player_assists |      76 |                0 | 43.4%      | -4.6%   | -43.0%   | 38.6%    |           1 | insufficient evidence (n=76 < 100)            |
| player_goals   |      39 |                1 | 10.3%      | -5.1%   | -77.6%   | 63.7%    |           1 | insufficient evidence (n=39 < 100)            |
| player_points  |     130 |                2 | 36.9%      | -23.0%  | -45.4%   | 3.4%     |           1 | no demonstrable edge on ROI (adjusted p=1.00) |
| player_sog     |     166 |                5 | 49.4%      | -7.3%   | -19.5%   | 6.4%     |           1 | no demonstrable edge on ROI (adjusted p=1.00) |
| spreads        |     147 |                0 | 55.8%      | -9.8%   | -18.4%   | -2.6%    |           1 | no demonstrable edge on ROI (adjusted p=1.00) |
| totals         |     130 |                5 | 45.4%      | -13.6%  | -34.0%   | 7.0%     |           1 | no demonstrable edge on ROI (adjusted p=1.00) |

CLV = margin-free move of the closing no-vig probability toward the pick (probability points; log-odds in the json). Player props have no closing capture.

| market         |   clv_n |   clv_pts |   clv_lo |   clv_hi |   clv_logodds |   p_clv_adj | verdict_clv                                    |   roi_at_closing_price |
|:---------------|--------:|----------:|---------:|---------:|--------------:|------------:|:-----------------------------------------------|-----------------------:|
| h2h            |     183 |    0.0425 |  -0.0394 |   0.1175 |        0.0018 |      0.1487 | no demonstrable edge on CLV (adjusted p=0.15)  |                -0.0971 |
| parlay         |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |
| player_assists |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |
| player_goals   |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |
| player_points  |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |
| player_sog     |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |
| spreads        |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |
| totals         |       0 |  nan      | nan      | nan      |      nan      |    nan      | CLV not measurable (no closing price captured) |               nan      |

## Per market: control picks

| market         |   picks |   pushes_or_void | win_rate   | roi    | roi_lo   | roi_hi   |   p_roi_adj | verdict_roi                                   |
|:---------------|--------:|-----------------:|:-----------|:-------|:---------|:---------|------------:|:----------------------------------------------|
| h2h            |     223 |                0 | 51.6%      | -5.8%  | -12.5%   | 0.3%     |        1    | no demonstrable edge on ROI (adjusted p=1.00) |
| parlay         |      12 |                0 | 25.0%      | -14.4% | -100.0%  | 76.1%    |        1    | insufficient evidence (n=12 < 100)            |
| player_assists |      39 |                1 | 56.4%      | 4.8%   | -1.8%    | 11.4%    |        0.58 | insufficient evidence (n=39 < 100)            |
| player_goals   |      40 |                0 | 32.5%      | -10.3% | -40.5%   | 20.9%    |        1    | insufficient evidence (n=40 < 100)            |
| player_points  |      75 |                0 | 53.3%      | -13.3% | -28.0%   | 1.7%     |        1    | insufficient evidence (n=75 < 100)            |
| player_sog     |     114 |                1 | 50.0%      | 2.4%   | -14.6%   | 19.5%    |        1    | no demonstrable edge on ROI (adjusted p=1.00) |
| spreads        |      87 |                0 | 57.5%      | -10.2% | -24.4%   | 1.6%     |        1    | insufficient evidence (n=87 < 100)            |
| totals         |     213 |                8 | 50.2%      | -2.9%  | -10.3%   | 5.0%     |        1    | no demonstrable edge on ROI (adjusted p=1.00) |

CLV = margin-free move of the closing no-vig probability toward the pick (probability points; log-odds in the json). Player props have no closing capture.

| market         |   clv_n |   clv_pts |   clv_lo |   clv_hi |   clv_logodds |   p_clv_adj | verdict_clv                                    |   roi_at_closing_price |
|:---------------|--------:|----------:|---------:|---------:|--------------:|------------:|:-----------------------------------------------|-----------------------:|
| h2h            |     210 |    0.0206 |  -0.0039 |   0.0606 |         0.001 |      0.0705 | no demonstrable edge on CLV (adjusted p=0.07)  |                  -0.03 |
| parlay         |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |
| player_assists |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |
| player_goals   |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |
| player_points  |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |
| player_sog     |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |
| spreads        |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |
| totals         |       0 |  nan      | nan      | nan      |       nan     |    nan      | CLV not measurable (no closing price captured) |                 nan    |

## Baselines (moneyline, totals and puck-line picks, same games)

```json
{
 "n": 478,
 "model_roi": -0.10656903765690376,
 "random_side_mean_roi": -0.03598308368200837,
 "random_side_roi_95": [
  -0.12373012552301255,
  0.05192730125523008
 ],
 "p_random_at_least_as_good": 0.9400119976004799,
 "market_favourite_roi": -0.0021338912133891225,
 "note": "random side and favourite are priced at the best decision-time price of that side; the market favourite is the side with the higher no-vig probability"
}
```

## Model against the no-vig market on the same finished games

**proportional** (n=122): log loss model 0.6426, market at decision 0.6297, market close 0.6274; Brier 0.2254 / 0.2194 / 0.2183. Model's log-loss gain over the closing market: -0.0152 (95% interval -0.0275 to +0.0003).

Reliability by model-probability quintile:

| bin            |   n |   model |   market_at_decision |   actual |
|:---------------|----:|--------:|---------------------:|---------:|
| (0.297, 0.474] |  25 |   0.414 |                0.405 |    0.36  |
| (0.474, 0.524] |  24 |   0.505 |                0.514 |    0.542 |
| (0.524, 0.571] |  24 |   0.546 |                0.533 |    0.583 |
| (0.571, 0.609] |  24 |   0.592 |                0.567 |    0.667 |
| (0.609, 0.733] |  25 |   0.659 |                0.658 |    0.84  |

**power** (n=122): log loss model 0.6426, market at decision 0.6269, market close 0.6245; Brier 0.2254 / 0.2183 / 0.2171. Model's log-loss gain over the closing market: -0.0180 (95% interval -0.0300 to -0.0029).

**shin** (n=122): log loss model 0.6426, market at decision 0.6279, market close 0.6255; Brier 0.2254 / 0.2187 / 0.2175. Model's log-loss gain over the closing market: -0.0171 (95% interval -0.0292 to -0.0018).

## Edge-threshold sensitivity (all model picks, then moneyline only)

|   min_edge |   picks | roi    | roi_lo   | roi_hi   |   clv_n |   clv_pts |
|-----------:|--------:|:-------|:---------|:---------|--------:|----------:|
|       0    |     365 | -15.0% | -25.2%   | -3.8%    |     122 |    0.0663 |
|       0.01 |     317 | -15.6% | -25.5%   | -4.7%    |     101 |    0.0627 |
|       0.02 |     279 | -16.7% | -26.3%   | -6.3%    |      83 |    0.1164 |
|       0.03 |     234 | -18.0% | -30.4%   | -6.5%    |      69 |    0.134  |
|       0.04 |     189 | -19.4% | -31.5%   | -8.1%    |      57 |    0.1387 |
|       0.05 |     150 | -23.4% | -37.1%   | -10.9%   |      41 |    0.1575 |
|       0.06 |     113 | -21.6% | -38.7%   | -5.3%    |      31 |    0.2415 |
|       0.08 |      61 | -20.9% | -47.0%   | 5.4%     |      14 |    0.2183 |

|   min_edge |   picks | roi    | roi_lo   | roi_hi   |   clv_n |   clv_pts |
|-----------:|--------:|:-------|:---------|:---------|--------:|----------:|
|       0    |     135 | -17.5% | -33.2%   | 0.4%     |     122 |    0.0663 |
|       0.01 |     112 | -18.7% | -35.3%   | 0.6%     |     101 |    0.0627 |
|       0.02 |      93 | -18.9% | -34.0%   | -0.1%    |      83 |    0.1164 |
|       0.03 |      77 | -18.0% | -33.1%   | -0.6%    |      69 |    0.134  |
|       0.04 |      64 | -11.2% | -32.0%   | 12.8%    |      57 |    0.1387 |
|       0.05 |      48 | -8.5%  | -30.4%   | 16.6%    |      41 |    0.1575 |
|       0.06 |      38 | -14.0% | -39.7%   | 23.0%    |      31 |    0.2415 |
|       0.08 |      19 | -1.7%  | -54.8%   | 49.8%    |      14 |    0.2183 |

## CLV under each no-vig method (is the result robust to the method?)

| method       | market   |   clv_n |   clv_pts |   clv_lo |   clv_hi |
|:-------------|:---------|--------:|----------:|---------:|---------:|
| proportional | h2h      |     183 |    0.0425 |  -0.0394 |   0.1175 |
| power        | h2h      |     183 |    0.0447 |  -0.0419 |   0.1239 |
| shin         | h2h      |     183 |    0.044  |  -0.041  |   0.1218 |

## Splits (one correction family over every cell with enough bets)

| split        | cell           |   n | roi     | roi_lo   | roi_hi   |   p_roi_adj |   clv_n |   clv_pts |   p_clv_adj | label                            |
|:-------------|:---------------|----:|:--------|:---------|:---------|------------:|--------:|----------:|------------:|:---------------------------------|
| market       | h2h            | 201 | -9.4%   | -16.6%   | -1.2%    |           1 |     183 |     0.043 |       0.877 | not significant after correction |
| market       | parlay         |  12 | -100.0% | -100.0%  | -100.0%  |         nan |       0 |   nan     |     nan     | insufficient evidence (n<30)     |
| market       | player_assists |  76 | -4.6%   | -43.0%   | 38.6%    |           1 |       0 |   nan     |     nan     | not significant after correction |
| market       | player_goals   |  39 | -5.1%   | -77.6%   | 63.7%    |           1 |       0 |   nan     |     nan     | not significant after correction |
| market       | player_points  | 130 | -23.0%  | -45.4%   | 3.4%     |           1 |       0 |   nan     |     nan     | not significant after correction |
| market       | player_sog     | 166 | -7.3%   | -19.5%   | 6.4%     |           1 |       0 |   nan     |     nan     | not significant after correction |
| market       | spreads        | 147 | -9.8%   | -18.4%   | -2.6%    |           1 |       0 |   nan     |     nan     | not significant after correction |
| market       | totals         | 130 | -13.6%  | -34.0%   | 7.0%     |           1 |       0 |   nan     |     nan     | not significant after correction |
| month        | 2026-09        |  10 | -13.6%  | -13.6%   | -13.6%   |         nan |       2 |     0     |     nan     | insufficient evidence (n<30)     |
| month        | 2026-10        | 891 | -12.3%  | -21.9%   | -2.0%    |           1 |     181 |     0.043 |       0.877 | not significant after correction |
| home_away    | away/under     | 473 | -14.0%  | -23.1%   | -4.8%    |           1 |      77 |     0.182 |       0.072 | not significant after correction |
| home_away    | home/over      | 428 | -10.3%  | -27.1%   | 9.8%     |           1 |     106 |    -0.058 |       1     | not significant after correction |
| fav_dog      | favourite      | 485 | -2.8%   | -9.9%    | 4.2%     |           1 |     101 |     0.084 |       0.868 | not significant after correction |
| fav_dog      | underdog       | 416 | -23.3%  | -42.3%   | -0.3%    |           1 |      82 |    -0.009 |       1     | not significant after correction |
| price_bucket | 1.5-1.8        | 205 | -1.4%   | -14.5%   | 10.6%    |           1 |      55 |     0.187 |       0.406 | not significant after correction |
| price_bucket | 1.8-2.2        | 319 | -9.6%   | -23.9%   | 3.8%     |           1 |      51 |    -0.037 |       1     | not significant after correction |
| price_bucket | 2.2-3.0        | 135 | -27.4%  | -49.9%   | -4.0%    |           1 |      41 |    -0.039 |       1     | not significant after correction |
| price_bucket | <1.5           | 122 | -3.2%   | -18.9%   | 11.3%    |           1 |      25 |     0.102 |     nan     | not significant after correction |
| price_bucket | >3.0           | 120 | -30.3%  | -79.5%   | 33.5%    |           1 |      11 |    -0.146 |     nan     | not significant after correction |

## How many bets would the observed CLV need?

- **h2h**: 183 CLV observations, mean +0.04 probability points; n = ((z_0.95 + z_0.80) * sd / mean)^2 = ((1.645 + 0.842) * 0.0090 / 0.0004)^2 = 2784; about 2,601 more bets
