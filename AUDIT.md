# Phase 1 Audit: legacy NHL prediction pipeline

Scope: the code at commit `7d0c7a1` (Random Forest, 11 features, `RealisticBacktester`). That code was
replaced in Phases 2-3; `audit/reproduce_legacy.py` explains how to check it out and re-run it.
Everything below is reproducible with the scripts in `audit/`:

| Script | What it does |
|---|---|
| `audit/leakage_probes.py` (`leakage_probes.txt`) | Feature-construction probes |
| `audit/reproduce_legacy.py` | Re-runs the legacy walk-forward backtest unchanged and saves every bet |
| `audit/statistics.py` (`statistics.txt/.json`) | Bootstrap CIs, significance tests, power, calibration, baselines |

## Bottom line

**The README claim of a "proven edge" with "11.69% ROI" is not supported.** The number is
reproducible (1,805 bets, 1,008 wins, 55.84%, +11.69%), but it is an arithmetic artefact
of the payout assumption, not evidence of betting edge.

| Question | Finding |
|---|---|
| Is there look-ahead leakage in the features? | **No.** Corrupting every game on/after the target date changed features in 0/200 samples. |
| Is the walk-forward time-ordered? | **Yes**, with caveats (stale test-time features, below). |
| Were real odds used? | **No. There are no odds anywhere in the repo.** Every win pays exactly +1.00 and every loss costs 1.00 (even money, no vig). |
| Is the model better than the trivial "always pick home" rule? | **No.** On the same 1,805 bets the model is 0.4 pts *worse* (95% CI −3.4 to +2.6). |
| Are the probabilities usable for bet sizing? | **No.** Mean stated confidence 64.3% vs realised 55.8%. Brier 0.2535 and log loss 0.7028 are *worse than a coin flip* (0.2500 / 0.6931). |
| Is 1,805 bets enough to separate edge from luck? | Enough to see a ~3-point win-rate edge; **not** enough for the 1-5% ROI a real market edge would look like (see below). |

## Corrected numbers

Legacy reported: **55.84% win rate, +11.69% ROI**, which is `2 × win rate − 1`. Any ROI
depends on the price actually paid, and no historical prices exist in this repo, so the
table below shows ROI under explicit flat-price assumptions. These are **assumptions, not
measurements.** Break-even win rate is the price's implied probability, vig included.

| Assumed price on every bet | Break-even | ROI | 95% bootstrap CI |
|---|---|---|---|
| +100 (legacy, no vig) | 50.00% | +11.69% | +7.4% to +16.2% |
| −105 | 51.22% | +9.03% | +4.8% to +13.5% |
| **−110 (standard vig)** | 52.38% | **+6.61%** | +2.5% to +10.9% |
| −120 | 54.55% | +2.38% | −1.6% to +6.5% |
| −130 | 56.52% | −1.20% | −5.0% to +2.8% |
| −150 | 60.00% | −6.93% | −10.5% to −3.1% |
| −175 | 63.64% | −12.24% | −15.6% to −8.7% |

The honest reading: the number is **somewhere between about +7% and about −7%, and I cannot
say where**. The model picks the home team 56% of the time and is right 60.7% of the time when
it does, so many picks are probably home favourites priced at −130 or shorter. At those
prices the same win rate loses money. A −110 flat price is the most generous realistic
assumption for a favourites-heavy bettor, so **+6.6% should be read as an upper-end
estimate, not a result.**

## Statistical tests (n = 1,805, 1,008 wins)

- Win rate 95% CI: Wilson [53.5%, 58.1%]; iid bootstrap [53.6%, 58.1%]; **cluster (by-week)
  bootstrap [53.4%, 58.1%]**. Bets on one slate are not independent, and clustering barely
  widens the interval.
- One-sided exact binomial test vs break-even:

| H0: win rate ≤ | p |
|---|---|
| 50.00% (legacy even-money) | 3.7e-07 |
| 52.38% (−110) | 0.0017 |
| 54.55% (−120) | 0.139 |
| 60.00% (−150) | 0.9998 |

- **Beating break-even only says the model beats a coin flip at fair prices. It does not show
  the model beats the market.** The relevant benchmark is the market's own probability,
  and that is unavailable here.
- The **"always pick home" baseline** hits 54.7% over all 2,400 tested games and 56.2% on the
  model's own bets. The model's 55.8% is statistically indistinguishable from it. Most of
  the "edge" is home-ice advantage, which every sportsbook already prices.
- **Power.** Minimum detectable win-rate edge at n = 1,805 (80% power, one-sided α = 0.05) is
  about +2.9 pts. To detect a true ROI of 5% you need ~2,500 bets, 3% ~6,900, 2% ~15,500 and
  1% ~61,800. A real market edge is likely 1-3%. **Backtests of this size, and even a
  full season of live bets, cannot verify it, which is why CLV (Phase 4) is the primary
  long-run metric.**
- **Stability.** Hit rate by season: 2023-24 57.9% (423 bets), 2024-25 57.7% (893),
  **2025-26 50.7% (489)**. The most recent, and most live-like, sample shows no edge at all.
- **Forking paths.** The 0.55 threshold and other choices were fixed constants with no
  documented tuning, but the number of variants tried before this one is unknown. Treat
  even the surviving p-values as optimistic.

## Calibration (bets placed)

| Stated confidence | n | Mean stated | Realised hit rate |
|---|---|---|---|
| 0.55-0.60 | 575 | 57.0% | 52.0% |
| 0.60-0.65 | 500 | 61.9% | 56.6% |
| 0.65-0.70 | 334 | 66.9% | 53.6% |
| 0.70-0.80 | 319 | 73.6% | 61.8% |
| 0.80-1.00 | 77 | 83.9% | 64.9% |

Every bin is overconfident by 5-19 points. Random Forest `predict_proba` from unpruned trees is
not a probability. Feeding it to Kelly, as `backtester.kelly_criterion` and `predictor` edge
maths do, would systematically overbet.

## Findings by area

### 1. Data leakage
- **Feature date filter is clean.** `calculate_team_features` uses `Date < game_date`
  (strict, so same-day games are excluded, conservatively). Verified by the corruption test.
  Labels come from `Winner`, and there is no shuffled split.
- **No closing lines, post-game stats or season aggregates including the target game are used.**
- The original `NHLBacktester.backtest()` (used by `src/test_backtest.py`) **is in-sample**: it
  trains on all history in `run_training`, then predicts those same games. Its output must
  never be quoted. The README figures come from the walk-forward one.

### 2. Feature bugs (not leakage, but they invalidate the model as described)
- **Home-perspective columns misapplied.** `Points`, `GoalDiff`, `GoalsFor` and
  `GoalsAgainst` in the CSV are always from the *home* team's view (verified 100%).
  `feature_engineer.py` sums them over all of a team's games, so **away games use the
  opponent's numbers.** Correlation of the legacy value vs the correct value across 28
  teams: `ppg` −0.07, `goal_diff_pg` +0.12, `gf_pg` +0.44, `ga_pg` +0.33. Only `win_pct`
  (built from `Winner`) is correct. The README's claim of goal-differential-based
  features does not hold in practice.
- **`_calculate_win_streak` is wrong.** It compares a team name to DataFrame index labels
  (`game['Home'] in team_games.index`), so it agrees with the correct streak in only 59 of
  300 sampled team-dates.
- **Career-to-date, not season-to-date.** No feature resets or decays across seasons, so
  "form" spans up to ~2.3 seasons of stale data. Preseason and early-season teams carry last
  year's strength unchanged.
- `home_away_split` is built from the corrupted `Points` column.
- **Unknown teams and missing history collapse to zeros**, which the model reads as a real
  value (`_zero_features`).

### 3. Walk-forward validity
- Time ordering and the expanding training window are correct: train on `history[:train_end]`,
  test on the next 100 games.
- **Test-time features are stale.** Games in the test window are featurised against
  `train_data` only, so the features for game *k* of 100 ignore the previous *k−1* results and
  rest-days are computed from the last *training* game. This is conservative rather than
  leaky, but it does not reflect live use, where history is current.
- Model is retrained every 100 games (~10 days), not daily as the README implies.
- **The inner "cross-validation" is not used for selection**, and the printed fold
  accuracies (51-55%) are much lower than the headline. They are also on very small folds.

### 4. Odds assumptions
- **No odds exist in the dataset or code.** Backtest payout is +100/−100. `predictor.py`
  hard-codes decimal 1.91 for "edge" on every game, regardless of the real price.
- `backtester.kelly_criterion` sizes stakes from the aggregate win rate at flat 1.91 with a
  25% cap. That is full Kelly at up to 25%, far above the intended fractional Kelly.
- Real closing lines could not be obtained during this audit. The sandbox network
  policy blocks api-web.nhle.com and the odds sources. Historical NHL lines for 2023-2026 are
  also not available from any reliable free bulk source I know of. **Phase 4 therefore
  records live odds snapshots going forward, and any ROI before that must be labelled as
  assumption-based.**

### 5. Data quality
- 2,987 games (2023-10-10 to 2026-02-02); **28 of 32 teams**: CBJ, SEA, TBL and UTA are missing.
  Their games were dropped by `prepare_data.py`, presumably from abbreviation mismatches
  (e.g. `TB` vs `TBL`; `config.yaml` also lists non-standard codes like `LA`, `NJ`, `SJ`). The
  file has ~1,080 / 1,155 / 751 games per season vs ~1,312 in a full regular season.
  Schedule strength and rest features are computed on an incomplete schedule, so they are
  biased (a team's real previous game may be absent).
- Playoff games are mixed in with the regular season with no flag.
- Home win rate is 54.0% over the file, the baseline to beat.

### 6. Automation and reporting
- `daily_run.yml` calls `nhl_tracker.py`, which does not exist, and `nhl_predictions.yml` runs
  `test_realistic.py`. **Neither fetches new data:** `NHLDataFetcher` is a stub returning empty
  results, so the history file never grows and the "daily retrain" retrains on identical data.
  `nhl_predictions.yml` also lacks `permissions: contents: write` and does not install
  `python-dotenv`, which `main.py` imports.
- `update_readme.py` reads `data/logs/nhl_predictions_log.csv`, which is never written, and
  recomputes ROI with the same even-money assumption. The README "Key Results" would be
  overwritten with the same inflated metric if it ever ran.
- `.pyc` files and the 9.8 MB model pickle are committed. Two dependency lists disagree
  (pandas 1.5.3 vs 2.1.3).
- There are no unit tests, only two scripts named `test_*.py` that run backtests.

## What this means for the next phases

1. Treat the legacy model as a **home-advantage detector**. The bar to beat is "always home"
   (~54-56%), not 50%.
2. The new backtest needs (a) fixed team-perspective features, (b) season-aware and
   as-of-daily features, (c) a complete 32-team schedule, (d) probabilities judged by
   Brier/log-loss against the **home-rate and Elo baselines**, and (e) ROI only against
   recorded prices, otherwise reported as scenario tables like the one above.
3. Bet-sizing and edge logic must use **calibrated** probabilities and be shrunk toward the
   market, because the current confidence numbers overstate reality by ~9 points.
4. Success will be judged on CLV and log-loss vs the closing line long before ROI can be
   trusted.

## Limitations of this audit
- No real odds were available, so ROI under real prices is bounded but not measured.
- Results are from one run of the legacy code (random_state=42), and the bootstrap seed is
  fixed at 42.
- The dataset itself was not cross-checked against the NHL API (network blocked).
