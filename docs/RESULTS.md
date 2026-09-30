# Results log (honest, cumulative)

> **Latest: see "Current results" below (8 seasons of real NHL API data).** The Phase 1/3 sections are kept as history; their numbers came from
> the old score-only CSV and have been superseded.

All numbers are out-of-sample, walk-forward, on the **legacy score-only dataset** (2,987 games, 28 of 32
teams) because the sandbox used for development cannot reach the NHL API. Advanced metrics, goaltending and
lineup features exist in code and tests but are NaN on this data. Expect the full-API run to change the picture.

## Phase 1: the old claim
55.8% win rate / +11.69% ROI came from even-money payouts with no odds or vig, features that were partly
corrupted, and a model no better than "always pick the home team". Details and corrected ROI table in `AUDIT.md`.

## Phase 3: model comparison (`reports/model_comparison.md`)

Protocol: features chosen and hyper-parameters tuned on games through **2024-06-30**; evaluated on
**1,856 regular-season games, 2024-10-01 to 2026-02-02**, retraining every 14 days on strictly earlier games.
Stackers and calibrators come from inner time-series out-of-fold predictions. Coin flip: log loss 0.6931.

| Model (log loss, lower is better) | Log loss | 95% CI | AUC |
|---|---|---|---|
| Home-rate baseline | 0.6896 | 0.686-0.693 | 0.49 |
| **Elo only** | **0.6817** | 0.673-0.690 | 0.572 |
| Logistic | 0.6836 | 0.675-0.692 | 0.563 |
| LightGBM | 0.6839 | 0.677-0.691 | 0.555 |
| Random forest (old model family, regularised) | 0.6854 | 0.676-0.695 | 0.562 |
| XGBoost | 0.6861 | 0.679-0.694 | 0.556 |
| Stack (logit meta-learner) | 0.6837 | 0.673-0.694 | 0.569 |
| Weighted average | 0.6818 | 0.673-0.690 | 0.570 |

Same-game comparison including **online calibration** (n = 1,492, 2024-12-10 to 2026-02-02):

| | Log loss | ECE | Calibration slope |
|---|---|---|---|
| Stack, online-calibrated (**production output**) | **0.6824** | **0.027** | **0.90** |
| Elo | 0.6841 | 0.046 | 0.63 |
| Home-rate | 0.6889 | 0.034 | n/a |

Paired tests (per-game log loss, week-cluster bootstrap):
- stack (online) vs home-rate: **-0.0065**, 95% CI [-0.0118, -0.0012] -> real skill over the base rate.
- stack (online) vs Elo: -0.0017, CI [-0.0059, +0.0027] -> **not distinguishable from Elo**.

### What this means
1. On score-only features, **nothing reliably beats a well-tuned Elo**. The extra models add complexity
   without demonstrated benefit. The ensemble is kept because it is best calibrated and is the vehicle
   for the richer features that need the API.
2. NHL games are close to coin flips: even the best model is only ~0.011 log-loss better than the base rate.
   For scale, sportsbook closing lines typically sit around 0.67-0.68 (external figure, not measured here), so
   **no edge over the market has been shown or should be assumed**. That question needs real odds (Phase 4).
3. **The drift monitor currently reports WARN**: over the most recent 200 games log loss is 0.7025 (worse than a
   coin flip, not significantly) and calibration slope about 0. That is consistent with the audit's finding that
   2025-26 shows no edge. It is surfaced in the registry and report rather than hidden.
4. Un-tuned models and inner-fold Platt calibration were over-confident (slope 0.5-0.65); stronger regularisation
   (paired one-SE rule) and calibrating on the model's own past out-of-sample predictions fixed most of it.

### Caveats on the evaluation itself
- I looked at one un-tuned (default hyper-parameter) run on the evaluation window before tuning; over-confidence
  seen there motivated the regularisation grid and the online-calibration stage. Online calibration is therefore a
  **post-hoc design choice**; its gain (0.0015 log loss) is inside the noise.
- Choosing a "production" model among several on the same evaluation games carries selection bias.
- 1,492-1,856 games cannot resolve differences smaller than ~0.005 log loss.
- Feature selection on ~1,080 games is unstable: `d_sos10` was kept with a sign that contradicts intuition
  (see `reports/feature_importance.csv`); treat the kept set as provisional until refit on full API data.

## Phase 5: staking risk (`docs/RISK.md`)
Monte Carlo, 3% claimed edge at ~1.95 odds, 1% stakes, 500 bets: with **no** real edge 87% of paths finish down (median $784
from $1,000); with the full edge 40% still finish down. Staking is deliberately small and gated behind raw-edge >= 3%,
shrinkage toward the market, a 12-point "model error" screen, and a drift-monitor kill switch. Because the model has not
been shown to beat the market (Phase 3), expect the policy to output **mostly "no bet"**; that is the intended behaviour.


## Current results: 8 seasons of real NHL API data (2018-19 to 2026-27)

Setup: 9,786 regular-season games with shots, expected goals, goalies, penalties (every game's play-by-play and boxscore; 6,323 older games
ingested in one 66-minute run with 0 failures). Features chosen and hyper-parameters tuned on the **7,157 games before 2024-07-01**
(previously 1,312); evaluated walk-forward on **2,629 games from 2024-10 to 2026-09-29**, retraining every 14 days on earlier games only.
(`reports/model_comparison.md`)

| Model | Log loss | AUC | Note |
|---|---|---|---|
| Home-rate baseline | 0.6897 | 0.49 | |
| Elo only | 0.6800 | 0.582 | |
| Regularised logistic (best single) | **0.6741** | 0.597 | heavily regularised (C = 0.003) on 14 selected features |
| Weighted average | 0.6742 | 0.596 | |
| LightGBM / XGBoost / Random forest | 0.6764 / 0.6763 / 0.6772 | ~0.592 | not better than the simple model |
| Stack (production ensemble, raw) | 0.6751 | 0.596 | |

On the 2,269 games that have online-calibration history: **production `stack__online` 0.6759** (slope 0.83, ECE 0.033) vs Elo 0.6812 and base rate 0.6896.
Paired, week-clustered bootstrap: vs Elo **-0.0053 (95% CI -0.0100 to -0.0003)**, vs base rate **-0.0137 (-0.0209 to -0.0061)**.

What changed vs the one-season-of-features result: the models now beat Elo with intervals that exclude zero (before: -0.0017, CI crossing zero), because
feature selection and tuning rest on 5x more games. Selected features: Elo difference, xG share, Corsi share, xG against, goalie save % (5-game and
season), back-to-backs, penalties drawn, recent workload, early-season weight, stakes. Model health: **OK** (recent 200-game log loss 0.6735).

### Caveats that still apply
- **Nothing here is measured against the betting market.** Real closing lines typically sit around 0.67-0.68 log loss (external figure, not measured here),
  so this model is plausibly near market level; whether it is better can only be answered by the closing-line value and model-vs-market log loss that
  the odds snapshots now being collected will produce over the coming weeks.
- The simple regularised logistic is as good as or slightly better than the ensemble; complexity is not earning its keep. The stack stays as the
  production output because it is better calibrated than the raw models, but the difference between stack, weighted average and logistic is inside the noise.
- Choosing a "best" model among several on the same evaluation games carries selection optimism; the 2024-10+ window was not used for selection or tuning.
- Differences of ~0.002 log loss cannot be resolved with ~2,300 games. The 95% CI for the stack vs Elo nearly touches zero.
