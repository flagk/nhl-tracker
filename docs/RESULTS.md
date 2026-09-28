# Results log (honest, cumulative)

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
