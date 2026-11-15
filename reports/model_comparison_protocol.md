# Phase 4 protocol (written before any Phase 4 result was produced)

Purpose: decide whether any simpler or alternative model should replace the current production model, and evaluate the holdout exactly once.

## Data
- Predictions: `reports/walkforward_predictions.csv`, every game scored by models trained on strictly earlier games (see `reports/phase1_leakage.md`).
- Research window: games dated before the holdout start in `reports/holdout_lock.json` (2026-01-05): 1,973 games from 2024-10-04. All choices below use this window only (`nhlbet.validity.holdout.research_view`).
- Holdout: the 656 games from 2026-01-05 to 2026-09-29. It is evaluated once, after the selection is fixed, by `evaluate_once`; a second evaluation raises.
- Live window: games after the holdout (from 2026-09-30) with market prices. Informational only: used for the market-blend check, never for selection.

## Reference and candidates
- Reference (current production output): `stack__online`.
- Candidates: `home_rate`, `elo`, `elo__online`, `logistic__online`, `rf__online`, `lgbm__online`, `xgb__online`, `wavg__online`, `stack__platt`.
- Market blend (informational, live window only): p = (1 - w) * p_market + w * p_model with w fixed in advance at 0, 0.25, 0.5, 0.75, 1. Weights are not tuned: no historical market prices exist.

## Decision rule
1. Metric: mean log loss per game on the research window (Brier and calibration slope are reported, not used to decide).
2. For each candidate, the paired per-game loss difference (reference minus candidate; positive = candidate better) gets (a) a date-clustered bootstrap one-sided p-value and (b) a one-sided Diebold-Mariano p-value with a Bartlett-kernel HAC variance, lag floor(n^(1/3)).
3. Both p-values are corrected across the candidates with Holm (policy `multiple_comparisons`).
4. A candidate replaces the reference only if BOTH adjusted p-values are below 0.05 (policy `alpha`). If several qualify, the one with the lowest research log loss wins. Otherwise the reference stays: an improvement must be statistically supported, and ties go to the status quo.
5. The holdout is then evaluated once for: the selected model, the reference, `elo`, and `home_rate`. The result is recorded whatever it is, with its timestamp, and is not used to change the selection.

## What would change my mind
Nothing in the holdout changes the selection. The holdout result is reported next to its research-window result; a large gap between them is reported as evidence of overfitting to the research window, not corrected for.
