# Phase 1: leakage and validity audit

Scope: items 1 to 8 of Phase 1. Every check has a test (`tests/test_validity.py`, `test_validity_realdata.py`, `test_settlement.py`, `test_data_integrity.py`, `test_no_leakage.py`,
`test_audit_phase1.py`) and the whole-history checks were run on the real database in GitHub Actions (`reports/phase1_realdata.md`, workflow run on branch `hardening`).

## Verdict
No leakage was found in the features, the walk-forward models, the stack or the calibrators. The numbers that describe the model's predictive skill did not change. Four real defects were
found and fixed in settlement, void-game handling and goalie-confirmation timing; they change how paper bets are graded, not what the model predicts.
The headline you asked about (about 56% win rate, double-digit ROI, about 1,800 bets) is the legacy README claim: 1,805 bets, 1,008 wins, 55.84%, +11.69%. It recomputes exactly from
`audit/legacy_predictions.csv`, and `AUDIT.md` shows it is an even-money artefact (every win paid +1.00, no odds or vig existed anywhere). The current system has no such record.

## Before and after
"Before" is the code at the merge commit `04b2c5c` (`reports/baseline/before.json`); "after" is this branch (`after.json`; the live section is recomputed from the committed exports).

| Metric | Before | After | Why |
|---|---|---|---|
| Legacy headline: bets / wins / win rate / ROI at even money | 1,805 / 1,008 / 55.84% / +11.69% | identical | recomputed from the committed per-bet file; not a market result. The code re-run was not completed and is not claimed |
| Walk-forward n, 2024-10-04 to 2026-09-29 | 2,629 | identical | |
| Walk-forward log loss: home rate / Elo / logistic / stack | 0.6897 / 0.6800 / 0.6741 / 0.6751 | identical | no leak found, so no change is expected; the truncation test passes to 1e-13 |
| Live moneyline picks / ROI (flat $10) | 201 / -9.4% | identical | moneyline settlement does not depend on the shootout rule |
| Live totals paper bets, profit under each shootout rule | 465 bets, $-348.9 (old rule) | 465 bets, $-518.0 (new default) | 15 of 465 results flip; recomputed from the database in CI |
| Bets with no named book / odds captured after puck drop | not measured | 0 / 0 (new checks) | every logged bet names a book and has a pre-game price |
| Pick-log files / tree hash | 689 / `2299adf38cbba895` | 689 / same | no log was modified |

## Item by item
| # | Requirement | Status | Proof |
|---|---|---|---|
| 1 | Features use only pre-decision information | already covered; extended | `tests/test_no_leakage.py` (future corruption, dropped future games, same-day, row order, goalie modes); added `nhlbet/validity/availability.py`, called inside `fit_stack`, so every walk-forward and production fit fails closed on an outcome column, a closing-line column, or a feature that reproduces the outcome |
| 1 | Scalers, imputers, calibrators fit on past data only | already covered; test added | `test_preprocessing_and_calibration_never_use_other_test_rows`: poisoning every other test row changes none of a row's outputs |
| 1 | Goalie confirmations used only if known at decision time | **fixed** | `starter_overrides` / `goalie_display` gained `as_of`; the loader kept re-stamping the capture time to "now" on every reload, now it keeps the first-seen time. `test_a_goalie_confirmation_captured_after_the_decision_time_is_ignored`, `test_reloading_the_confirmation_file_keeps_the_first_seen_time` |
| 2 | Truncation invariance | added | `test_predictions_for_games_up_to_D_do_not_change_when_later_data_is_added` plus a power check that catches a planted look-ahead feature; real data: 959 games, max difference 1.1e-13 |
| 3 | Tuning, shrinkage, selection, calibration use the past only | already covered by a stricter design | tuning and selection stop at 2024-06-30, before every test fold (2024-10 on); `test_selection_and_tuning_cutoffs_precede_the_holdout` reads every cutoff default; stack and calibrators come from inner expanding folds (`models/walkforward.py::fit_stack`); online calibration uses earlier out-of-sample predictions only |
| 4 | Odds validity: timestamped, before start, named book; decision-time prices primary | already covered; two checks added | `log_integrity`: 0 rows at or after puck drop, 0 odds captured after the run, plus new `bets_without_book` and `odds_captured_after_puck_drop` (both 0). Backtests have no historical prices, so decision-time versus closing prices can only be compared on the live record (Phase 3) |
| 5 | Duplicates, postponements, mapping, time zones | partly covered; **fixed** | duplicates and result consistency: `hygiene.validate_inputs`, `test_input_validation_catches...`. Postponed or cancelled games were not excluded from the slate and a bet on a game moved to another date was graded against the later result: both fixed (void, stake refunded). Team-name mapping (8 spot checks plus a 32-club check) and Eastern-day boundaries including both DST changes now have tests |
| 6 | Settlement rules per book | **fixed / added** | `config/policy.yaml` `settlement`, `nhlbet/settle.py`, `docs/SETTLEMENT.md`, `tests/test_settlement.py`. The repo excluded the shootout goal from totals; house rules found via search say a shootout adds one goal. Default is now `so_adds_goal`; every book is flagged `verified: false` because the rules pages could not be read |
| 7a | Shuffled-outcome null | added | `test_shuffling_outcomes...`, `..._betting_loses_about_the_vig...`; real data, three seeds: shuffled AUC 0.48 to 0.50 (real world 0.60), ROI against a fair market with a 4.5% margin -3.4% to -4.8% (expected -4.3%). CLV is outcome-independent, so a shuffle cannot test it; the random-pick CLV baseline in Phase 3 does |
| 7b | Planted leak sentinel | added | `test_a_planted_outcome_feature_is_caught_before_any_model_is_fit`; real data: copy, complement and noisy copy all caught |
| 8 | Holdout lock | added | `reports/holdout_lock.json`, `nhlbet/validity/holdout.py`, policy `holdout`: 656 games from 2026-01-05 to 2026-09-29 (larger of 20% and a half-season; slightly more than the 526 games proposed in Phase 0, as the brief specifies); evaluated once via `evaluate_once`; allowlist test for who may read it |

## Caveats stated plainly
- The holdout block was examined in reports written before the lock (model comparison, ablations). The lock constrains new decisions; it does not make the block pristine. The live record since 2026-09-30 is the only unseen data.
- The settlement default is a judgement from search snippets, not from any book's rules page. The totals market stays RESEARCH ONLY until a person verifies each book (docs/SETTLEMENT.md).
- The weighted-average blend differs between truncated and full runs at about 4e-7 on synthetic data (solver tie-order noise, not data); on real data the difference was 1.1e-13.
- The real-data run uses Elo, logistic and the stack inputs available in a fast zoo (home rate, Elo, logistic); tree models share the same feature frame and guard, but were not part of the truncation run.
