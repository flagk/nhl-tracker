# Phase 3: leakage attack

Everything here is reproducible: unit tests `tests/test_audit_leakage.py`, `tests/test_audit_placebo.py`; mutation harness `audit/scripts/p3_mutation.py`; real-data run `audit/scripts/p3_realdata.py` (GitHub Actions, `audit/jobs.txt`); results in `audit/results/`. Research window only (nothing after 2026-01-04 is read).

## 1. Verdict

- **No leakage found in the production pipeline.** Truncation invariance, availability guard and the outcome-shuffle null all pass on the real database (10,694 completed games); all 14 production feature columns pass the stronger guard.
- **The test suite is not vacuous.** Mutation testing: 13 of 13 deliberately planted leaks are caught by at least one test; survivors after closing the one gap found (M6, see section 3): none; unmutated control: SURVIVED.
- **One weakness in the defences was found and fixed:** the availability guard only flagged inputs correlated above 0.98 with the label, so a feature that ranks winners perfectly (AUC 1.0, correlation about 0.8) passed. A univariate-AUC limit of 0.75 was added (`nhlbet/validity/availability.py`); it was confirmed to leave all 14 production features alone on the real data.
- **Two things look like leaks to a naive reader and are not:** the adversarial-validation AUC (section 5) and the flat stale-value curve (section 4). Both are explained there.

## 2. Planted canaries and who catches them

Canary features built from forbidden information, run through six independent detectors on the real data (2,059 games, 2024-07-01 to 2026-01-04). The clean builder is the control row: it must raise no alarm.

| Canary | guard | truncation | corruption | same-day | shuffle null | too good | caught by any |
|---|---|---|---|---|---|---|---|
| clean pipeline (control) | - | - | - | - | - | - | no alarm (correct) |
| own_result | caught | - | - | - | caught | caught | yes |
| noisy_own_result | caught | - | - | - | caught | caught | yes |
| next_game | - | caught | caught | - | - | - | yes |
| same_day_other | - | - | - | caught | - | - | yes |

Reading the matrix:
- `own_result` / `noisy_own_result` (peeking at the game's own margin, with or without noise) survive truncation and corruption of the future, because they use no future games. Only the guard, the shuffled world and the univariate-AUC check see them.
- `next_game` (the next game's result) is invisible to the shuffled world and to any univariate check (it is unrelated to this game's label) but is the one the truncation and corruption tests exist for.
- `same_day_other` (another game on the same date) is invisible to everything except the same-date perturbation test. It needs no future games and carries no signal about its own label.
- **No single detector is sufficient** (each misses at least one canary): the defences are a set, and removing any member opens a specific hole. This is asserted in `test_no_single_detector_is_sufficient`.

## 3. Mutation testing of the real code

Each mutation re-introduces one leak into a copy of the package and runs the leakage test subset. KILLED = at least one test fails.

| ID | Planted leak | Result | Failing tests (first) |
|---|---|---|---|
| M0 | control: nothing changed | SURVIVED (must be SURVIVED) | |
| M1 | results of a date folded into the state before that date's features are emitted | **KILLED** | test_future_corruption_never_changes_earlier_or_same_day_features[2023-11-15]; test_future_corruption_never_changes_earlier_or_same_day_features[2024-02-20] (9 failing) |
| M2 | a game's own result included in its rolling window | **KILLED** | test_future_corruption_never_changes_earlier_or_same_day_features[2023-11-15]; test_future_corruption_never_changes_earlier_or_same_day_features[2024-02-20] (6 failing) |
| M3 | default goalie mode switched to 'actual' (post-game starter identity) | **KILLED** | test_default_mode_ignores_tonights_starter_identity; test_default_configuration_is_morning_safe (2 failing) |
| M4 | market features use snapshots after the decision time (closing line leaks in) | **KILLED** | test_market_features_ignore_snapshots_after_the_decision_time (1 failing) |
| M5 | walk-forward training window includes the test block | **KILLED** | test_predictions_for_games_up_to_D_do_not_change_when_later_data_is_added; test_labels_inside_a_walk_forward_test_block_never_change_that_blocks_predictions (2 failing) |
| M6 | online calibrator fitted on the block it is applied to | **KILLED** | test_online_calibrator_never_sees_the_labels_of_the_block_it_calibrates (1 failing) |
| M7 | availability guard disabled | **KILLED** | test_a_planted_outcome_feature_is_caught_before_any_model_is_fit; test_the_guard_runs_inside_every_model_fit (2 failing) |
| M8 | outcome shuffle is a no-op (the null world is not null) | **KILLED** | test_shuffling_outcomes_within_season_removes_skill_and_the_pipeline_notices (1 failing) |
| M9 | goalie confirmations ignore the decision time | **KILLED** | test_a_goalie_confirmation_captured_after_the_decision_time_is_ignored (1 failing) |
| M10 | point-in-time resolver ignores knowledge time | **KILLED** | test_asof_resolver_hides_later_versions_and_unknown_games; test_a_late_correction_changes_only_decisions_after_it_was_fetched (2 failing) |
| M11 | snapshot as_of returns versions fetched after the decision time | **KILLED** | test_snapshots_are_immutable_versioned_and_verifiable (1 failing) |
| M12 | walk-forward row order left to the input order (unstable ties) | **KILLED** | test_predictions_for_games_up_to_D_do_not_change_when_later_data_is_added (1 failing) |
| M13 | league goalie-start counters updated with the current game before it is emitted | **KILLED** | test_default_mode_ignores_tonights_starter_identity; test_changing_the_games_own_starter_does_not_move_features_in_default_mode_but_does_in_actual_mode (2 failing) |

**Test gap found and closed (M6):** on the first run this mutant SURVIVED: no test failed. Added tests/test_audit_leakage.py::test_online_calibrator_never_sees_the_labels_of_the_block_it_calibrates It was killed on the rerun.

The harness drops the editable-install import hook so the tests really import the mutated copy, and includes an unmutated control that must pass: an earlier run without those two safeguards reported mutants as killed for an unrelated reason (missing config files in the temporary tree) and was discarded.

## 4. Placebo: time-shifted features

AUC of home-minus-away of a pre-game feature re-timed by k games (k < 0 reads FUTURE values; k > 0 stale ones), 95% bootstrap interval.

| Feature | k = -2 (future) | k = -1 (future) | **k = 0 (as built)** | k = 1 | k = 5 | k = 10 |
|---|---|---|---|---|---|---|
| gd_l10 | 0.744 | 0.745 | **0.572** | 0.574 | 0.572 | 0.570 |
| gd_ewm | 0.751 | 0.763 | **0.594** | 0.596 | 0.596 | 0.598 |
| elo | 0.676 | 0.676 | **0.618** | 0.619 | 0.618 | 0.619 |

Positive control: shifting into the future lifts the AUC by 0.07 to 0.17, so this evaluation would show a leak if one existed, and the as-built column sits on the honest side. **Finding that is not a leak but matters for the modelling story:** stale values (up to 10 games old) rank winners as well as current ones. The signal in these features is slow-moving team quality, not recent form.

## 5. Adversarial validation (can a classifier tell train rows from test rows?)

| Split | rows before / after | AUC | Reading | Top features |
|---|---|---|---|---|
| 2024-10-04 | 7,157 / 1,973 | 0.614 | moderate shift | d_ga_season_shrunk (0.0787), h_early_w (0.0729), d_elo (0.0687) |
| 2025-07-01 | 8,469 / 661 | 0.768 | strong shift: the test period is distinguishable from training | h_early_w (0.2712), d_ga_season_shrunk (0.0587), d_xg_share (0.0398) |

An AUC of 0.5 would mean the test period looks like the past. These are higher, and the dominant feature is `h_early_w` (games played this season / (games played + 20)): the later test slice covers the first months of a season, so the rows differ by **season phase**, not because anything is wrong. It is covariate shift by construction, not leakage, but it means walk-forward blocks do not share the same mixture of season phases as the training set, which matters for calibration (Phase 4 evaluates by season phase).

## 6. Placebo: shuffled odds

Live record after the locked block, 174 games with a decision-time market price, 97 bets chosen as the policy chooses, flat stake, fair odds less a 4.5% margin. Real-market ROI **-11.0%**; the same selection against market probabilities permuted across games earns **+28.0%** (95% range +11% to +44%).

Reading: prices unrelated to the game make almost any model look brilliant, which is the ceiling an evaluation flaw (a price that does not belong to its game) could manufacture. Against the right prices the model loses about the margin and slightly more: the share of the model's information already in the price is 1.20 (above 1 means the model's disagreements with the market were worse than random). On this sample the model's deviations from the market add no value, which agrees with the closing-line-value result from the hardening branch. Small sample (97 bets): treat as consistent, not decisive.

## 7. Placebo: shuffled labels

Already in place and re-run on the real data in this phase (`audit/results/p1check`): results are reassigned within each season and the whole pipeline is rerun; a clean pipeline finds no skill. Real world AUC elo 0.584; shuffled worlds: 0.497, 0.496, 0.497. Planted outcome copies were caught by the guard.

## 8. What this phase does NOT show

- It shows the feature and model pipeline is free of the leaks it was tested for. It says nothing about whether the model has an edge over the market (Phase 4) and nothing about the sealed-holdout question (there is none; see PHASE1_RECON section 0).
- The point-in-time data revision problem (Phase 1 item 10) cannot be tested on history; only the forward snapshots from Phase 2 can measure it.
- Walk-forward still has no purge or embargo (training uses every game dated before the block). Labels here resolve the same day, so the exposure is small, but Phase 4 adds both.
- Goalie `actual` mode and lineup mode remain reachable by configuration; the tests pin the defaults and prove the modes would leak, they do not forbid someone enabling them.
