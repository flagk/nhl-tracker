# Pre-registration of the confirmatory test

Written and committed BEFORE any of the data it will be run on exists. The commit that adds this file, and `audit/prereg_lock.json` (its hash, the frozen code and policy hashes), is the pre-registration time. Nothing in this file may be edited after that commit; a change is a new pre-registration with a new period.

## What is being tested

Whether the production moneyline model carries information about game outcomes that the betting market does not already contain, and whether picks made on that information beat the closing line.

## The sealed sample

There is no sealed historical season (see `audit/PHASE1_RECON.md`, section 0): the 2026-01-05 to 2026-09-29 block and the live record through 2026-10-26 have all been examined. The sealed sample is therefore **every NHL regular-season game that starts after the pre-registration commit** and for which the repository holds a decision-time prediction committed before puck drop (`audit/scripts/p6_forward_verify.py` classifies each as verified) and a decision-time market price. Playoff games, postponed games and games without a verified pre-puck-drop commit are excluded.

## Frozen procedure

The code is frozen, not the weights: the production model retrains every morning by design. Frozen: the feature builder, model zoo, calibration, de-vig method and decision rules (`nhlbet/features`, `models`, `odds`, `risk`, `train.py`) and the policy sections that change what is bet (`staking`, `evaluation`, `settlement`, `risk`), as recorded by their hashes in `audit/prereg_lock.json`. A change to any of them during the period is a protocol deviation, is recorded in the result file, and ends the confirmatory period unless the change is a bug fix that is documented before the evaluation. The evaluation script reports, but does not decide, whether the frozen paths changed.

## Hypotheses and decision rules (fixed now)

Two co-primary tests, Bonferroni-corrected (each at one-sided alpha 0.025):

1. **P1, information beyond the market.** On the same games, the log loss of the model's logged pre-game probability against the log loss of the Shin-devigged market probability at the decision time. Gain = market loss minus model loss, positive means the model is better. Passes if the one-sided Diebold-Mariano test (Bartlett HAC variance, lag floor(n^(1/3))) gives p < 0.025 **and** the mean gain is at least **0.007** log-loss units. The threshold is the smallest effect detectable with 80% power at the planned sample size, given the per-game standard deviation of 0.101 measured on the live record (audit/PHASE4_EVALUATION.md); a smaller effect cannot be confirmed by this design and is not claimed.
2. **P2, closing-line value.** Moneyline picks of the model group (the paper strategies tagged "model", which exist because the official policy places almost no bets), restricted to picks whose closing capture is at most 15 minutes before puck drop. Passes if there are at least 300 such picks, the mean margin-free CLV is at least **+0.5 probability points** (the value of `betting_gate.min_mean_clv_pts` in the policy), and the lower end of the 95% date-clustered bootstrap interval of the mean is above zero.

Reported but not tested: ROI (it is noise on this sample size), the other markets (no closing prices), calibration slope.

## Verdict categories (fixed now)

* **EDGE SUPPORTED:** P1 and P2 both pass.
* **EDGE UNSUPPORTED:** P1's 95% interval for the gain has an upper end below 0.007 (a gain worth confirming is excluded), or the mean gain is negative with p > 0.975 against the model.
* **INCONCLUSIVE:** anything else, including P1 passing while P2 has fewer than 300 qualifying picks.

## When the evaluation happens, and once only

The first of: (a) 1,312 qualifying forward games (one regular season), or (b) 2027-06-30. It runs once with `scripts/prereg_eval.py`, which refuses to run before the trigger, refuses a second run, and checks that this file still matches its recorded hash. There are no interim significance tests, no early stopping for success, and no peeking: interim monitoring exists only to halt betting for safety (the kill switch and the decay thresholds in `config/policy.yaml` and `audit/PHASE7_OPERATIONS.md`) and its numbers are not used to declare anything.

## What would change the plan

Nothing about the thresholds above. The sample size, closing-capture quality and the number of picks are properties of the data; if they fall short by the trigger date the verdict is INCONCLUSIVE, and that is a legitimate result.
