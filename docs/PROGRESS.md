# Progress (hardening branch)

Resume rule: a new session reads this file and continues from "Next". Do not redo finished steps.

## Status
- Branch: `hardening` (from `main`, with the review-agent branch merged in; 327 tests green at the merge).
- Approved decisions (owner): (1) real-data checks run in GitHub Actions on `hardening`, (2) holdout rule below, (3) ntfy.sh push + Healthchecks.io dead-man switch, (4) policy immutability via hash + owner-email check + manual branch protection, (5) proposed policy values.
- Phase 0: done (inventory, corrections, plan approved).
- Phase 1: DONE (checkpoint). Real-data run passed: truncation invariance 1e-13 over 959 games, sentinels caught, shuffled null AUC 0.48-0.50, shuffled ROI -3.4% to -4.8%. Report: `reports/phase1_leakage.md`.
- Credits spent on the odds API: 0 (no live calls made).

## Phase 1 findings so far
1. The "headline" (1,805 bets, 55.84%, +11.69%) is the legacy README claim. The committed per-bet file recomputes exactly (1,008 wins). It is an even-money artefact (AUDIT.md). Re-running the legacy code was started in a scratch copy and NOT finished; it is not claimed as reproduced.
2. REAL FINDING, settlement: the repo excluded the shootout goal from game totals; the industry convention (house rules found via search) adds one goal. Both are now rule sets in `config/policy.yaml` (default `so_adds_goal`, unverified for every book). Model target and PMF follow the same switch.
3. REAL FINDING, void games: postponed/cancelled games were not excluded from the slate, and a bet on a game moved to another date was graded against the later result. Both fixed (void, stake refunded).
4. REAL FINDING, goalie confirmations: reloading the manual file re-stamped `captured_at` to "now" and the slate did not filter confirmations by decision time. Fixed: first-seen time is kept and `as_of` filtering added.
5. Availability guard added to every model fit; planted-outcome sentinels are caught; truncation invariance holds for features, all models, the stack and calibrators (weighted average differs ~4e-7 from solver tie-order noise only).
6. Holdout locked: 656 games, 2026-01-05 to 2026-09-29 (rule: larger of 20% and a half-season), previously examined.
7. Settlement sensitivity on the real database: 465 totals paper bets, 15 results flip, profit -$348.90 (old rule) vs -$518.00 (new default).
8. `reports/baseline/before.json` holds the pre-change baseline (code at the merge commit); live record: moneyline 144 picks, ROI -7.2% (CI -15.8% to +1.8%).

## Next
- Phase 2 (count models and the joint distribution). Note: the totals goal model now follows the shootout rule switch, so its out-of-sample metrics must be re-run in CI (goals workflow) and compared with the old rule.
- Real-data jobs: push a change to `ci/phase1_trigger.txt` style trigger files on `hardening` (workflow files only exist on the branch, so push-triggered).

## Open blockers
- None blocking. Manual items accumulating for OPERATIONS.md: verify each book's shootout/total rules; branch protection + CODEOWNERS for `config/policy.yaml` and `.github/**`.
