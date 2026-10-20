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

- Phase 2: DONE (checkpoint). Quarter-line fixes (puck line, props), PMF/line/direct-summation tests, totals PMF scoring (RPS, PIT), grid-edge audit. Real data: totals show NO skill over the base rate; puck line weak; saves RESEARCH ONLY. Report `reports/phase2_counts.md`.
- Phase 3: evaluation module and real-data run DONE (`reports/evaluation.md`, `evaluation.json`). Moneyline model picks: 201 bets, ROI -9.4%, CLV +0.04 points (interval -0.04 to +0.12, adjusted p=0.15): no demonstrable edge. Closing price exists for moneylines only (the closing capture is h2h-only). Remaining for the checkpoint: summary in VALIDATION_REPORT at the end.
- CI: the test suite was flaky because the synthetic league used process-random `hash(team)` ids; fixed with crc32 (commit `675966a`); CI now echoes failing test names as annotations.

- Phase 4: DONE. Protocol committed first (`reports/model_comparison_protocol.md`); nothing qualified to replace `stack__online`; single holdout evaluation recorded (`reports/holdout_eval.json`): stack log loss 0.6826, Elo 0.6785, home rate 0.6916; gain over home rate +0.009 (CI -0.004 to +0.021). Live market-blend check: every model weight above 0 scored worse than the market alone (122 games). Disclosure: a local dry run printed the holdout numbers before the recorded run (selection unaffected). Report `reports/model_comparison_phase4.md`.
- Phase 5: DONE. Policy-driven staking, Kelly ceiling, push-aware stake/EV, correlated-exposure caps (`risk/caps.py`), 10,000-path simulation on the real moneyline template (`reports/risk.md`): with no skill the median bankroll ends at 98% of its start after 500 bets, P(20% drawdown kill switch) 0.7%, no ruin.
- Phase 6: DONE except the failure alert (needs the Phase 7 notifier). Action pinned to SHA (v1.0.242), settings checked against the action source at that SHA, shadow mode default, odds credits shared with the daily tally, CLV-first agent evaluation with permutation test and flag rate.
- CI: pytest step fails with no FAILED lines in the latest runs: the workflow now also echoes the last lines of the pytest output as annotations; diagnosing.

## Next
- Phase 4 (model comparison; only if Phases 1 to 3 are clean: they are, but nothing shows an edge, so Phase 4 is a check that no simpler or market-blended model is better, then ONE holdout evaluation).
- Real-data jobs: push a change to `ci/phase1_trigger.txt` style trigger files on `hardening` (workflow files only exist on the branch, so push-triggered).

## Open blockers
- None blocking. Manual items accumulating for OPERATIONS.md: verify each book's shootout/total rules; branch protection + CODEOWNERS for `config/policy.yaml` and `.github/**`.
