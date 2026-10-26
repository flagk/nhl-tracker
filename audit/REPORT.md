# Audit report: is there a real, durable betting edge?

Branch `audit` (built on `hardening`). Research and education only; nothing here is betting advice or a promise of profit.

## 1. Verdict

**EDGE UNSUPPORTED.** Nothing in the evidence shows that this system beats the betting market, several independent checks point the other way, and the sample is too small to prove that no small edge exists. In the categories the brief asked for (supported, unsupported, inconclusive): not supported, not demonstrated absent.

Applying the decision rules that are now pre-registered (section 5) to the exploratory live sample, the result would be *unsupported* on the first co-primary test and *not met* on the second. That is an illustration, not the confirmatory test: the confirmatory sample does not exist yet.

## 2. Evidence

| Question | Finding | Where |
|---|---|---|
| Does the pipeline leak the future? | **No leak found.** Truncation invariance, outcome-shuffle null and the availability guard pass on 10,694 real games. 13 of 13 deliberately planted leaks in the code are caught by at least one test (after one test gap, a calibrator fitted on its own block, was closed); 4 planted leaky features are each caught by at least one of six detectors, and no single detector catches all four. The correlation-only guard missed a feature that ranks winners perfectly: a univariate-AUC limit was added and leaves all 14 production features untouched | [Phase 3](PHASE3_LEAKAGE.md) |
| Are the timestamps honest? | Decisions are made a median of 11 h (morning) or 4 h (late) before puck drop. **16% of games have a last odds capture more than 60 minutes before the start**, so earlier "closing" figures mixed true closes with stale ones. All 464 logged predictions were committed to git before their games (median 473 minutes before puck drop) | [Phase 1](PHASE1_RECON.md), [Phase 6](PHASE6_VERIFICATION.md) |
| Which model is best, with purged and embargoed walk-forward? | The simplest competent one. On 1,973 research-window games: logistic regression 0.6714 log loss, production stack 0.6722, random forest 0.6764, Elo 0.6782, hierarchical Bayes 0.6807, home rate 0.6891. The ensemble is not better than its logistic component; the Bayesian challenger does not beat Elo. Embargo (0, 1, 7 days) and recalibration (Platt, isotonic, beta) change nothing material | [Phase 4](PHASE4_EVALUATION.md) |
| Does any model beat the devigged market? | **No.** On 161 live games every model, including the production model, is worse than the market at the decision time and at the close, under all three devig methods (production model: gain -0.012 log loss, 95% interval -0.023 to +0.001; every Holm-adjusted Diebold-Mariano p = 1.0) | Phase 4 |
| Closing-line value per bet | +0.04 points (all captures), **+0.06 points with a close within 15 minutes** (108 picks, interval -0.05 to +0.16, adjusted p = 0.11); ROI -9% on 260 paper picks. Not distinguishable from zero | Phase 4 |
| Is the apparent best strategy a product of searching? | Yes, in so far as there is a best one: it is a control that always bets the home team. White's reality check p = 0.80, Hansen SPA p = 0.46, deflated Sharpe 0.001 (29 strategies) and 3e-7 (324-trial register), probability of backtest overfitting 0.57. These rest on 25 days of paper trading and are weak, but they all point the same way | [Phase 5](PHASE5_OVERFITTING_RISK.md) |
| What could happen under the policy? | With stakes on claims shrunk toward the market (the policy), drawdowns stay shallow (median 4 to 8% over 500 bets) because the policy bets very little; unshrunk, aggressive staking under no skill hits a 20% drawdown in about 84% of 500-bet paths. The live record cannot say how much of the model's disagreement with the market is real: theta = -0.6, standard error 0.6 | Phase 5 |
| Does the decision code do what it claims? | A blind reimplementation from a prose spec agrees on 8,007 of 8,007 decisions, 9,000 de-vig cases and 2,987 Elo probabilities; a 0.001 change in one threshold shows up as 17 disagreements. This checks implementation, not design, and not the feature or model pipeline | Phase 6 |

## 3. What the audit changed in the code

- Availability guard: univariate-AUC limit (`nhlbet/validity/availability.py`).
- Walk-forward: optional embargo (`nhlbet/splits.py`, default unchanged); beta calibration; a hierarchical Bayesian model (`nhlbet/models/hier.py`, challenger only, not in production).
- Point-in-time: snapshot store with fetch time, as-of mode in the feature builder (identical output when nothing was revised; a late correction affects only later decisions), goalie confirmation event log wired into the daily run, odds archiver (off).
- Operations: pandera validation, rolling CLV / calibration / market-gain trackers, a market-decay gate, a gates dead-man's switch pinged only when every gate passed, concurrency groups and timeouts on every workflow.
- New test files: 10; the whole suite is 540 tests and passes.

## 4. Pre-registered thresholds (frozen before the data exists)

Full text: [preregistration.md](preregistration.md); hashes in `prereg_lock.json`; evaluation script `audit/scripts/p6_prereg_eval.py` (once only, after the trigger).

| Test | Passes if |
|---|---|
| P1: information beyond the market | one-sided Diebold-Mariano p < 0.025 and mean log-loss gain over the Shin market at least 0.007, on the same games |
| P2: closing-line value | at least 300 picks with a close within 15 minutes, mean CLV at least +0.5 points, 95% date-clustered interval above zero |
| Verdict | both pass: EDGE SUPPORTED. P1 interval's upper end below 0.007 (or clearly negative): EDGE UNSUPPORTED. Otherwise INCONCLUSIVE |
| Trigger | 1,312 qualifying forward games or 2027-06-30, whichever is first; one evaluation only; no interim tests |

## 5. What would change the verdict

- **Towards supported:** the confirmatory test passes both P1 and P2 on the forward sample. A model with a true log-loss edge of 0.007 over the market would need about 1,300 games (one regular season) to be confirmed; an edge of 0.005 would need about 2,500.
- **Towards firmly unsupported:** the same forward sample with a P1 interval whose upper end stays below 0.007.
- **Evidence that does not exist yet and would matter:** closing prices for totals, puck lines and props (none are captured, so those markets cannot be judged by CLV at all); real fills at a named book (the official policy has placed no bets, so nothing has been tried against a price you could actually take); per-book settlement rules (none verified).
- **A different kind of result:** an information source the market lacks, for example confirmed goalies earlier than the market reprices. The pipeline only has a manual confirmation file; no feed exists.

## 6. Limits of this audit, stated plainly

- The live sample is small (161 games with prices, 25 days of paper trading). It cannot detect an edge under about 0.01 log loss.
- There is no sealed historical season. The 2026-01-05 to 2026-09-29 block and the live record through 2026-10-26 were examined; the sealed test is the future.
- The model and feature pipeline were not independently reimplemented (the decision layer was).
- The point-in-time rebuild cannot be applied to history, only forward. Ingestion into the snapshot store and DVC have not been run; DVC could not be installed in the authoring sandbox. NHL endpoints (schedule, play-by-play, boxscore, shift charts) were verified live on 2026-10-26 through GitHub Actions.
- Healthchecks and ntfy were never exercised against the live services (the sandbox cannot reach them); alerting is tested with injected senders.
- The odds archive "every few minutes" is not switched on: at one credit per call it would cost 1,440 to 2,900 credits a month against a plan of 500.
- The forward sample only measures the frozen code once the audit branch is merged into production.

## 7. Decisions and actions for the owner

1. Merge the hardening pull request, then the audit branch, before relying on the forward sample.
2. Provide a private DVC remote (`DVC_REMOTE_URL` plus credentials) if the point-in-time archive is wanted; decide on a larger odds plan or a slower archive cadence.
3. Create the second Healthchecks check and the secret `HC_GATES_PING_URL`; set the other secrets and variables listed in `OPERATIONS.md`.
4. Decide whether to capture closing prices for totals, puck lines and props.
5. Review `audit/preregistration.md` before the first forward games count; after the audit branch is merged it cannot be edited without starting a new period.
