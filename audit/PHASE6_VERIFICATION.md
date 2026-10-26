# Phase 6: independent verification

Three parts. (a) A blind reimplementation of the decision layer, diffed against production. (b) A forward ledger and its verifier. (c) A pre-registered final test, locked now and runnable once, later.

## (a) Blind reimplementation

**What was done.** `audit/SPEC.md` describes, in prose and formulas with no code names, the parts of the pipeline that can be specified exactly and checked against recorded data: de-vigging (proportional, power, Shin), the Elo model, the bet/no-bet decision and stake for one game (screens, shrinkage toward the market, Kelly, caps) and the portfolio caps. A separate subagent was given only that file and asked to implement it (`audit/reimpl/reimpl.py`) and to record every ambiguity it resolved (`audit/reimpl/NOTES.md`, 12 entries). `audit/scripts/p6_diff.py` compares it with production.

**Result** (`audit/results/p6_diff.json`):

| Comparison | Cases | Differences |
|---|---|---|
| De-vig, 3 methods, 2-way and 3-way markets with random margins | 9,000 | largest difference 1.2e-13 (power method, bisection tolerance) |
| Elo pre-game probability, real score history in the repository database | 2,987 games | none (largest difference 0.0) |
| Bet / no-bet, side, odds and stake over random slates with edge cases | 8,007 games in 1,500 slates, 865 of them bets | **0** |

The comparison can fail: changing one constant in a copy of the reimplementation (minimum edge 0.03 to 0.031) produces 17 disagreements in the same 8,007 games, and `tests/test_audit_reimpl.py` keeps both the agreement and this sensitivity in the suite.

**What this does and does not prove.**
- It proves the spec is sufficient to reproduce the decision layer: the reimplementer made 12 interpretation choices (tie-breaks, rounding, where caps apply) and none changed an outcome.
- It does **not** prove the decision layer is wise, only that it does what the spec says. The spec was written by the same author who read the code, so the independence is of implementation, not of design: a flaw in the idea would be reproduced faithfully.
- **The model itself was not reimplemented.** Features, the model zoo, the stack and the calibrators depend on play-by-play data that exists only in the Actions cache and are far too large to specify blind. The leakage tests (Phase 3) and the model comparison (Phase 4) are the checks that stand in for that, and they are not an independent reproduction. A second reimplementation of the feature builder against raw snapshots is the natural next step once the Phase 2 snapshots exist.
- The subagent's own report came back empty (the single word "placeholder"); its work product was read from the files it wrote, which is why the notes above are quoted from `NOTES.md` rather than from its message. The subagent was instructed not to read the repository; the instruction was not technically enforced, but its code contains nothing beyond what the spec says and it made only 4 tool calls.

## (b) Forward ledger

**Definition.** The ledger is what the daily job already commits: `data/logs/recommendations` (model probability, no-vig market probability, best price, decision, stake, odds capture time, one row per game per run), `data/logs/odds_consensus`, `data/logs/goalie_events`, hash-chained in `data/logs/chain.jsonl` (added on the hardening branch) and mirrored without force to `log-backup`. What was missing was a check that each row reached git before puck drop. `audit/scripts/p6_forward_verify.py` provides it.

**Result on the existing record** (default branch, `audit/results/forward_verification.json`): all **464 of 464** logged predictions were in a commit dated before their game's start; median 473 minutes before puck drop; the commit followed the run by a median of 1.2 minutes. The hash-chain check on that checkout reports no problems, but the ledger file does not exist on the default branch yet (it is part of the unmerged hardening branch), so that part is vacuous until the merge.

**Trust assumption.** The commit date is written by the machine that commits (the Actions runner). This proves "committed before the game" against anyone without push rights, not against the repository owner. An external timestamp (for example OpenTimestamps on each day's ledger hash) would close that gap; it is not built.

## (c) The pre-registered final test

`audit/preregistration.md` fixes the hypotheses, thresholds, verdict rules, trigger and stopping rule. `audit/prereg_lock.json` records its hash and the hashes of the frozen code and policy sections; `audit/scripts/p6_prereg_eval.py` checks the hash, refuses to run before the trigger (1,312 qualifying forward games, or 2027-06-30) and refuses a second run. `--count-only` prints the number of qualifying games and nothing else.

**It has not been run and cannot be run yet:** it needs games that have not been played. There is no sealed historical season (Phase 1, section 0), so the sealed sample is the future, starting at the pre-registration commit. Two consequences the owner must act on:
1. The frozen code is the code on the `audit` branch. Production runs from the default branch, so the forward period only measures the frozen code once that branch is merged; until then the evaluation script will report the frozen paths as changed. **Merge the hardening and audit branches before relying on the forward sample.**
2. The design's power is modest: 1,312 games detect a log-loss advantage of about 0.007 with 80% power. A smaller true edge would be reported as inconclusive, not as absent.

The exact thresholds: P1 passes if the Diebold-Mariano p < 0.025 and the mean log-loss gain over the Shin market is at least 0.007; P2 passes if at least 300 picks with a closing capture within 15 minutes have mean CLV of at least +0.5 points with a 95% interval above zero. Verdict rules are in the file and unit-tested (`tests/test_audit_prereg.py`).
