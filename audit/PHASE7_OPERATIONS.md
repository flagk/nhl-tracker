# Phase 7: operations

The daily run is a chain of gates. Any failing gate withholds every pick, alerts the owner once, and (for the gates check) withholds the success ping. Nothing here adjusts a threshold; thresholds live in `config/policy.yaml` and change only by the owner. The state machine (OK, HALT_SOFT, HALT_HARD, auto-resume after 3 healthy runs for soft halts, owner acknowledgement for hard ones) and the alerting were built on the hardening branch (`OPERATIONS.md`); this phase added the items below.

## Gates run on every daily run

| Gate | Level | What it checks | Tested | Checked against real data |
|---|---|---|---|---|
| `pandera` (new) | soft | schemas for games, goalie boxscores, team stats and the last 30 days of odds: ranges, uniqueness, score consistency, price range | `tests/test_ops_schemas.py` | **yes**: first probe flagged my own bound on goalie minutes (multi-overtime games run past 4,000 s); fixed, second probe clean on 11,859 games, 22,756 goalie rows, 21,388 team rows, 73,946 odds rows |
| `calibration` | soft | rolling calibration slope of the last 200 resolved predictions outside the halt range | `tests/test_ops.py` | rolling slope is reported every run in `state/rolling_metrics.json` |
| `model_market_disagreement` | soft | mean absolute gap between model and market probability above the halt limit | `tests/test_ops.py` | (policy `autonomy.drift`) |
| `market_decay` (new) | soft | rolling 150-game log-loss gain over the market has fallen more than 0.03 below the audit baseline of -0.012 | `tests/test_ops.py::test_market_decay_is_a_change_detector_not_a_level_test` | baseline is the measured value; not yet seen to trip |
| `live_clv` | hard | at least 100 official picks with mean CLV below -1.0 points and the 95% interval below zero | `tests/test_ops.py` | the official policy has placed no bets, so it has nothing to test yet |
| schema, freshness, row counts, input validation, cross-source, odds source, quota | soft | database and feed health | hardening branch tests | running daily |
| `drawdown`, `log_integrity`, `policy_unchanged` | hard | 20% drawdown, hash-chained logs, policy file author | hardening branch tests | log chain exists only after the hardening merge |
| Leakage and calibration tests | blocking | the Publish gate step runs `test_no_leakage`, `test_audit_leakage`, `test_audit_reimpl`, `test_ops_schemas` and the earlier gate tests before the job may publish | CI | every run |

Design note on the decay detector: a test of the *level* of the model's gain over the market would halt the system permanently (the model is already worse than the market, by 0.012 log loss on the live record). Betting is separately held at RESEARCH ONLY by the betting gate. The decay gate therefore detects *worsening relative to the audit baseline*, which is what "decay" means.

## Decay thresholds, set before they can trip

| Tracker (in `state/rolling_metrics.json`) | Window | Action | Policy key |
|---|---|---|---|
| closing-line value of the picks | last 100 | hard halt when the mean is below -1.0 points and the 95% interval is entirely below zero (from 100 picks) | `autonomy.kill_switch.clv` |
| calibration slope | last 200 | warn outside 0.6 to 1.4, soft halt outside 0.3 to 2.0 | `autonomy.drift` |
| model vs market log-loss gain | last 150 | soft halt when below baseline minus 0.03 (about 3.5 standard errors of a 150-game mean) | `autonomy.decay` |
| feature drift (population stability index) | last 60 days against history | handled by the existing monitor at retrain (`nhlbet/monitor.py`: status ALERT withholds all picks); the stricter `ops.drift.feature_drift_check` with policy limits 0.2 / 0.5 exists and is tested but is **not wired into the daily gate** | `autonomy.drift` |
| drawdown | since the start of the official record | hard halt at 20% | `risk.kill_switch` |

Soft halts resume after 3 consecutive healthy runs; hard halts only when the owner lists the halt id in `autonomy.acknowledged_halts`.

## Dead-man's switches

Two Healthchecks checks, so that one cannot mask the other:

| Check | Secret | Pinged by | Meaning of silence | Suggested settings |
|---|---|---|---|---|
| heartbeat | `HC_PING_URL` | `scripts/heartbeat.py`, every 10 minutes | the scheduler chain is dead | period 60 min, grace 6 h |
| gates (new) | `HC_GATES_PING_URL` | `run_daily`: success ping **only if every gate passed**, `/fail` if any gate failed | the daily run did not finish, or a gate is failing | period 2 days, grace 12 h; **pause it in the off-season** (no games, no run) |

Tested with an injected sender (`tests/test_ops_pipeline.py`): a halted run pings `/fail` on the gates URL and never the success URL; a fully healthy run pings success. **Not tested against the live Healthchecks or ntfy servers** (the sandbox cannot reach them); the first real run is the test.

## Per-run caps

Every workflow has a concurrency group and every job a timeout (`tests/test_ops_workflows.py` enforces it for all current and future workflows). The review agent has `--max-turns 45`, a 20-minute step timeout inside a 40-minute job, and shares the `nhl-data` group with the daily job so nothing overlaps. A crash, timeout or turn-limit stop publishes picks as UNREVIEWED and alerts.

## Forward ledger

Every prediction is verified as committed before puck drop (`audit/scripts/p6_forward_verify.py`; 464 of 464 so far). Rows later than puck drop would show as `LATE_commit_after_puck_drop` and make the script exit non-zero.

## Blocking bets

Any open halt sets the model status to ALERT, which the slate builder turns into "no usable picks"; the betting gate independently keeps the page at RESEARCH ONLY unless 300 live CLV picks, a CLV interval above zero, a mean of at least +0.5 points and a drawdown under 10% hold. No LLM output reaches a probability or a stake: the review agent can only keep, flag or veto.

## Owner actions

1. Create the second Healthchecks check and add the secret `HC_GATES_PING_URL` (plus the earlier `NTFY_TOPIC`, `HC_PING_URL`, and the variables `POLICY_OWNER_LOGINS` and `POLICY_OWNER_EMAILS`).
2. `pandera>=0.24` is now in `requirements.txt` (a new dependency, requested by the audit brief).
3. Merge the hardening and audit pull requests, in that order; until then the new gates do not run in production.
