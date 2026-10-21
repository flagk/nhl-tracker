# Operations

Research and education only. The system never places bets and never touches a sportsbook account. Nothing here promises profit.

Prime directive: when anything looks wrong, the system stops publishing and alerts the owner. It never changes its own thresholds, rules or model settings to make a problem go away. A halted system that pages the owner is a success.

## Statuses

| Status | Meaning | Picks | Clears |
|---|---|---|---|
| `OK` | every check passed | published | n/a |
| `HALT_SOFT` | stale or missing data, source outage, quota exhausted, drift beyond halt limits, impossible values | withheld | by itself after `autonomy.soft_resume_healthy_runs` (3) consecutive fully healthy runs |
| `HALT_HARD` | drawdown at the limit, live CLV clearly negative, log integrity failure, policy file changed by a non-owner | withheld | never by itself: only when the owner adds the halt id to `autonomy.acknowledged_halts` in `config/policy.yaml` |

Betting status on the page is separate: `RESEARCH ONLY` unless every criterion in `betting_gate` (policy file) holds, then `ELIGIBLE` for moneylines only. It reverts the run any criterion fails.

State lives in `state/system_state.json` (committed by the daily job), never in the policy file.

## What triggers what

| Check (halt name) | Level | Trigger | Next step |
|---|---|---|---|
| `schema`, `row_counts`, `input_validation` | soft | database tables or columns missing, fewer completed games than `autonomy.health.min_completed_games`, inconsistent results | run the Backfill workflow |
| `freshness` | soft | NHL refresh failed and data older than `max_data_age_days` | wait; check api-web.nhle.com |
| `cross_source` | soft | odds feed events do not match the NHL schedule | check team mapping in `nhlbet/teams.py` |
| `odds_source`, `odds_quota` | soft | odds fetch failed or stale, credits below the reserve | wait for renewal on the 1st or upgrade the plan |
| `value_ranges` | soft | a probability or price outside its range in the built page | read the daily log |
| `feature_drift`, `calibration`, `model_market_disagreement` | soft | beyond the halt limits in `autonomy.drift` | inspect; retrain if the shift is real |
| `drawdown` | hard | drawdown from the high-water mark reaches `risk.kill_switch.max_drawdown_pct` | read `reports/evaluation.md`, decide, then acknowledge |
| `live_clv` | hard | at least 100 live picks, mean CLV below -1.0 points and the 95% interval entirely below zero | same |
| `log_integrity` | hard | ledger `data/logs/chain.jsonl` does not verify (edited, truncated or deleted log) | restore from the `log-backup` branch, find the cause, acknowledge |
| `policy_unchanged` | hard | the hash of `config/policy.yaml` changed and the newest commit touching it was not authored by an owner | review the diff; revert, or fix the owner variables and acknowledge |

To acknowledge a hard halt: copy its id (shown in the alert, on the page and in `state/system_state.json`) into `autonomy.acknowledged_halts` and commit as the owner. After that the same failing check does not page every run: one low-priority reminder per day, until the check passes once.

## Alerts and the dead-man's switch

| Signal | Channel | Source |
|---|---|---|
| halt opened, escalated or cleared | ntfy push, deduplicated by key | `nhlbet/ops/notify.py` |
| daily job crashed, review agent failed | ntfy push | `scripts/notify.py` steps in `daily.yml` and `review.yml` |
| halt in force | Healthchecks `/fail` ping | `run_daily` |
| system alive | Healthchecks success ping every 10 minutes | `scripts/heartbeat.py` |
| token expiry, low credits, keepalive health, weekly digest | ntfy push | `weekly-digest.yml`, `scripts/ops_watch.py` |

Healthchecks settings (docs: https://healthchecks.io/docs/http_api/): period 60 minutes, grace 6 hours (GitHub cron has run 3 to 6 hours late here). Silence means the heartbeat or its runner is dead.

## Secrets and variables to configure

Secrets: `NTFY_TOPIC` (a hard-to-guess topic name; on the public ntfy server the name is the only protection), `HC_PING_URL` (the Healthchecks ping URL), `ODDS_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` (review agent, optional).

Repository variables: `POLICY_OWNER_LOGINS` and `POLICY_OWNER_EMAILS` (comma separated; who may change `config/policy.yaml`), optional `REVIEW_ENABLED`, `BANKROLL`.

Domains to allowlist for outbound traffic: `ntfy.sh`, `hc-ping.com`, `api.the-odds-api.com`, `api-web.nhle.com`, `api.github.com`.

## Manual to-do list (only the owner can do these)

1. Set the repository variables above, then run the Policy guard workflow once.
2. Branch protection on `main` with a CODEOWNERS entry for `config/policy.yaml` and `.github/**`, requiring owner review. A git author field can be forged by anyone who can push, so this is the real control.
3. Create the Healthchecks check (period 60 minutes, grace 6 hours) and the ntfy topic; subscribe your phone to the topic.
4. Record the date the Claude OAuth token was created in `autonomy.credentials.claude_oauth_token_issued`.
5. Verify, per sportsbook you use, whether a shootout goal counts in totals and whether overtime counts in props; update `settlement.books` (`verified: true`) only after checking the book's rules.
6. Review and approve the proposed values in `config/policy.yaml` (`meta.status` is `proposed`).
7. Merge the hardening pull request yourself. The first run on `main` records the policy hash.
8. Decide whether to spend credits on a closing capture for totals, puck lines and props. Without it their closing-line value cannot be measured and they stay `RESEARCH ONLY`.

## Known single points of failure

Raw per-book odds quotes exist only in the Actions cache database (the public-safe export omits them). If the cache is evicted, per-book history is gone; the committed logs keep the no-vig consensus and decisions. The `log-backup` branch is pushed without force, so a rewritten history is detectable.

## Rerunning safely

A rerun of a day adds no duplicate alerts (alert keys are stored in the state), opens no second halt for the same check, and cannot break its own ledger (tests: `tests/test_ops_pipeline.py`).
