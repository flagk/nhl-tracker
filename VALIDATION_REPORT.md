# Validation report (hardening branch)

> Research and education only. Not betting advice. Nothing here guarantees profit. The project never places bets.

**Question:** does this system have a credible, durable edge against the betting market? **Answer: no edge has been demonstrated in any market, and the evidence that exists leans negative.** The system is now built so that it cannot publish on bad data silently, and so that the question gets answered by live closing-line value instead of by a flattering backtest.

Status key: **already covered** (existed and was verified, with proof), **fixed** (existed but was flawed; only the gap changed), **added** (did not exist). The earlier report, kept for its Phase 0 to 2 detail, is `docs/VALIDATION_REPORT_prior.md`. Progress log: `docs/PROGRESS.md`. Running it: `OPERATIONS.md`. Policy: `config/policy.yaml` (status `proposed`, the owner approves).

## 1. The old headline is not valid

The "55.8% win rate, +11.69% ROI, 1,805 bets" figure recomputes exactly (1,008 wins) but is `2 x win rate - 1`, an even-money artefact: the data had no odds. On the same bets the model was 0.4 points worse than always picking the home team (`AUDIT.md`). No document on the page or in the README makes the claim any more.

## 2. Requirements by phase

| Phase | Requirement | Status | Proof |
|---|---|---|---|
| 1 | No feature, split, scaler or calibrator sees the future | already covered | `tests/test_no_leakage.py`, truncation invariance now exactly 0 (stable sort fix) on 959 real games, `reports/phase1_leakage.md` |
| 1 | Availability guard on every fit (outcome columns, closing columns, near-perfect correlation) | added | `nhlbet/validity/availability.py`, planted-leak sentinels caught (`tests/test_validity.py`, real-data run) |
| 1 | Outcome-shuffle null | added | shuffled AUC 0.48 to 0.50, shuffled ROI -3.4% to -4.8% (`reports/phase1_realdata.md`) |
| 1 | Holdout locked | added | `reports/holdout_lock.json`: 656 games, 2026-01-05 to 2026-09-29, labelled previously examined |
| 1 | Settlement per book (shootout goal in totals, void games, props overtime) | fixed | the repo excluded the shootout goal; both rule sets are now in the policy, default `so_adds_goal`, unverified for every book (`docs/SETTLEMENT.md`) |
| 1 | Postponed or moved games void | fixed | a bet on a moved game was graded against the later result; now void with refund |
| 1 | Goalie confirmations as of decision time | fixed | reloading re-stamped capture time; first-seen time kept, `as_of` filter |
| 2 | Discrete counts, one joint score distribution | already covered / fixed | quarter lines for puck line and props, push-aware probabilities, PMF scoring (RPS, PIT), grid-edge audit (`reports/phase2_counts.md`) |
| 2 | Saves | added, research only | prior pinned at the edge of its grid, so not priced |
| 3 | Evaluation against the market, multiplicity-corrected | added | `reports/evaluation.md` (Holm across markets, date-clustered bootstrap, random-side and favourite baselines) |
| 3 | Margin-free CLV from raw quotes | added | `nhlbet/report/evaluation.py`; raw quotes exist only in the Actions cache |
| 4 | Model comparison with a pre-registered rule, one holdout evaluation | added | `reports/model_comparison_protocol.md` committed first; nothing qualified; `reports/holdout_eval.json` |
| 5 | Staking from the policy, Kelly ceiling, push-aware, correlated caps | fixed / added | `nhlbet/risk/*`, `tests/test_staking.py`, `reports/risk.md` |
| 6 | Review agent veto-only, shadow by default, pinned action | added | `AGENT.md`, `tests/test_review_*.py`, action pinned to a full SHA with doc citations |
| 7 | Halts, alerts, dead-man's switch, ledger, policy guard | added | `nhlbet/ops/*`, `tests/test_ops*.py`, `OPERATIONS.md` |
| 8 | Betting gate | added | `config/policy.yaml` `betting_gate`, `nhlbet/betting_gate.py`, `tests/test_betting_gate.py` |
| 9 | Page transparency | fixed / added | banners, live record kept apart from backtests, per-pick fields without JavaScript, half Kelly removed, caps cannot exceed the policy (`tests/test_site_transparency.py`) |
| 10 | Tests and docs | added | 492 tests green locally; `OPERATIONS.md` |
| All | Single policy file, read-only to code | added | `config/policy.yaml`, `nhlbet/policy.py`, `tests/test_policy_readonly.py`, `policy-guard.yml` |

## 3. Baseline versus final

The production moneyline model was deliberately not changed: nothing in Phases 1 to 4 qualified a replacement, so its probabilities are identical before and after. What changed is settlement, totals and puck-line calibration, the checks around the model, and how it is judged.

| Measure (log loss, lower is better) | Before | Final |
|---|---|---|
| Walk-forward 2,629 games, stack | 0.6751 | 0.6751 (unchanged by design) |
| Live games with results (135), model | 0.6531 | 0.6531 |
| Live, market at decision time | 0.6379 | 0.6379 |
| Live, market closing price | 0.6360 | 0.6360 |
| Matched live set (122 games), model / market at decision / market close | n/a | 0.6426 / 0.6297 / 0.6274 |
| Model minus market at decision (gain, 95% interval) | n/a | -0.0129 (-0.0260 to +0.0031) |
| Model minus market close (gain, 95% interval) | n/a | -0.0152 (-0.0275 to +0.0003) |
| Holdout (656 games, evaluated once): stack / Elo / home rate | n/a | 0.6826 / 0.6785 / 0.6916 |
| Totals paper bets, profit under the old rule vs the corrected default | -$348.90 | -$518.00 (15 of 465 results flip) |

The model is worse than the market at decision time and at the close, on every measure above. Blending toward the model never beat the market alone (122 games).

## 4. Conclusions per market (live picks, through 2026-10-19)

| Market | Settled picks | ROI (95% interval) | CLV | Plain conclusion |
|---|---|---|---|---|
| Moneyline | 201 | -9.4% (-16.6% to -1.2%) | +0.04 points (-0.04 to +0.12), 183 with a closing price, adjusted p = 0.15 | No demonstrable edge. Random-side baseline ROI -3.6%, market favourite -0.2%, model -10.7% (478 comparisons) |
| Totals | 130 | -13.6% (-34% to +7%) | not measurable | Insufficient evidence for any claim; model has no skill over the base rate |
| Puck line | 147 | -9.8% (-18% to -2.6%) | not measurable | Insufficient evidence; weak skill against the base rate only |
| Player shots | 166 | -7.3% (-20% to +6%) | not measurable | Insufficient evidence |
| Player points | 130 | -23.0% (-45% to +3%) | not measurable | Insufficient evidence |
| Player assists | 76 | -4.6% | not measurable | Insufficient evidence (n below 100) |
| Anytime goals | 39 | -5.1% | not measurable | Insufficient evidence (n below 100) |
| Parlays | 12 | -100% | not measurable | Insufficient evidence; parlays multiply margins |

Closing prices are captured for moneylines only, so closing-line value cannot be measured for any other market until a closing capture exists (an owner decision, see section 7).

## 5. Verdict on credible edge

Primary measure: closing-line value, date-clustered, multiplicity-corrected. Moneyline CLV is +0.04 probability points with an interval that includes zero (adjusted p = 0.15), and ROI is negative. **There is no credible edge.** The betting gate therefore reads `RESEARCH ONLY`, and the page says so.

How many more live bets would be needed to detect the observed CLV if it were real: mean +0.00043 in probability, standard deviation 0.0090, 95% confidence and 80% power: n = ((1.645 + 0.842) x 0.0090 / 0.00043)^2 = 2,784 bets, so **about 2,600 more** than the 183 measured. At the current pace this is years, which is itself the finding: the volume the policy can place cannot settle the question quickly, and the pre-registered gate (300 live CLV picks, interval above zero, mean at least +0.5 points, drawdown at most 10%) is the practical bar. A true +0.5 point CLV would be detected far sooner than the observed +0.04.

Risk if the policy were followed with no skill (`reports/risk.md`, 10,000 paths, quarter Kelly, 2% cap): median bankroll 98% of start after 500 bets, 61% chance of being behind, 0.7% chance of tripping the 20% drawdown kill switch, no ruin.

## 6. Docs vs reality

| Item | Docs say | Reality / decision |
|---|---|---|
| Claude OAuth token in Actions | The action accepts `claude_code_oauth_token` from `claude setup-token`; subscription terms for unattended CI use and usage limits are not spelled out in the pages read | Not verified end to end: the agent workflow has never run with a real token. Review is in shadow mode by default and failures publish `UNREVIEWED` and alert |
| Token lifetime | `setup-token` tokens last about a year | The issue date is not recorded; `autonomy.credentials.claude_oauth_token_issued` is null and a weekly warning says so |
| Shootout goal in totals | Several sources say the shootout goal counts toward totals; others differ by book | Unverified for every book (`verified: false` in the policy); both rules implemented, default `so_adds_goal`; owner must verify per book |
| ntfy | 250 messages per day on the public server, 4,096-byte messages, the topic name is the only protection | Implemented with deduplication; the topic is a secret. Not tested against the live server from this environment (blocked network) |
| Healthchecks | Ping URL GET means alive, `/fail` signals failure, at most 5 pings per minute | Implemented; heartbeat pings every 10 minutes. Not tested against the live service |
| GitHub schedules | Cron is best effort; public repos disable schedules after 60 days of inactivity | Observed 3 to 6 hours late: grace set to 6 hours; keepalive workflow and weekly warning added |
| Shallow checkouts | `git log -- path` cannot see policy history in a shallow clone | Policy owner check uses the GitHub commits API instead |
| Raw odds history | The public-safe export omits per-book quotes | Quotes live only in the Actions cache: single point of failure, documented |

## 7. Secrets, domains and assumptions to confirm

Secrets: `NTFY_TOPIC`, `HC_PING_URL`, `ODDS_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` (optional). Variables: `POLICY_OWNER_LOGINS`, `POLICY_OWNER_EMAILS`. Domains to allowlist: `ntfy.sh`, `hc-ping.com`, `api.the-odds-api.com`, `api-web.nhle.com`, `api.github.com`. Services: Healthchecks.io check with period 60 minutes and grace 6 hours; ntfy topic.

Assumptions to confirm:
1. The odds plan has 500 credits a month and resets on the 1st.
2. Every policy value marked `proposed` (staking, kill switch, betting gate, drift limits) is acceptable.
3. Per-book settlement rules (section 6) are verified by you before any `verified: true`.
4. Whether to spend credits on closing captures for totals, puck lines and props (otherwise they stay `RESEARCH ONLY` permanently).
5. The `policy-guard` workflow is red on this branch because the owner variables are not set yet; set them and re-run.

## 8. Known limitations and open items

- Empty-net goals are not modelled; a bivariate Poisson was not fitted.
- The holdout was examined in earlier reports and once as a local dry run before the recorded run (disclosed in `reports/model_comparison_phase4.md`).
- The log ledger and the heartbeat have been tested with synthetic data and a round trip over the 688 real log files, but not yet across a real week of daily runs. A false positive would show as a loud hard halt `log_integrity`, which is the intended failure mode.
- Branch protection and CODEOWNERS for `config/policy.yaml` and `.github/**` cannot be set from code and are the real control against policy tampering.
