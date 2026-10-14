# Validation report

> Research and education only. Not betting advice. Past performance does not guarantee future results.

**Question:** do this project's picks have a real, durable edge? **Answer, up front: no edge has been demonstrated in any market.** The models have genuine skill against trivial baselines (base rate, Elo, each player's own rate), but against the betting market the live evidence is either neutral or negative, and every sample is far too small to say more. Details, numbers and what would change that verdict are below. Live numbers are from the committed logs through **2026-10-14** and are regenerated on every daily run (`reports/VALIDATION.json`, `site/validation.html`).

Status key: **already covered** = existed and was verified correct (with the proof); **fixed** = existed but was flawed or incomplete, only the gap changed; **added** = did not exist; **not done** = deliberately not built, with the reason.

---

## Phase 0: orientation

| Item | Status | Evidence |
|---|---|---|
| Architecture documentation | already covered | README "Under the hood" module map and `docs/` (FEATURES, ODDS, RISK, RESULTS, MARKETS, PAPER_TRADING, SITE); no ARCHITECTURE.md written |
| Trace of every number the page claims | added (below) | table that follows |

| Page claim | Produced by |
|---|---|
| Model health, drift (OK / WARN / ALERT) | `nhlbet/train.py` (`retrain`, now fed live games) -> `nhlbet/monitor.py` (`performance_drift`, `feature_drift`) |
| Win probabilities, edge, EV per game | `nhlbet/report/slate.py` -> `nhlbet/odds/edge.py` -> `nhlbet/risk/policy.py` (`assess_sides`, `recommend_slate`) |
| Track record: resolved games, model log loss, market-close log loss | `nhlbet/report/betlog.py:53` `performance` <- `resolved` (:31) = last pre-game recommendation per game + final score + last consensus before puck drop |
| Real bets, ROI, win rate | `betlog.performance` <- `nhlbet/risk/bankroll.py:28` `summarize_bets` (the live policy has placed 0 bets so far) |
| CLV | `nhlbet/odds/clv.py` `bet_clv`, `summarize_clv`; margin-free move added in `betlog.resolved` |
| Fake-bet table (bets, ROI, 95% CI, win rate, CLV) | `betlog.shadow_performance` <- `shadow_resolved` <- game/prop rows (`all_shadow_rows`, flat $10 via `_flat_stakes`) + parlays (`parlay_resolved`); settlement in `betlog.alt_value` |
| Pick ranking and trust weights | `nhlbet/report/ranking.py` (`rank_picks`, `learn_weights`) |
| Parlays | `nhlbet/risk/parlays.py` |
| Evidence page, verdicts | `nhlbet/report/validation.py` (`run_validation`) -> `nhlbet/site/validation_page.py` |

---

## Phase 1: leakage and validity

| Item | Status | Evidence / result |
|---|---|---|
| Every feature uses only pre-puck-drop information | **already covered, re-verified** | `features/builder.py:184-189`: all games of a date get features from pre-date state, and only then are that date's results folded in. Elo snapshot taken before the date (:182). Season aggregates shrink to the *previous* season (`_roll_season`). Goalie starts come from past starts only; in the default `probable` mode tonight's starter is never read (:371), in `actual` mode only his *identity* is (:371-383). Lineup features are off in training and live (:423). Corruption tests: `tests/test_no_leakage.py` (future games scrambled -> earlier and same-day features identical; truncation; same-day outcome swaps; row order; goalie-mode tests). The xG model has fixed geometry coefficients and is never fitted on outcomes (`data/xg.py`) |
| Walk-forward never touches future data (splits, scaling, calibration) | **already covered, re-verified** | `splits.py:26` trains only on dates strictly before the block; imputer and scaler sit inside the sklearn pipeline and are fitted on the training fold (`models/base.py:91`); stacker and calibrators come from inner expanding folds (`walkforward.py:33`); online calibration fits only on earlier blocks (`calibration.py:100`). Tests: `test_walk_forward_predictions_never_depend_on_the_future`, `test_online_calibration_uses_only_the_past`, `test_walk_forward_is_strictly_time_ordered` |
| Tuning and feature selection stay before the evaluation window | **fixed (test added)** | Provenance was recorded in `feature_selection.json` / `hyperparams.json` (both end 2024-06-30, evaluation starts 2024-10-04) but nothing enforced it. `tests/test_audit_phase1.py::test_feature_selection_and_tuning_ended_before_the_walk_forward_evaluation_began` now fails if either file is regenerated on a window overlapping the backtest. Disclosed forking path (docs/RESULTS.md): online calibration was chosen after seeing one un-tuned evaluation run; its gain (~0.0015 log loss) is inside the noise |
| Backtest odds were obtainable at pick time, timestamped | **already covered; limitation stated** | The long backtest contains **no market prices** (none exist), so it never claims betting profit. Live picks use the last odds capture before the run: across 309 logged recommendations the odds were captured a median of 0.1 minutes (max 8.2) before the run, none after, none missing (`log_integrity` in `reports/VALIDATION.json`). The closing line is defined as the last consensus no later than puck drop (`odds/clv.py:13`) and is never a feature (`features/market.py:29,37`) |
| Logs are not contaminated by in-play data | **already covered, verified** | 0 of 283 recommendation rows, 0 of 4,331 paper-bet rows, 0 of 1,293 prop rows and 0 of 5,907 alt-quote rows were logged at or after puck drop; 0 duplicate keys in any log; every pick carries a model version (17 versions in use). `hygiene.purge_inplay` and `odds/consensus.pregame_only` guard it |
| Duplicates, postponements, OT / shootout | **already covered, verified; one gap fixed** | Game ids are primary keys; legacy rows duplicating API games are dropped (`data/loaders.py:24-27`). Unplayed or postponed games have no result (`parsers.py:51-60`) and stay pending, never counted. Moneyline settles on the official result including shootout; totals subtract the shootout goal; the puck line uses the official margin (`betlog.alt_value`, `goals.regulation_goals`; hand-checked tests in `test_alt_markets.py`, `test_goals.py`). **Gap fixed:** no schema/sanity validation existed on the game table, so `hygiene.validate_inputs` now withholds picks on duplicate games, inconsistent `home_win`, negative scores or missing columns |
| **Gap found: the monitor and the production calibrator never saw a live game** | **fixed** | `train.retrain` called `resolved_history` with only the static backtest file, so "model health" could not react to how the deployed model did on games played after deployment. `train.live_predictions` now appends the last pre-game raw stack probability of every resolved live game. Before: drift judged on 2,629 backtest games only (recent log loss 0.6716). After: backtest + 100 live games (recent log loss 0.6577, calibration slope 1.10, status still OK). The status did not change; the point is that it now can |
| **Gap found: stale data could still produce published picks** | **fixed** | If the NHL refresh failed the run only added a note. `hygiene.data_freshness` now marks the model ALERT (no bets, no ranking) when the refresh failed *and* the newest completed game is more than 3 days old while games are scheduled |
| Train / serve mismatch to be aware of | **limitation, not a leak** | Training and the backtest use `probable` goalies; late live runs with a confirmed starter use `actual`. Same semantics, a different mixture of the same features. The backtest therefore measures morning-mode skill; late-mode skill is only measured by the live log |

---

## Phase 2: discrete count modelling (totals and stat lines)

| Item | Status | Evidence / result |
|---|---|---|
| No market compares a continuous point estimate to a line | **already covered, verified** | Game totals and puck lines come from a 16x16 joint grid of integer goals (`models/goals.py` `ScoreDistribution`); player stats use negative-binomial PMFs (`models/props.py`). No regression output is compared with a line anywhere |
| Distribution chosen by out-of-sample likelihood | **added** | `reports/GOALS.md` "Which joint score distribution": out-of-sample log-likelihood of the observed regulation score. Production (negative binomial + tie correction) beats independent negative binomial / Poisson by **+0.0117 per game (95% CI 0.0081 to 0.0153)**: the Dixon-Coles-style correction that moves mass onto tied scores earns its place. Independent NB and Poisson are identical (the fitted dispersion for team goals is 0), so the extra parameter is unnecessary. A full bivariate Poisson was not tried |
| One joint distribution for every game market, coherence test | **fixed / added** | The moneyline and the grid were already one distribution. Added: team totals, both-teams-to-score and quarter lines from the same grid, `goals.coherence_issues` (moneyline = puck line -0.5, three outcomes sum to 1, half lines never push, over probability falls with the line, team totals reproduce the game mean, BTTS agrees with inclusion-exclusion), run on every game in the daily slate (an incoherent game is not priced); `tests/test_counts.py` |
| Half-point / whole-number / quarter-line settlement and push-aware EV | **fixed** | Totals already handled pushes. Gaps: **whole-number player prop lines were skipped entirely**, quarter lines unsupported. Now: props priced with (over, push, under) from the PMF, `p_over` conditional on no push to match the market's no-vig pair, EV uses the push-aware probabilities (`models/props.over_push_under`, `report/props.py`); quarter lines are the average of the two adjacent legs (EV is linear, so this is exact). Hand-checked in `tests/test_counts.py` |
| Calibrate P(over) per line, compare with market at the same line | **fixed (calibration), limitation (market)** | **Flaw:** the totals / puck-line recalibration was one pooled map across lines. Pooling 5.5 and 6.5 (57% and 44% over) hid per-line overconfidence (pooled slope ~1.0 vs per-line 0.67-0.76). Now line-aware (`goals_eval.fit_platt_line`, `markets.recalibrate`). Before / after on 6,000+ walk-forward games: puck line calibration slope 1.45 -> **0.90**, log-loss gain vs base rate -0.0185 -> **-0.0208**; totals unchanged (slopes 0.4-0.56 +/- 0.22 before and after, log loss 0.6820/0.6929/0.6864 vs 0.6821/0.6928/0.6859). Added `cal_slope_se`: totals probabilities barely vary, so their slopes cannot be measured to better than +/- 0.22 and "overconfident" there is not established. Comparison with the market at the same line is only possible live (no historical lines exist) |
| Shots, points, assists, goals: count models, shrinkage, PMF scoring | **already covered (models) / added (evaluation)** | Negative-binomial PMFs driven by shrunk own rate, opponent, venue, ice time, rest. Added to `scripts/props_backtest.py` / `reports/PROPS.md`: PMF log-likelihood and **ranked probability score**, reliability of P(over) at every line, a shrinkage grid chosen on games before 2024-07-01 and scored after, and `n_to_confirm` with an "insufficient evidence" verdict. Result over ~190,000 player-games: every stat beats the player's own shrunk rate with intervals far from zero (e.g. shots NLL -0.0174, CI -0.0184 to -0.0163) and the reliability tables track the diagonal. **Found in the first run:** the shrinkage optimum for points / assists / goals sat at the top of the grid (50); the grid now runs to 400 |
| Anytime scorer = 1 - P(0); shot-rate x shooting-% structure | **already covered (different but equivalent)** | Modelled directly as a goal count with a position-mean-shrunk rate rather than shots x shooting %; evaluated with the same PMF scores (goals NLL -0.0058, CI -0.0062 to -0.0055). Market probability for the one-sided "yes" price is estimated with an assumed 7% margin (`odds/props.ONE_SIDED_MARGIN`) and labelled approximate |
| Goalie saves | **added (research only)** | `models/saves.py`: shots against ~ NB (opponent and team shot rates, home), thinned by the goalie's shrunk save percentage (a thinned NB is NB with the same dispersion, so the full PMF and pushes are exact). Walk-forward over 10,577 starts: saves NLL -0.064 vs the goalie's own average (CI -0.071 to -0.057). **Underconfident in the upper range** (actual > predicted at high P(over)), so it is not priced. Not wired to live prices: needs a verified book rule for a goalie who starts and is replaced, and the odds-credit budget does not allow another market |
| Overtime / empty-net / DNP settlement for props; confirmed lineups only | **limitation, partly fixed** | Official stats include overtime and empty-net goals but not shootout goals; a player who did not dress is void (`betlog.alt_value`, tested). **No lineup confirmation source exists**, so props are priced before lineups are known: the page now says so on every prop and the bet voids if he does not play. Per-book settlement differences are not modelled |
| CLV on totals and props | **not measurable** | Only one pre-game price capture exists per game for totals, puck lines and props (closing captures fetch moneylines only; adding spreads / totals costs two credits per capture). Stated on the evidence page |

---

## Phase 3: honest evaluation

All numbers: `python scripts/validate.py` (committed files only), flat $10 paper bets, intervals resample whole days.

**Game level, 100 settled games (Sep 30 - Oct 14):**

| Forecast | Log loss | Brier |
|---|---|---|
| Model (production, calibrated) | 0.6584 | 0.2332 |
| Elo | 0.6580 | 0.2330 |
| Market at pick time (no-vig) | 0.6458 | 0.2273 |
| Market close (no-vig) | 0.6431 | 0.2260 |
| Home-rate baseline | 0.6899 | 0.2484 |

Model minus market close: **+0.0153 (95% CI -0.0019 to +0.0298)**. The point estimate favours the market; the data cannot yet separate them.

**Moneyline picks (the model's side of every game, 100 picks):** win rate 58.0% vs a break-even of 58.3%; ROI **-2.0% (95% CI -20.7% to +15.3%)**; -2.0 units of $10; max drawdown 7.8 units; longest losing run 7; the model claimed a mean +0.9 point edge over the no-vig price. Random sides in the same games at the same margin match or beat this result 38% of the time. Closing-line value: the no-vig line moved 0.08 points toward our side on average, **identical to the no-skill market-favourite control (0.08)**. Flat bets only where the model sees a positive edge (`flat_model_side`, claimed edge +4.6 points): win rate 44.0%, ROI -11.8% (CI -28.6% to +4.2%), line moved -0.02 points against; the claimed edge did not materialise.

**By market (distinct picks; a pick shared by several strategies counts once) vs the no-skill controls:**

| Market | Picks | ROI (95% CI) | Verdict | Picks needed to confirm a 5% ROI |
|---|---|---|---|---|
| Moneyline | 144 | -7.2% (-15.8% to +1.8%) | insufficient evidence | 2,253 |
| Totals | 97 | -8.0% (-35% to +16%) | insufficient evidence | 2,255 |
| Puck line | 106 | -8.0% (-18% to +0.6%) | insufficient evidence | 1,834 |
| Player shots | 145 | -5.7% (-20% to +10%) | insufficient evidence | 2,259 |
| Player points | 103 | -17.3% (-44% to +15%) | insufficient evidence | 3,126 |
| Player assists | 76 | -4.6% (-43% to +39%) | insufficient evidence | 5,134 |
| Anytime goalscorer | 39 | -5.1% (-78% to +64%) | insufficient evidence | 20,657 |
| Parlays (14 tickets) | 14 | -100% | too few picks | n/a |

The controls (no model, flat $10) do about as well: moneyline -4.9%, totals -2.9%, shots +2.4%, assists +4.8%, points -8.2%, puck line -18% (CI -34% to -7%). No model strategy beats its control by more than noise.

| Item | Status |
|---|---|
| Win rate, ROI, units, max drawdown, longest losing streak, no-vig edge per bet | **added** (`validation.summarize_picks`, `pick_table`) |
| Splits by market type, month, home/away, favourite/underdog, odds bucket | **added** (`validation.split_table`, `market_summary`; the odds-bucket and role splits show the claimed edge concentrating in underdogs, where the realised result is worst: 8.1 claimed points, 44% wins) |
| Bootstrap CIs, random-pick baseline, probability the result is luck | **added** (`cluster_bootstrap`, `random_pick_baseline`, one-sided t-tests) |
| CLV as primary evidence | **fixed.** The existing CLV (`clv_ev`) pays the margin of the price taken, so it sits near **-2.1% even for the no-skill favourite control** and was being read as a signal. Added the margin-free line move (`clv_move_pts`, in the real-bet track record too) and report both |
| Calibration vs the market's no-vig probability, fit on training folds only | **added** (`validation.score_vs_market`; calibrators verified fold-safe in Phase 1) |
| Ablations and parameter sensitivity | **added** (`nhlbet/analysis/ablation.py`, workflow `validation.yml`, `reports/ABLATION.md`) |

Ablation highlights (walk-forward, 2,729 games, logistic): removing the **advanced** (xG/Corsi) group costs +0.0030 log loss (CI 0.0004 to 0.0054) and the **rest/schedule** group +0.0028 (0.0004 to 0.0051); goaltending, special teams, situational and strength groups are indistinguishable from noise individually; Elo-only costs +0.0059 and the home rate +0.0165. Regularisation C from 0.0003 to 0.3 moves log loss between 0.6730 and 0.6743: the result does not hinge on it. **Stability:** logistic minus Elo is -0.0082 (CI -0.0133 to -0.0026) in 2024-25 but only -0.0040 (CI -0.0105 to +0.0034) in 2025-26, and the logistic is 0.6868 vs a home-rate baseline of 0.6927 that season: the skill over trivial baselines was concentrated in the first season and has faded to within noise.

---

## Phase 4: model improvement

**Not done, by the stated gate.** Phase 3 is not "clean" in the sense the brief requires (there is no demonstrable edge against the market), and adding features to chase one would only add forking paths. Already covered by the existing comparison (`reports/model_comparison.md`, same walk-forward protocol, 2,629 games): logistic 0.6741, weighted average 0.6742, stack 0.6751, LightGBM 0.6764, XGBoost 0.6763, random forest 0.6772, Elo 0.6800, home rate 0.6897; the simplest regularised model is as good as the ensemble. A market-plus-adjustment model cannot be backtested (no historical prices); the live policy already shrinks the model toward the market and requires a raw edge of 3 points, a 12-point disagreement ceiling and a healthy drift monitor before staking (`risk/policy.py`, `risk/shrink.py`). The ranking and stakes of the experimental markets are now sized by trust weights learned from each market's own settled bets (`ranking.learn_weights`), which is the uncertainty-aware part.

---

## Phase 5: staking and risk

| Item | Status | Evidence |
|---|---|---|
| Quarter-Kelly, per-bet and daily caps, shared with the page | **already covered, verified** | `RiskConfig`: Kelly fraction 0.25, 2% per bet, 5% per day, 5 bets a day (`risk/policy.py`); the page recomputes with `site/core.js`, parity-tested against Python in `tests/test_site.py` |
| Push-aware Kelly | **added** | `risk/kelly.kelly_fraction_push`, verified against a numerical optimum. Not used by the live policy, which stakes moneylines only |
| Correlated exposure | **not needed today** | Real stakes are one moneyline per game with a daily cap; all other markets are flat-$10 paper bets, so same-game correlation cannot compound real exposure. If any other market is ever staked for real, a per-game cap must be added first |
| Monte Carlo ruin / drawdown | **already covered, run** | `scripts/risk_sim.py`. 500 bets, 1% stakes, a claimed 3% edge: with **no** real edge 87% of paths end below the start, median final $784 of $1,000, P(ever -50%) 2.5%, P(drawdown >= 30%) 55%; with half the claimed edge 66% end below; with all of it 40% |

---

## Phase 6: production reliability

| Item | Status | Evidence |
|---|---|---|
| Stale-data and invalid-data refusal | **fixed** | `hygiene.data_freshness`, `hygiene.validate_inputs`, both wired into `pipeline.run_daily` |
| Missing odds, stale odds, unconfirmed goalies | **already covered** | no odds -> no bet; odds older than the limit -> no bet; unconfirmed goalie -> higher edge threshold (`risk/policy.py`, `RiskConfig.min_edge_unconfirmed_goalie`) |
| Idempotent runs | **already covered** | run ids are `date-type-HHMM` with upserts keyed on them; the workflow gate (`gate.py`, heartbeat) prevents duplicate runs |
| Append-only, timestamped log, model version per pick | **already covered, verified** | per-run CSV partitions under `data/logs/`, git history; checked in Phase 1 (no duplicates, no post-drop rows, version on every row). `purge_inplay` can delete contaminated rows; that is the only deletion path |
| Automatic grading | **already covered** | results from the official NHL API; settlement rules above. Per-book push/void differences are not modelled |
| CI gate that blocks publishing | **added** | `daily.yml` runs leakage, audit, count-model, validation, gate and risk tests before anything is generated; a failure stops the job (and the Pages deploy, which only follows a successful daily run) |
| **Gap found: the odds-credit budget would have run out** | **fixed** | 23 credits a day used against a 500-credit month (196 left on Oct 14) would have ended the moneyline feed around Oct 22. Closing snapshots are now one per start wave instead of one every ~20 minutes (17 on Oct 13), fresh player markets are not re-bought, and player prices get a daily budget (`pipeline.props_budget`); moneylines, totals and puck lines are never rationed |
| Heartbeat scheduler, stuck-deploy recovery, morning run timing | **added earlier in this engagement** | see git history |

---

## Phase 7: page transparency

| Item | Status | Evidence |
|---|---|---|
| Live record from the logs only, separate from the backtest, with CIs, by market | **added** | `site/validation.html` ("Evidence" in the nav) built from the committed logs on every run; backtest in its own section |
| Per pick: market, line, model probability, no-vig market probability, odds, timestamp, stake, edge | **fixed** | the ranking now shows line, P(push) and the price-capture time; moneyline cards already showed model, market, edge, price. **Book names are deliberately not published** (the repository is public; `privacy.py`), so "book" is the one requested field that stays hidden |
| Methodology, limitations, research-only disclaimer | **added / already covered** | evidence page sections "Method" and "Known limitations"; disclaimer on every page |
| Core table works without JavaScript, on mobile | **added** | `<noscript>` static table of every game (model P, market P, best price, edge, EV, price time, decision) in `index.html`; the evidence and history pages are script-free; tables collapse to cards on phones (`theme.css`) |

---

## Before / after summary

| Metric | Before | After |
|---|---|---|
| Drift monitor input | 2,629 backtest games, no live game | backtest + live games (n=2,729 today), recent log loss 0.6577 |
| Puck-line calibration slope (walk-forward recalibrated) | 1.45 | 0.90 (SE 0.06) |
| Puck-line log-loss gain vs base rate | -0.0185 | -0.0208 |
| Totals calibration slope | 0.54-0.56 | 0.42-0.56 (SE 0.22: unchanged within noise) |
| Joint score distribution | chosen by convention | tie correction beats independent NB/Poisson by 0.0117 per game |
| Whole-number prop lines | skipped | priced with a push |
| Closing-line value shown | -2.1% for everything (pays the margin) | plus margin-free move: +0.08 pts for the model and for the no-skill control alike |
| Odds credits per day | ~23 (plan ends ~Oct 22) | budgeted to last the month |
| Tests | 204 -> 240 before this audit | 266, plus a 5-minute publish gate on every daily run |

---

## Verdict

**Is there statistically credible evidence of edge? No, in any market.**

| Market | Live evidence | Size of any edge | Live bets needed before it could be trusted |
|---|---|---|---|
| Moneyline | 100 games, ROI -2% (CI -21% to +15%); model log loss 0.658 vs market close 0.643 (CI -0.002 to +0.030); line movement identical to a no-skill control | none shown; the model is as good as Elo and slightly behind the market | ~5,000 bets for a true 3% ROI, ~1,800 for 5%; closing-line value would show it far sooner (about 20 bets per 1% of CLV at the observed spread, once the margin-free CLV is positive and above the control) |
| Totals | 97 picks, ROI -8% | none; backtest skill vs the base rate is 0.001-0.002 log loss and the calibration cannot be measured precisely | ~2,300 picks for 5% |
| Puck line | 106 picks, ROI -8% (CI -18% to +0.6%) | backtest skill vs base rate is real (-0.021 log loss) but that is not a market edge; the no-skill puck-line control lost 18% | ~1,800 picks for 5% |
| Player shots / points / assists | 145 / 103 / 76 picks, ROI -6% / -17% / -5% | none; the models describe player outcomes well (calibrated over 190,000 player-games) but the market already knows what they know | 2,300 / 3,100 / 5,100 picks for 5% |
| Anytime goalscorer | 39 picks | none; the market probability itself is only estimated | ~20,000 picks |
| Goalie saves | not priced | n/a | n/a |
| Parlays | 14 tickets, all lost | none possible: they multiply the margin | n/a |

What would change the verdict: positive margin-free closing-line value that beats the no-skill control over a few hundred picks (visible in weeks), then a positive ROI interval over a few thousand. What would end it: closing-line value at or below the control after several hundred picks. Until then the right reading of every number on the site is "research output", and the ranking's trust weights (now learned from results) already push the experimental markets toward the floor.

## Not done / open items

- A bivariate-Poisson or Dixon-Coles likelihood with an explicit correlation parameter was not fitted; the tie-correction variant already beats independence by a clear margin.
- CLV for totals, puck lines and props needs closing captures of those markets (2-3 more credits per capture).
- Lineup confirmation for props needs a lineup source the project does not have; props stay labelled unconfirmed.
- Goalie saves are a research model only (see above).
- Per-book settlement rules are not modelled.
- The 2024-25 / 2025-26 contrast (strong skill, then within noise) deserves a longer look once more seasons accumulate; one season of fade is suggestive, not conclusive.
- Props shrinkage was re-run on the wider grid (3 to 400 pseudo-games), chosen on games through 2024-06-30 and scored on later games (`reports/PROPS.md`, `reports/props_shrink.json`, which feeds the live player models): shots 25, points 50, assists 50, goals 100. None of the four sits at the edge of the grid any more (the first run chose the old top, 50, for points, assists and goals). On the later games the shots optimum was 12 rather than the chosen 25 (NLL 1.54512 vs 1.54537), a difference too small to matter. The goalie-saves prior (research model only) still chose the top of its own grid (1600), so that choice is not yet bracketed; nothing is priced or staked from it.

Reproduce everything: `python -m pytest -q` (leakage, settlement, count and evaluation tests), `python scripts/validate.py` (live evidence from the committed logs), and the `Goals model backtest`, `Player props backtest` and `Validation (ablations)` workflows for the backtest reports.
