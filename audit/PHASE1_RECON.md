# Phase 1: reconnaissance

Branch `audit` (from `hardening`). Scripts: `audit/scripts/`. Data: `audit/data/`. Nothing in this phase touches the holdout.

## 0. Read this first: the "sealed holdout" is not sealed

The standing rule is "never modify or peek at the sealed holdout season until Phase 6". **There is no sealed season in this repository.** Facts, from the history in `docs/PROGRESS.md` and `reports/`:

- The data span 2023-10-10 to now. Features, hyperparameters and feature selection were fixed on games before 2024-07-01.
- The walk-forward evaluation (2024-10-04 to 2026-09-29, 2,629 games) has been looked at repeatedly across the project, including the earlier audit (`AUDIT.md`) and the hardening phases.
- The hardening branch locked a "holdout" (656 games, 2026-01-05 to 2026-09-29, `reports/holdout_lock.json`) and evaluated it once (`reports/holdout_eval.json`), but those games had already been examined, and a local dry run printed the numbers before the recorded run. It is labelled "previously examined" everywhere. It cannot serve as a pristine test.

Consequence for this audit: the only genuinely unseen data is the **future**. The plan in Phase 6 therefore uses the forward ledger (games from the lock date onward, with odds committed before puck drop) as the sealed test, and the old holdout stays a documented, already-spent sample. Nothing below reads `reports/holdout_eval.json`, and no new script evaluates 2026-01-05 to 2026-09-29 as a test.

## 1. Environment limits that shape the plan

| Limit | Evidence | Consequence |
|---|---|---|
| NHL APIs (`api-web.nhle.com`, `api.nhle.com`) are not reachable from the authoring sandbox | `curl` returns `CONNECT tunnel failed, response 403` for all of them | Endpoint verification and ingestion run in GitHub Actions (`audit-realdata.yml`, push-triggered by `ci/audit_trigger.txt`); tests use recorded fixtures only |
| The full database (play-by-play-derived stats, start times, raw odds quotes) exists only in the Actions cache | local `data/nhl.db` holds 2,987 scores-only legacy games, no start times, no odds | Anything needing real data runs via the same workflow |
| Raw per-book odds quotes are never committed (public repo) | `nhlbet/privacy.py`, `export_logs` | Per-book timestamps exist only in the cache: a single point of failure addressed in Phase 2 |

## 2. Pipeline map

1. **Ingest** (`nhlbet/data/ingest.py`, `client.py`): schedule, boxscore, play-by-play, landing from `api-web.nhle.com/v1`, cached forever (`ttl=None`) in `data/cache/raw`, parsed into `games`, `team_game`, `goalie_game`, `skater_game` (SQLite `data/nhl.db`).
2. **Features** (`nhlbet/features/builder.py`): one calendar date at a time; features for all games of a date are emitted from pre-date state, then that date's results are folded in. Games are processed in `(game_date, game_id)` order.
3. **Models** (`nhlbet/models/`): logistic, random forest, LightGBM, XGBoost, Elo, stacked by inner expanding folds, online Platt calibration (`walkforward.py`, `calibration.py`).
4. **Odds** (`nhlbet/odds/`): The Odds API snapshots, per-book h2h quotes, consensus no-vig probability (Shin per book, mean across books), best price per side.
5. **Policy** (`nhlbet/risk/`): edge floor, fractional Kelly, caps (`config/policy.yaml`).
6. **Logs** (`data/logs/`): append-only CSVs of recommendations, consensus, paper bets, with a hash-chain ledger.

## 3. Feature inventory: source and knowledge time

The full table is `audit/data/feature_inventory.md` (and `.csv`), produced by `audit/scripts/p1_feature_inventory.py`, which **fails if the builder emits a base feature the table does not cover**. 57 base features; 13 are in the production model: `elo`, `xg_share`, `cf_pct`, `b2b`, `xga_pg`, `early_w`, `g_sv_l5`, `stakes`, `pen_drawn_pg`, `games_last_7d`, `ga_season_shrunk`, `homestand_n`, `g_sv_season`.

Summary of knowledge times:

| Class | Source | Becomes knowable | Notes |
|---|---|---|---|
| Strength, Elo, standings, situational | final scores | final horn of the team's previous game | the builder uses date granularity: a game on date D sees only games dated before D |
| Shot-attempt, xG, high-danger, PDO | play-by-play events | after the previous game's play-by-play is final | the NHL corrects coordinates and attribution for roughly a day afterwards; the repo caches the **first fetch forever**, so the stored value can be a preliminary one that differs from the settled value (knowledge time is not stored) |
| Special teams, penalties | boxscore | after the previous game | same caveat |
| Goaltending (save %, GSAx, rest) | goalie boxscores | after each prior game | **which** goalie starts tonight is a probability in the default `probable` mode |
| Rest, travel, time zones, streaks | schedule + static geography | when the schedule is published | postponements move it late |
| Lineup features | tonight's dressed skaters | about 1 h before puck drop at the earliest | off in training and live |
| Market features `mkt_*` | odds snapshots up to decision time | at capture | not among the 13 production features |

**Not stored anywhere: knowledge time.** Every table row has an event date, not the moment the repo first saw the value. Point-in-time correctness is by construction (the date loop), not by data. Phase 2 adds `knowledge_time` to every raw record.

## 4. Which odds the backtest uses

| Question | Answer | Evidence |
|---|---|---|
| Odds in the long backtest (2,629 games) | **none**. The walk-forward evaluates probabilities only (log loss, calibration); it makes no profit claim | `reports/walkforward_predictions.csv` has no price column; `features/market.py` is unused by the 13 production features |
| Odds behind live decisions | consensus no-vig probability (Shin per book, mean across books) from the last snapshot captured at the run (morning ~09:00 to 12:00 ET, late 20 min to 3 h before the first unstarted game) | `recommendations.odds_captured_at`, `slate.py` |
| Price used for staking and ROI | best decimal price across books in that snapshot, books whose own update is older than 90 minutes dropped | `odds/consensus.latest_book_prices` |
| "Closing" price | the last consensus capture at or before puck drop. It is a true close only if a capture landed in the final 25 minutes; `gate.py --kind close` tries for one per start wave (about 1 to 2 captures per wave), and GitHub cron has been 3 to 6 hours late | `odds/clv.py`, `nhlbet/gate.py` (`CLOSE_WINDOW_MIN = 25`) |
| Open | the first capture seen, not the bookmakers' opening line | `features/market.py` |

Measured on the real database by `audit/scripts/p1_odds_timing.py` (GitHub Actions run, result `audit/data/odds_timing.json`; 387 logged recommendations, 146 games with captures):

| Quantity (minutes before puck drop) | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| Decision time, morning runs (n=197) | 200 | 476 | **656** | 716 | 867 |
| Decision time, late runs (n=190) | 96 | 177 | **238** | 388 | 627 |
| Last capture before start (the "close") | 5.8 | 10.8 | **13.2** | 23.6 | 658 |
| Captures per game | 5 | 9 | 13 | 19 | 42 |

Findings:
1. **Decisions are made 4 to 11 hours before puck drop** (median), not near it. Closing-line value therefore measures how far the market moves over the last several hours; it is the right test for an early-price edge, but a model decision at a late run is not comparable with a morning one.
2. **The "close" is a true close for about half the games.** 48.6% of games have no capture in their final 15 minutes, and **16.4% have a last capture more than 60 minutes before the start**, where the "close" is really a mid-afternoon price. CLV for those games understates the market's final move; the sign of the bias is unknown. Every CLV number reported so far (including the +0.04 points moneyline figure) mixes true closes with stale ones. Phase 4 therefore reports CLV with and without the games whose last capture is older than 15 minutes.

## 5. Every place a future-leaking value could enter

Each item has an owner test or a Phase 3 attack. "Open" means no test currently catches it.

| # | Surface | Mechanism | Status |
|---|---|---|---|
| 1 | Same-date results | a game's features reading another same-date game's result (doubleheaders do not exist, but late games ending after midnight UTC share `game_date` logic) | covered: `test_no_leakage.py` (same-day outcome swaps); Phase 3 mutation M1 |
| 2 | Fold order | emitting after folding, or sorting ties unstably | covered (stable sort, truncation difference exactly 0); mutation M2 |
| 3 | Goalie `actual` mode | starter identity read from the post-game boxscore in backtests | default is `probable`; `actual` is a live-only path. **Open**: nothing prevents someone training with `actual` |
| 4 | Lineup `actual` mode | dressed skaters from the post-game boxscore | off by default; open as in 3 |
| 5 | Closing odds as a feature | `mkt_*` or any `close` column | `assert_no_closing_columns`; attach uses snapshots at or before decision time |
| 6 | Target-derived columns | `hr, ar, ot, so, tot, mar` | dropped before candidate columns; availability guard (Phase 1 of hardening) |
| 7 | Scaling, imputation, selection | fitted on test rows | pipeline-internal; feature selection and hyperparameters end 2024-06-30 (test enforces) |
| 8 | Stacker / calibrator | fitted on rows that include the evaluated game | inner expanding folds; online calibration uses earlier blocks only |
| 9 | Production calibrator in hindsight | the calibrator fitted on all history is in-sample for any retrospective rescoring of old games | by design for live use; **any evaluation that rescored old games with the current bundle would be in-sample**. Open: nothing forbids it |
| 10 | Data revisions | API values corrected after the first fetch; the repo keeps the first fetch if the cache survives and silently refetches (revised) if the cache is evicted | **Open**: no knowledge-time column, no snapshot immutability (Phase 2) |
| 11 | Postponed games | a game moved to a later date keeps its old date's features | fixed in hardening (void) |
| 12 | `days_into_season` / `_season_start` | first date seen in the data, not the schedule | not a leak; wrong only for partial first seasons |
| 13 | Rolling league counters (`league[...]`) | updated after each date | covered by truncation invariance |
| 14 | Odds timing | decision odds captured after the information used by the model, or "closing" odds captured long before puck drop | unquantified: see section 4 |
| 15 | Live vs backtest mismatch | backtest uses morning-mode goalie probabilities; late live runs use confirmed starters | stated limitation; not a leak |
| 16 | Holdout contamination | selection decisions informed by the holdout | **already happened** (section 0); disclosed |
| 17 | Evaluation on rows the model trained on | reports that average over train and test | walk-forward only scores test blocks; Phase 3 adversarial validation checks the split shape |

## 6. What Phase 1 changes in the plan

- Phase 2 must add `knowledge_time` and snapshot immutability, because items 10 and 14 are open and cannot be tested without it.
- Phase 3 must include a canary for items 3 and 9 (a model trained in `actual` mode must be flagged, and the production calibrator must refuse in-sample rescoring), and mutation tests for items 1, 2, 7 and 8.
- The sealed test is the forward ledger, not the old holdout (section 0).
