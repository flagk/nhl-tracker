# Feature catalogue

Every feature is computed **as of the game date**: it uses only games on strictly earlier dates
(enforced structurally by `nhlbet/features/builder.py` and proven by `tests/test_no_leakage.py`,
which also includes a deliberate-leak mutation check). Each per-team feature appears as `h_*` (home),
`a_*` (away) and `d_*` (home minus away, except flags/counters).

| Group | Features | Source | Notes |
|---|---|---|---|
| **strength** | `gd_ewm`, `gf_ewm`, `ga_ewm` (halflife 10 games), `gd_l10`, `win_pct_l10`, `gd/gf/ga_season_shrunk`, `gd_venue` (home/away split), `elo`, `sos10`, game-level `elo_prob` | scores | Season stats are shrunk toward last season's (half-regressed) rating with k=15 pseudo-games, so early season is not noisy. `sos10` = mean pre-game Elo of the last 10 opponents. Elo: K=6, goal-margin multiplier, +35 home-ice, 30% regression each season. |
| **advanced** | `cf_pct`, `ev_cf_pct` (5v5), `ff_pct`, `sog_share`, `hd_share`, `xg_share`, `xgf_pg`, `xga_pg`, `sh_pct`, `sv_pct_team`, `pdo_dev`, `adv_n` | play-by-play | Last 20 games, ratio of sums. `pdo_dev` = shrunk (SH% + SV%) - 1: a regression-to-the-mean signal, not a strength signal. Blocked shots are attributed to the shooter's team via the roster, not the blocker. xG is a transparent distance/angle/shot-type model (`nhlbet/data/xg.py`) that can be refit on stored shots. |
| **special_teams** | `pp_pct`, `pk_pct`, `st_net`, `pen_taken_pg`, `pen_drawn_pg`, `pen_diff_pg` | play-by-play | Last 30 games; PP%/PK% shrunk to league ~20% with 20 pseudo-opportunities. Coincident minors are not counted as power plays. Penalties per game ~ per 60 (regulation). |
| **goaltending** | `g_sv_season`, `g_sv_l5`, `g_sv_l10`, `g_gsax100_season`, `g_gsax100_l10`, `g_rest`, `g_b2b`, `g_p_primary`, `g_confirmed` | boxscore + play-by-play | GSAx per 100 shots = (xG faced - goals against). All shrunk toward league (.905 SV%, 0 GSAx). **Starter**: by default a probability-weighted blend of the team's primary and backup goalie, where P(primary) is the league's *as-of* rate for that rest situation (back-to-back vs rested) - no tonight information. With `goalie_mode='actual'` the confirmed starter is used (`g_confirmed=1`). |
| **rest_schedule** | `rest_days`, `b2b`, `g3in4`, `g4in6`, `games_last_7d`, `travel_km`, `tz_shift`, `road_trip_n`, `homestand_n` | schedule | Travel = great-circle km from the venue of the team's previous game to tonight's arena. Time zones use standard-time offsets. |
| **lineup** | `pts_share_missing`, `core_missing_n` | boxscore skaters | Share of the team's core-10 skaters' (recent points/game) production not dressed tonight. **Off by default** (`lineup_mode='off'`) because dressed lineups are not known in the morning. Free NHL data has no official injury feed; live use needs `data/manual/` inputs. |
| **situational** | `is_rivalry` (division + short curated list), `is_playoff`, `days_into_season`, team `gp_season`, `early_w`, `cutoff_gap`, `stakes` | schedule + standings-as-of | `cutoff_gap` = points% minus the conference's 8th-best points% (approximate wild-card line); `stakes` = late season and within 6 points% of it. |
| **market** | `mkt_open_p`, `mkt_move` | odds snapshots (Phase 4) | Opening no-vig probability and movement to the latest snapshot **no later than decision time**. The closing line is never a feature (`features/market.py`, tested). Not part of the default model. |

## Availability by data source

| Data | Legacy CSV bootstrap | Full API ingest |
|---|---|---|
| strength, rest/schedule, situational | yes (28 of 32 teams, so rest/travel are biased where a hidden game exists) | yes |
| advanced, special teams, goaltending, lineup | **NaN** | yes |

The legacy CSV has no shots, penalties, goalies or players. Columns that are mostly missing are
dropped by the selection step, so experiments on the CSV only exercise the first and last rows.

## Feature selection (`nhlbet/analysis/importance.py`)

1. drop constant / >50% missing columns; 2. use differentials in place of raw `h_`/`a_` duplicates;
3. greedy Spearman pruning (|rho|>0.85), keeping the more predictive twin; 4. walk-forward
permutation importance (log-loss increase on *future* folds, 8 repeats); keep a feature only if it
is positive in >=60% of folds; 5. SHAP (out-of-sample) for direction. Selection uses an **early
window only**; later data is reserved for evaluation.
