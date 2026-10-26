"""Phase 1: feature inventory. For every feature the builder can emit: its data source and the moment its value becomes knowable.

The table is hand-written (it is a claim about the code, checked against it) and this script FAILS if the builder emits a base feature the table does not cover,
so the inventory cannot silently go stale. Writes audit/data/feature_inventory.csv and audit/data/feature_inventory.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nhlbet.features.builder import GAME_LEVEL, GROUPS, NO_DIFF  # noqa: E402

SCORES = ("games.home_score/away_score (NHL schedule/boxscore)", "final horn of the team's previous game, plus the API's lag until the score is final (minutes)")
PBP = ("team_game columns derived from play-by-play shot events (api-web.nhle.com play-by-play) and xG geometry (nhlbet/data/xg.py, fixed coefficients)",
       "after the previous game's play-by-play is final; coordinates and event attribution can be corrected for ~24-48 h afterwards, so the first-fetched value may differ from the settled one")
BOX = ("team_game / goalie_game columns from the boxscore (api-web.nhle.com boxscore)", "final horn of the previous game, plus API lag; boxscore corrections possible within ~24 h")
SCHED = ("schedule dates and venues (games table) and static geography (nhlbet/teams.py)", "when the schedule is published (weeks ahead); postponements change it late")

# base feature -> (group note, source, knowledge time, windowing, leakage note)
T = {
 "gd_ewm": (*SCORES, "EWM half-life 10 over the last 40 games", "past games only"),
 "gf_ewm": (*SCORES, "EWM over last 40", "past games only"),
 "ga_ewm": (*SCORES, "EWM over last 40", "past games only"),
 "gd_l10": (*SCORES, "last 10 games", "past games only"),
 "win_pct_l10": (*SCORES, "last 10 games", "regulation+OT result of past games only"),
 "gd_season_shrunk": (*SCORES, "season to date, shrunk (k=15) toward the previous season's 50/50 blend with the league mean", "previous-season prior is a completed season"),
 "gf_season_shrunk": (*SCORES, "as above", "as above"),
 "ga_season_shrunk": (*SCORES, "as above", "as above"),
 "gd_venue": (*SCORES, "season-to-date same-venue goal difference, shrunk (k=10)", "past games only"),
 "elo": ("Elo ratings updated from final scores (nhlbet/features/elo.py), K=6, home adv 35, regress 0.30", "ratings snapshot taken before any game of the date is folded in", "all prior dates", "snapshot copied before the date's updates"),
 "sos10": (*SCORES, "mean opponent Elo over the last 10 games", "uses opponent Elo as of each past game"),
 "cf_pct": (*PBP, "last 20 games, ratio of sums", "past games only"),
 "ev_cf_pct": (*PBP, "last 20", "past games only"),
 "ff_pct": (*PBP, "last 20", "past games only"),
 "sog_share": (*BOX, "last 20", "past games only"),
 "hd_share": (*PBP, "last 20", "past games only"),
 "xg_share": (*PBP, "last 20", "past games only"),
 "xgf_pg": (*PBP, "last 20, needs 5 valid games", "past games only"),
 "xga_pg": (*PBP, "last 20, needs 5 valid games", "past games only"),
 "pdo_dev": (*PBP, "last 20, shrunk to league mean", "past games only"),
 "sh_pct": (*PBP, "last 20, shrunk", "past games only"),
 "sv_pct_team": (*PBP, "last 20, shrunk", "past games only"),
 "adv_n": (*PBP, "count of valid advanced-stat games in the last 20", "past games only"),
 "pp_pct": (*BOX, "last 30 games, shrunk to 20%", "past games only"),
 "pk_pct": (*BOX, "last 30 games, shrunk to 20%", "past games only"),
 "st_net": (*BOX, "pp_pct + pk_pct - 1", "past games only"),
 "pen_taken_pg": (*BOX, "last 30 games", "past games only"),
 "pen_drawn_pg": (*BOX, "last 30 games", "past games only"),
 "pen_diff_pg": (*BOX, "last 30 games", "past games only"),
 "g_sv_season": ("goalie_game saves/shots against from boxscores of the goalies' past starts, weighted by P(start)", "after each prior game; WHICH goalie starts tonight is a probability in 'probable' mode (default)", "season to date, shrunk (400 shots)", "in 'actual' mode the starter identity comes from tonight's boxscore: knowable only once the starter is announced (~1 h before puck drop), NOT at a morning decision"),
 "g_sv_l5": ("as g_sv_season", "as above", "last 5 starts, shrunk (150)", "same actual-mode caveat"),
 "g_sv_l10": ("as g_sv_season", "as above", "last 10 starts, shrunk (250)", "same actual-mode caveat"),
 "g_gsax100_season": ("goalie_game xg_faced (PBP geometry) minus goals against", "after each prior game", "season to date", "same actual-mode caveat"),
 "g_gsax100_l10": ("as above", "as above", "last 10 starts", "same actual-mode caveat"),
 "g_rest": ("goalie's previous appearance date", "after the previous game", "capped at 14 days", "same actual-mode caveat"),
 "g_b2b": ("goalie's previous appearance date", "after the previous game", "binary", "same actual-mode caveat"),
 "g_p_primary": ("probability the team's most frequent recent starter plays: league rate of primary starters on rest / back-to-back, from past games", "after each prior game", "running league counters", "league counters are updated after each date, never before"),
 "g_confirmed": ("0 in 'probable' mode, 1 when the starter identity is supplied (live: manual/confirmed file with first-seen timestamp)", "live: the moment the confirmation is first seen (stored); backtest 'actual' mode: boxscore after the game", "binary", "in a backtest 'actual' mode this flag is set from after-the-fact data"),
 "rest_days": (*SCHED, "days since the previous game, capped at 7", "none"),
 "b2b": (*SCHED, "rest == 1", "none"),
 "g3in4": (*SCHED, "games in the last 3 days among the last 4", "none"),
 "g4in6": (*SCHED, "games in the last 5 days among the last 5", "none"),
 "travel_km": (*SCHED, "haversine from the previous game's venue", "none"),
 "tz_shift": (*SCHED, "timezone hours from the previous venue", "none"),
 "road_trip_n": (*SCHED, "consecutive away games including tonight", "none"),
 "homestand_n": (*SCHED, "consecutive home games including tonight", "none"),
 "games_last_7d": (*SCHED, "count", "none"),
 "pts_share_missing": ("skater_game: who dressed tonight (boxscore)", "lineups: ~1 h before puck drop at the earliest (live: no confirmation source exists)", "last 20 games", "OFF in training and live (lineup_mode='off'); in 'actual' mode it is after-the-fact"),
 "core_missing_n": ("as pts_share_missing", "as above", "as above", "as above"),
 "gp_season": (*SCORES, "count of games this season", "none"),
 "early_w": (*SCORES, "n/(n+20)", "none"),
 "cutoff_gap": (*SCORES, "own points% minus the 8th-best conference points% (needs 10 games)", "table built from regular-season results of earlier dates"),
 "stakes": (*SCORES, "binary: within 0.06 of the cutoff after 60 games", "as above"),
 "elo_prob": ("Elo expectation from the pre-date snapshot", "as elo", "none", "none"),
 "is_rivalry": ("static list in nhlbet/teams.py", "always known", "none", "none"),
 "is_playoff": ("games.game_type", "schedule", "none", "none"),
 "days_into_season": ("games table: first game date of the season in the data", "schedule", "none", "if the data begins mid-season the zero point is wrong (not a leak)"),
}


def main() -> None:
    base = {b for bs in GROUPS.values() for b in bs} | set(GAME_LEVEL)
    missing = sorted(base - set(T))
    extra = sorted(set(T) - base)
    if missing or extra:
        raise SystemExit(f"inventory out of date: missing {missing}, stale {extra}")
    sel = json.loads((ROOT / "data/models/feature_selection.json").read_text())["kept"]
    used = {c[2:] if c[:2] in ("h_", "a_", "d_") else c for c in sel}
    gof = {b: g for g, bs in GROUPS.items() for b in bs} | {k: v for k, v in GAME_LEVEL.items()}
    rows = []
    for b, v in T.items():
        src, when, win, note = v
        rows.append({"feature": b, "group": gof[b], "emitted_as": "h_/a_ and d_ (home minus away)" if b not in NO_DIFF and b not in GAME_LEVEL else "h_ and a_" if b in NO_DIFF else "game level",
                     "in_production_model": b in used, "source": src, "knowable_when": when, "window": win, "leak_note": note})
    out = ROOT / "audit" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "feature_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| Feature | Group | In production model | Source | Knowable when | Window | Leak note |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['feature']} | {r['group']} | {'yes' if r['in_production_model'] else 'no'} | {r['source']} | {r['knowable_when']} | {r['window']} | {r['leak_note']} |" for r in rows]
    (out / "feature_inventory.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} base features, {sum(r['in_production_model'] for r in rows)} in the production model; wrote audit/data/feature_inventory.(csv|md)")


if __name__ == "__main__":
    main()
