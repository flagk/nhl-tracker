"""Validate the NHL API parsers against the LIVE API (needs network access to api-web.nhle.com).

Fetches several completed games and cross-checks what we parse from play-by-play against the
official numbers in the boxscore/schedule. Exit code 1 if any check fails, so it can gate CI.

    python scripts/check_api.py --season 20242025 --team BOS --n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nhlbet.data.client import NHLClient
from nhlbet.data.parsers import parse_boxscore, parse_play_by_play, parse_schedule_games

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=20242025)
ap.add_argument("--team", default="BOS")
ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()

c = NHLClient()
games = [g for g in parse_schedule_games(c.club_schedule_season(a.team, a.season)) if g["home_score"] is not None]
print(f"{len(games)} completed games for {a.team} {a.season}")
bad = 0
for g in games[-a.n:]:
    box, pbp = c.boxscore(g["game_id"]), c.play_by_play(g["game_id"])
    p, b = parse_play_by_play(pbp), parse_boxscore(box)
    home, away = p["stats"][g["home"]], p["stats"][g["away"]]
    goals_h = sum(s["is_goal"] for s in p["shots"] if s["team"] == g["home"])
    goals_a = sum(s["is_goal"] for s in p["shots"] if s["team"] == g["away"])
    reg_score_ok = (goals_h, goals_a) == (g["home_score"], g["away_score"]) or g["last_period"] == "SO"
    box_sog_h, box_sog_a = box["homeTeam"].get("sog"), box["awayTeam"].get("sog")
    sog_ok = (home.get("sog_for") == box_sog_h and away.get("sog_for") == box_sog_a) if box_sog_h is not None else None
    starters = [x for x in b["goalies"] if x["started"]]
    checks = {
        "pbp goals == official score (SO games excepted)": reg_score_ok,
        "pbp SOG == boxscore SOG": sog_ok,
        "exactly 2 starting goalies": len(starters) == 2,
        # ingestion joins play-by-play xG to boxscore goalies by player id; this proves the two payloads use the same ids
        "goalie xG faced joins by player id": all(p["goalie_xg"].get(x["player_id"]) is not None for x in starters if (x["shots_against"] or 0) > 0),
        "attempts > SOG > goals": home.get("att_for", 0) >= home.get("sog_for", 0) >= goals_h,
        "skaters parsed (>=30)": len(b["skaters"]) >= 30,
    }
    print(f"\n{g['game_date']} {g['away']} @ {g['home']}  {g['away_score']}-{g['home_score']}  "
          f"CF% {home['att_for'] / max(1, home['att_for'] + home['att_against']):.2f}  xG {home.get('xg_for', 0):.2f}-{home.get('xg_against', 0):.2f}  "
          f"PP {home.get('pp_goals', 0):.0f}/{home.get('pp_opps', 0):.0f}")
    for k, v in checks.items():
        print(f"   [{'PASS' if v else ('SKIP' if v is None else 'FAIL')}] {k}")
        bad += v is False
print("\nALL CHECKS PASSED" if not bad else f"\n{bad} CHECK(S) FAILED - parsers need adjusting to the live payload")
sys.exit(1 if bad else 0)
