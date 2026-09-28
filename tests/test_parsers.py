from nhlbet.data.parsers import mmss, parse_boxscore, parse_play_by_play, parse_schedule_games, team_game_rows
from tests import fakes as F

H, A = F.HOME_ID, F.AWAY_ID


def test_mmss():
    assert mmss("18:22") == 1102 and mmss(None) == 0 and mmss("bad") == 0


def test_schedule_parses_final_and_future_and_filters_types():
    payload = {"games": [F.schedule_game(1, hs=4, as_=1), F.schedule_game(2, state="FUT"), F.schedule_game(3, gtype=1),
                         F.schedule_game(4, home="TB", away="LA", period="SO", hs=2, as_=3)]}
    rows = {r["game_id"]: r for r in parse_schedule_games(payload)}
    assert set(rows) == {1, 2, 4}                      # preseason dropped
    assert rows[1]["home_win"] == 1 and rows[1]["home_score"] == 4
    assert rows[2]["home_score"] is None and rows[2]["home_win"] is None   # unplayed: no fake result
    assert rows[4]["home"] == "TBL" and rows[4]["away"] == "LAK" and rows[4]["last_period"] == "SO" and rows[4]["home_win"] == 0


def test_gameweek_payload_and_dedupe():
    day = {"date": "2023-10-10", "games": [F.schedule_game(1), F.schedule_game(1)]}
    assert len(parse_schedule_games({"gameWeek": [day]})) == 1


def test_boxscore_goalies_and_skaters():
    b = parse_boxscore(F.boxscore())
    starters = {g["team"]: g["player_id"] for g in b["goalies"] if g["started"]}
    assert starters == {"BOS": 31, "TOR": 41}
    assert not any(g["player_id"] == 32 for g in b["goalies"])          # unused backup dropped
    assert any(g["player_id"] == 42 for g in b["goalies"])              # relief goalie kept, not started
    assert {s["player_id"] for s in b["skaters"]} == {11, 13, 21}       # 0:00 TOI (scratch/DNP) excluded


def test_boxscore_starter_fallback_without_flag():
    b = parse_boxscore(F.boxscore(with_flag=False))
    starters = {g["team"]: g["player_id"] for g in b["goalies"] if g["started"]}
    assert starters == {"BOS": 31, "TOR": 41}                           # most TOI


def test_pbp_attempts_shots_and_blocked_attribution():
    plays = [
        F.play(1, "shot-on-goal", H, x=80, y=5, shotType="wrist", goalieInNetId=41, shootingPlayerId=1),
        F.play(2, "missed-shot", A, x=-70, y=10, shotType="snap", shootingPlayerId=2),
        # blocked: owner reported as the BLOCKING team (TOR) but the shooter (player 1) is BOS -> BOS attempt
        F.play(3, "blocked-shot", A, x=60, y=0, shootingPlayerId=1),
        F.play(4, "goal", H, x=85, y=2, shotType="tip-in", goalieInNetId=41, scoringPlayerId=1),
        F.play(5, "shot-on-goal", A, x=-80, y=0, sit="1551", shotType="wrist", goalieInNetId=31, shootingPlayerId=2, period=4),
        F.play(6, "goal", H, period=5, x=89, y=0, goalieInNetId=41),                    # shootout: ignored
    ]
    r = parse_play_by_play(F.pbp(plays))
    h, a = r["stats"]["BOS"], r["stats"]["TOR"]
    assert h["att_for"] == 3 and h["att_against"] == 2                  # BOS: sog, blocked(attributed), goal
    assert h["sog_for"] == 2 and a["sog_for"] == 1
    assert h["fen_for"] == 2                                            # blocked shot excluded from Fenwick
    assert h["hd_for"] >= 1
    assert len(r["shots"]) == 5                                         # shootout excluded
    assert r["goalie_xg"][41] > 0 and r["goalie_xg"][31] > 0
    assert next(s for s in r["shots"] if s["event_id"] == 3)["xg"] is None   # no xG for blocked attempts


def test_pbp_empty_net_goal_excluded_from_goalie_xg_and_ev_flag():
    plays = [F.play(1, "goal", H, x=80, y=0, sit="0651", shotType="wrist", scoringPlayerId=1),   # away goalie pulled, no goalieInNetId
             F.play(2, "shot-on-goal", H, x=80, y=0, sit="1551", goalieInNetId=41, shotType="wrist")]
    r = parse_play_by_play(F.pbp(plays))
    en = next(s for s in r["shots"] if s["event_id"] == 1)
    assert en["xg"] is None and en["ev"] == 0
    assert list(r["goalie_xg"]) == [41]
    assert next(s for s in r["shots"] if s["event_id"] == 2)["ev"] == 1


def test_pbp_special_teams():
    plays = [
        F.play(1, "penalty", A, t="02:00", typeCode="MIN", duration=2),                  # TOR minor -> BOS PP
        F.play(2, "goal", H, t="02:40", sit="1451", x=80, y=0, goalieInNetId=41, shotType="wrist"),  # BOS PP goal (5 v 4)
        F.play(3, "penalty", H, t="10:00", typeCode="MIN", duration=2),                  # coincident minors -> no PP
        F.play(4, "penalty", A, t="10:00", typeCode="MIN", duration=2),
        F.play(5, "penalty", H, t="15:00", typeCode="MIS", duration=10),                 # misconduct: not counted
        F.play(6, "goal", A, t="16:00", sit="1551", x=-80, y=0, goalieInNetId=31, shotType="wrist"),
    ]
    st = parse_play_by_play(F.pbp(plays))["stats"]
    assert st["BOS"]["pp_opps"] == 1 and st["BOS"]["pp_goals"] == 1
    assert st["TOR"]["pk_opps"] == 1 and st["TOR"]["pk_goals_against"] == 1
    assert st["TOR"].get("pp_opps", 0) == 0
    assert st["BOS"]["pen_taken"] == 1 and st["TOR"]["pen_taken"] == 2 and st["BOS"]["pen_drawn"] == 2


def test_team_game_rows_shape():
    game = parse_schedule_games({"games": [F.schedule_game(1, hs=3, as_=2)]})[0]
    rows = team_game_rows(game, parse_play_by_play(F.pbp([])))
    assert [r["team"] for r in rows] == ["BOS", "TOR"] and rows[0]["goals"] == 3 and rows[1]["goals_against"] == 3
    assert team_game_rows(game, None)[0]["att_for"] is None
