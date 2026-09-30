"""Pure functions turning NHL API payloads into flat rows. No I/O, so they are unit-testable.

The parsers are written defensively (``.get`` everywhere, tolerant of missing keys) because the
web API is unofficial and its shape drifts. Anything they cannot parse is skipped, never guessed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from nhlbet.data.xg import XGModel, is_high_danger
from nhlbet.teams import canon

FINAL_STATES = {"OFF", "FINAL"}
KEEP_GAME_TYPES = {2, 3}  # regular season, playoffs
SHOT_EVENTS = {"shot-on-goal": "sog", "missed-shot": "miss", "blocked-shot": "block", "goal": "goal"}
PENALTY_CODES = {"MIN", "MAJ", "BEN"}  # penalties that (normally) create a power play


def _name(v: Any) -> str | None:
    if isinstance(v, dict):
        return v.get("default")
    return v


def mmss(v: Any) -> int:
    """'18:22' -> 1102 seconds; tolerant of None / bad input."""
    try:
        m, s = str(v).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0


def parse_schedule_games(payload: dict) -> list[dict]:
    """Rows for ``games`` from a club-schedule-season or /schedule/{date} payload."""
    raw: list[dict] = list(payload.get("games", []))
    for day in payload.get("gameWeek", []):
        for g in day.get("games", []):
            g = dict(g)
            g.setdefault("gameDate", day.get("date"))
            raw.append(g)
    out, seen = [], set()
    for g in raw:
        gid = g.get("id")
        if gid is None or gid in seen or g.get("gameType") not in KEEP_GAME_TYPES:
            continue
        seen.add(gid)
        home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
        hs, as_ = home.get("score"), away.get("score")
        final = g.get("gameState") in FINAL_STATES and hs is not None and as_ is not None
        out.append({
            "game_id": int(gid), "season": g.get("season"), "game_type": g.get("gameType"),
            "game_date": g.get("gameDate") or str(g.get("startTimeUTC", ""))[:10],
            "start_utc": g.get("startTimeUTC"),
            "home": canon(home.get("abbrev", "")), "away": canon(away.get("abbrev", "")),
            "home_score": hs if final else None, "away_score": as_ if final else None,
            "status": g.get("gameState"),
            "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType"),
            "home_win": (int(hs > as_) if final else None),
            "source": "nhl_api", "updated_at": None,
        })
    return out


def parse_boxscore(box: dict) -> dict:
    """Goalie and skater rows from a boxscore payload."""
    gid = box.get("id")
    teams = {"home": canon(box.get("homeTeam", {}).get("abbrev", "")),
             "away": canon(box.get("awayTeam", {}).get("abbrev", ""))}
    goalies, skaters = [], []
    pbg = box.get("playerByGameStats", {})
    for side in ("home", "away"):
        block = pbg.get(f"{side}Team", {})
        team_goalies = block.get("goalies", [])
        has_flag = any("starter" in g for g in team_goalies)
        top_toi = max((mmss(g.get("toi")) for g in team_goalies), default=0)
        for g in team_goalies:
            toi = mmss(g.get("toi"))
            started = bool(g.get("starter")) if has_flag else (toi == top_toi and toi > 0)
            if toi == 0 and not started:
                continue  # dressed backup who did not play
            goalies.append({"game_id": gid, "team": teams[side], "player_id": g.get("playerId"),
                            "name": _name(g.get("name")), "started": int(started), "toi_sec": toi,
                            "shots_against": g.get("shotsAgainst"), "saves": g.get("saves"),
                            "goals_against": g.get("goalsAgainst"), "xg_faced": None})
        for grp in ("forwards", "defense"):
            for p in block.get(grp, []):
                toi = mmss(p.get("toi"))
                if toi <= 0:
                    continue
                skaters.append({"game_id": gid, "team": teams[side], "player_id": p.get("playerId"),
                                "name": _name(p.get("name")), "position": p.get("position"),
                                "toi_sec": toi, "goals": p.get("goals", 0), "assists": p.get("assists", 0),
                                "points": p.get("points", 0)})
    return {"goalies": goalies, "skaters": skaters}


def _sec(play: dict) -> int:
    per = (play.get("periodDescriptor") or {}).get("number", 1) or 1
    return (per - 1) * 1200 + mmss(play.get("timeInPeriod"))


def _strength(code: Any) -> tuple[int, int, int, int] | None:
    """situationCode 'abcd' = away goalie, away skaters, home skaters, home goalie."""
    s = str(code)
    if len(s) != 4 or not s.isdigit():
        return None
    a, b, c, d = (int(ch) for ch in s)
    return a, b, c, d


def parse_play_by_play(pbp: dict, xg_model: XGModel | None = None) -> dict:
    """Shot rows, per-team advanced stats and per-goalie xG faced from a play-by-play payload."""
    xgm = xg_model or XGModel()
    gid = pbp.get("id")
    hid, aid = pbp.get("homeTeam", {}).get("id"), pbp.get("awayTeam", {}).get("id")
    id2abbr = {hid: canon(pbp.get("homeTeam", {}).get("abbrev", "")),
               aid: canon(pbp.get("awayTeam", {}).get("abbrev", ""))}
    home, away = id2abbr[hid], id2abbr[aid]
    roster = {r.get("playerId"): r.get("teamId") for r in pbp.get("rosterSpots", [])}

    def opp(t: str) -> str:
        return away if t == home else home

    stats: dict[str, dict[str, float]] = {t: defaultdict(float) for t in (home, away)}
    shots, goalie_xg = [], defaultdict(float)
    pen_events: list[tuple[int, str, str]] = []  # (sec, team, code)

    for p in pbp.get("plays", []):
        key = p.get("typeDescKey")
        d = p.get("details") or {}
        if (p.get("periodDescriptor") or {}).get("periodType") == "SO":
            continue  # shootouts are not hockey; also excluded from goals stats
        if key in SHOT_EVENTS:
            kind = SHOT_EVENTS[key]
            owner = id2abbr.get(d.get("eventOwnerTeamId"))
            if owner is None:
                continue
            team = owner
            if kind == "block":  # attribute to the shooter's team, not the blocker's
                shooter_team = id2abbr.get(roster.get(d.get("shootingPlayerId")))
                team = shooter_team if shooter_team else opp(owner)
            x, y = d.get("xCoord"), d.get("yCoord")
            st = _strength(p.get("situationCode"))
            ev = int(bool(st) and st[1] == 5 and st[2] == 5 and st[0] == 1 and st[3] == 1)
            goalie = d.get("goalieInNetId")
            xg = None
            if kind != "block" and (goalie or kind == "miss"):
                xg = xgm.xg(x, y, d.get("shotType"), kind)
            hd = kind != "block" and x is not None and y is not None and is_high_danger(x, y)
            o = opp(team)
            s_for, s_ag = stats[team], stats[o]
            s_for["att_for"] += 1; s_ag["att_against"] += 1
            if ev:
                s_for["ev_att_for"] += 1; s_ag["ev_att_against"] += 1
            if kind in ("sog", "goal"):
                s_for["sog_for"] += 1; s_ag["sog_against"] += 1
            if kind != "block":
                s_for["fen_for"] += 1; s_ag["fen_against"] += 1
                if hd:
                    s_for["hd_for"] += 1; s_ag["hd_against"] += 1
                if xg is not None:
                    s_for["xg_for"] += xg; s_ag["xg_against"] += xg
                    if ev:
                        s_for["ev_xg_for"] += xg; s_ag["ev_xg_against"] += xg
            if kind in ("sog", "goal") and goalie and xg is not None:
                goalie_xg[goalie] += xg
            shots.append({"game_id": gid, "event_id": p.get("eventId"), "team": team,
                          "period": (p.get("periodDescriptor") or {}).get("number"), "sec": _sec(p),
                          "x": x, "y": y, "shot_type": d.get("shotType"), "kind": kind,
                          "is_goal": int(kind == "goal"), "ev": ev, "goalie_id": goalie, "xg": xg})
            if kind == "goal":
                st = _strength(p.get("situationCode"))
                if st:
                    own_sk, opp_sk, own_g = ((st[2], st[1], st[3]) if team == home else (st[1], st[2], st[0]))
                    if own_sk > opp_sk and own_g == 1:
                        stats[team]["pp_goals"] += 1
        elif key == "penalty":
            code = d.get("typeCode")
            owner = id2abbr.get(d.get("eventOwnerTeamId"))
            if owner and code in PENALTY_CODES:
                stats[owner]["pen_taken"] += 1
                stats[opp(owner)]["pen_drawn"] += 1
                pen_events.append((_sec(p), owner, code))

    # power-play opportunities: a penalty gives the opponent a PP unless offset by a simultaneous
    # penalty on the other team (coincident minors -> 4v4, no advantage)
    by_sec: dict[int, set[str]] = defaultdict(set)
    for sec, team, _ in pen_events:
        by_sec[sec].add(team)
    for sec, team, _ in pen_events:
        if len(by_sec[sec]) == 1:
            stats[opp(team)]["pp_opps"] += 1

    for t in (home, away):
        stats[t]["pk_opps"] = stats[opp(t)]["pp_opps"]
        stats[t]["pk_goals_against"] = stats[opp(t)]["pp_goals"]
    return {"home": home, "away": away, "shots": shots, "stats": {t: dict(s) for t, s in stats.items()},
            "goalie_xg": dict(goalie_xg)}


TEAM_STAT_COLS = ["att_for", "att_against", "sog_for", "sog_against", "fen_for", "fen_against", "hd_for",
                  "hd_against", "xg_for", "xg_against", "ev_att_for", "ev_att_against", "ev_xg_for",
                  "ev_xg_against", "pen_taken", "pen_drawn", "pp_opps", "pp_goals", "pk_opps",
                  "pk_goals_against"]


def team_game_rows(game: dict, pbp_parsed: dict | None) -> list[dict]:
    """Two ``team_game`` rows (home, away) combining the game result with parsed pbp stats."""
    rows = []
    for side, team, opp_t, gf, ga in (("home", game["home"], game["away"], game["home_score"], game["away_score"]),
                                      ("away", game["away"], game["home"], game["away_score"], game["home_score"])):
        row = {"game_id": game["game_id"], "team": team, "opp": opp_t, "is_home": int(side == "home"),
               "goals": gf, "goals_against": ga}
        st = (pbp_parsed or {}).get("stats", {}).get(team)
        for c in TEAM_STAT_COLS:
            row[c] = (st.get(c, 0.0) if st is not None else None)
        rows.append(row)
    return rows


def iter_chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
