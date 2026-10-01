"""Hand-built payloads that mimic the api-web.nhle.com structure (built from the documented shape;
NOT recorded from the live API, which is unreachable from the development sandbox)."""
from __future__ import annotations

HOME_ID, AWAY_ID = 6, 10  # BOS, TOR


def schedule_game(gid=2023020001, home="BOS", away="TOR", date="2023-10-10", state="OFF", hs=3, as_=2, gtype=2, period="REG"):
    g = {"id": gid, "season": 20232024, "gameType": gtype, "gameDate": date, "startTimeUTC": f"{date}T23:00:00Z",
         "gameState": state, "homeTeam": {"abbrev": home, "score": hs}, "awayTeam": {"abbrev": away, "score": as_}}
    if state in ("OFF", "FINAL"):
        g["gameOutcome"] = {"lastPeriodType": period}
    else:
        g["homeTeam"].pop("score"); g["awayTeam"].pop("score")
    return g


def play(eid, key, owner, period=1, t="05:00", x=None, y=None, sit="1551", **det):
    d = {"eventOwnerTeamId": owner, **det}
    if x is not None:
        d.update(xCoord=x, yCoord=y)
    return {"eventId": eid, "typeDescKey": key, "periodDescriptor": {"number": period, "periodType": "REG" if period < 4 else ("OT" if period == 4 else "SO")},
            "timeInPeriod": t, "situationCode": sit, "details": d}


def pbp(plays):
    return {"id": 2023020001, "homeTeam": {"id": HOME_ID, "abbrev": "BOS"}, "awayTeam": {"id": AWAY_ID, "abbrev": "TOR"},
            "rosterSpots": [{"playerId": 1, "teamId": HOME_ID}, {"playerId": 2, "teamId": AWAY_ID}], "plays": plays}


def boxscore(with_flag=True):
    def goalie(pid, toi, sa, sv, ga, starter):
        g = {"playerId": pid, "name": {"default": f"G{pid}"}, "toi": toi, "shotsAgainst": sa, "saves": sv, "goalsAgainst": ga}
        if with_flag:
            g["starter"] = starter
        return g

    def skater(pid, toi, pts):
        return {"playerId": pid, "name": {"default": f"S{pid}"}, "position": "C", "toi": toi, "goals": pts, "assists": 0, "points": pts, "sog": pid % 4 + 1}

    return {"id": 2023020001, "homeTeam": {"abbrev": "BOS"}, "awayTeam": {"abbrev": "TOR"},
            "playerByGameStats": {
                "homeTeam": {"forwards": [skater(11, "18:00", 1), skater(12, "00:00", 0)], "defense": [skater(13, "22:10", 0)],
                             "goalies": [goalie(31, "60:00", 30, 28, 2, True), goalie(32, "00:00", 0, 0, 0, False)]},
                "awayTeam": {"forwards": [skater(21, "17:00", 0)], "defense": [],
                             "goalies": [goalie(41, "40:00", 20, 17, 3, True), goalie(42, "20:00", 10, 10, 0, False)]}}}


class FakeResp:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._d, self.headers = status, data, headers or {}
        self.text = str(data)

    def json(self):
        return self._d


class FakeSession:
    """Maps URL substring -> payload (or list of responses to pop in order)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, timeout=None, params=None):
        self.calls.append(url)
        self.params = params
        for frag, val in self.routes.items():
            if frag in url:
                if isinstance(val, list):
                    r = val.pop(0) if len(val) > 1 else val[0]
                else:
                    r = FakeResp(200, val)
                if isinstance(r, Exception):
                    raise r
                return r
        return FakeResp(404, {})


def odds_event(eid="e1", home="Boston Bruins", away="Toronto Maple Leafs", commence="2023-10-11T23:00:00Z", books=None, extra_markets=False):
    """Odds API v4 event. books: [(key, home_dec, away_dec, last_update)]."""
    books = books or [("bookA", 1.80, 2.10, "2023-10-11T15:00:00Z"), ("bookB", 1.87, 2.00, "2023-10-11T15:00:00Z")]
    out = []
    for key, h, a, upd in books:
        markets = [{"key": "h2h", "last_update": upd, "outcomes": [{"name": home, "price": h}, {"name": away, "price": a}]}]
        if extra_markets:
            markets += [{"key": "spreads", "last_update": upd, "outcomes": [{"name": home, "price": 2.2, "point": -1.5}, {"name": away, "price": 1.7, "point": 1.5}]},
                        {"key": "totals", "last_update": upd, "outcomes": [{"name": "Over", "price": 1.9, "point": 6.5}, {"name": "Under", "price": 1.9, "point": 6.5}]}]
        out.append({"key": key, "title": key, "last_update": upd, "markets": markets})
    return {"id": eid, "sport_key": "icehockey_nhl", "commence_time": commence, "home_team": home, "away_team": away, "bookmakers": out}
