"""Static team metadata: arena location, time zone, division; abbreviation aliases."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    abbrev: str
    name: str
    lat: float
    lon: float
    tz: int  # standard-time UTC offset (hours); DST shifts all NA zones together so differences hold
    division: str


_T = [
    Team("ANA", "Anaheim Ducks", 33.808, -117.877, -8, "Pacific"),
    Team("ARI", "Arizona Coyotes", 33.425, -111.933, -7, "Central"),
    Team("BOS", "Boston Bruins", 42.366, -71.062, -5, "Atlantic"),
    Team("BUF", "Buffalo Sabres", 42.875, -78.876, -5, "Atlantic"),
    Team("CGY", "Calgary Flames", 51.037, -114.052, -7, "Pacific"),
    Team("CAR", "Carolina Hurricanes", 35.803, -78.722, -5, "Metropolitan"),
    Team("CHI", "Chicago Blackhawks", 41.881, -87.674, -6, "Central"),
    Team("COL", "Colorado Avalanche", 39.749, -105.008, -7, "Central"),
    Team("CBJ", "Columbus Blue Jackets", 39.969, -83.006, -5, "Metropolitan"),
    Team("DAL", "Dallas Stars", 32.790, -96.810, -6, "Central"),
    Team("DET", "Detroit Red Wings", 42.341, -83.055, -5, "Atlantic"),
    Team("EDM", "Edmonton Oilers", 53.547, -113.498, -7, "Pacific"),
    Team("FLA", "Florida Panthers", 26.158, -80.326, -5, "Atlantic"),
    Team("LAK", "Los Angeles Kings", 34.043, -118.267, -8, "Pacific"),
    Team("MIN", "Minnesota Wild", 44.945, -93.101, -6, "Central"),
    Team("MTL", "Montreal Canadiens", 45.496, -73.569, -5, "Atlantic"),
    Team("NSH", "Nashville Predators", 36.159, -86.779, -6, "Central"),
    Team("NJD", "New Jersey Devils", 40.734, -74.171, -5, "Metropolitan"),
    Team("NYI", "New York Islanders", 40.723, -73.590, -5, "Metropolitan"),
    Team("NYR", "New York Rangers", 40.751, -73.994, -5, "Metropolitan"),
    Team("OTT", "Ottawa Senators", 45.297, -75.927, -5, "Atlantic"),
    Team("PHI", "Philadelphia Flyers", 39.901, -75.172, -5, "Metropolitan"),
    Team("PIT", "Pittsburgh Penguins", 40.439, -79.989, -5, "Metropolitan"),
    Team("SEA", "Seattle Kraken", 47.622, -122.354, -8, "Pacific"),
    Team("SJS", "San Jose Sharks", 37.333, -121.901, -8, "Pacific"),
    Team("STL", "St. Louis Blues", 38.627, -90.203, -6, "Central"),
    Team("TBL", "Tampa Bay Lightning", 27.943, -82.452, -5, "Atlantic"),
    Team("TOR", "Toronto Maple Leafs", 43.643, -79.379, -5, "Atlantic"),
    Team("UTA", "Utah Hockey Club", 40.768, -111.901, -7, "Central"),
    Team("VAN", "Vancouver Canucks", 49.278, -123.109, -8, "Pacific"),
    Team("VGK", "Vegas Golden Knights", 36.103, -115.178, -8, "Pacific"),
    Team("WPG", "Winnipeg Jets", 49.893, -97.144, -6, "Central"),
    Team("WSH", "Washington Capitals", 38.898, -77.021, -5, "Metropolitan"),
]
TEAMS: dict[str, Team] = {t.abbrev: t for t in _T}

ALIASES = {"TB": "TBL", "LA": "LAK", "NJ": "NJD", "SJ": "SJS", "VEG": "VGK", "PHX": "ARI",
           "WIN": "WPG", "TBL.": "TBL", "T.B": "TBL", "L.A": "LAK", "N.J": "NJD", "S.J": "SJS"}

# Curated, deliberately short list of historic rivalries outside of same-division pairs.
RIVALRIES = {frozenset(p) for p in [
    ("BOS", "MTL"), ("TOR", "MTL"), ("EDM", "CGY"), ("PIT", "PHI"), ("NYR", "NYI"),
    ("NYR", "NJD"), ("CHI", "STL"), ("COL", "DAL"), ("VAN", "SEA"), ("TOR", "OTT"),
    ("WSH", "PIT"), ("BOS", "TOR"), ("LAK", "ANA"), ("SJS", "VGK"),
]}


def canon(abbrev: str) -> str:
    """Normalise a team abbreviation to the NHL API's three-letter code."""
    a = str(abbrev).strip().upper()
    return ALIASES.get(a, a)


def is_rivalry(a: str, b: str) -> bool:
    a, b = canon(a), canon(b)
    if frozenset((a, b)) in RIVALRIES:
        return True
    ta, tb = TEAMS.get(a), TEAMS.get(b)
    return bool(ta and tb and ta.division == tb.division)


def haversine_km(a: str, b: str) -> float:
    """Great-circle distance between two teams' arenas in km (0 if unknown)."""
    from math import asin, cos, radians, sin, sqrt

    ta, tb = TEAMS.get(canon(a)), TEAMS.get(canon(b))
    if not ta or not tb:
        return 0.0
    la1, lo1, la2, lo2 = map(radians, (ta.lat, ta.lon, tb.lat, tb.lon))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def tz_shift(frm: str, to: str) -> int:
    """Signed time-zone hours crossed travelling from one arena to another."""
    tf, tt = TEAMS.get(canon(frm)), TEAMS.get(canon(to))
    return (tt.tz - tf.tz) if tf and tt else 0
