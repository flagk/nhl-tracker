"""Leak-free, as-of-game-date feature construction.

Design guarantee (tested in ``tests/test_no_leakage.py``):
    the features of a game depend only on games with a strictly earlier date.

How it is enforced structurally: games are processed one *calendar date* at a time. For a given
date, features for **every** game are emitted first from the team/goalie/league state, and only
afterwards are that date's results folded into the state. Results are never read while emitting.

Two optional "late-information" switches use identities known shortly before puck drop but not
in the morning: ``goalie_mode='actual'`` (who started) and ``lineup_mode='actual'`` (who dressed).
Even then only *identities* are used, never tonight's stats or score.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nhlbet.features.elo import Elo
from nhlbet.teams import TEAMS, haversine_km, is_rivalry, tz_shift

NAN = float("nan")
LEAGUE_SV, LEAGUE_SH = 0.905, 0.095

GROUPS: dict[str, list[str]] = {
    "strength": ["gd_ewm", "gf_ewm", "ga_ewm", "gd_l10", "win_pct_l10", "gd_season_shrunk", "gf_season_shrunk",
                 "ga_season_shrunk", "gd_venue", "elo", "sos10"],
    "advanced": ["cf_pct", "ev_cf_pct", "ff_pct", "sog_share", "hd_share", "xg_share", "xgf_pg", "xga_pg",
                 "pdo_dev", "sh_pct", "sv_pct_team", "adv_n"],
    "special_teams": ["pp_pct", "pk_pct", "st_net", "pen_taken_pg", "pen_drawn_pg", "pen_diff_pg"],
    "goaltending": ["g_sv_season", "g_sv_l5", "g_sv_l10", "g_gsax100_season", "g_gsax100_l10", "g_rest", "g_b2b",
                    "g_p_primary", "g_confirmed"],
    "rest_schedule": ["rest_days", "b2b", "g3in4", "g4in6", "travel_km", "tz_shift", "road_trip_n",
                      "homestand_n", "games_last_7d"],
    "lineup": ["pts_share_missing", "core_missing_n"],
    "situational_team": ["gp_season", "early_w", "cutoff_gap", "stakes"],
}
NO_DIFF = {"b2b", "g3in4", "g4in6", "g_b2b", "g_confirmed", "stakes", "adv_n", "gp_season", "early_w",
           "road_trip_n", "homestand_n", "tz_shift", "g_p_primary", "core_missing_n"}
GAME_LEVEL = {"elo_prob": "strength", "is_rivalry": "situational", "is_playoff": "situational",
              "days_into_season": "situational"}
META = ["game_date", "season", "game_type", "home", "away", "home_score", "away_score", "home_win", "start_utc"]


def feature_group_map(columns: list[str]) -> dict[str, str]:
    """column -> group name (``h_``/``a_``/``d_`` prefixes stripped)."""
    base_group = {b: g for g, bs in GROUPS.items() for b in bs}
    out = {}
    for c in columns:
        b = c[2:] if c[:2] in ("h_", "a_", "d_") else c
        if b in base_group:
            out[c] = base_group[b]
        elif c in GAME_LEVEL:
            out[c] = GAME_LEVEL[c]
        elif c.startswith("mkt_"):
            out[c] = "market"
    return out


@dataclass
class BuilderConfig:
    halflife: float = 10.0
    goalie_mode: str = "probable"   # 'probable' (morning-safe) | 'actual' (late: starter identity known)
    lineup_mode: str = "off"        # 'off' (morning-safe) | 'actual' (late: dressed players known)
    shrink_k: float = 15.0
    adv_window: int = 20
    elo_k: float = 6.0
    elo_home_adv: float = 35.0
    elo_regress: float = 0.30


@dataclass
class Rec:
    day: int
    season: int
    is_home: bool
    opp: str
    venue: str
    gf: int
    ga: int
    otl: bool
    opp_elo: float
    adv: dict
    skaters: dict
    starter: int | None


@dataclass
class TeamState:
    recs: list = field(default_factory=list)
    prev: dict = field(default_factory=dict)  # previous-season summary priors
    starts: list = field(default_factory=list)  # [(day, goalie_id)] of started games


@dataclass
class GoalieState:
    apps: list = field(default_factory=list)  # (day, season, sa, saves, ga, xg_faced)


def _ewm(vals: list[float], hl: float) -> float:
    a = np.asarray(vals, float)
    if a.size == 0:
        return NAN
    w = 0.5 ** ((a.size - 1 - np.arange(a.size)) / hl)
    m = ~np.isnan(a)
    return float(np.sum(a[m] * w[m]) / np.sum(w[m])) if m.any() else NAN


def _ratio(recs: list[Rec], num: str, den: str, min_games: int = 5) -> float:
    n = d = 0.0
    k = 0
    for r in recs:
        a, b = r.adv.get(num), r.adv.get(den)
        if a is None or b is None or a != a or b != b:
            continue
        n += a; d += b; k += 1
    return n / d if k >= min_games and d > 0 else NAN


def _share(recs: list[Rec], f: str, a: str, min_games: int = 5) -> float:
    xf = xa = 0.0
    k = 0
    for r in recs:
        u, v = r.adv.get(f), r.adv.get(a)
        if u is None or v is None or u != u or v != v:
            continue
        xf += u; xa += v; k += 1
    return xf / (xf + xa) if k >= min_games and (xf + xa) > 0 else NAN


class FeatureBuilder:
    def __init__(self, config: BuilderConfig | None = None) -> None:
        self.cfg = config or BuilderConfig()
        self._season_start = 0

    # ---------------------------------------------------------------------------------------
    def build(self, tables: dict[str, pd.DataFrame], market: pd.DataFrame | None = None,
              starter_override: dict | None = None) -> pd.DataFrame:
        cfg = self.cfg
        games = tables["games"].sort_values(["game_date", "game_id"]).reset_index(drop=True)
        if games.empty:
            return pd.DataFrame()
        tg = tables["team_game"].set_index(["game_id", "team"])
        adv_cols = [c for c in tg.columns if c not in ("opp", "is_home", "goals", "goals_against")]
        adv_lookup = {k: {c: v for c, v in row.items() if c in adv_cols} for k, row in tg[adv_cols].to_dict("index").items()}
        gg = tables.get("goalie_game", pd.DataFrame())
        sk = tables.get("skater_game", pd.DataFrame())
        starter_of, appear = {}, defaultdict(list)
        if len(gg):
            for r in gg.itertuples(index=False):
                appear[r.game_id].append(r)
                if r.started:
                    starter_of[(r.game_id, r.team)] = r.player_id
        if starter_override:  # confirmed starters for upcoming games: {(game_id, team): goalie_id}; identity only
            starter_of.update({k: v for k, v in starter_override.items() if k not in starter_of})
        skaters_of: dict[tuple[int, str], dict] = defaultdict(dict)
        if len(sk):
            for r in sk.itertuples(index=False):
                skaters_of[(r.game_id, r.team)][r.player_id] = r.points or 0

        teams: dict[str, TeamState] = defaultdict(TeamState)
        goalies: dict[int, GoalieState] = defaultdict(GoalieState)
        elo = Elo(k=cfg.elo_k, home_adv=cfg.elo_home_adv, regress=cfg.elo_regress)
        league = defaultdict(float)  # running league counters (goalie start rates, goals)
        season_pts: dict[tuple[int, str], list] = defaultdict(lambda: [0, 0])  # (season, team) -> [pts, gp]
        pending: dict[tuple[int, str], tuple] = {}
        cur_season = None
        rows = []

        for day_ts, day_games in games.groupby("game_date", sort=True):
            day = day_ts.toordinal()
            season = int(day_games.season.iloc[0])
            if season != cur_season:
                if cur_season is not None:
                    self._roll_season(teams, season_pts, cur_season)
                    elo.new_season()
                cur_season = season
                self._season_start = day
            snapshot = dict(elo.ratings)
            table = self._cutoff_table(season_pts, season)
            # ---- 1) emit features for every game on this date, from pre-date state only ----
            for g in day_games.itertuples(index=False):
                row = self._game_row(g, day, season, teams, goalies, elo, snapshot, table, league, adv_lookup,
                                     starter_of, skaters_of, pending, cfg)
                rows.append(row)
            # ---- 2) only now fold this date's completed results into the state ----
            for g in day_games.itertuples(index=False):
                if g.home_score != g.home_score or g.home_score is None:
                    continue  # not played yet
                self._update(g, day, season, teams, goalies, elo, snapshot, league, season_pts, adv_lookup,
                             starter_of, skaters_of, appear, pending)

        out = pd.DataFrame(rows).set_index("game_id")
        if market is not None and len(market):
            from nhlbet.features.market import attach_market_features
            out = attach_market_features(out, market)
        return out

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _roll_season(teams, season_pts, old_season) -> None:
        gf_all = ga_all = n_all = 0.0
        for t, ts in teams.items():
            rs = [r for r in ts.recs if r.season == old_season]
            if rs:
                gf_all += sum(r.gf for r in rs); ga_all += sum(r.ga for r in rs); n_all += len(rs)
        lg = gf_all / n_all if n_all else 3.0
        for t, ts in teams.items():
            rs = [r for r in ts.recs if r.season == old_season]
            if rs:
                n = len(rs)
                ts.prev = {"gf": 0.5 * sum(r.gf for r in rs) / n + 0.5 * lg,
                           "ga": 0.5 * sum(r.ga for r in rs) / n + 0.5 * lg}
            else:
                ts.prev = {"gf": lg, "ga": lg}

    @staticmethod
    def _cutoff_table(season_pts, season) -> dict[str, dict]:
        """As-of points% per team and the conference playoff-cutoff pace (8th best points%)."""
        pct = {}
        for (s, t), (pts, gp) in season_pts.items():
            if s == season and gp >= 10:
                pct[t] = pts / (2 * gp)
        cutoff = {}
        for conf, divs in (("E", {"Atlantic", "Metropolitan"}), ("W", {"Central", "Pacific"})):
            vals = sorted((v for t, v in pct.items() if TEAMS[t].division in divs), reverse=True)
            cutoff[conf] = vals[7] if len(vals) >= 8 else NAN
        return {"pct": pct, "cutoff": cutoff}

    # ---------------------------------------------------------------------------------------
    def _game_row(self, g, day, season, teams, goalies, elo, snapshot, table, league, adv_lookup,
                  starter_of, skaters_of, pending, cfg) -> dict:
        home, away = g.home, g.away
        feats = {}
        for side, team, opp in (("h", home, away), ("a", away, home)):
            f = self._team_feats(team, opp, side == "h", g, day, season, teams, goalies, snapshot, table, league,
                                 starter_of, skaters_of, pending, cfg)
            for k, v in f.items():
                feats[f"{side}_{k}"] = v
        for base in GROUPS_FLAT:
            if base in NO_DIFF:
                continue
            a, b = feats.get(f"h_{base}", NAN), feats.get(f"a_{base}", NAN)
            feats[f"d_{base}"] = a - b if a == a and b == b else NAN
        rh, ra = snapshot.get(home, elo.start), snapshot.get(away, elo.start)
        feats["elo_prob"] = elo.expected_home(home, away, rh, ra)
        feats["is_rivalry"] = int(is_rivalry(home, away))
        feats["is_playoff"] = int(g.game_type == 3)
        feats["days_into_season"] = day - self._season_start
        meta = {"game_id": g.game_id, "game_date": g.game_date, "season": season, "game_type": g.game_type,
                "home": home, "away": away, "home_score": g.home_score, "away_score": g.away_score,
                "home_win": g.home_win, "start_utc": g.start_utc}
        return {**meta, **feats}

    def _team_feats(self, team, opp, is_home, g, day, season, teams, goalies, snapshot, table, league,
                    starter_of, skaters_of, pending, cfg) -> dict:
        ts = teams[team]
        recs = ts.recs
        srecs = [r for r in recs if r.season == season]
        f: dict[str, float] = {}
        n_s = len(srecs)
        # ---------- strength ----------
        gd = [r.gf - r.ga for r in recs[-40:]]
        f["gd_ewm"] = _ewm(gd, cfg.halflife)
        f["gf_ewm"] = _ewm([r.gf for r in recs[-40:]], cfg.halflife)
        f["ga_ewm"] = _ewm([r.ga for r in recs[-40:]], cfg.halflife)
        l10 = recs[-10:]
        f["gd_l10"] = float(np.mean([r.gf - r.ga for r in l10])) if l10 else NAN
        f["win_pct_l10"] = float(np.mean([r.gf > r.ga for r in l10])) if l10 else NAN
        lg = league["gf"] / league["n"] if league["n"] else 3.0
        pgf, pga = ts.prev.get("gf", lg), ts.prev.get("ga", lg)
        k = cfg.shrink_k
        sgf = (sum(r.gf for r in srecs) + k * pgf) / (n_s + k)
        sga = (sum(r.ga for r in srecs) + k * pga) / (n_s + k)
        f["gf_season_shrunk"], f["ga_season_shrunk"], f["gd_season_shrunk"] = sgf, sga, sgf - sga
        venue = [r.gf - r.ga for r in srecs if r.is_home == is_home]
        f["gd_venue"] = (sum(venue) + 10 * (sgf - sga)) / (len(venue) + 10)
        f["elo"] = snapshot.get(team, 1500.0)
        f["sos10"] = float(np.mean([r.opp_elo for r in l10]) - 1500.0) if l10 else NAN
        # ---------- advanced ----------
        aw = recs[-cfg.adv_window:]
        f["cf_pct"] = _share(aw, "att_for", "att_against")
        f["ev_cf_pct"] = _share(aw, "ev_att_for", "ev_att_against")
        f["ff_pct"] = _share(aw, "fen_for", "fen_against")
        f["sog_share"] = _share(aw, "sog_for", "sog_against")
        f["hd_share"] = _share(aw, "hd_for", "hd_against")
        f["xg_share"] = _share(aw, "xg_for", "xg_against")
        valid = [r for r in aw if r.adv.get("xg_for") == r.adv.get("xg_for") and r.adv.get("xg_for") is not None]
        f["xgf_pg"] = float(np.mean([r.adv["xg_for"] for r in valid])) if len(valid) >= 5 else NAN
        f["xga_pg"] = float(np.mean([r.adv["xg_against"] for r in valid])) if len(valid) >= 5 else NAN
        gsum = sum(r.gf for r in aw if r.adv.get("sog_for") is not None and r.adv["sog_for"] == r.adv["sog_for"])
        sfor = sum(r.adv["sog_for"] for r in aw if r.adv.get("sog_for") is not None and r.adv["sog_for"] == r.adv["sog_for"])
        gag = sum(r.ga for r in aw if r.adv.get("sog_against") is not None and r.adv["sog_against"] == r.adv["sog_against"])
        sag = sum(r.adv["sog_against"] for r in aw if r.adv.get("sog_against") is not None and r.adv["sog_against"] == r.adv["sog_against"])
        if len(valid) >= 5:
            sh = (gsum + LEAGUE_SH * 100) / (sfor + 100)
            sv = 1 - (gag + (1 - LEAGUE_SV) * 100) / (sag + 100)
            f["sh_pct"], f["sv_pct_team"], f["pdo_dev"] = sh, sv, (sh + sv) - 1.0
        else:
            f["sh_pct"] = f["sv_pct_team"] = f["pdo_dev"] = NAN
        f["adv_n"] = float(len(valid))
        # ---------- special teams ----------
        pw = recs[-30:]
        ppg_ = sum(r.adv.get("pp_goals") or 0 for r in pw if r.adv.get("pp_opps") is not None and r.adv["pp_opps"] == r.adv["pp_opps"])
        ppo = sum(r.adv["pp_opps"] for r in pw if r.adv.get("pp_opps") is not None and r.adv["pp_opps"] == r.adv["pp_opps"])
        pko = sum(r.adv["pk_opps"] for r in pw if r.adv.get("pk_opps") is not None and r.adv["pk_opps"] == r.adv["pk_opps"])
        pkg = sum(r.adv.get("pk_goals_against") or 0 for r in pw if r.adv.get("pk_opps") is not None and r.adv["pk_opps"] == r.adv["pk_opps"])
        have_st = any(r.adv.get("pp_opps") is not None and r.adv["pp_opps"] == r.adv["pp_opps"] for r in pw)
        f["pp_pct"] = (ppg_ + 0.20 * 20) / (ppo + 20) if have_st else NAN
        f["pk_pct"] = 1 - (pkg + 0.20 * 20) / (pko + 20) if have_st else NAN
        f["st_net"] = f["pp_pct"] + f["pk_pct"] - 1 if have_st else NAN
        tk = [r.adv["pen_taken"] for r in pw if r.adv.get("pen_taken") is not None and r.adv["pen_taken"] == r.adv["pen_taken"]]
        dr = [r.adv["pen_drawn"] for r in pw if r.adv.get("pen_drawn") is not None and r.adv["pen_drawn"] == r.adv["pen_drawn"]]
        f["pen_taken_pg"] = float(np.mean(tk)) if len(tk) >= 5 else NAN  # ~ per 60 (one game = 60 min)
        f["pen_drawn_pg"] = float(np.mean(dr)) if len(dr) >= 5 else NAN
        f["pen_diff_pg"] = f["pen_drawn_pg"] - f["pen_taken_pg"]
        # ---------- rest / schedule ----------
        last = recs[-1] if recs else None
        rest = min(day - last.day, 7) if last else 7
        f["rest_days"] = float(rest)
        f["b2b"] = float(rest == 1)
        d3 = sum(1 for r in recs[-4:] if 0 < day - r.day <= 3)
        d5 = sum(1 for r in recs[-5:] if 0 < day - r.day <= 5)
        f["g3in4"], f["g4in6"] = float(d3 >= 2), float(d5 >= 3)
        f["games_last_7d"] = float(sum(1 for r in recs[-8:] if 0 < day - r.day <= 7))
        tonight = g.home
        origin = last.venue if last else team
        f["travel_km"] = haversine_km(origin, tonight)
        f["tz_shift"] = float(tz_shift(origin, tonight))
        streak = 0
        for r in reversed(recs):
            if r.is_home != is_home:
                break
            streak += 1
        # consecutive games at this venue type including tonight
        f["road_trip_n"] = 0.0 if is_home else float(streak + 1)
        f["homestand_n"] = float(streak + 1) if is_home else 0.0
        # ---------- situational (team) ----------
        f["gp_season"] = float(n_s)
        f["early_w"] = n_s / (n_s + 20.0)
        conf = "E" if TEAMS[team].division in ("Atlantic", "Metropolitan") else "W"
        pct, cut = table["pct"].get(team, NAN), table["cutoff"][conf]
        gap = pct - cut if pct == pct and cut == cut else NAN
        f["cutoff_gap"] = gap
        f["stakes"] = float(gap == gap and n_s >= 60 and abs(gap) < 0.06)
        # ---------- goaltending ----------
        f.update(self._goalie_feats(team, is_home, g, day, season, ts, goalies, league, starter_of, pending, cfg, rest))
        # ---------- lineup ----------
        f.update(self._lineup_feats(team, g, ts, skaters_of, cfg))
        return f

    def _goalie_feats(self, team, is_home, g, day, season, ts, goalies, league, starter_of, pending, cfg, rest) -> dict:
        keys = GROUPS["goaltending"]
        empty = {k: NAN for k in keys}
        recent_starts = [pid for _, pid in ts.starts[-20:]]
        if not recent_starts:
            pending[(g.game_id, team)] = (None, False)
            return empty
        cnt = defaultdict(int)
        for pid in recent_starts:
            cnt[pid] += 1
        ranked = sorted(cnt, key=lambda p: (-cnt[p], -max(i for i, q in enumerate(recent_starts) if q == p)))
        primary = ranked[0]
        backup = ranked[1] if len(ranked) > 1 else None
        last_started = ts.starts[-1][1]
        b2b_after_primary = rest == 1 and last_started == primary
        pending[(g.game_id, team)] = (primary, b2b_after_primary)
        actual = starter_of.get((g.game_id, team)) if cfg.goalie_mode == "actual" else None
        if actual is not None:
            weights, confirmed = {actual: 1.0}, 1.0
        else:
            if b2b_after_primary:
                p = league["b2b_pri"] / league["b2b_n"] if league["b2b_n"] >= 30 else 0.35
            else:
                p = league["rest_pri"] / league["rest_n"] if league["rest_n"] >= 30 else 0.85
            weights = {primary: p}
            if backup is not None:
                weights[backup] = 1 - p
            else:
                weights = {primary: 1.0}
            confirmed = 0.0
        acc = {k: 0.0 for k in keys if k not in ("g_p_primary", "g_confirmed")}
        wsum = {k: 0.0 for k in acc}
        for pid, w in weights.items():
            gf = self._goalie_stats(goalies[pid], day, season)
            for k, v in gf.items():
                if v == v:
                    acc[k] += w * v; wsum[k] += w
        out = {k: (acc[k] / wsum[k] if wsum[k] > 0 else NAN) for k in acc}
        out["g_p_primary"] = weights.get(primary, 0.0)
        out["g_confirmed"] = confirmed
        return out

    @staticmethod
    def _goalie_stats(gs: GoalieState, day: int, season: int) -> dict:
        apps = gs.apps
        if not apps:
            return {"g_sv_season": NAN, "g_sv_l5": NAN, "g_sv_l10": NAN, "g_gsax100_season": NAN,
                    "g_gsax100_l10": NAN, "g_rest": NAN, "g_b2b": NAN}

        def sv(a, k):
            sa, sv_ = sum(x[2] for x in a), sum(x[3] for x in a)
            return (sv_ + LEAGUE_SV * k) / (sa + k)

        def gsax(a, k):
            valid = [x for x in a if x[5] == x[5] and x[5] is not None]
            if not valid:
                return NAN
            sa = sum(x[2] for x in valid)
            return 100.0 * sum(x[5] - x[4] for x in valid) / (sa + k)

        cur = [a for a in apps if a[1] == season]
        rest = min(day - apps[-1][0], 14)
        return {"g_sv_season": sv(cur, 400), "g_sv_l5": sv(apps[-5:], 150), "g_sv_l10": sv(apps[-10:], 250),
                "g_gsax100_season": gsax(cur, 400) if cur else 0.0, "g_gsax100_l10": gsax(apps[-10:], 250),
                "g_rest": float(rest), "g_b2b": float(rest == 1)}

    def _lineup_feats(self, team, g, ts, skaters_of, cfg) -> dict:
        out = {"pts_share_missing": NAN, "core_missing_n": NAN}
        if cfg.lineup_mode != "actual":
            return out
        dressed = skaters_of.get((g.game_id, team))
        window = [r for r in ts.recs[-20:] if r.skaters]
        if not dressed or len(window) < 5:
            return out
        gp, pts = defaultdict(int), defaultdict(float)
        for r in window:
            for pid, p in r.skaters.items():
                gp[pid] += 1; pts[pid] += p
        regulars = [pid for pid in gp if gp[pid] >= 0.5 * len(window)]
        core = sorted(regulars, key=lambda p: -pts[p] / gp[p])[:10]
        tot = sum(pts[p] / gp[p] for p in core)
        miss = [p for p in core if p not in dressed]
        out["pts_share_missing"] = sum(pts[p] / gp[p] for p in miss) / tot if tot > 0 else 0.0
        out["core_missing_n"] = float(len(miss))
        return out

    # ---------------------------------------------------------------------------------------
    def _update(self, g, day, season, teams, goalies, elo, snapshot, league, season_pts, adv_lookup,
                starter_of, skaters_of, appear, pending) -> None:
        hs, as_ = int(g.home_score), int(g.away_score)
        ot = str(getattr(g, "last_period", "") or "") in ("OT", "SO")
        rh, ra = snapshot.get(g.home, elo.start), snapshot.get(g.away, elo.start)
        for team, opp, is_home, gf, ga in ((g.home, g.away, True, hs, as_), (g.away, g.home, False, as_, hs)):
            ts = teams[team]
            starter = starter_of.get((g.game_id, team))
            ts.recs.append(Rec(day, season, is_home, opp, g.home, gf, ga, ot and gf < ga,
                               ra if is_home else rh, adv_lookup.get((g.game_id, team), {}),
                               dict(skaters_of.get((g.game_id, team), {})), starter))
            if starter is not None:
                ts.starts.append((day, starter))
            league["gf"] += gf; league["n"] += 1
            if g.game_type == 2:
                sp = season_pts[(season, team)]
                sp[0] += 2 if gf > ga else (1 if ot else 0); sp[1] += 1
            primary, b2b_pri = pending.pop((g.game_id, team), (None, False))
            if primary is not None and starter is not None:
                if b2b_pri:
                    league["b2b_n"] += 1; league["b2b_pri"] += int(starter == primary)
                else:
                    league["rest_n"] += 1; league["rest_pri"] += int(starter == primary)
        for r in appear.get(g.game_id, []):
            xg = r.xg_faced if r.xg_faced is not None and r.xg_faced == r.xg_faced else NAN
            goalies[r.player_id].apps.append((day, season, r.shots_against or 0, r.saves or 0, r.goals_against or 0, xg))
        elo.update(g.home, g.away, hs, as_, rh, ra)


GROUPS_FLAT = [b for bs in GROUPS.values() for b in bs]


def build_features(store, config: BuilderConfig | None = None, market: pd.DataFrame | None = None,
                   starter_override: dict | None = None) -> pd.DataFrame:
    """Convenience: load tables from a ``Store`` and build the full feature frame."""
    from nhlbet.data.loaders import load_tables
    return FeatureBuilder(config).build(load_tables(store), market, starter_override)
