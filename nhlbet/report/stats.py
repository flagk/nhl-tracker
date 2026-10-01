"""Plain-language view of the numbers behind each game: both teams' as-of stats side by side, the model's main inputs, and per-game drivers.

Everything here is *read from* the features the model already used (computed as of the game date); nothing new is modelled.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

# key (without h_/a_), label, group, format, which direction is better for the team ("high"/"low"/None), what it tells you
STATS: list[tuple[str, str, str, str, str | None, str]] = [
    ("elo", "Elo rating", "Strength", "{:.0f}", "high", "Overall team strength from results, goal margin and home ice."),
    ("gd_season_shrunk", "Goal differential per game (season)", "Strength", "{:+.2f}", "high", "Goals for minus against, blended with last season early in the year."),
    ("gd_l10", "Goal differential, last 10 games", "Strength", "{:+.2f}", "high", "Recent form."),
    ("win_pct_l10", "Win % last 10", "Strength", "{:.0%}", "high", "Recent results."),
    ("cf_pct", "Corsi % (shot attempts)", "Puck possession and chances", "{:.1%}", "high", "Share of all shot attempts, last 20 games. Above 50% means they control play."),
    ("ev_cf_pct", "Corsi %, 5-on-5", "Puck possession and chances", "{:.1%}", "high", "Same, at even strength only."),
    ("xg_share", "Expected-goals share", "Puck possession and chances", "{:.1%}", "high", "Share of shot quality (distance, angle, type), last 20 games."),
    ("hd_share", "High-danger chances share", "Puck possession and chances", "{:.1%}", "high", "Share of the best scoring chances."),
    ("xgf_pg", "Expected goals for per game", "Puck possession and chances", "{:.2f}", "high", "How much quality offence they create."),
    ("xga_pg", "Expected goals against per game", "Puck possession and chances", "{:.2f}", "low", "How much quality offence they allow."),
    ("sh_pct", "Shooting %", "Luck check", "{:.1%}", None, "Very high or low values tend to move back toward the league average."),
    ("sv_pct_team", "Team save %", "Luck check", "{:.1%}", None, "Same idea for goaltending results."),
    ("pdo_dev", "PDO vs league (shooting % + save %)", "Luck check", "{:+.3f}", None, "Positive = running hot, which usually fades; negative = running cold."),
    ("g_sv_season", "Starting goalie save % (season)", "Goaltending", "{:.3f}", "high", "Shrunk toward the league average so small samples don't mislead."),
    ("g_sv_l5", "Starting goalie save %, last 5", "Goaltending", "{:.3f}", "high", "Recent form."),
    ("g_gsax100_season", "Goalie goals saved above expected / 100 shots", "Goaltending", "{:+.2f}", "high", "Saves vs the quality of shots faced; positive is better than average."),
    ("g_rest", "Goalie days of rest", "Goaltending", "{:.0f}", None, "Backups usually start on the second night of a back-to-back."),
    ("pp_pct", "Power play %", "Special teams", "{:.1%}", "high", "Shrunk toward the league average."),
    ("pk_pct", "Penalty kill %", "Special teams", "{:.1%}", "high", "Shrunk toward the league average."),
    ("pen_taken_pg", "Penalties taken per game", "Special teams", "{:.1f}", "low", "More penalties means more time shorthanded."),
    ("pen_drawn_pg", "Penalties drawn per game", "Special teams", "{:.1f}", "high", "More penalties drawn means more power plays."),
    ("rest_days", "Days of rest", "Schedule and travel", "{:.0f}", "high", "Days since the last game."),
    ("b2b", "Second game in two nights", "Schedule and travel", "{:yesno}", "low", "Back-to-backs lower win chances a little."),
    ("games_last_7d", "Games in the last 7 days", "Schedule and travel", "{:.0f}", "low", "Schedule congestion."),
    ("travel_km", "Travel since last game (km)", "Schedule and travel", "{:,.0f}", "low", "Distance from the previous game's arena."),
    ("tz_shift", "Time-zone change", "Schedule and travel", "{:+.0f} h", None, "Hours of time-zone shift for the trip."),
    ("gp_season", "Games played this season", "Context", "{:.0f}", None, "Early in the season the model leans on last year's ratings."),
]
GROUP_ORDER = ["Strength", "Puck possession and chances", "Goaltending", "Special teams", "Schedule and travel", "Luck check", "Context"]
_LABELS = {k: lab for k, lab, *_ in STATS}
_LABELS.update({"early_w": "Early-season weight", "stakes": "Playoff-race stakes", "homestand_n": "Games into a homestand", "road_trip_n": "Games into a road trip",
                "gd_ewm": "Goal differential (recent, weighted)", "ga_season_shrunk": "Goals against per game (season)", "gf_season_shrunk": "Goals for per game (season)",
                "g_sv_l10": "Goalie save %, last 10", "cutoff_gap": "Points-% gap to the playoff line", "sos10": "Strength of recent opponents"})


def _fmt(spec: str, v: float) -> str:
    if spec == "{:yesno}":
        return "yes" if v >= 0.5 else "no"
    return spec.format(v)


def _num(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def game_stats(row) -> list[dict]:
    """Rows for the side-by-side table: both teams' values, display strings, and who has the better number (None when neither or it is not a quality measure)."""
    out = []
    for key, label, group, spec, better, tip in STATS:
        h, a = _num(row.get("h_" + key)), _num(row.get("a_" + key))
        if h is None and a is None:
            continue
        edge = None
        if better and h is not None and a is not None and abs(h - a) > 1e-9:
            edge = "home" if ((h > a) == (better == "high")) else "away"
        out.append({"key": key, "label": label, "group": group, "tip": tip, "home": h, "away": a, "home_s": "–" if h is None else _fmt(spec, h),
                    "away_s": "–" if a is None else _fmt(spec, a), "better": edge})
    return out


def feature_label(name: str) -> str:
    """Readable name for a model input such as ``d_elo`` or ``h_b2b``."""
    base = name[2:] if name[:2] in ("d_", "h_", "a_") else name
    lab = _LABELS.get(base, base.replace("_", " "))
    if name.startswith("d_"):
        return f"{lab} (home minus away)"
    if name.startswith("h_"):
        return f"{lab} (home team)"
    if name.startswith("a_"):
        return f"{lab} (away team)"
    return lab


def model_inputs(features: list[str], importance_csv: str | Path = "reports/feature_importance.csv") -> list[dict]:
    """The inputs the model actually uses, with their out-of-sample importance (mean |SHAP|, share of the total) and plain-language labels."""
    imp: dict[str, dict] = {}
    p = Path(importance_csv)
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                imp[r["feature"]] = r
    rows = []
    for name in features:
        r = imp.get(name, {})
        shap = _num(r.get("shap_mean_abs")) or 0.0
        group = r.get("group") or ("rest_schedule" if name[:2] in ("h_", "a_") else "")
        rows.append({"feature": name, "label": feature_label(name), "group": group.replace("_", " "), "importance": shap, "direction": _num(r.get("shap_direction"))})
    tot = sum(r["importance"] for r in rows)
    for r in rows:
        r["share"] = r["importance"] / tot if tot > 0 else None          # no importance data: list the inputs without pretending to rank them
    return sorted(rows, key=lambda r: -(r["share"] or 0))
