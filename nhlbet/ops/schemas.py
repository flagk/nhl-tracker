"""Pandera schemas for the tables the model reads. A violation is a SOFT halt (the data source or the database cache is broken); nothing is repaired or dropped silently.

Docs: https://pandera.readthedocs.io/en/stable/dataframe_schemas.html (``lazy=True`` collects every failure instead of stopping at the first).
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from nhlbet.ops.checks import CheckResult, failed, passed

nonneg = Check.ge(0)
GAMES = DataFrameSchema({
    "game_id": Column(int, Check.gt(0), unique=True),
    "game_date": Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}"), coerce=True),
    "home": Column(str, Check.str_length(2, 4)), "away": Column(str, Check.str_length(2, 4)),
    "game_type": Column(int, Check.isin([1, 2, 3, 4]), coerce=True),
    "home_score": Column(float, nonneg, nullable=True, coerce=True), "away_score": Column(float, nonneg, nullable=True, coerce=True),
    "home_win": Column(float, Check.isin([0, 1]), nullable=True, coerce=True),
}, checks=[Check(lambda d: ~((d.home == d.away)), error="a team plays itself", element_wise=False),
           Check(lambda d: d.home_score.isna() == d.away_score.isna(), error="one score present, the other missing"),
           Check(lambda d: ~((d.home_score == d.away_score) & d.home_score.notna() & (d.game_type != 1)), error="completed regular-season or playoff game with equal scores")],
    strict=False)

GOALIE_GAME = DataFrameSchema({
    "game_id": Column(int, Check.gt(0), coerce=True), "player_id": Column(int, Check.gt(0), coerce=True),
    "started": Column(int, Check.isin([0, 1]), coerce=True), "toi_sec": Column(float, Check.in_range(0, 12000), nullable=True, coerce=True),
    "shots_against": Column(float, nonneg, nullable=True, coerce=True), "saves": Column(float, nonneg, nullable=True, coerce=True),
}, checks=[Check(lambda d: (d.saves.isna() | d.shots_against.isna() | (d.saves <= d.shots_against)), error="more saves than shots against")], strict=False)

ODDS = DataFrameSchema({
    "captured_at": Column(str), "event_id": Column(str), "book": Column(str, nullable=True),
    "market": Column(str, Check(lambda m: m.isin(["h2h", "totals", "spreads"]) | m.str.startswith("player_"), error="unknown market key"), coerce=True),
    "price": Column(float, Check.in_range(1.0001, 1000.0), coerce=True),
}, strict=False)

TEAM_GAME = DataFrameSchema({
    "game_id": Column(int, Check.gt(0), coerce=True), "team": Column(str),
    "goals": Column(float, nonneg, nullable=True, coerce=True),
    "sog_for": Column(float, nonneg, nullable=True, coerce=True), "att_for": Column(float, nonneg, nullable=True, coerce=True),
    "xg_for": Column(float, Check.in_range(0, 15), nullable=True, coerce=True),
}, strict=False)

TABLES = {"games": (GAMES, "SELECT * FROM games WHERE source = 'nhl_api'"), "goalie_game": (GOALIE_GAME, "SELECT * FROM goalie_game"),
          "team_game": (TEAM_GAME, "SELECT * FROM team_game"), "odds_snapshots": (ODDS, "SELECT * FROM odds_snapshots WHERE captured_at >= date('now', '-30 day')")}


def validate_frame(schema: DataFrameSchema, df: pd.DataFrame) -> list[str]:
    """Human-readable failures (empty = valid)."""
    if df.empty:
        return []
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        return [f"{r.column}: {r.check} ({r.failure_case})" for r in fc.head(5).itertuples()] + ([f"... and {len(fc) - 5} more"] if len(fc) > 5 else [])


def schema_validation_check(store) -> CheckResult:
    problems = []
    for name, (schema, sql) in TABLES.items():
        errs = validate_frame(schema, store.df(sql))
        if errs:
            problems.append(f"{name}: " + "; ".join(errs[:3]))
    if problems:
        return failed("pandera", "soft", "table validation failed: " + " | ".join(problems)[:600],
                      "A data source or the cached database returned malformed rows: run the Backfill workflow (or delete the Actions cache so the database is rebuilt); the halt clears after healthy runs.")
    return passed("pandera", "soft", "games, goalie_game, team_game and recent odds snapshots validate")
