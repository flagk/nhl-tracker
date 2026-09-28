"""SQLite store with idempotent upserts. All ingestion writes go through ``Store.upsert``."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY, season INTEGER, game_type INTEGER, game_date TEXT NOT NULL,
    start_utc TEXT, home TEXT NOT NULL, away TEXT NOT NULL, home_score INTEGER, away_score INTEGER,
    status TEXT, last_period TEXT, home_win INTEGER, source TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_games_date ON games(game_date);

-- one row per team per game; *_for / *_against are from this team's perspective
CREATE TABLE IF NOT EXISTS team_game (
    game_id INTEGER NOT NULL, team TEXT NOT NULL, opp TEXT NOT NULL, is_home INTEGER NOT NULL,
    goals INTEGER, goals_against INTEGER,
    sog_for INTEGER, sog_against INTEGER,
    att_for INTEGER, att_against INTEGER,            -- Corsi (all shot attempts), all strengths
    fen_for INTEGER, fen_against INTEGER,            -- Fenwick (unblocked attempts)
    hd_for INTEGER, hd_against INTEGER,              -- high-danger unblocked attempts
    xg_for REAL, xg_against REAL,
    ev_att_for INTEGER, ev_att_against INTEGER,      -- 5v5 Corsi
    ev_xg_for REAL, ev_xg_against REAL,
    pen_taken INTEGER, pen_drawn INTEGER,            -- minor+major penalties
    pp_opps INTEGER, pp_goals INTEGER, pk_opps INTEGER, pk_goals_against INTEGER,
    PRIMARY KEY (game_id, team)
);
CREATE TABLE IF NOT EXISTS goalie_game (
    game_id INTEGER NOT NULL, team TEXT NOT NULL, player_id INTEGER NOT NULL, name TEXT,
    started INTEGER, toi_sec INTEGER, shots_against INTEGER, saves INTEGER, goals_against INTEGER,
    xg_faced REAL, PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS skater_game (
    game_id INTEGER NOT NULL, team TEXT NOT NULL, player_id INTEGER NOT NULL, name TEXT,
    position TEXT, toi_sec INTEGER, goals INTEGER, assists INTEGER, points INTEGER,
    PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS shots (
    game_id INTEGER NOT NULL, event_id INTEGER NOT NULL, team TEXT, period INTEGER, sec INTEGER,
    x REAL, y REAL, shot_type TEXT, kind TEXT, is_goal INTEGER, ev INTEGER, goalie_id INTEGER,
    xg REAL, PRIMARY KEY (game_id, event_id)
);
CREATE TABLE IF NOT EXISTS ingest_log (
    game_id INTEGER PRIMARY KEY, stage TEXT, ok INTEGER, note TEXT, ts TEXT
);
"""


class Store:
    def __init__(self, path: str | Path = "data/nhl.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def upsert(self, table: str, rows: Iterable[Mapping], keys: Sequence[str]) -> int:
        """Insert or update ``rows`` keyed on ``keys``. Safe to call repeatedly (idempotent)."""
        rows = list(rows)
        if not rows:
            return 0
        cols = list(rows[0].keys())
        upd = [c for c in cols if c not in keys]
        sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
               f"ON CONFLICT({','.join(keys)}) DO "
               + (f"UPDATE SET {','.join(f'{c}=excluded.{c}' for c in upd)}" if upd else "NOTHING"))
        with self.tx() as c:
            c.executemany(sql, [tuple(r[k] for k in cols) for r in rows])
        return len(rows)

    def df(self, sql: str, params: Sequence = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def known_final_games(self) -> set[int]:
        rows = self.conn.execute("SELECT game_id FROM ingest_log WHERE ok=1 AND stage='detail'").fetchall()
        return {r[0] for r in rows}

    def close(self) -> None:
        self.conn.close()
