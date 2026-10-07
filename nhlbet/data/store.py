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
    position TEXT, toi_sec INTEGER, goals INTEGER, assists INTEGER, points INTEGER, sog INTEGER,
    PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS shots (
    game_id INTEGER NOT NULL, event_id INTEGER NOT NULL, team TEXT, period INTEGER, sec INTEGER,
    x REAL, y REAL, shot_type TEXT, kind TEXT, is_goal INTEGER, ev INTEGER, goalie_id INTEGER,
    xg REAL, PRIMARY KEY (game_id, event_id)
);
-- every odds observation we ever capture; nothing is overwritten (needed for opening line and CLV)
CREATE TABLE IF NOT EXISTS odds_snapshots (
    captured_at TEXT NOT NULL, event_id TEXT NOT NULL, commence_time TEXT, home TEXT, away TEXT,
    book TEXT NOT NULL, market TEXT NOT NULL, outcome TEXT NOT NULL, point REAL NOT NULL DEFAULT 0,
    price REAL NOT NULL, book_updated TEXT, game_id INTEGER,
    PRIMARY KEY (captured_at, event_id, book, market, outcome, point)
);
CREATE INDEX IF NOT EXISTS ix_odds_game ON odds_snapshots(game_id, captured_at);
-- derived, publishable summary of each odds capture: mean no-vig home probability across books (no per-book quotes)
CREATE TABLE IF NOT EXISTS odds_consensus (
    game_id INTEGER NOT NULL, captured_at TEXT NOT NULL, home_prob_novig REAL NOT NULL, n_books INTEGER,
    PRIMARY KEY (game_id, captured_at)
);
CREATE TABLE IF NOT EXISTS odds_fetch_log (
    captured_at TEXT PRIMARY KEY, ok INTEGER, source TEXT, remaining INTEGER, used INTEGER, events INTEGER, note TEXT
);
-- every model run's verdict on every game (bets AND no-bets): needed for calibration, Brier and honest ROI
CREATE TABLE IF NOT EXISTS recommendations (
    run_id TEXT NOT NULL, run_at TEXT NOT NULL, run_type TEXT, game_id INTEGER NOT NULL, game_date TEXT, home TEXT, away TEXT,
    home_goalie TEXT, away_goalie TEXT, goalie_status TEXT, model_version TEXT, p_model REAL, p_adj REAL, p_market REAL,
    p_stack_raw REAL, action TEXT, side TEXT, team TEXT, book TEXT, decimal REAL, stake REAL, edge REAL, ev REAL,
    reasons TEXT, odds_captured_at TEXT, odds_stale INTEGER, model_status TEXT,
    PRIMARY KEY (run_id, game_id)
);
-- fake-money "paper trading" strategies evaluated on every slate (never real bets): used to measure what works, faster than real bets
CREATE TABLE IF NOT EXISTS shadow_bets (
    run_id TEXT NOT NULL, run_at TEXT NOT NULL, game_id INTEGER NOT NULL, strategy TEXT NOT NULL, game_date TEXT,
    action TEXT, side TEXT, team TEXT, book TEXT, decimal REAL, stake REAL, p_model REAL, p_adj REAL, p_market REAL, edge REAL, ev REAL,
    market TEXT, point REAL, label TEXT,              -- NULL market = moneyline; 'totals' / 'spreads' for the goals-model strategies
    PRIMARY KEY (run_id, game_id, strategy)
);
-- player-prop bets (several per game, hence their own table): fake money, settled from skater_game.sog like every other paper bet
CREATE TABLE IF NOT EXISTS prop_bets (
    run_id TEXT NOT NULL, run_at TEXT NOT NULL, game_id INTEGER NOT NULL, strategy TEXT NOT NULL, game_date TEXT, player_id INTEGER NOT NULL, name TEXT,
    side TEXT NOT NULL, label TEXT, point REAL, book TEXT, decimal REAL, stake REAL, p_model REAL, p_market REAL, edge REAL, ev REAL, lam REAL, market TEXT,
    PRIMARY KEY (run_id, game_id, strategy, player_id, side)
);
-- every run's model-vs-market view of totals and puck lines for EVERY game with odds (bets and passes): for calibration and paper trading
CREATE TABLE IF NOT EXISTS alt_quotes (
    run_id TEXT NOT NULL, run_at TEXT NOT NULL, game_id INTEGER NOT NULL, game_date TEXT, market TEXT NOT NULL, side TEXT NOT NULL, label TEXT,
    point REAL, p_model REAL, p_market REAL, p_push REAL, book TEXT, decimal REAL, edge REAL, ev REAL, n_books INTEGER,
    lam_home REAL, lam_away REAL, exp_total REAL, odds_captured_at TEXT, player_id INTEGER,
    PRIMARY KEY (run_id, game_id, market, side)
);
CREATE TABLE IF NOT EXISTS goalie_confirmations (
    game_date TEXT NOT NULL, team TEXT NOT NULL, player_id INTEGER, name TEXT, source TEXT, captured_at TEXT,
    PRIMARY KEY (game_date, team)
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
        self._migrate()

    # columns added after first release: CREATE TABLE IF NOT EXISTS leaves an existing (cached) database unchanged, so add them here
    MIGRATIONS = {"shadow_bets": {"market": "TEXT", "point": "REAL", "label": "TEXT"}, "skater_game": {"sog": "INTEGER"}, "alt_quotes": {"player_id": "INTEGER"}, "prop_bets": {"market": "TEXT"}}

    def _migrate(self) -> None:
        for table, cols in self.MIGRATIONS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        self.conn.commit()

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
