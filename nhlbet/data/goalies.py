"""Starting-goalie information for upcoming games.

Reality check: the free NHL API does not publish confirmed starters before puck drop. Confirmations
(morning-skate rotations, beat reports, DailyFaceoff) are therefore an *input*:

* ``data/manual/confirmed_goalies.csv`` with columns ``date,team,name`` (``player_id`` optional), filled by hand or by
  any integration you trust; ``load_confirmations`` ingests it into ``goalie_confirmations``.
* Otherwise the **probable** starter is the goalie who started most of the team's last 20 games, adjusted for
  back-to-backs by the model (see ``nhlbet.features.builder``).

Confirmed identities feed the model only as *who is in net* (never stats about tonight).
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nhlbet.data.store import Store
from nhlbet.teams import canon

log = logging.getLogger(__name__)
MANUAL_PATH = Path("data/manual/confirmed_goalies.csv")


def load_confirmations(store: Store, path: str | Path = MANUAL_PATH) -> int:
    """Upsert rows of the manual CSV into ``goalie_confirmations`` (names resolved to ids from goalie history)."""
    p = Path(path)
    if not p.exists():
        return 0
    df = pd.read_csv(p)
    need = {"date", "team", "name"}
    if not need <= set(df.columns):
        raise ValueError(f"{p} must have columns {sorted(need)} (optional: player_id); found {list(df.columns)}")
    ids = store.df("SELECT player_id, name FROM goalie_game GROUP BY player_id").drop_duplicates("name").set_index("name").player_id.to_dict()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = {(r.game_date, r.team): (r.player_id, r.captured_at) for r in store.df("SELECT game_date, team, player_id, captured_at FROM goalie_confirmations").itertuples()}
    rows = []
    for r in df.itertuples(index=False):
        pid = getattr(r, "player_id", None)
        pid = int(pid) if pid is not None and pd.notna(pid) else ids.get(r.name)
        if pid is None:
            log.warning("confirmed goalie %r (%s) not found in goalie history; report will show the name only", r.name, r.team)
        key = (str(r.date)[:10], canon(r.team))
        seen = old.get(key)
        stamp = seen[1] if seen and seen[0] == pid and seen[1] else now            # an unchanged confirmation keeps the time it was FIRST seen, so reloading the file never makes it look earlier
        rows.append({"game_date": key[0], "team": key[1], "player_id": pid, "name": r.name, "source": "manual", "captured_at": stamp})
    return store.upsert("goalie_confirmations", rows, ["game_date", "team"])


def starter_overrides(store: Store, games: pd.DataFrame, as_of=None) -> dict:
    """{(game_id, team): player_id} for confirmed starters on the given upcoming games. With ``as_of`` only confirmations captured at or before that time
    count: a confirmation that arrived after the decision time must not shape the pick."""
    conf = store.df("SELECT * FROM goalie_confirmations WHERE player_id IS NOT NULL")
    if conf.empty:
        return {}
    if as_of is not None:
        conf = conf[pd.to_datetime(conf.captured_at, utc=True, errors="coerce") <= pd.Timestamp(as_of).tz_convert("UTC")]
        if conf.empty:
            return {}
    key = {(r.game_date, r.team): int(r.player_id) for r in conf.itertuples()}
    out = {}
    for g in games.itertuples():
        d = pd.Timestamp(g.game_date).strftime("%Y-%m-%d")
        for team in (g.home, g.away):
            if (d, team) in key:
                out[(g.game_id, team)] = key[(d, team)]
    return out


def goalie_display(store: Store, team: str, date: str, as_of=None) -> tuple[str, str]:
    """(name, 'confirmed' | 'probable' | 'unknown') for the report; with ``as_of`` a confirmation captured later than that does not count."""
    c = store.df("SELECT name, captured_at FROM goalie_confirmations WHERE game_date=? AND team=?", [date, team])
    if as_of is not None and len(c):
        c = c[pd.to_datetime(c.captured_at, utc=True, errors="coerce") <= pd.Timestamp(as_of).tz_convert("UTC")]
    if len(c):
        return str(c.name.iloc[0]), "confirmed"
    recent = store.df("SELECT gg.player_id, gg.name FROM goalie_game gg JOIN games g ON g.game_id=gg.game_id "
                      "WHERE gg.team=? AND gg.started=1 AND g.game_date < ? ORDER BY g.game_date DESC LIMIT 20", [team, date])
    if recent.empty:
        return "unknown", "unknown"
    top = Counter(recent.player_id).most_common(1)[0][0]
    return str(recent[recent.player_id == top].name.iloc[0]), "probable"
