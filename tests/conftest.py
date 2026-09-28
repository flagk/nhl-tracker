"""Shared fixtures: a synthetic league with results, goalies, skaters and advanced stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS8 = ["BOS", "TOR", "NYR", "PIT", "CHI", "COL", "EDM", "VGK"]


def make_league(n_days: int = 150, seed: int = 0, start: str = "2023-10-10") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS8, rng.normal(0, 0.4, len(TEAMS8))))
    days = pd.date_range(start, periods=n_days, freq="D")
    games, tg, gg, sk = [], [], [], []
    gid = 1000
    for d in days:
        if rng.random() < 0.25:  # off day
            continue
        order = list(rng.permutation(TEAMS8))
        for i in range(0, len(order) - 1, 2):
            if rng.random() < 0.15:
                continue
            h, a = order[i], order[i + 1]
            mu = 3.0 + 0.5 * (strength[h] - strength[a]) + 0.15
            hs, as_ = rng.poisson(max(mu, 0.5)), rng.poisson(max(6.0 - mu, 0.5))
            if hs == as_:
                hs += 1
            gid += 1
            ot = bool(rng.random() < 0.2)
            games.append(dict(game_id=gid, game_date=d, season=int(d.year - (d.month < 8)), game_type=2, home=h, away=a,
                              home_score=hs, away_score=as_, home_win=int(hs > as_), status="FINAL",
                              last_period="OT" if ot else "REG",
                              start_utc=(d + pd.Timedelta(hours=23)).tz_localize("UTC"), source="nhl_api"))
            for team, opp, ishome, gf, ga in ((h, a, 1, hs, as_), (a, h, 0, as_, hs)):
                sog_f, sog_a = int(rng.integers(20, 40)), int(rng.integers(20, 40))
                tg.append(dict(game_id=gid, team=team, opp=opp, is_home=ishome, goals=gf, goals_against=ga,
                               sog_for=sog_f, sog_against=sog_a, att_for=sog_f + 20, att_against=sog_a + 20,
                               fen_for=sog_f + 10, fen_against=sog_a + 10, hd_for=int(rng.integers(5, 20)),
                               hd_against=int(rng.integers(5, 20)), xg_for=float(rng.uniform(1.5, 4)),
                               xg_against=float(rng.uniform(1.5, 4)), ev_att_for=sog_f + 12, ev_att_against=sog_a + 12,
                               ev_xg_for=float(rng.uniform(1, 3)), ev_xg_against=float(rng.uniform(1, 3)),
                               pen_taken=int(rng.integers(1, 5)), pen_drawn=int(rng.integers(1, 5)),
                               pp_opps=int(rng.integers(1, 5)), pp_goals=int(rng.integers(0, 2)),
                               pk_opps=int(rng.integers(1, 5)), pk_goals_against=int(rng.integers(0, 2))))
                g1, g2 = (hash(team) % 9000 + 100), (hash(team) % 9000 + 101)
                pid = g1 if rng.random() < 0.7 else g2
                sa = sog_a
                gg.append(dict(game_id=gid, team=team, player_id=pid, name="G", started=1, toi_sec=3600,
                               shots_against=sa, saves=max(sa - ga, 0), goals_against=ga, xg_faced=float(rng.uniform(2, 3.5))))
                for k in range(18):
                    sk.append(dict(game_id=gid, team=team, player_id=hash(team) % 5000 * 100 + k, name="S", position="C",
                                   toi_sec=900, goals=0, assists=0, points=int(rng.random() < 0.15 * (1 + (k < 6)))))
    return {"games": pd.DataFrame(games), "team_game": pd.DataFrame(tg), "goalie_game": pd.DataFrame(gg),
            "skater_game": pd.DataFrame(sk)}


@pytest.fixture(scope="session")
def league():
    return make_league()


def pytest_configure(config):
    import os
    os.environ["PYTHONHASHSEED"] = "0"
