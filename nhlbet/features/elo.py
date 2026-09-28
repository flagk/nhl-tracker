"""Elo rating with home-ice advantage, goal-margin multiplier and between-season regression."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Elo:
    k: float = 6.0
    home_adv: float = 35.0     # ~54% home win rate for equal teams
    regress: float = 0.30      # fraction pulled to the mean each new season
    start: float = 1500.0
    ratings: dict = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    def expected_home(self, home: str, away: str, rh: float | None = None, ra: float | None = None) -> float:
        rh = self.rating(home) if rh is None else rh
        ra = self.rating(away) if ra is None else ra
        return 1.0 / (1.0 + 10 ** (-(rh + self.home_adv - ra) / 400.0))

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               rh: float | None = None, ra: float | None = None) -> None:
        """Update from a result. ``rh``/``ra`` are the pre-game ratings (defaults: current)."""
        rh = self.rating(home) if rh is None else rh
        ra = self.rating(away) if ra is None else ra
        exp = self.expected_home(home, away, rh, ra)
        margin = abs(home_goals - away_goals)
        won = home_goals > away_goals
        diff = (rh + self.home_adv - ra) * (1 if won else -1)
        mult = math.log(margin + 1.0) * (2.2 / (diff * 0.001 + 2.2))
        delta = self.k * mult * ((1.0 if won else 0.0) - exp)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta

    def new_season(self) -> None:
        for t in list(self.ratings):
            self.ratings[t] = (1 - self.regress) * self.ratings[t] + self.regress * self.start
