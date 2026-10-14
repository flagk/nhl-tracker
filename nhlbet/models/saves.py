"""Goalie saves as a count: shots against (a count) thinned by the goalie's save probability (shrunk toward the league mean).

    saves = shots against - goals against.  Shots against ~ negative binomial with a mean from a Poisson regression on the team's own recent shots-against rate,
    the opponent's recent shots-for rate and home ice; each shot is saved with the goalie's shrunk save percentage. A binomially thinned negative binomial is
    again negative binomial with the same dispersion, so the whole PMF (and every over/under line) is available in closed form, pushes included.

Everything is as-of: a goalie's save percentage and the teams' rates come from strictly earlier games. Only goalies who STARTED are modelled; a goalie who is pulled
mid-game simply has fewer shots against in the history the distribution is fitted on, which is how the risk of a short night enters the price. Books differ on
how they settle a goalie who starts but is replaced (most let the bet stand on the saves he made; some void under a minimum time on ice), so this is a research
model only: nothing is priced or staked from it until a verified book rule and live prices exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

LEAGUE_SV = 0.905
SV_PRIOR_SHOTS = 400.0          # pseudo-shots of league-average goaltending blended into a goalie's own save percentage
HALFLIFE = 8
SA_BOUNDS = (10.0, 50.0)
DISPERSIONS = (0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15)


def goalie_starts(store) -> pd.DataFrame:
    """One row per goalie start (regular season, API games) with the goalie's saves and the two teams' shot volume. Needs ``goalie_game`` and ``team_game``."""
    d = store.df("""
        SELECT gg.game_id, g.game_date, gg.player_id, gg.name, gg.team, tg.opp, tg.is_home, gg.toi_sec, gg.shots_against, gg.saves,
               tg.sog_against AS team_sa, tg.sog_for AS team_sf, ot.sog_for AS opp_sf, ot.sog_against AS opp_sa
        FROM goalie_game gg
        JOIN games g ON g.game_id = gg.game_id
        JOIN team_game tg ON tg.game_id = gg.game_id AND tg.team = gg.team
        JOIN team_game ot ON ot.game_id = gg.game_id AND ot.team = tg.opp
        WHERE gg.started = 1 AND g.game_type = 2 AND g.source = 'nhl_api' AND gg.shots_against IS NOT NULL AND gg.saves IS NOT NULL""")
    d["game_date"] = pd.to_datetime(d.game_date)
    return d


def _ewm_prior(s: pd.Series, hl: float = HALFLIFE) -> pd.Series:
    return s.ewm(halflife=hl, adjust=True, ignore_na=True).mean().shift(1)


def add_asof_features(d: pd.DataFrame, sv_prior: float = SV_PRIOR_SHOTS) -> pd.DataFrame:
    """Add as-of features: ``team_sa_ewm`` (team's shots against), ``opp_sf_ewm`` (opponent's shots for), the goalie's shrunk save percentage ``sv`` and ``n_prev`` starts."""
    d = d.sort_values(["game_date", "game_id"]).copy()
    t = d.sort_values(["team", "game_date"])
    d["team_sa_ewm"] = t.groupby("team").team_sa.transform(_ewm_prior).reindex(d.index)
    d["opp_sf_ewm"] = d.assign(_o=d.opp_sf).sort_values(["opp", "game_date"]).groupby("opp")._o.transform(_ewm_prior).reindex(d.index)
    g = d.sort_values(["player_id", "game_date", "game_id"]).groupby("player_id")
    cs = g.shots_against.cumsum() - d.shots_against
    cv = g.saves.cumsum() - d.saves
    d["sv"] = ((cv.reindex(d.index) + LEAGUE_SV * sv_prior) / (cs.reindex(d.index) + sv_prior))
    d["n_prev"] = g.cumcount().reindex(d.index)
    return d


def _nb_pmf(mu: np.ndarray, k: float, n: int) -> np.ndarray:
    x = np.arange(n)[None, :]
    mu = np.asarray(mu, float)[:, None]
    if k <= 1e-9:
        return poisson.pmf(x, mu)
    r = 1.0 / k
    return nbinom.pmf(x, r, r / (r + mu))


class SavesModel:
    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha

    def _X(self, d: pd.DataFrame) -> np.ndarray:
        lg_sa = self.lg_sa_
        return np.column_stack([np.log(d.team_sa_ewm.fillna(lg_sa).clip(10, 50)), np.log(d.opp_sf_ewm.fillna(lg_sa).clip(10, 50)), d.is_home.to_numpy(float)])

    def fit(self, d: pd.DataFrame) -> "SavesModel":
        d = d[d.team_sa_ewm.notna() & d.opp_sf_ewm.notna()]
        self.lg_sa_ = float(d.shots_against.mean())
        self.reg_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(self._X(d), d.shots_against.to_numpy(float))
        lam = self.lam_sa(d)
        mu = lam * d.sv.to_numpy(float)
        y = d.saves.to_numpy(int)
        ll = {k: float(np.sum(np.log(np.maximum(_nb_pmf(mu, k, 90)[np.arange(len(y)), np.minimum(y, 89)], 1e-12)))) for k in DISPERSIONS}
        self.dispersion_ = max(ll, key=ll.get)
        self.mean_saves_ = float(y.mean())
        return self

    def lam_sa(self, d: pd.DataFrame) -> np.ndarray:
        return np.clip(self.reg_.predict(self._X(d)), *SA_BOUNDS)

    def lam(self, d: pd.DataFrame) -> np.ndarray:
        """Expected saves: expected shots against times the goalie's shrunk save percentage."""
        return self.lam_sa(d) * d.sv.to_numpy(float)

    def over_push_under(self, d: pd.DataFrame, line: float):
        from nhlbet.models.props import over_push_under
        return over_push_under(self.dispersion_, self.lam(d), line)


def walk_forward_saves(d: pd.DataFrame, first_test: str, step_days: int = 28, min_train: int = 1500, min_prev: int = 3, sv_prior: float = SV_PRIOR_SHOTS) -> pd.DataFrame:
    """Out-of-sample rows (``y`` = saves, ``lam``, ``base``, ``k``) retrained on strictly earlier games; same layout as ``props_eval.walk_forward_props``."""
    from nhlbet.splits import walk_forward_windows
    f = add_asof_features(d, sv_prior).sort_values("game_date").reset_index(drop=True)
    parts = []
    for tr, te in walk_forward_windows(f.game_date, first_test, step_days, min_train):
        m = SavesModel().fit(f.iloc[tr])
        t = f.iloc[te]
        t = t[t.team_sa_ewm.notna() & t.opp_sf_ewm.notna() & (t.n_prev >= min_prev)]
        if t.empty:
            continue
        hist = f.iloc[tr]
        n_prior = np.clip(t.n_prev.to_numpy(float), 0, 40)
        # baseline: the goalie's own shrunk saves per start (no opponent, venue or shot-volume information), prior = the league mean saves per start
        per_start = (hist.groupby("player_id").saves.mean()).reindex(t.player_id).fillna(m.mean_saves_).to_numpy(float)
        base = (n_prior * per_start + 6.0 * m.mean_saves_) / (n_prior + 6.0)
        parts.append(pd.DataFrame({"game_id": t.game_id.to_numpy(), "game_date": t.game_date.to_numpy(), "player_id": t.player_id.to_numpy(), "pos": "G", "y": t.saves.to_numpy(float),
                                   "lam": m.lam(t), "base": base, "k": m.dispersion_}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
