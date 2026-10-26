"""Deliberately leaky canary features and the detectors that must catch them.

A leak detector that has never seen a leak proves nothing. ``LeakyBuilder`` adds one extra feature, ``canary``, built from information a pre-game feature must not have, and
``detect`` runs every independent detector on it. The expected result (asserted in tests/test_audit_leakage.py) is that EVERY canary is caught by at least one detector and the
clean builder by none. Which detector catches which canary is itself a finding: no single detector is sufficient.

Canaries (all signed from the home team's point of view):
  own_result        the game's own goal difference
  noisy_own_result  own goal difference plus Gaussian noise (sd 2 goals): evades an "is it a copy of the label" check
  next_game         the home team's NEXT game's goal difference minus the away team's: future, but unrelated to this game's label
  same_day_other    the goal difference of another game on the same date (the home team of the first listed game): peeking sideways in time
Detectors:
  guard             nhlbet.validity.availability.assert_features_clean (outcome copies, |corr| > 0.98)
  truncation        the canary of games up to D changes when later games are deleted
  corruption        the canary of games up to D changes when later results are scrambled
  shuffle_null      univariate AUC of the canary stays above the chance band (Bonferroni, alpha 0.001) when results are shuffled within a season (no real signal can remain)
  same_day          the canary of the other games on a date changes when one game's result is altered
  too_good          univariate AUC above 0.70 (the best honest pre-game hockey features sit near 0.55 to 0.60)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.validity.availability import LeakageError, assert_features_clean
from nhlbet.validity.null import shuffle_outcomes

MODES = ("own_result", "noisy_own_result", "next_game", "same_day_other")


class LeakyBuilder(FeatureBuilder):
    def __init__(self, mode: str | None, config: BuilderConfig | None = None, seed: int = 0) -> None:
        super().__init__(config)
        self.mode, self._rng = mode, np.random.default_rng(seed)

    def build(self, tables, *a, **k):
        g = tables["games"]
        self._gd = {int(r.game_id): float(r.home_score - r.away_score) for r in g.itertuples() if r.home_score == r.home_score}
        seq = {}
        for r in g.sort_values(["game_date", "game_id"]).itertuples():
            if r.home_score == r.home_score:
                gd = float(r.home_score - r.away_score)
                seq.setdefault(r.home, []).append((r.game_date, gd)); seq.setdefault(r.away, []).append((r.game_date, -gd))
        self._seq = seq
        self._by_day = {d: [int(x) for x in grp.sort_values("game_id").game_id] for d, grp in g.groupby("game_date")}
        return super().build(tables, *a, **k)

    def _game_row(self, g, *a, **k):
        row = super()._game_row(g, *a, **k)
        if self.mode is None:
            return row
        own = self._gd.get(int(g.game_id), np.nan)
        if self.mode == "own_result":
            row["canary"] = own
        elif self.mode == "noisy_own_result":
            row["canary"] = own + self._rng.normal(0, 2.0)
        elif self.mode == "next_game":
            nxt = lambda t: next((gd for d, gd in self._seq.get(t, []) if d > g.game_date), np.nan)
            row["canary"] = nxt(g.home) - nxt(g.away)
        elif self.mode == "same_day_other":
            others = [i for i in self._by_day[g.game_date] if i != int(g.game_id)]
            row["canary"] = self._gd.get(others[0], np.nan) if others else np.nan
        return row


def _max_change(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference; a value that appears or disappears (NaN against a number) counts as an infinite change."""
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return float("inf")
    both = ~na
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0


def _auc(x, y) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(np.unique(y[m])) < 2 or np.std(x[m]) == 0:
        return 0.5
    a = roc_auc_score(y[m], x[m])
    return max(a, 1 - a)                               # direction does not matter for a leak


def _scramble_future(tables, cutoff, seed=1):
    rng = np.random.default_rng(seed)
    t = {k: v.copy() for k, v in tables.items()}
    g = t["games"]; late = set(g.loc[g.game_date >= cutoff, "game_id"]); m = g.game_id.isin(late)
    g.loc[m, "home_score"] = rng.integers(0, 9, m.sum()).astype(float)
    g.loc[m, "away_score"] = rng.integers(0, 9, m.sum()).astype(float)
    g.loc[m, "home_score"] = np.where(g.loc[m, "home_score"] == g.loc[m, "away_score"], g.loc[m, "home_score"] + 1, g.loc[m, "home_score"])
    g.loc[m, "home_win"] = (g.loc[m, "home_score"] > g.loc[m, "away_score"]).astype(int)
    return t


def _frame(tables, mode, seed=0):
    F = LeakyBuilder(mode, seed=seed).build(tables)
    return F[(F.game_type == 2) & F.home_win.notna()] if "canary" in F or mode is None else F


def detect(tables: dict, mode: str | None, cut_date, features: list[str] | None = None) -> dict:
    """Run every detector on the canary ``mode`` (or on the clean builder when None). Returns {detector: flagged?} plus the numbers behind each verdict."""
    cut_date = pd.Timestamp(cut_date)
    F = LeakyBuilder(mode).build(tables)
    F = F[(F.game_type == 2) & F.home_win.notna()]
    y = F.home_win.astype(int).to_numpy()
    feats = features or [c for c in F.columns if c[:2] == "d_"][:12]
    cols = feats + (["canary"] if mode else [])
    out, num = {}, {}
    try:
        assert_features_clean(F, y, cols)
        out["guard"] = False
    except LeakageError:
        out["guard"] = True
    # truncation / corruption: compare the same games' values with the future removed / scrambled
    ids = F.index[F.game_date <= cut_date]
    keep = set(tables["games"].loc[tables["games"].game_date <= cut_date, "game_id"])
    trunc = {k: (v[v.game_id.isin(keep)] if "game_id" in v else v) for k, v in tables.items()}
    for name, tb in (("truncation", trunc), ("corruption", _scramble_future(tables, cut_date + pd.Timedelta(days=1)))):
        G = LeakyBuilder(mode).build(tb)
        both = [i for i in ids if i in G.index]
        d = _max_change(F.loc[both, cols].to_numpy(float), G.loc[both, cols].to_numpy(float)) if both else float("nan")
        num[f"{name}_max_abs_diff"], out[name] = d, bool(d > 1e-9)
    # same-day perturbation: changing the result of one game must not move the features of the OTHER games on its date
    day = F.game_date.value_counts().loc[lambda v: v >= 2].index.sort_values()[len(F.game_date.unique()) // 3]
    ids_day = [int(i) for i in tables["games"].loc[tables["games"].game_date == day, "game_id"]]
    moved = 0.0
    for victim in ids_day[:3]:
        t2 = {k: v.copy() for k, v in tables.items()}
        t2["games"].loc[t2["games"].game_id == victim, ["home_score", "away_score"]] = [9.0, 0.0]
        H = LeakyBuilder(mode).build(t2)
        others = [i for i in ids_day if i != victim and i in F.index and i in H.index]
        if others:
            moved = max(moved, _max_change(F.loc[others, cols].to_numpy(float), H.loc[others, cols].to_numpy(float)))
    num["same_day_max_abs_diff"], out["same_day"] = moved, bool(moved > 1e-9)
    # shuffle null
    Fs = LeakyBuilder(mode).build(shuffle_outcomes(tables, seed=1))
    Fs = Fs[(Fs.game_type == 2) & Fs.home_win.notna()]
    ys = Fs.home_win.astype(int).to_numpy()
    n = len(ys)
    auc_s = _auc(Fs["canary"].to_numpy(float), ys) if mode else max(_auc(Fs[c].to_numpy(float), ys) for c in feats)
    k = 1 if mode else len(feats)                       # Bonferroni across the features tested, two-sided, family alpha 0.001
    from scipy.stats import norm
    n1 = int(ys.sum()); n0 = n - n1
    se = np.sqrt((n + 1) / (12.0 * max(n1, 1) * max(n0, 1)))                         # standard error of an AUC under no association
    num["shuffled_world_auc"], num["shuffled_world_auc_chance_band"] = auc_s, 0.5 + norm.isf(0.001 / (2 * k)) * se
    out["shuffle_null"] = bool(auc_s > num["shuffled_world_auc_chance_band"])
    auc_r = _auc(F["canary"].to_numpy(float), y) if mode else max(_auc(F[c].to_numpy(float), y) for c in feats)
    num["real_world_auc"], out["too_good"] = auc_r, bool(auc_r > 0.70)
    return {"flags": out, "numbers": {k: float(v) for k, v in num.items()}, "caught": any(out.values())}
