"""Phase 1 validity tests: model-level truncation invariance, in-fold fitting, outcome shuffle (null), planted leak, holdout lock."""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhlbet.features.builder import BuilderConfig, FeatureBuilder
from nhlbet.models.base import EloOnly, HomeRate, Logistic
from nhlbet.models.walkforward import fit_stack, predict_stack, walk_forward
from nhlbet.validity import holdout
from nhlbet.validity.availability import LeakageError, assert_features_clean
from nhlbet.validity.null import shuffle_outcomes
from tests.conftest import make_league

ROOT = Path(__file__).resolve().parent.parent
FEATS = ["d_elo", "h_b2b", "a_b2b", "d_xg_share", "d_gf_ewm"]


def build(tables):
    F = FeatureBuilder(BuilderConfig()).build(tables)
    return F[F.game_type == 2]


def zoo(features=None):
    f = features or [c for c in FEATS]
    return {"home_rate": lambda: HomeRate(), "elo": lambda: EloOnly(), "logistic": lambda: Logistic(f)}


@pytest.fixture(scope="module")
def big_league():
    return make_league(n_days=300, seed=4)


def usable(F):
    return [c for c in FEATS if c in F.columns]


# ---------------------------------------------------------------- 2. truncation invariance, whole pipeline
def test_predictions_for_games_up_to_D_do_not_change_when_later_data_is_added(big_league):
    D = pd.Timestamp("2024-05-01")
    full = build(big_league)
    trunc_tables = {k: v[v.game_id.isin(big_league["games"].loc[big_league["games"].game_date <= D, "game_id"])] for k, v in big_league.items()}
    cut = build(trunc_tables)
    z = zoo(usable(full))
    a = walk_forward(full, z, "2024-02-01", 14, min_train=120, inner_folds=3, n_jobs=1)
    b = walk_forward(cut, z, "2024-02-01", 14, min_train=120, inner_folds=3, n_jobs=1)
    a = a[a.game_date <= D]
    assert len(a) > 60 and set(a.index) == set(b.index)
    a, b = a.sort_index(), b.sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)         # every model, the stack, every calibrator and the weighted average: identical


def test_the_truncation_test_would_catch_a_leak(big_league):
    """Power check: a feature that peeks at the NEXT game's result makes truncated and full predictions differ."""
    D = pd.Timestamp("2024-05-01")
    full = build(big_league)
    full["peek"] = full.groupby("home").home_win.shift(-1).fillna(0.5)       # the same home team's next result: only available with later data
    cut = full[full.game_date <= D].copy()
    cut["peek"] = cut.groupby("home").home_win.shift(-1).fillna(0.5)         # in the truncated world the last games have no "next" game
    z = zoo(usable(full) + ["peek"])
    a = walk_forward(full, z, "2024-02-01", 14, min_train=120, inner_folds=3, n_jobs=1)
    b = walk_forward(cut, z, "2024-02-01", 14, min_train=120, inner_folds=3, n_jobs=1)
    a = a[a.game_date <= D].sort_index()
    assert not np.allclose(a["logistic"].to_numpy(), b.loc[a.index, "logistic"].to_numpy(), atol=1e-9)


# ---------------------------------------------------------------- 3. everything is fitted inside the fold, from the past only
def test_preprocessing_and_calibration_never_use_other_test_rows(big_league):
    F = build(big_league)
    cols = usable(F)
    F = F.sort_values("game_date")
    tr, te = F.iloc[:300], F.iloc[300:340]
    fitted, stack, wavg, cals = fit_stack(zoo(cols), tr, tr.home_win.astype(int).to_numpy(), tr.game_date.reset_index(drop=True), inner_folds=3, stack_inputs=("elo", "logistic"))
    base = predict_stack(fitted, stack, wavg, cals, te, stack_inputs=("elo", "logistic"))
    poisoned = te.copy()
    poisoned.iloc[1:, [te.columns.get_loc(c) for c in cols if c != "d_elo"]] = 1e6          # extreme values in every OTHER row (scaler / imputer would shift if they were refit on test)
    poisoned.iloc[1:, poisoned.columns.get_loc("elo_prob")] = 0.99
    alt = predict_stack(fitted, stack, wavg, cals, poisoned, stack_inputs=("elo", "logistic"))
    pd.testing.assert_series_equal(base.iloc[0], alt.iloc[0], check_exact=False, rtol=1e-12, atol=1e-12)    # row 0's every output (models, stack, calibrators) is unchanged


def test_selection_and_tuning_cutoffs_precede_the_holdout():
    lock = holdout.load_lock()
    found = []
    for f in list((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args and isinstance(node.args[0], ast.Constant):
                name = str(node.args[0].value)
                if re.search(r"(tune|select|sel)[-_]?end|tune_end", name):
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            found.append((f.name, name, kw.value.value))
    assert found, "no tuning / selection cutoffs found: the test would be vacuous"
    for fname, name, val in found:
        assert pd.Timestamp(val) < pd.Timestamp(lock["start"]), f"{fname} {name}={val} reaches into the holdout"


# ---------------------------------------------------------------- 7. null tests and the planted leak
def test_shuffling_outcomes_within_season_removes_skill_and_the_pipeline_notices(big_league):
    from sklearn.metrics import roc_auc_score
    real = build(big_league)
    null = build(shuffle_outcomes(big_league, seed=11))
    z = zoo(usable(real))
    pr = walk_forward(real, z, "2024-01-15", 14, min_train=120, inner_folds=3, n_jobs=1)
    pn = walk_forward(null, z, "2024-01-15", 14, min_train=120, inner_folds=3, n_jobs=1)
    auc_real, auc_null = roc_auc_score(pr.y, pr.elo), roc_auc_score(pn.y, pn.elo)
    assert auc_real > 0.55, "power check: the unshuffled league has real strength differences and Elo should find them"
    assert abs(auc_null - 0.5) < 0.08 and auc_null < auc_real - 0.04
    n = len(pn)
    base = pn.y.mean()
    ll = lambda p: float(-np.mean(pn.y * np.log(p) + (1 - pn.y) * np.log(1 - p)))
    assert ll(pn.logistic.clip(1e-6, 1 - 1e-6)) > ll(np.full(n, base)) - 0.02                     # no out-of-sample log-loss gain over the base rate


def test_shuffled_world_betting_loses_about_the_vig_and_beats_random_picks_no_better_than_chance(big_league):
    null = build(shuffle_outcomes(big_league, seed=12))
    z = zoo(usable(null))
    p = walk_forward(null, z, "2024-01-15", 14, min_train=120, inner_folds=3, n_jobs=1)
    vig = 0.045
    base = float(p.y.mean())                                               # the market knows the base rate (home advantage): fair probabilities, 4.5% margin on top
    dec_home, dec_away = 1 / (base * (1 + vig)), 1 / ((1 - base) * (1 + vig))
    y = p.y.to_numpy()
    home_pick = (p.logistic >= 0.5).to_numpy()
    dec = np.where(home_pick, dec_home, dec_away)
    roi = float(np.mean((home_pick == (y == 1)) * dec - 1))
    rng = np.random.default_rng(0)
    def random_roi():
        pick = rng.integers(0, 2, len(p)).astype(bool)
        return np.mean((pick == (y == 1)) * np.where(pick, dec_home, dec_away) - 1)
    rand = np.array([random_roi() for _ in range(4000)])
    p_value = float(np.mean(rand >= roi))
    assert -0.20 < roi < 0.04, "against a fair market with a margin, random outcomes cost about the vig (-4.5%)"
    assert abs(float(rand.mean()) + 0.043) < 0.01
    assert p_value > 0.01


def test_a_planted_outcome_feature_is_caught_before_any_model_is_fit(big_league):
    F = build(big_league)
    y = F.home_win.astype(int).to_numpy()
    assert_features_clean(F, y, usable(F))                                                  # the real feature set passes
    for name, col in (("planted", y.astype(float)), ("planted_flipped", 1.0 - y), ("planted_noisy", y + np.random.default_rng(0).normal(0, 0.05, len(y)))):
        G = F.assign(**{name: col})
        with pytest.raises(LeakageError):
            assert_features_clean(G, y, usable(F) + [name])
    with pytest.raises(LeakageError):
        assert_features_clean(F, y, usable(F) + ["home_score"])                             # an outcome column by name
    with pytest.raises(LeakageError):
        assert_features_clean(F.assign(closing_prob=0.5), y, usable(F) + ["closing_prob"])
    with pytest.raises(LeakageError):
        assert_features_clean(F, y, usable(F) + ["not_a_column"])


def test_the_guard_runs_inside_every_model_fit(big_league):
    F = build(big_league).sort_values("game_date").iloc[:300]
    F = F.assign(planted=F.home_win.astype(float))
    with pytest.raises(LeakageError):
        fit_stack(zoo(usable(F) + ["planted"]), F, F.home_win.astype(int).to_numpy(), F.game_date.reset_index(drop=True), inner_folds=3, stack_inputs=("elo", "logistic"))


# ---------------------------------------------------------------- 8. holdout lock
def test_lock_file_matches_the_policy_and_the_committed_predictions():
    from nhlbet.policy import policy_get
    lock = holdout.load_lock()
    again = holdout.compute_lock()
    assert again["hash_sha256"] == lock["hash_sha256"] and again["start"] == lock["start"] and again["n_games"] == lock["n_games"]
    for k_pol, k_lock in (("start", "start"), ("end", "end"), ("n_games", "n_games"), ("hash_sha256", "hash_sha256")):
        assert policy_get("holdout", k_pol) == lock[k_lock]
    w = pd.read_csv(ROOT / "reports" / "walkforward_predictions.csv", index_col=0, parse_dates=["game_date"])
    assert lock["n_games"] >= max(0.2 * len(w), 656) - 1 and lock["previously_examined"] is True


def test_research_view_excludes_the_holdout_and_touching_it_raises():
    lock = holdout.load_lock()
    F = pd.DataFrame({"game_date": pd.date_range("2025-12-01", "2026-03-01"), "x": 1})
    v = holdout.research_view(F)
    assert v.game_date.max() < pd.Timestamp(lock["start"]) and len(v) < len(F)
    holdout.assert_research_only(v.game_date, "ok")
    with pytest.raises(holdout.HoldoutViolation):
        holdout.assert_research_only(F.game_date, "tuning code")


def test_the_holdout_can_be_evaluated_exactly_once(tmp_path):
    ev = tmp_path / "eval.json"
    out = holdout.evaluate_once(lambda lock: {"n": lock["n_games"]}, path=ev)
    assert out["n"] == holdout.load_lock()["n_games"] and ev.exists()
    with pytest.raises(holdout.HoldoutViolation):
        holdout.evaluate_once(lambda lock: {"n": 0}, path=ev)
    assert json.loads(ev.read_text())["result"]["n"] == out["n"]                        # the first result stands


def test_only_allowlisted_code_reads_the_holdout_lock():
    allow = {"nhlbet/validity/holdout.py", "scripts/holdout_lock.py", "scripts/holdout_eval.py", "scripts/compare.py"}
    pat = re.compile(r"holdout_lock|load_lock|evaluate_once|nhlbet\.validity\.holdout")
    offenders = []
    for f in list((ROOT / "nhlbet").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        rel = str(f.relative_to(ROOT))
        if rel not in allow and pat.search(f.read_text()):
            offenders.append(rel)
    assert not offenders, f"{offenders} read holdout lock information without being on the allowlist (selection code must use research_view)"
