"""Phase 3 evaluation: multiplicity corrections, no-vig methods, closing-line value from raw quotes, sample-size labels, baselines, model-vs-market scoring."""
import numpy as np
import pandas as pd
import pytest

from nhlbet.data.store import Store
from nhlbet.odds.math import devig
from nhlbet.report import evaluation as ev

CONTROL = "every_game"      # replaced below by a real control name if this one is not a control
from nhlbet.report.validation import control_names

CTRL = sorted(control_names())[0]


def test_holm_and_benjamini_hochberg_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert ev.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])                      # sorted 0.005, 0.01, 0.03, 0.04 x (4, 3, 2, 1), made monotone
    assert ev.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert (ev.holm(p) >= np.asarray(p)).all() and (ev.benjamini_hochberg(p) >= np.asarray(p)).all() and (ev.holm(p) >= ev.benjamini_hochberg(p) - 1e-12).all()
    assert np.isnan(ev.correct([0.01, np.nan])[1]) and ev.correct([0.01, np.nan])[0] == pytest.approx(0.01)
    assert ev.correct([0.02, 0.03], "bh")[0] == pytest.approx(0.03)


def test_the_corrected_threshold_is_stricter_with_more_tests():
    assert ev.holm([0.03])[0] == pytest.approx(0.03)
    assert ev.holm([0.03] + [0.5] * 9)[0] == pytest.approx(0.3)                          # one nominal discovery among ten tests is no longer one


@pytest.mark.parametrize("dec", [[1.5, 2.7], [1.91, 1.91], [2.2, 3.4, 3.1]])
def test_every_method_sums_to_one_for_two_way_and_three_way_markets(dec):
    for m in ("proportional", "power", "shin"):
        p = devig(dec, m)
        assert p.sum() == pytest.approx(1.0, abs=1e-9) and (p > 0).all()
    assert devig([1.91, 1.91], "proportional") == pytest.approx([0.5, 0.5])
    p = devig([1.4, 3.2], "proportional"), devig([1.4, 3.2], "power"), devig([1.4, 3.2], "shin")
    assert p[1][0] > p[0][0] and p[2][0] > p[0][0]                                      # power and Shin shift mass to the favourite (longshot bias), proportional does not


def snap(cap, gid, rows):
    out = []
    for book, market, outcome, point, price in rows:
        out.append(dict(captured_at=cap, event_id=f"e{gid}", commence_time=f"2026-10-08T23:00:00Z", home="HHH", away="AAA", book=book, market=market, outcome=outcome, point=point, price=price, book_updated=None, game_id=gid))
    return out


def world(n_days=6, win_prob=0.5, seed=1):
    st = Store(":memory:")
    rng = np.random.default_rng(seed)
    games, shadow, snaps, recs = [], [], [], []
    gid = 0
    for d in range(n_days):
        day = f"2026-10-{8 + d:02d}"
        for k in range(8):
            gid += 1
            hw = int(rng.random() < win_prob)
            hs, as_ = (4, 2) if hw else (1, 3)
            games.append(dict(game_id=gid, season=1, game_type=2, game_date=day, start_utc=f"{day}T23:00:00Z", home="HHH", away="AAA", home_score=hs, away_score=as_, status="FINAL", last_period="REG",
                              home_win=hw, source="nhl_api", updated_at=None))
            # decision capture at 21:00 (home 50/50 at 1.95), closing capture at 22:55 where the market has moved TOWARD the home side
            for book in ("bk1", "bk2"):
                snaps += snap(f"{day}T21:00:00+00:00", gid, [(book, "h2h", "HHH", 0, 1.95), (book, "h2h", "AAA", 0, 1.95),
                                                           (book, "totals", "Over", 6.5, 1.95), (book, "totals", "Under", 6.5, 1.95)])
                snaps += snap(f"{day}T22:55:00+00:00", gid, [(book, "h2h", "HHH", 0, 1.80), (book, "h2h", "AAA", 0, 2.10),
                                                           (book, "totals", "Over", 6.5, 1.95), (book, "totals", "Under", 6.5, 1.95)])
            for strat, side in (("edge_1pct", "home"), (CTRL, "away")):
                dec = 1.95
                shadow.append(dict(run_id=f"r{day}", run_at=f"{day}T21:05:00+00:00", game_id=gid, strategy=strat, game_date=day, action="BET", side=side, team="x", book="bk1", decimal=dec, stake=10.0,
                                   p_model=0.56, p_adj=0.56, p_market=0.5, edge=0.06, ev=0.0, market=None, point=None, label="x"))
            shadow.append(dict(run_id=f"r{day}", run_at=f"{day}T21:05:00+00:00", game_id=gid, strategy="totals_edge", game_date=day, action="BET", side="over", team="x", book="bk1", decimal=1.95, stake=10.0,
                               p_model=0.55, p_adj=0.55, p_market=0.5, edge=0.05, ev=0.0, market="totals", point=6.5, label="x"))
            recs.append(dict(run_id=f"r{day}", run_at=f"{day}T21:05:00+00:00", run_type="late", game_id=gid, game_date=day, home="HHH", away="AAA", model_version="v", p_model=0.58, p_adj=0.55, p_market=0.5,
                             action="BET", side="home", team="HHH", book="bk1", decimal=1.95, stake=10.0, edge=0.08, ev=0.1, odds_captured_at=f"{day}T21:00:00+00:00"))
    st.upsert("games", games, ["game_id"])
    st.upsert("odds_snapshots", snaps, ["captured_at", "event_id", "book", "market", "outcome", "point"])
    st.upsert("shadow_bets", shadow, ["run_id", "game_id", "strategy"])
    st.upsert("recommendations", recs, ["run_id", "game_id"])
    return st


def test_clv_from_raw_quotes_is_the_margin_free_move_of_the_no_vig_probability():
    st = world()
    d = ev.pick_frame(st)
    m = d[(d.group == "model") & (d.market == "h2h")]
    assert len(m) == 48 and m.p_dec_proportional.eq(0.5).all()
    close_home = (1 / 1.80) / (1 / 1.80 + 1 / 2.10)                                              # proportional no-vig at the close
    assert m.p_close_proportional.iloc[0] == pytest.approx(close_home)
    assert m.clv_pts_proportional.iloc[0] == pytest.approx(100 * (close_home - 0.5)) and m.clv_pts_proportional.iloc[0] > 0
    c = d[(d.group == "control") & (d.market == "h2h")]
    assert c.clv_pts_proportional.iloc[0] == pytest.approx(-100 * (close_home - 0.5))             # the away side lost the same amount: CLV of a pick and its opposite sum to zero
    assert m.clv_lo_proportional.iloc[0] == pytest.approx(np.log(close_home / (1 - close_home)))      # log-odds move from 50/50
    t = d[(d.market == "totals")]
    assert t.clv_pts_proportional.abs().max() < 1e-12                                              # the total did not move: zero CLV, not a missing value
    assert (d.p_dec_shin.notna() & d.p_dec_power.notna()).all()


def test_a_capture_far_from_the_start_is_not_a_closing_price():
    st = world()
    with st.tx() as c:
        c.execute("DELETE FROM odds_snapshots WHERE captured_at LIKE '%22:55:00%'")
    d = ev.pick_frame(st)
    assert d[d.market == "h2h"].clv_pts_proportional.isna().all()                               # only the 21:00 capture remains: 2 hours before the start, no closing price


def test_market_table_labels_small_samples_and_unmeasurable_clv_and_applies_holm():
    st = world(n_days=3)
    d = ev.pick_frame(st)
    t = ev.market_table(d, "proportional").set_index("market")
    assert t.loc["h2h", "picks"] == 24 and t.loc["h2h", "verdict_roi"].startswith("insufficient evidence (n=24 < 100)")
    assert t.loc["h2h", "p_clv_adj"] >= t.loc["h2h", "p_clv"] - 1e-12
    d2 = d.copy()
    d2.loc[d2.market == "totals", "clv_pts_proportional"] = np.nan
    assert ev.market_table(d2, "proportional").set_index("market").loc["totals", "verdict_clv"].startswith("CLV not measurable")


def test_a_real_clv_edge_in_a_big_sample_is_called_credible_and_zero_is_called_no_edge():
    st = world(n_days=14, seed=3)
    d = ev.pick_frame(st)
    t = ev.market_table(d, "proportional").set_index("market")
    assert t.loc["h2h", "clv_pts"] > 3 and "credible positive CLV" in t.loc["h2h", "verdict_clv"]
    assert "no demonstrable edge on CLV" in t.loc["totals", "verdict_clv"] or "CLV" in t.loc["totals", "verdict_clv"]        # a flat line has zero CLV: never "credible"
    assert "credible positive" not in t.loc["totals", "verdict_clv"]


def test_split_cells_use_one_holm_family_and_flag_small_cells():
    d = ev.pick_frame(world(n_days=14, seed=2))
    cells = ev.split_cells(d, "proportional")
    assert set(cells.split) == {"market", "month", "home_away", "fav_dog", "price_bucket"}
    big = cells[cells.n >= 30]
    assert (cells.family_size == len(big)).all() and (big.p_roi_adj >= big.p_roi - 1e-12).all()
    assert cells[cells.n < 30].label.str.startswith("insufficient evidence").all() if (cells.n < 30).any() else True


def test_edge_threshold_table_is_monotone_in_n_and_method_sensitivity_covers_each_method():
    d = ev.pick_frame(world(n_days=10, seed=4))
    t = ev.edge_threshold_table(d, "proportional")
    assert t.picks.is_monotonic_decreasing and t.picks.iloc[0] > 0
    ms = ev.method_sensitivity(d)
    assert set(ms.method) == {"proportional", "power", "shin"}


def test_baselines_random_sides_lose_about_the_margin_and_the_favourite_is_priced_from_raw_quotes():
    st = world(n_days=10, seed=5)
    d = ev.pick_frame(st)
    b = ev.baselines(st, d, "proportional", B=2000)
    assert b["n"] > 100 and -0.12 < b["random_side_mean_roi"] < 0.12 and 0 < b["p_random_at_least_as_good"] <= 1
    assert b["market_favourite_roi"] == pytest.approx(b["market_favourite_roi"])


def test_model_vs_market_scoring_on_the_same_games():
    st = world(n_days=14, win_prob=0.6, seed=6)
    s = ev.score_games(st, "proportional")
    assert s["n"] == 112 and set(s) >= {"model", "market_at_decision", "market_close", "reliability"}
    assert 0 < s["model"]["log_loss"] < 1 and s["market_close"]["brier"] > 0
    assert s["model_vs_market_close_log_loss_gain"]["ci"][0] <= s["model_vs_market_close_log_loss_gain"]["mean_gain"] <= s["model_vs_market_close_log_loss_gain"]["ci"][1]


def test_official_picks_go_through_the_same_clv_machinery():
    st = world(n_days=4)
    o = ev.official_picks(st)
    assert len(o) == 32 and set(o.group) == {"official"} and o.p_dec_proportional.eq(0.5).all()
    assert o.clv_pts_proportional.iloc[0] > 0 and o.settled.all()
    t = ev.market_table(o, "proportional", group="official")
    assert t.loc[0, "picks"] == 32 and t.loc[0, "verdict_roi"].startswith("insufficient evidence")
