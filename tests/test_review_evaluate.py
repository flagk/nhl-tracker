import json

import numpy as np
import pandas as pd

from nhlbet.review.evaluate import MIN_N, compare, grade, group_of, load_log, render_md, report


def rec(pid, group_decision, by="agent", status="REVIEWED", date="2026-10-14", game=1, odds=2.0, market="h2h", side="home", line=None, logged="2026-10-14T21:00:00+00:00"):
    return {"pick_id": pid, "date": date, "game_id": game, "market": market, "line": line, "side": side, "label": pid, "odds": odds, "p_model": 0.6, "p_market": 0.5, "edge": 0.1,
            "stake": 10.0, "player_id": 0, "logged_at": logged, "review": {"status": status, "decision": group_decision, "decided_by": by, "reason": "r", "run_id": "rv-1"}}


def write(tmp, final=(), vetoed=()):
    d = tmp / "data/logs/review"
    d.mkdir(parents=True)
    (d / "final_picks.jsonl").write_text("\n".join(json.dumps(r) for r in final))
    (d / "vetoed_picks.jsonl").write_text("\n".join(json.dumps(r) for r in vetoed))


def test_groups():
    assert group_of(rec("a", "keep")) == "kept" and group_of(rec("a", "flag")) == "flagged"
    assert group_of(rec("a", "veto")) == "vetoed_by_agent" and group_of(rec("a", "veto", by="code")) == "vetoed_by_code"
    assert group_of(rec("a", None, by=None, status="UNREVIEWED")) == "unreviewed"
    assert group_of(rec("a", "flag", by="code", status="UNREVIEWED")) == "flagged_by_code"


def test_latest_record_per_pick_wins_so_a_good_rerun_supersedes_unreviewed(tmp_path):
    write(tmp_path, final=[rec("p1", None, by=None, status="UNREVIEWED", logged="2026-10-14T21:00:00+00:00"), rec("p1", "keep", logged="2026-10-14T21:30:00+00:00")])
    d = load_log(tmp_path)
    assert len(d) == 1 and d.group.iloc[0] == "kept"


def games_frame():
    return pd.DataFrame({"game_id": [1, 2, 3], "home": ["A", "C", "E"], "away": ["B", "D", "F"], "home_score": [3, 1, 2], "away_score": [1, 4, 2], "last_period": ["REG", "REG", "OT"],
                         "home_win": [1.0, 0.0, 1.0]})


def test_grading_uses_the_same_settlement_rules_and_a_flat_stake():
    d = pd.DataFrame([{**{k: v for k, v in rec("a", "keep").items() if k not in ("review",)}, "group": "kept"},
                      {**{k: v for k, v in rec("b", "keep", game=2).items() if k != "review"}, "group": "kept"},
                      {**{k: v for k, v in rec("c", "veto", game=3, market="totals", side="over", line=4.0).items() if k != "review"}, "group": "vetoed_by_agent"},
                      {**{k: v for k, v in rec("d", "keep", game=9).items() if k != "review"}, "group": "kept"}])
    g = grade(d, games_frame(), pd.DataFrame(columns=["game_id", "player_id", "sog", "points", "goals", "assists"]))
    assert list(g.result) == ["won", "lost", "push", "pending"]                       # 2+2 = 4 goals on a 4.0 line pushes; the game 9 result is not in yet
    assert list(g.profit) == [10.0, -10.0, 0.0, 0.0]


def synth(n_kept, n_veto, kept_roi_wins, veto_roi_wins, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for grp, n, pw in (("kept", n_kept, kept_roi_wins), ("vetoed_by_agent", n_veto, veto_roi_wins)):
        for i in range(n):
            won = rng.random() < pw
            rows.append({"group": grp, "result": "won" if won else "lost", "profit": 10.0 if won else -10.0, "stake": 10.0, "date": f"2026-{1 + i % 11:02d}-{1 + i % 27:02d}", "odds": 2.0,
                         "edge": 0.05, "clv_move_pts": np.nan, "market": "h2h"})
    return pd.DataFrame(rows)


def test_too_few_picks_is_insufficient_data():
    c = compare(synth(30, 12, 0.55, 0.45))
    assert c["verdict"].startswith("insufficient data") and "30 kept and 12" in c["verdict"]


def test_a_real_difference_in_a_large_sample_is_reported_as_value():
    c = compare(synth(MIN_N * 6, MIN_N * 6, 0.60, 0.40))
    assert c["diff"] > 0.2 and c["diff_ci"][0] > 0 and "did credibly worse" in c["verdict"]


def test_no_difference_in_a_large_sample_is_reported_as_no_demonstrated_value():
    c = compare(synth(MIN_N * 3, MIN_N * 3, 0.50, 0.50, seed=3))
    assert c["diff_ci"][0] <= 0 and c["verdict"].startswith("the agent adds no demonstrated value")


def test_vetoed_doing_better_than_kept_is_also_no_value():
    c = compare(synth(MIN_N * 3, MIN_N * 3, 0.40, 0.60))
    assert c["diff"] < 0 and c["verdict"].startswith("the agent adds no demonstrated value")


def test_report_renders_with_empty_and_small_logs():
    assert "Insufficient data" in render_md(report(pd.DataFrame(columns=["group"])))
    md = render_md(report(synth(20, 20, 0.5, 0.5)))
    assert "| kept |" in md and "insufficient data" in md


def clv_frame(n_kept, n_veto, mean_kept, mean_veto, days=40, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for grp, n, m in (("kept", n_kept, mean_kept), ("vetoed_by_agent", n_veto, mean_veto)):
        for i in range(n):
            rows.append({"group": grp, "clv_move_pts": rng.normal(m, sd), "date": f"2026-{1 + (i % days) // 28:02d}-{1 + (i % days) % 28:02d}", "result": "won", "profit": 1.0, "stake": 10.0, "pick_id": f"{grp}{i}"})
    return pd.DataFrame(rows)


def test_clv_comparison_is_insufficient_below_the_policy_minimum_and_does_not_read_a_verdict():
    from nhlbet.review.evaluate import compare_clv
    c = compare_clv(clv_frame(40, 10, 0.5, -0.5), n_perm=500)
    assert c["verdict"].startswith("insufficient data") and "at least 100 and 30" in c["verdict"]


def test_vetoes_that_really_do_worse_on_clv_are_detected_with_an_interval_and_a_permutation_test():
    from nhlbet.review.evaluate import compare_clv
    c = compare_clv(clv_frame(300, 60, 0.3, -0.3, sd=1.0), n_perm=1500)
    assert c["diff"] > 0.4 and c["diff_ci"][0] > 0 and c["p_permutation_one_sided"] < 0.01 and "carry information" in c["verdict"]


def test_no_difference_or_a_reverse_difference_is_reported_as_no_demonstrated_value():
    from nhlbet.review.evaluate import compare_clv
    same = compare_clv(clv_frame(300, 60, 0.0, 0.0, seed=2), n_perm=1500)
    assert same["verdict"].startswith("the agent adds no demonstrated value") and same["p_permutation_one_sided"] > 0.05
    rev = compare_clv(clv_frame(300, 60, -0.3, 0.3, seed=3), n_perm=800)
    assert rev["diff"] < 0 and rev["verdict"].startswith("the agent adds no demonstrated value")


def test_permutation_labels_are_shuffled_within_dates_so_a_day_effect_alone_is_not_a_finding():
    from nhlbet.review.evaluate import compare_clv
    rng = np.random.default_rng(5)
    rows = []
    for day in range(60):
        shift = rng.normal(0, 1.5)                                    # a big day effect shared by both groups
        for grp, n in (("kept", 6), ("vetoed_by_agent", 1)):
            for i in range(n):
                rows.append({"group": grp, "clv_move_pts": shift + rng.normal(0, 0.1), "date": f"2026-{1 + day // 28:02d}-{1 + day % 28:02d}", "result": "won", "profit": 1.0, "stake": 10.0, "pick_id": f"{grp}{day}{i}"})
    c = compare_clv(pd.DataFrame(rows), n_perm=1000)
    assert abs(c["diff"]) < 0.3 and c["p_permutation_one_sided"] > 0.05


def test_flag_rate_is_reported_per_day():
    from nhlbet.review.evaluate import flag_rate_table
    d = pd.DataFrame({"date": ["2026-10-14"] * 4 + ["2026-10-15"] * 2, "group": ["kept", "flagged", "flagged", "vetoed_by_agent", "kept", "kept"], "pick_id": list("abcdef")})
    t = flag_rate_table(d).set_index("date")
    assert t.loc["2026-10-14", "flag_rate"] == 0.5 and t.loc["2026-10-14", "veto_rate"] == 0.25 and t.loc["2026-10-15", "flag_rate"] == 0.0 and t.loc["2026-10-14", "picks"] == 4


def test_the_review_schema_has_no_field_that_could_carry_a_probability_price_stake_or_new_pick():
    import json
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parent.parent / "agent" / "picks_review.schema.json").read_text())
    names = set()

    def walk(x):
        if isinstance(x, dict):
            names.update(x.get("properties", {}).keys())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(schema)
    banned = {"p_model", "p_market", "p_adj", "p_win", "probability", "prob", "price", "prices", "odds", "stake", "stakes", "edge", "ev", "line", "point", "new_pick", "new_picks", "picks", "bet", "bets",
              "kelly", "units", "decimal", "side", "market", "pick"}
    assert not (names & banned), names & banned                       # exact names: "odds_movement" and "lineup_changes" are the allowed check names
    def closed(x):
        if isinstance(x, dict):
            if x.get("type") == "object":
                assert x.get("additionalProperties") is False, x
            for v in x.values():
                closed(v)
        elif isinstance(x, list):
            for v in x:
                closed(v)
    closed(schema)
