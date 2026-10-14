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
