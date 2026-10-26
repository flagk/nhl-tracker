"""When the review runs: from the schedule in site/picks.json, once per start wave, never from a fixed clock time."""
import json
from datetime import datetime, timezone

from nhlbet.gate import decide_files, review_target, slate_waves


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def repo(tmp_path, starts, date="2026-10-14", cand_games=None, passes=()):
    (tmp_path / "site").mkdir(exist_ok=True)
    (tmp_path / "site" / "picks.json").write_text(json.dumps({"date": date, "games": [{"game_id": i + 1, "start_utc": s} for i, s in enumerate(starts)]}))
    (tmp_path / "data" / "review").mkdir(parents=True, exist_ok=True)
    ids = cand_games if cand_games is not None else list(range(1, len(starts) + 1))
    (tmp_path / "data/review/picks_candidate.json").write_text(json.dumps({"date": date, "picks": [{"game_id": i, "pick_id": f"pk_{i:012x}"} for i in ids]}))
    if passes:
        (tmp_path / "data/review/picks_final.json").write_text(json.dumps({"date": date, "passes": [{"run_id": p, "status": "reviewed"} for p in passes]}))
    return tmp_path


TWO_WAVES = ("2026-10-14T23:00:00Z", "2026-10-14T23:30:00Z", "2026-10-15T02:00:00Z")


def test_waves_are_clusters_of_start_times(tmp_path):
    d, w = slate_waves(repo(tmp_path, TWO_WAVES))
    assert d == "2026-10-14" and [x["game_ids"] for x in w] == [[1, 2], [3]]
    assert [x["key"] for x in w] == ["rv-2026-10-14-2300", "rv-2026-10-14-0200"]


def test_review_opens_75_minutes_before_the_first_puck_drop(tmp_path):
    r = repo(tmp_path, TWO_WAVES)
    assert not decide_files("review", utc("2026-10-14T21:30:00"), r)[0]                     # 90 min before
    ok, why = decide_files("review", utc("2026-10-14T21:45:00"), r)                         # exactly 75
    assert ok and "rv-2026-10-14-2300" in why
    assert decide_files("review", utc("2026-10-14T22:40:00"), r)[0]                         # 20 min before: still worth it
    assert review_target(utc("2026-10-14T21:50:00"), r)["game_ids"] == [1, 2]


def test_a_game_that_is_about_to_start_is_too_late_and_the_next_wave_takes_over(tmp_path):
    r = repo(tmp_path, TWO_WAVES)
    assert not decide_files("review", utc("2026-10-14T22:58:00"), r)[0]                     # 2 min before the first wave, next wave is 3 h away
    t = review_target(utc("2026-10-15T00:50:00"), r)                                         # 70 min before the 02:00 wave
    assert t["key"] == "rv-2026-10-14-0200" and t["game_ids"] == [3]


def test_second_pass_for_the_later_wave_after_the_first_is_done(tmp_path):
    r = repo(tmp_path, TWO_WAVES, passes=("rv-2026-10-14-2300",))
    assert not decide_files("review", utc("2026-10-14T22:00:00"), r)[0]                     # wave 1 done, wave 2 not open yet
    assert decide_files("review", utc("2026-10-15T00:50:00"), r)[0]


def test_a_failed_pass_is_not_retried_automatically(tmp_path):
    r = repo(tmp_path, TWO_WAVES, passes=("rv-2026-10-14-2300",))
    (r / "data/review/picks_final.json").write_text(json.dumps({"date": "2026-10-14", "passes": [{"run_id": "rv-2026-10-14-2300", "status": "unreviewed"}]}))
    assert not decide_files("review", utc("2026-10-14T22:00:00"), r)[0]


def test_no_games_wrong_day_or_no_candidates(tmp_path):
    r = repo(tmp_path, [])
    assert decide_files("review", utc("2026-10-14T21:00:00"), r) == (False, "no games today")
    r = repo(tmp_path, TWO_WAVES)
    ok, why = decide_files("review", utc("2026-10-15T21:45:00"), r)
    assert not ok and "not built" in why
    r = repo(tmp_path, TWO_WAVES, cand_games=[3])
    ok, why = decide_files("review", utc("2026-10-14T21:50:00"), r)
    assert not ok and "no candidate" in why
