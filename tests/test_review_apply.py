import json
from datetime import datetime, timezone

import pytest

from nhlbet.review.apply import (ReviewIntegrityError, append_log, apply_review, assert_subset, attach_to_payload, load_state, merge_state, prepare_input, strip_review,
                                  write_all)
from nhlbet.review.contract import validate_review
from tests.review_helpers import NOW, candidate, good_review, payload

KEEPFACT = {"floor": "keep", "summary": "price 2.00 -> 2.00; edge +8.0 -> +8.0 pts", "veto": False}


def facts_for(c, **override):
    f = {p["pick_id"]: dict(KEEPFACT) for p in c["picks"]}
    f.update(override)
    return f


def test_prepare_input_covers_only_the_wave_and_skips_code_vetoes():
    c = candidate()
    ids = [p["pick_id"] for p in c["picks"]]
    f = facts_for(c, **{ids[0]: {"floor": "veto", "summary": "gone"}})
    inp = prepare_input(c, {1}, f, NOW)
    assert [p["pick_id"] for p in inp["picks"]] == [ids[1]]                  # game 2 is a later wave; the code-vetoed pick needs no agent
    assert inp["picks"][0]["odds_movement_fact"] and set(inp["games"]) == {"1"}


def test_decisions_split_final_and_vetoed_and_never_change_numbers():
    c = candidate()
    inp = prepare_input(c, {1, 2}, facts_for(c), NOW)
    rv = good_review(inp)
    rv["reviews"][0]["decision"] = "veto"
    rv["reviews"][1]["decision"] = "flag"
    assert validate_review(rv, inp, NOW) == []
    res = apply_review(c, {1, 2}, facts_for(c), rv, [], "rv-test", NOW)
    assert res["status"] == "reviewed"
    assert [r["review"]["decision"] for r in res["final"]] == ["flag", "keep"] and len(res["vetoed"]) == 1
    flagged = res["final"][0]
    assert flagged["review"]["warning"] and flagged["p_model"] == c["picks"][1]["p_model"] and flagged["odds"] == c["picks"][1]["odds"]
    assert res["vetoed"][0]["review"]["reason"] and res["vetoed"][0]["pick_id"] == c["picks"][0]["pick_id"]


def test_odds_movement_in_the_log_is_the_codes_fact_not_the_agents_text():
    c = candidate()
    inp = prepare_input(c, {1, 2}, facts_for(c), NOW)
    rv = good_review(inp)
    rv["reviews"][0]["checks"]["odds_movement"] = {"found": "the agent made this up", "source_urls": []}
    res = apply_review(c, {1, 2}, facts_for(c), rv, [], "rv-test", NOW)
    assert res["final"][0]["review"]["checks"]["odds_movement"] == KEEPFACT["summary"]


def test_a_code_veto_beats_an_agent_keep_and_survives_an_agent_failure():
    c = candidate()
    pid = c["picks"][0]["pick_id"]
    f = facts_for(c, **{pid: {"floor": "veto", "summary": "edge +1.0 pts < 3 minimum"}})
    inp = prepare_input(c, {1, 2}, f, NOW)
    res = apply_review(c, {1, 2}, f, good_review(inp), [], "rv-test", NOW)
    assert [r["pick_id"] for r in res["vetoed"]] == [pid] and res["vetoed"][0]["review"]["decided_by"] == "code"
    bad = apply_review(c, {1, 2}, f, None, ["picks_review.json is missing"], "rv-test", NOW)
    assert [r["pick_id"] for r in bad["vetoed"]] == [pid]


def test_failure_publishes_every_pick_as_unreviewed():
    c = candidate()
    res = apply_review(c, {1, 2}, facts_for(c), None, ["picks_review.json is missing"], "rv-test", NOW)
    assert res["status"] == "unreviewed" and len(res["final"]) == 3 and not res["vetoed"]
    assert all(r["review"]["status"] == "UNREVIEWED" and r["review"]["decision"] is None and "UNREVIEWED" in r["review"]["warning"] for r in res["final"])
    assert [r["p_model"] for r in res["final"]] == [p["p_model"] for p in c["picks"]]
    assert all("did not complete" in r["review"]["reason"] and "picks_review.json is missing" in r["review"]["reason"] for r in res["final"])


def test_invalid_output_is_a_failure_even_if_a_parsed_object_is_passed():
    c = candidate()
    inp = prepare_input(c, {1, 2}, facts_for(c), NOW)
    rv = good_review(inp)
    errs = validate_review({**rv, "reviews": rv["reviews"][:1]}, inp, NOW)
    assert errs
    res = apply_review(c, {1, 2}, facts_for(c), rv, errs, "rv-test", NOW)               # the caller hands over the errors: everything is unreviewed
    assert res["status"] == "unreviewed" and all(r["review"]["status"] == "UNREVIEWED" for r in res["final"])


def test_subset_assertion_catches_any_change():
    c = candidate()
    final = [dict(p) for p in c["picks"]]
    assert_subset(c, final)
    for k, v in (("p_model", 0.99), ("odds", 3.0), ("stake", 50.0), ("line", 7.5), ("edge", 0.5)):
        bad = [dict(p) for p in c["picks"]]
        bad[0][k] = v
        with pytest.raises(ReviewIntegrityError):
            assert_subset(c, bad)
    with pytest.raises(ReviewIntegrityError):
        assert_subset(c, final + [{**final[0], "pick_id": "pk_ffffffffffff"}])
    with pytest.raises(ReviewIntegrityError):
        assert_subset(c, final + final[:1])


def test_logs_are_append_only_and_idempotent(tmp_path):
    c = candidate()
    inp = prepare_input(c, {1, 2}, facts_for(c), NOW)
    res = apply_review(c, {1, 2}, facts_for(c), good_review(inp, "veto"), [], "rv-1", NOW)
    r1 = write_all(res, tmp_path, NOW)
    first = (tmp_path / "data/logs/review/vetoed_picks.jsonl").read_text()
    r2 = write_all(res, tmp_path, NOW)                                              # re-running the same review adds nothing
    assert r1["added_vetoed"] == 3 and r2["added_vetoed"] == 0
    assert (tmp_path / "data/logs/review/vetoed_picks.jsonl").read_text() == first
    assert len(first.splitlines()) == 3
    state = load_state(tmp_path / "data/review/picks_final.json")
    assert len(state["vetoed"]) == 3 and len(state["passes"]) == 1


def test_a_successful_rerun_after_a_failure_adds_the_reviewed_rows_and_keeps_history(tmp_path):
    c = candidate()
    bad = apply_review(c, {1, 2}, facts_for(c), None, ["timeout"], "rv-1", NOW)
    write_all(bad, tmp_path, NOW)
    inp = prepare_input(c, {1, 2}, facts_for(c), NOW)
    good = apply_review(c, {1, 2}, facts_for(c), good_review(inp), [], "rv-1", NOW)
    r = write_all(good, tmp_path, NOW)
    rows = [json.loads(x) for x in (tmp_path / "data/logs/review/final_picks.jsonl").read_text().splitlines()]
    assert r["added_final"] == 3 and len(rows) == 6
    assert {x["review"]["status"] for x in rows} == {"UNREVIEWED", "REVIEWED"}
    st = load_state(tmp_path / "data/review/picks_final.json")
    assert all(v["review"]["status"] == "REVIEWED" for v in st["picks"].values()) and len(st["passes"]) == 1


def test_dry_run_writes_nothing(tmp_path):
    c = candidate()
    res = apply_review(c, {1, 2}, facts_for(c), None, ["x"], "rv-1", NOW)
    write_all(res, tmp_path, NOW, dry_run=True)
    assert not list(tmp_path.rglob("*.json*"))


def test_new_day_resets_state():
    c = candidate()
    res = apply_review(c, {1}, facts_for(c), None, ["x"], "rv-1", NOW)
    st = merge_state({"date": "2026-10-13", "picks": {"old": {}}, "vetoed": {}, "passes": []}, res)
    assert "old" not in st["picks"] and st["date"] == "2026-10-14"


def test_page_payload_shows_status_and_moves_vetoes_and_is_reversible():
    c = candidate()
    inp = prepare_input(c, {1}, facts_for(c), NOW)
    rv = good_review(inp)
    rv["reviews"][0]["decision"] = "veto"
    res = apply_review(c, {1}, facts_for(c), rv, [], "rv-1", NOW)
    state = merge_state({}, res)
    p = payload()
    n = len(p["ranking"])
    attach_to_payload(p, state)
    by_label = {r["pick"]: r for r in p["ranking"]}
    assert "BBB moneyline" not in by_label and [r["pick"] for r in p["vetoed"]] == ["BBB moneyline"]
    assert by_label["Over 6.5"]["review"]["decision"] == "keep"
    assert by_label["CCC moneyline"]["review"]["status"] == "UNREVIEWED"             # game 2 is a later wave: not reviewed yet
    assert "review" not in by_label["Tiny edge"]                                       # below the minimum edge: not a candidate at all
    assert p["review"]["n_vetoed"] == 1 and p["review"]["n_unreviewed"] == 1
    again = attach_to_payload(json.loads(json.dumps(p)), state)                        # idempotent
    assert again["ranking"] == p["ranking"] and again["vetoed"] == p["vetoed"]
    strip_review(p)
    assert len(p["ranking"]) == n and "vetoed" not in p and all("review" not in r for r in p["ranking"])


def test_other_days_state_is_ignored_on_the_page():
    c = candidate()
    res = apply_review(c, {1, 2}, facts_for(c), None, ["x"], "rv-1", NOW)
    state = merge_state({}, res)
    state["date"] = "2026-10-13"
    p = attach_to_payload(payload(), state)
    assert p["review"]["passes"] == [] and all(r["review"]["run_id"] is None for r in p["ranking"] if "review" in r)
