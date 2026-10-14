import copy

from nhlbet.review.contract import build_candidates, pick_id, validate_review
from tests.review_helpers import NOW, candidate, good_review, payload


def test_candidates_are_the_ranked_picks_with_enough_edge():
    c = candidate()
    assert [p["label"] for p in c["picks"]] == ["BBB moneyline", "Over 6.5", "CCC moneyline"]      # the 1-point edge is not a candidate
    p = c["picks"][0]
    for k in ("pick_id", "game_id", "market", "line", "side", "odds", "book", "p_model", "edge"):
        assert k in p
    assert abs(p["edge"] - 0.08) < 1e-9 and p["book"] == "bk1"
    assert set(c["games"]) == {"1", "2"} and c["games"]["1"]["home_goalie"] == "H. Goalie"


def test_pick_ids_are_stable_across_runs_and_distinct_across_bets():
    a, b = candidate(), build_candidates(payload())
    assert [p["pick_id"] for p in a["picks"]] == [p["pick_id"] for p in b["picks"]]
    assert len({p["pick_id"] for p in a["picks"]}) == 3
    assert pick_id("2026-10-14", 1, "totals", 6.5, "over") != pick_id("2026-10-14", 1, "totals", 7.0, "over")
    assert pick_id("2026-10-14", 1, "h2h", None, "home") != pick_id("2026-10-15", 1, "h2h", None, "home")


def test_candidate_id_changes_when_a_price_changes():
    p2 = payload()
    p2["ranking"][0]["price"] = 1.9
    assert build_candidates(p2)["candidate_id"] != candidate()["candidate_id"]


def test_a_complete_review_is_valid():
    c = candidate()
    assert validate_review(good_review(c), c, NOW) == []


def test_missing_extra_and_duplicate_reviews_fail():
    c = candidate()
    r = good_review(c)
    short = {**r, "reviews": r["reviews"][:2]}
    assert any("missing" in e for e in validate_review(short, c, NOW))
    extra = copy.deepcopy(r)
    extra["reviews"].append({**extra["reviews"][0], "pick_id": "pk_000000000000"})
    assert any("not asked" in e for e in validate_review(extra, c, NOW))
    dup = {**r, "reviews": r["reviews"] + r["reviews"][:1]}
    assert any("duplicate" in e for e in validate_review(dup, c, NOW))


def test_the_agent_cannot_smuggle_in_probabilities_prices_or_new_picks():
    c = candidate()
    r = good_review(c)
    r["reviews"][0]["p_model"] = 0.9
    assert validate_review(r, c, NOW)
    r = good_review(c)
    r["new_picks"] = [{"label": "x"}]
    assert validate_review(r, c, NOW)


def test_wrong_candidate_id_and_garbage_fail():
    c = candidate()
    r = good_review(c)
    assert validate_review({**r, "candidate_id": "0" * 12}, c, NOW)
    assert validate_review({"hello": 1}, c, NOW)
    assert validate_review([], c, NOW)


def test_keep_requires_confirmation_but_flag_may_be_unconfirmed():
    c = candidate()
    r = good_review(c)
    r["reviews"][0]["checks"]["injuries"] = "unconfirmed"
    assert any("flag" in e for e in validate_review(r, c, NOW))
    r["reviews"][0]["decision"] = "flag"
    assert validate_review(r, c, NOW) == []


def test_every_claim_needs_a_listed_source_and_a_plausible_time():
    c = candidate()
    r = good_review(c)
    r["reviews"][0]["checks"]["injuries"]["source_urls"] = ["https://other.example/x"]
    assert any("not in sources" in e for e in validate_review(r, c, NOW))
    r = good_review(c)
    r["reviews"][0]["checks"]["injuries"]["source_urls"] = []
    assert any("no source" in e for e in validate_review(r, c, NOW))
    r = good_review(c)
    r["reviews"][0]["sources"][0]["retrieved_at"] = "2026-10-10T00:00:00Z"
    assert any("implausible" in e for e in validate_review(r, c, NOW))
    r = good_review(c)
    r["reviews"][0]["sources"][0]["retrieved_at"] = "2026-10-15T09:00:00Z"
    assert any("implausible" in e for e in validate_review(r, c, NOW))
    r = good_review(c)
    r["reviews"][0]["sources"][0]["url"] = "http://insecure.example"
    assert validate_review(r, c, NOW)


def test_a_veto_needs_a_confirmed_finding_and_a_source():
    c = candidate()
    r = good_review(c, decision="veto")
    assert validate_review(r, c, NOW) == []
    r = good_review(c, decision="veto")
    for x in r["reviews"]:
        x["checks"] = {k: "unconfirmed" for k in ("confirmed_goalie", "lineup_changes", "injuries", "odds_movement")}
    assert any("confirmed finding" in e for e in validate_review(r, c, NOW))
    r = good_review(c, decision="veto")
    for x in r["reviews"]:
        x["sources"] = []
        x["checks"]["confirmed_goalie"]["source_urls"] = []
    assert validate_review(r, c, NOW)


def test_reason_must_be_one_line():
    c = candidate()
    r = good_review(c)
    r["reviews"][0]["reason"] = "line one\nline two"
    assert validate_review(r, c, NOW)
