"""The page shows review status, reason and sources without JavaScript, and escapes everything the agent wrote."""
from nhlbet.review.apply import apply_review, attach_to_payload, merge_state, prepare_input
from nhlbet.review.contract import build_candidates
from nhlbet.site.build import render_html
from tests.review_helpers import NOW, good_review, payload


def page(decisions, failure=False):
    p = payload()
    for g in p["games"]:
        g.update({"p_home": 0.55, "blocked": None, "odds_captured_at": "x", "sides": {}})
    p["model"] = {"version": "v", "status": "OK"}
    c = build_candidates(p)
    facts = {x["pick_id"]: {"floor": "keep", "summary": "s"} for x in c["picks"]}
    inp = prepare_input(c, {1, 2}, facts, NOW)
    rv = good_review(inp)
    for r, d in zip(rv["reviews"], decisions):
        r["decision"] = d
    rv["reviews"][1]["reason"] = "<script>alert(1)</script> goalie unconfirmed"
    rv["reviews"][1]["sources"][0]["url"] = "https://example.com/a?x=1&y='2'"
    res = apply_review(c, {1, 2}, facts, None if failure else rv, ["agent timed out"] if failure else [], "rv-1", NOW)
    attach_to_payload(p, merge_state({}, res))
    return p, render_html(p)


def noscript(h):
    return h[h.index("<noscript>"):h.index("</noscript>")]


def test_no_javascript_table_shows_status_reason_sources_and_vetoes():
    p, h = page(["veto", "flag", "keep"])
    n = noscript(h)
    assert "Picks and their pre-game review" in n and "<b>Flagged</b>" in n and "<b>Kept</b>" in n
    assert "Vetoed by the pre-game review (1)" in n and "BBB moneyline" in n.split("Vetoed by the pre-game review")[1]
    assert "<script>alert(1)" not in n and "&lt;script&gt;alert(1)" in n          # the agent's text is data, never markup
    assert "href='https://example.com/a?x=1&amp;y=&#x27;2&#x27;'" in n


def test_unreviewed_picks_are_labelled_on_the_page():
    p, h = page([], failure=True)
    n = noscript(h)
    assert n.count("<b>UNREVIEWED</b>") == 3 and "Kept" not in n and "agent did not complete" in n
    assert p["review"]["n_unreviewed"] == 3


def test_javascript_payload_carries_the_review_fields_and_ui_renders_them():
    p, h = page(["veto", "flag", "keep"])
    assert '"vetoed"' in h and '"warning"' in h
    ui = open("nhlbet/site/ui.js").read()
    for needle in ("reviewCell", "vetoedSection", "UNREVIEWED", "noopener noreferrer", "safeUrl"):
        assert needle in ui
