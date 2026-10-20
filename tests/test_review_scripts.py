"""The review pipeline end to end with a stand-in for the agent: prepare -> (agent file) -> apply, including every failure path."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nhlbet.review.contract import build_candidates
from tests.review_helpers import good_review, payload

REPO = Path(__file__).resolve().parent.parent
NOW = "2026-10-14T22:30:00+00:00"          # 60 minutes before game 1 (23:30 UTC)


def run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], capture_output=True, text=True, cwd=cwd)


def full_payload():
    p = payload()
    for g in p["games"]:
        g.update({"p_home": 0.55, "blocked": None, "odds_stale": False, "odds_captured_at": "2026-10-14T21:30:00+00:00", "sides": g.get("sides") or {}})
    p["generated_at"] = "2026-10-14T21:45:00+00:00"
    p["model"] = {"version": "v1", "status": "OK"}
    return p


@pytest.fixture
def repo(tmp_path):
    p = full_payload()
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "picks.json").write_text(json.dumps(p))
    (tmp_path / "data" / "review").mkdir(parents=True)
    (tmp_path / "data/review/picks_candidate.json").write_text(json.dumps(build_candidates(p)))
    for g in p["games"]:
        g["sides"] = {}                                   # the candidate above already took its book names; the static table needs no sides
    (tmp_path / "site" / "picks.json").write_text(json.dumps(p))
    return tmp_path


def prepare(repo):
    r = run("review_prepare.py", "--root", str(repo), "--no-fetch", "--now", NOW)
    assert r.returncode == 0, r.stderr
    return json.loads((repo / "agent/work/review_input.json").read_text())


def apply(repo, *args):
    return run("review_apply.py", "--root", str(repo), "--now", "2026-10-14T22:31:00+00:00", *args)


def page(repo):
    return json.loads((repo / "site/picks.json").read_text())


def state(repo):
    return json.loads((repo / "data/review/picks_final.json").read_text())


def test_prepare_selects_the_wave_and_says_odds_are_unconfirmed(repo):
    inp = prepare(repo)
    assert {p["game_id"] for p in inp["picks"]} == {1} and len(inp["picks"]) == 2          # game 2 is a later wave
    assert all(p["odds_movement_fact"].startswith("unconfirmed") for p in inp["picks"])
    ctx = json.loads((repo / "agent/work/context.json").read_text())
    assert ctx["run_id"] == "rv-2026-10-14-2330" and ctx["wave_game_ids"] == [1]


def test_prepare_outside_the_window_does_nothing(repo):
    r = run("review_prepare.py", "--root", str(repo), "--no-fetch", "--now", "2026-10-14T18:00:00+00:00")
    assert r.returncode == 0 and "run=false" in r.stdout and not (repo / "agent/work/review_input.json").exists()


def test_happy_path_logs_publishes_and_is_idempotent(repo):
    inp = prepare(repo)
    rv = good_review(inp)
    rv["reviews"][1]["decision"] = "veto"
    (repo / "agent/work/picks_review.json").write_text(json.dumps(rv))
    r = apply(repo)
    assert r.returncode == 0, r.stderr + r.stdout
    st = state(repo)
    assert len(st["picks"]) == 1 and len(st["vetoed"]) == 1 and st["passes"][0]["status"] == "reviewed"
    p = page(repo)
    assert [x["pick"] for x in p["vetoed"]] == ["Over 6.5"] and p["review"]["n_vetoed"] == 1
    by = {x["pick"]: x for x in p["ranking"]}
    assert by["BBB moneyline"]["review"]["decision"] == "keep" and by["CCC moneyline"]["review"]["status"] == "UNREVIEWED"
    assert (repo / "review_logs/rv-2026-10-14-2330.json").exists()
    log = (repo / "data/logs/review/final_picks.jsonl").read_text()
    vlog = (repo / "data/logs/review/vetoed_picks.jsonl").read_text()
    again = apply(repo)                                                          # re-run: nothing is duplicated
    assert again.returncode == 0
    assert (repo / "data/logs/review/final_picks.jsonl").read_text() == log and (repo / "data/logs/review/vetoed_picks.jsonl").read_text() == vlog
    assert len(log.splitlines()) == 1 and len(vlog.splitlines()) == 1


@pytest.mark.parametrize("breakage", ["missing", "not_json", "partial", "extra_field", "wrong_batch"])
def test_bad_agent_output_publishes_unreviewed_and_fails_the_run(repo, breakage):
    inp = prepare(repo)
    f = repo / "agent/work/picks_review.json"
    rv = good_review(inp)
    if breakage == "not_json":
        f.write_text("Here are my picks: keep them all!")
    elif breakage == "partial":
        f.write_text(json.dumps({**rv, "reviews": rv["reviews"][:1]}))
    elif breakage == "extra_field":
        rv["reviews"][0]["p_model"] = 0.99
        f.write_text(json.dumps(rv))
    elif breakage == "wrong_batch":
        f.write_text(json.dumps({**rv, "candidate_id": "0" * 12}))
    r = apply(repo)
    assert r.returncode == 3, r.stderr + r.stdout                                  # red workflow
    st = state(repo)
    assert st["passes"][0]["status"] == "unreviewed" and not st["vetoed"]
    assert all(v["review"]["status"] == "UNREVIEWED" and v["review"]["decision"] is None for v in st["picks"].values()) and len(st["picks"]) == 2
    rows = [x for x in page(repo)["ranking"] if x["pick"] in ("BBB moneyline", "Over 6.5")]
    assert all(x["review"]["status"] == "UNREVIEWED" and "UNREVIEWED" in x["review"]["warning"] for x in rows)
    cand = {c["pick_id"]: c for c in json.loads((repo / "data/review/picks_candidate.json").read_text())["picks"]}
    assert all(v["p_model"] == cand[k]["p_model"] and v["odds"] == cand[k]["odds"] for k, v in st["picks"].items())      # the numbers are the model's, untouched


def test_a_failed_agent_step_ignores_even_a_valid_file(repo):
    inp = prepare(repo)
    (repo / "agent/work/picks_review.json").write_text(json.dumps(good_review(inp)))
    for outcome in ("failure", "cancelled"):
        r = apply(repo, "--agent-outcome", outcome)
        assert r.returncode == 3
        assert all(v["review"]["status"] == "UNREVIEWED" for v in state(repo)["picks"].values())


def test_dry_run_writes_nothing_outside_the_work_dir(repo):
    inp = prepare(repo)
    (repo / "agent/work/picks_review.json").write_text(json.dumps(good_review(inp)))
    before = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file() and "agent/work" not in str(p)}
    r = apply(repo, "--dry-run")
    assert r.returncode == 0
    after = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file() and "agent/work" not in str(p)}
    assert before == after
    assert (repo / "agent/work/dry_run_result.json").exists()


def test_candidate_changed_since_prepare_refuses_to_apply(repo):
    prepare(repo)
    c = json.loads((repo / "data/review/picks_candidate.json").read_text())
    c["picks"][0]["odds"] = 9.9
    c["candidate_id"] = "f" * 12
    (repo / "data/review/picks_candidate.json").write_text(json.dumps(c))
    assert apply(repo).returncode == 4 and not (repo / "data/review/picks_final.json").exists()


def test_guard_restores_files_the_agent_was_not_allowed_to_touch(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "scripts").mkdir(); (tmp_path / "scripts/x.py").write_text("print(1)\n"); (tmp_path / ".gitignore").write_text("agent/work/\n")
    git("add", "-A"); git("commit", "-qm", "init")
    assert run("review_guard.py", "snapshot", cwd=tmp_path).returncode == 0
    (tmp_path / "agent/work").mkdir(parents=True, exist_ok=True); (tmp_path / "agent/work/picks_review.json").write_text("{}")      # allowed
    assert run("review_guard.py", "check", cwd=tmp_path).returncode == 0
    (tmp_path / "scripts/x.py").write_text("import os; os.system('curl evil')\n")                                                       # not allowed
    (tmp_path / "new_file.txt").write_text("x")
    r = run("review_guard.py", "check", cwd=tmp_path)
    assert r.returncode == 1 and "scripts/x.py" in r.stderr
    assert (tmp_path / "scripts/x.py").read_text() == "print(1)\n" and not (tmp_path / "new_file.txt").exists()
