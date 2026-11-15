"""Phase 7 end to end: a halt withholds the picks and alerts once, a rerun repeats nothing, and edited logs open a HARD halt. No network: ntfy and the heartbeat are recorded."""
from __future__ import annotations

import json

import pytest

from nhlbet.data.store import Store
from nhlbet.ops import chain
from nhlbet.pipeline import run_daily
from tests.test_report_pipeline import _e2e_setup, _noon


class Net:
    def __init__(self): self.posts, self.gets = [], []

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts.append((url, headers["Title"], data.decode()))
        return type("R", (), {"status_code": 200})()

    def get(self, url, timeout=None):
        self.gets.append(url)
        return type("R", (), {"status_code": 200})()


@pytest.fixture
def net(monkeypatch):
    import requests
    n = Net()
    monkeypatch.setattr(requests, "post", n.post)
    monkeypatch.setattr(requests, "get", n.get)
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    monkeypatch.setenv("HC_PING_URL", "https://hc.example/abc")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    return n


def go(day, tmp_path, k=0, run_type="morning"):
    return run_daily(day, run_type, db=str(tmp_path / "t.db"), refresh=False, odds=False, log_root="data/logs", report_dir="reports", model_dir="data/models", now=_noon(day, k))


def test_halt_withholds_picks_alerts_once_and_rerun_is_idempotent(league, tmp_path, monkeypatch, net):
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    r = go(day, tmp_path)
    state = json.loads((tmp_path / "state/system_state.json").read_text())
    assert state["status"] == "HALT_SOFT" and [h["check"] for h in state["halts"]] == ["row_counts"]       # the synthetic history is far below the policy minimum
    assert r["status"] == "ALERT"
    assert (Store(str(tmp_path / "t.db")).df("SELECT action FROM recommendations").action != "BET").all()
    pj = json.loads((tmp_path / "site/picks.json").read_text())
    assert pj["system"]["status"] == "HALT_SOFT" and pj["system"]["halts"][0]["next_step"]
    assert pj["betting"]["status"] == "RESEARCH ONLY" and pj["betting"]["criteria"]
    assert len(net.posts) == 1 and "HALT (soft): row_counts" in net.posts[0][1]
    assert net.gets and net.gets[-1].endswith("/fail")                                                   # heartbeat reports the halt
    go(day, tmp_path, k=1, run_type="late")
    assert len(net.posts) == 1                                                                           # same halt, same alert: nothing repeated
    state2 = json.loads((tmp_path / "state/system_state.json").read_text())
    assert len(state2["halts"]) == 1 and state2["halts"][0]["id"] == state["halts"][0]["id"]
    assert chain.verify(tmp_path) == []                                                                  # reruns never break their own ledger


def test_edited_log_opens_a_hard_halt_that_does_not_clear_itself(league, tmp_path, monkeypatch, net):
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    go(day, tmp_path)
    f = next((tmp_path / "data/logs/recommendations").glob("*.csv"))
    f.write_text(f.read_text().replace("BOS", "XXX", 1))
    go(day, tmp_path, k=1, run_type="late")
    state = json.loads((tmp_path / "state/system_state.json").read_text())
    assert state["status"] == "HALT_HARD" and any(h["check"] == "log_integrity" and h["kind"] == "hard" for h in state["halts"])
    assert any("HALT (hard): log_integrity" in p[1] for p in net.posts)
    for k in range(2, 8):
        go(day, tmp_path, k=k, run_type="late")
    assert json.loads((tmp_path / "state/system_state.json").read_text())["status"] == "HALT_HARD"


def test_policy_change_by_a_non_owner_opens_a_hard_halt(league, tmp_path, monkeypatch, net):
    import nhlbet.ops.gate as gate
    st, day = _e2e_setup(tmp_path, league, monkeypatch)
    go(day, tmp_path)
    s = json.loads((tmp_path / "state/system_state.json").read_text()); s["policy_sha256"] = "0" * 64
    (tmp_path / "state/system_state.json").write_text(json.dumps(s))
    monkeypatch.setattr(gate.policyguard, "last_policy_commit_github", lambda repo, token: {"login": "github-actions[bot]", "email": "", "sha": "abc1234"})
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("POLICY_OWNER_LOGINS", "the-owner")
    go(day, tmp_path, k=1, run_type="late")
    s = json.loads((tmp_path / "state/system_state.json").read_text())
    assert s["status"] == "HALT_HARD" and any(h["check"] == "policy_unchanged" for h in s["halts"])
