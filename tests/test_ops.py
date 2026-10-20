"""Phase 7: halt state machine, ledger, policy guard, kill switch, health, drift, alerts and heartbeat (no network: senders are injected)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nhlbet.ops import chain, digest, drift, health, killswitch, notify, policyguard, state as S, watch
from nhlbet.ops.checks import CheckResult, failed, passed

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def run(st, results, k=0, soft=3, ack=()):
    return S.update(st, results, T0 + timedelta(hours=k), soft, list(ack))


def soft_fail(name="freshness"):
    return failed(name, "soft", "stale", "wait")


def hard_fail(name="drawdown"):
    return failed(name, "hard", "too deep", "decide")


def test_check_level_validated():
    with pytest.raises(ValueError):
        CheckResult("x", "medium", True)


def test_soft_halt_opens_withholds_and_auto_resumes_after_n_healthy_runs():
    st, ev = run(S.empty_state(), [soft_fail(), passed("a")], 0)
    assert st["status"] == "HALT_SOFT" and not S.can_publish(st) and [e["type"] for e in ev] == ["halt_opened"]
    st, ev = run(st, [soft_fail()], 1)
    assert ev == [] and len(st["halts"]) == 1                              # persisting: nothing new, no repeated alert
    for k in (2, 3):
        st, ev = run(st, [passed("freshness")], k)
        assert st["status"] == "HALT_SOFT"                                  # 1 and 2 healthy runs: still halted
    st, ev = run(st, [passed("freshness")], 4)
    assert st["status"] == "OK" and S.can_publish(st) and [e["type"] for e in ev] == ["halt_cleared"]


def test_unhealthy_run_resets_the_healthy_streak():
    st, _ = run(S.empty_state(), [soft_fail()], 0)
    st, _ = run(st, [passed("freshness")], 1)
    st, _ = run(st, [passed("freshness")], 2)
    st, _ = run(st, [soft_fail()], 3)
    assert st["healthy_streak"] == 0
    st, _ = run(st, [passed("freshness")], 4)
    assert st["status"] == "HALT_SOFT"


def test_hard_halt_never_auto_resumes():
    st, ev = run(S.empty_state(), [hard_fail()], 0)
    assert st["status"] == "HALT_HARD" and ev[0]["kind"] == "hard"
    for k in range(1, 30):
        st, _ = run(st, [passed("drawdown")], k)
    assert st["status"] == "HALT_HARD"


def test_soft_escalates_to_hard():
    st, _ = run(S.empty_state(), [failed("x", "soft", "a", "b")], 0)
    st, ev = run(st, [failed("x", "hard", "worse", "c")], 1)
    assert st["status"] == "HALT_HARD" and ev[0]["type"] == "halt_escalated"


def test_hard_halt_clears_only_with_its_listed_id_then_suppresses_until_check_passes():
    st, ev = run(S.empty_state(), [hard_fail()], 0)
    hid = ev[0]["id"]
    st, ev = run(st, [hard_fail()], 1, ack=["H-unknown"])
    assert st["status"] == "HALT_HARD"                                      # a wrong id changes nothing
    st, ev = run(st, [hard_fail()], 2, ack=[hid])
    assert st["status"] == "OK" and [e["type"] for e in ev] == ["halt_cleared", "acknowledged_but_persisting"]
    st, ev = run(st, [hard_fail()], 3, ack=[hid])
    assert st["status"] == "OK" and len(st["halts"]) == 1                   # accepted condition: no new halt
    st, ev = run(st, [passed("drawdown")], 4, ack=[hid])
    assert st["suppressed"] == []
    st, ev = run(st, [hard_fail()], 5, ack=[hid])
    assert st["status"] == "HALT_HARD" and len(st["halts"]) == 2             # it came back after passing: a new halt


def test_acknowledgement_never_clears_a_soft_halt():
    st, ev = run(S.empty_state(), [soft_fail()], 0)
    st, _ = run(st, [soft_fail()], 1, ack=[ev[0]["id"]])
    assert st["status"] == "HALT_SOFT"


def test_update_is_pure_and_state_round_trips(tmp_path):
    st0 = S.empty_state()
    before = json.dumps(st0, sort_keys=True)
    st, _ = run(st0, [hard_fail()])
    assert json.dumps(st0, sort_keys=True) == before
    S.save_state(st, tmp_path / "s" / "state.json")
    assert S.load_state(tmp_path / "s" / "state.json") == st
    assert S.load_state(tmp_path / "missing.json")["status"] == "OK"


def test_ledger_detects_edit_truncate_delete_and_allows_growth(tmp_path):
    d = tmp_path / "data" / "logs" / "recs"
    d.mkdir(parents=True)
    (tmp_path / "data" / "logs" / "bet_log.csv").write_text("derived\n")            # top-level aggregate: not covered
    f, g = d / "a.csv", d / "b.jsonl"
    f.write_text("x,y\n1,2\n"); g.write_text('{"a":1}\n')
    assert chain.append_new(tmp_path) == 2 and chain.verify(tmp_path) == []
    assert chain.append_new(tmp_path) == 0
    g.write_text('{"a":1}\n{"a":2}\n')                                               # growing log is fine
    assert chain.verify(tmp_path) == [] and chain.append_new(tmp_path) == 1 and chain.verify(tmp_path) == []
    f.write_text("x,y\n1,3\n")
    assert chain.verify(tmp_path) == [f"{f.relative_to(tmp_path)} was modified"]
    f.write_text("x")
    assert "truncated" in chain.verify(tmp_path)[0]
    f.unlink()
    assert "deleted" in chain.verify(tmp_path)[0]
    with pytest.raises(RuntimeError):
        chain.append_new(tmp_path)


def test_ledger_detects_rewritten_chain(tmp_path):
    d = tmp_path / "data" / "logs" / "r"
    d.mkdir(parents=True)
    (d / "a.csv").write_text("1\n"); chain.append_new(tmp_path)
    (d / "b.csv").write_text("2\n"); chain.append_new(tmp_path)
    led = tmp_path / "data" / "logs" / chain.LEDGER
    lines = led.read_text().splitlines()
    led.write_text(lines[1] + "\n")                                                  # drop the first link
    assert chain.verify(tmp_path)


def test_policy_guard():
    owner = {"login": "alice", "email": "a@x.org", "sha": "abc1234"}
    bot = {"login": "github-actions[bot]", "email": "b@x.org", "sha": "def5678"}
    kw = dict(owner_logins={"alice"}, owner_emails=set())
    r, rec = policyguard.check_policy(None, "h1", lambda: None, **kw)
    assert r.ok and rec == "h1"
    r, rec = policyguard.check_policy("h1", "h1", lambda: None, **kw)
    assert r.ok
    r, rec = policyguard.check_policy("h1", "h2", lambda: owner, **kw)
    assert r.ok and rec == "h2"
    r, rec = policyguard.check_policy("h1", "h2", lambda: bot, **kw)
    assert not r.ok and r.level == "hard" and rec == "h1"                              # old hash kept so it keeps failing
    r, _ = policyguard.check_policy("h1", "h2", lambda: None, **kw)
    assert not r.ok                                                                    # unknown author is not trusted
    r, _ = policyguard.check_policy("h1", "h2", lambda: owner, owner_logins=set(), owner_emails=set())
    assert not r.ok                                                                    # no owners configured: nothing is trusted


def test_policy_owners_from_env(monkeypatch):
    monkeypatch.setenv("POLICY_OWNER_LOGINS", "Alice, bob")
    monkeypatch.setenv("POLICY_OWNER_EMAILS", "")
    assert policyguard.owners() == ({"alice", "bob"}, set())


def test_drawdown_kill_switch():
    assert killswitch.drawdown_check([]).ok
    assert killswitch.drawdown_check([100, 110, 95], 0.20).ok
    r = killswitch.drawdown_check([100, 120, 96], 0.20)
    assert not r.ok and r.level == "hard"
    assert killswitch.drawdown_check([100, 120, 96, 118], 0.20).ok                      # recovered above the limit


def test_clv_kill_switch_needs_volume_and_a_clearly_negative_interval():
    rng = np.random.default_rng(1)
    assert killswitch.clv_check(rng.normal(-3, 1, 20)).ok                                 # too few picks
    bad = rng.normal(-3, 2, 300)
    r = killswitch.clv_check(bad, dates=np.arange(300) // 3)
    assert not r.ok and r.level == "hard"
    assert killswitch.clv_check(rng.normal(0, 2, 300), dates=np.arange(300) // 3).ok
    z = rng.normal(0, 40, 150); noisy = z - z.mean() - 1.2                                                     # mean below limit but interval spans zero
    assert killswitch.clv_check(noisy, dates=np.arange(150)).ok


def test_integrity_check():
    assert killswitch.integrity_check([]).ok
    r = killswitch.integrity_check(["a was modified"])
    assert not r.ok and r.level == "hard"


def test_range_check():
    good = {"games": [{"game_id": 1, "p_home": 0.55, "sides": {"home": {"p_model": 0.55, "p_market": 0.5, "best_decimal": 1.9}}}], "ranking": [{"pick": "x", "p_model": 0.6, "p_market": 0.5, "price": 2.0}]}
    assert health.range_check(good).ok
    for patch in ({"games": [{"game_id": 1, "p_home": float("nan")}]}, {"games": [{"game_id": 1, "p_home": 1.0}]}, {"ranking": [{"pick": "x", "p_model": 0.5, "p_market": 0.5, "price": 0.9}]}):
        r = health.range_check(patch)
        assert not r.ok and r.level == "soft"


def test_odds_source_and_quota_checks():
    assert health.odds_source_check(None, 5).ok
    assert health.odds_source_check({"enabled": True}, 5).ok
    assert not health.odds_source_check({"enabled": True, "error": "401"}, 5).ok
    assert health.odds_source_check({"enabled": True, "error": "401"}, 0).ok          # nothing scheduled: nothing to price
    assert not health.odds_source_check({"enabled": True, "stale": True}, 3).ok
    assert health.quota_check(None).ok and health.quota_check(500).ok and not health.quota_check(3).ok


def _store(tmp_path):
    from nhlbet.data.store import Store
    return Store(str(tmp_path / "t.db"))


def test_schema_and_row_count_checks_on_an_empty_store(tmp_path):
    st = _store(tmp_path)
    assert health.schema_check(st).ok
    r = health.row_count_check(st)
    assert not r.ok and r.level == "soft"


def test_cross_source_check(tmp_path):
    st = _store(tmp_path)
    with st.tx() as c:
        for i in range(4):
            c.execute("INSERT INTO games (game_id, game_date, home, away, game_type, source, start_utc) VALUES (?, '2026-03-01', 'A', 'B', 2, 'nhl_api', '2026-03-02T00:00:00Z')", [i])
        for i in range(10):
            c.execute("INSERT INTO odds_snapshots (captured_at, event_id, book, market, outcome, price, game_id) VALUES ('2026-03-01T12:00:00Z', ?, 'bk', 'h2h', 'A', 1.9, NULL)", [f"e{i}"])
    r = health.cross_source_check(st, "2026-03-01")
    assert not r.ok and "match no NHL game" in r.detail


def test_drift_checks():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    assert drift.feature_drift_check(ref, pd.DataFrame({"a": rng.normal(0, 1, 500)}), ["a"])[0].ok
    r, _ = drift.feature_drift_check(ref, pd.DataFrame({"a": rng.normal(6, 1, 500)}), ["a"])
    assert not r.ok and r.level == "soft"
    p = rng.uniform(0.3, 0.7, 400); y = (rng.uniform(size=400) < p).astype(int)
    assert drift.calibration_check(p, y)[0].ok
    assert drift.calibration_check(p[:10], y[:10])[0].ok
    m = rng.uniform(0.4, 0.6, 300)
    assert drift.disagreement_check(m, m)[0].ok
    r, _ = drift.disagreement_check(np.clip(m + 0.2, 0, 1), m)
    assert not r.ok
    _, w = drift.disagreement_check(m + 0.09, m)
    assert w


class Resp:
    def __init__(self, code): self.status_code = code


def test_send_uses_topic_headers_and_never_raises(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert notify.send("t", "m", post=lambda *a, **k: Resp(200)) is False             # no topic: not delivered, no crash
    monkeypatch.setenv("NTFY_TOPIC", "secret-topic")
    seen = {}

    def post(url, data, headers, timeout):
        seen.update(url=url, data=data, headers=headers)
        return Resp(200)
    assert notify.send("Title — x", "body", "urgent", ("warning",), "k1", post=post)
    assert seen["url"] == "https://ntfy.sh/secret-topic" and seen["headers"]["Priority"] == "urgent" and seen["headers"]["X-Sequence-ID"] == "k1"
    assert seen["headers"]["Title"].isascii()
    def boom(*a, **k): raise OSError("down")
    assert notify.send("t", "m", post=boom, sleep=lambda s: None) is False
    assert notify.send("t", "m", post=lambda *a, **k: Resp(500), sleep=lambda s: None) is False
    big = {}
    notify.send("t", "x" * 10000, post=lambda url, data, headers, timeout: big.setdefault("n", len(data)) and Resp(200))
    assert big["n"] <= 4096


def test_dispatch_is_idempotent_and_retries_failed_deliveries(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "t")
    st, ev = run(S.empty_state(), [hard_fail()])
    calls = []
    ok = lambda *a, **k: calls.append(1) or Resp(200)
    st2 = notify.dispatch(ev, st, post=ok)
    assert len(calls) == 1 and st2["alerts_sent"]
    notify.dispatch(ev, st2, post=ok)
    assert len(calls) == 1                                                             # re-running the day does not re-alert
    st3 = notify.dispatch(ev, st, post=lambda *a, **k: Resp(503) if not calls.append(1) else None)
    assert st3["alerts_sent"] == [] or st3["alerts_sent"] == st["alerts_sent"]         # failed delivery is not remembered
    assert notify.event_to_alert({"type": "unknown"}) is None


def test_hard_alert_says_how_to_resume():
    a = notify.event_to_alert({"type": "halt_opened", "id": "H-1", "check": "drawdown", "kind": "hard", "reason": "r", "next_step": "n"})
    assert a["priority"] == "urgent" and "acknowledged_halts" in a["message"] and "H-1" in a["message"]
    s = notify.event_to_alert({"type": "halt_opened", "id": "H-2", "check": "freshness", "kind": "soft", "reason": "r", "next_step": "n"})
    assert "clears by itself" in s["message"]


def test_ping(monkeypatch):
    monkeypatch.delenv("HC_PING_URL", raising=False)
    assert notify.ping() is False
    urls = []
    get = lambda u, timeout: urls.append(u) or Resp(200)
    assert notify.ping("ok", "https://hc-ping.com/abc", get=get) and notify.ping("fail", "https://hc-ping.com/abc/", get=get)
    assert urls == ["https://hc-ping.com/abc", "https://hc-ping.com/abc/fail"]
    assert notify.ping("ok", "https://x", get=lambda u, timeout: Resp(500), sleep=lambda s: None) is False
    assert notify.should_ping(None, 0, 10) and not notify.should_ping(0, 300, 10) and notify.should_ping(0, 601, 10)


def test_credential_and_keepalive_watch():
    ev = watch.credential_events(T0, remaining_credits=20)
    assert {e["check"] for e in ev} == {"claude_oauth_token", "odds_credits"}                # issue date unset in the policy: warned
    assert all(e["key"] for e in ev)
    assert watch.credential_events(T0, remaining_credits=20) == ev                          # stable keys: dedupe works
    assert not [e for e in watch.credential_events(T0, remaining_credits=900) if e["check"] == "odds_credits"]
    assert len(watch.keepalive_events(55, 10, T0)) == 1
    assert len(watch.keepalive_events(1, 45, T0)) == 1
    assert len(watch.keepalive_events(1, None, T0)) == 1
    assert watch.keepalive_events(1, 10, T0) == []


def test_digest():
    st, _ = run(S.empty_state(), [hard_fail()])
    title, body = digest.build_digest(st, None, None, 120, ["token expires in 20 days"], T0)
    assert "HALT_HARD" in body and "RESEARCH ONLY" in body and "120" in body and "token expires" in body and "—" not in body
