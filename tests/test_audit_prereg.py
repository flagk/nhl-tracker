import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "audit/scripts"))
spec = importlib.util.spec_from_file_location("pe", ROOT / "audit/scripts/p6_prereg_eval.py")
pe = importlib.util.module_from_spec(spec); sys.modules["pe"] = pe; spec.loader.exec_module(pe)


def P1(mean, p, lo, hi):
    return {"mean_gain": mean, "dm_p": p, "ci_low": lo, "ci_high": hi, "n": 1300}


def P2(n, mean, lo):
    return {"n": n, "mean": mean, "ci_low": lo, "ci_high": mean + 0.5}


def test_verdict_rules_are_exactly_the_preregistered_ones():
    assert pe.verdict(P1(0.009, 0.01, 0.002, 0.016), P2(400, 0.7, 0.1)) == "EDGE SUPPORTED"
    assert pe.verdict(P1(0.009, 0.01, 0.002, 0.016), P2(250, 0.7, 0.1)) == "INCONCLUSIVE"        # too few CLV picks
    assert pe.verdict(P1(0.009, 0.03, 0.0, 0.016), P2(400, 0.7, 0.1)) == "INCONCLUSIVE"           # p above 0.025
    assert pe.verdict(P1(0.003, 0.2, -0.004, 0.0065), P2(400, 0.7, 0.1)) == "EDGE UNSUPPORTED"   # a gain of 0.007 is excluded
    assert pe.verdict(P1(-0.012, 0.99, -0.02, -0.004), P2(400, 0.0, -0.2)) == "EDGE UNSUPPORTED"
    assert pe.verdict(P1(0.004, 0.2, -0.01, 0.018), P2(400, 0.0, -0.2)) == "INCONCLUSIVE"


def test_the_committed_preregistration_matches_its_lock_and_cannot_be_edited():
    if not (ROOT / "audit/prereg_lock.json").exists():
        pytest.skip("not locked yet")
    pe.check_lock()
    lock = json.loads((ROOT / "audit/prereg_lock.json").read_text())
    assert lock["trigger"]["qualifying_games"] == 1312 and "—" not in (ROOT / "audit/preregistration.md").read_text()


def test_tampered_preregistration_is_refused(tmp_path):
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit/preregistration.md").write_text("x")
    (tmp_path / "audit/prereg_lock.json").write_text(json.dumps({"preregistration_sha256": "0" * 64}))
    with pytest.raises(pe.PreregViolation):
        pe.check_lock(tmp_path)
