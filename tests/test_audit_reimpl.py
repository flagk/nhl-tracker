"""The blind reimplementation (written from audit/SPEC.md only) must keep agreeing with the production decision code; if either changes, one of them is wrong or the spec is stale."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("p6_diff", ROOT / "audit/scripts/p6_diff.py")
p6 = importlib.util.module_from_spec(spec); sys.modules["p6_diff"] = p6; spec.loader.exec_module(p6)


def test_reimplementation_agrees_on_devig_and_on_decisions():
    R = p6.load(ROOT / "audit/reimpl/reimpl.py")
    dv = p6.diff_devig(R, n=300)
    assert all(v["mismatches_over_1e-7"] == 0 for v in dv.values())
    d = p6.diff_decisions(R, slates=120)
    assert d["bets_by_our_code"] > 20 and d["games_with_any_disagreement"] == 0, d["examples"]


def test_the_comparison_can_fail(tmp_path):
    src = (ROOT / "audit/reimpl/reimpl.py").read_text()
    assert "min_edge=0.03," in src
    (tmp_path / "r.py").write_text(src.replace("min_edge=0.03,", "min_edge=0.045,"))
    d = p6.diff_decisions(p6.load(tmp_path / "r.py"), slates=120)
    assert d["games_with_any_disagreement"] > 0
