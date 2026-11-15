"""config/policy.yaml is written only by its owner. Code reads it (nhlbet/policy.py) and nothing writes it."""
import re
from pathlib import Path

from nhlbet import policy
from nhlbet.policy import load_policy, policy_get

ROOT = Path(__file__).resolve().parent.parent
WRITE = re.compile(r"write_text|write_bytes|open\([^)]*[\"'][wax]\+?b?[\"']|shutil\.(copy|move)|os\.(rename|replace)|\.unlink|yaml\.(safe_)?dump")


def test_policy_loads_and_has_the_sections_phase_1_relies_on():
    p = load_policy()
    assert p["meta"]["status"] in ("proposed", "approved") and "holdout" in p and "settlement" in p
    assert policy_get("settlement", "default_ruleset") in policy_get("settlement", "rulesets")
    assert policy_get("no", "such", "key", default=7) == 7


def test_no_code_path_writes_the_policy_file():
    offenders = []
    for f in list((ROOT / "nhlbet").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        text = f.read_text()
        if "policy.yaml" in text or "POLICY_PATH" in text:
            for i, line in enumerate(text.splitlines(), 1):
                if WRITE.search(line) and ("policy" in line.lower()):
                    offenders.append(f"{f.relative_to(ROOT)}:{i}")
    assert not offenders, offenders


def test_the_loader_opens_the_file_read_only():
    src = Path(policy.__file__).read_text()
    assert 'open(path, "r")' in src and not WRITE.search(src)
