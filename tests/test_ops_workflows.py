"""Phase 7 workflow wiring: every workflow parses, the alerting steps exist, and no workflow writes the policy file."""
from __future__ import annotations

from pathlib import Path

import yaml

W = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def test_all_workflows_parse():
    for f in W.glob("*.yml"):
        assert isinstance(yaml.safe_load(f.read_text()), dict), f.name


def test_daily_alerts_backs_up_and_commits_state():
    t = (W / "daily.yml").read_text()
    assert "scripts/notify.py" in t and "log-backup" in t and "--force" not in t.split("log-backup")[1].split("\n")[0] and "state README.md" in t
    assert "NTFY_TOPIC" in t and "HC_PING_URL" in t and "POLICY_OWNER_LOGINS" in t


def test_review_and_heartbeat_report_failures_and_liveness():
    assert "scripts/notify.py" in (W / "review.yml").read_text()
    assert "HC_PING_URL" in (W / "heartbeat.yml").read_text()


def test_no_workflow_stages_the_policy_file():
    for f in W.glob("*.yml"):
        for line in f.read_text().splitlines():
            if "git add" in line:
                assert "config" not in line, f"{f.name} stages config/: {line}"


def test_policy_guard_and_digest_exist():
    assert "POLICY_OWNER_LOGINS" in (W / "policy-guard.yml").read_text()
    assert "ops_watch.py" in (W / "weekly-digest.yml").read_text()
