import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fv", ROOT / "audit/scripts/p6_forward_verify.py")
fv = importlib.util.module_from_spec(spec); sys.modules["fv"] = fv; spec.loader.exec_module(fv)


def git(cwd, *a, date=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if date:
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = date
    subprocess.run(["git", *a], cwd=cwd, env=env, check=True, capture_output=True)


def test_rows_committed_after_puck_drop_are_flagged_and_before_are_verified(tmp_path):
    git(tmp_path, "init", "-q")
    d = tmp_path / "data/logs/recommendations"; d.mkdir(parents=True)
    hdr = "run_id,run_at,game_id,game_date\n"
    (d / "early.csv").write_text(hdr + "r1,2026-10-27T15:00:00+00:00,1,2026-10-27\n")
    git(tmp_path, "add", "-A"); git(tmp_path, "commit", "-q", "-m", "a", date="2026-10-27T15:05:00+00:00")
    (d / "late.csv").write_text(hdr + "r2,2026-10-27T15:00:00+00:00,2,2026-10-27\n")
    git(tmp_path, "add", "-A"); git(tmp_path, "commit", "-q", "-m", "b", date="2026-10-27T23:30:00+00:00")
    starts = {1: pd.Timestamp("2026-10-27T23:00:00Z"), 2: pd.Timestamp("2026-10-27T23:00:00Z")}
    r = fv.verify(tmp_path, starts)
    assert r["status_counts"] == {"verified_before_puck_drop": 1, "LATE_commit_after_puck_drop": 1}
    assert r["late_examples"][0]["game_id"] == 2 and abs(r["median_minutes_before_puck_drop_when_verified"] - 475) < 1
    assert fv.verify(tmp_path, {})["status_counts"] == {"no_start_time": 2}
