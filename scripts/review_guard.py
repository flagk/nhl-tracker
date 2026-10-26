"""Tamper guard around the review agent. The agent is given no shell and a single permitted output file, but the rules are enforced by permission settings,
so this script checks the result instead of trusting them: it records which files differ from git before the agent runs (``snapshot``) and afterwards
(``check``) lists anything else that changed, restores it to the committed version and exits 1. ``agent/work/`` (git-ignored) is the only place the agent may write.

    python scripts/review_guard.py snapshot
    python scripts/review_guard.py check
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path("agent/work/baseline_status.txt")


def status() -> set[str]:
    out = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    return {ln for ln in out.splitlines() if ln.strip()}


def restore(lines: set[str]) -> None:
    for ln in sorted(lines):
        code, path = ln[:2], ln[3:].strip().strip('"')
        if code == "??":
            Path(path).unlink(missing_ok=True)
        else:
            subprocess.run(["git", "checkout", "HEAD", "--", path], check=False)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("snapshot", "check"):
        print(__doc__)
        return 2
    if argv[0] == "snapshot":
        BASE.parent.mkdir(parents=True, exist_ok=True)
        BASE.write_text("\n".join(sorted(status())))
        return 0
    before = set(BASE.read_text().splitlines()) if BASE.exists() else set()
    changed = status() - before
    if changed:
        print("the review agent changed files it may not touch; restoring them:\n  " + "\n  ".join(sorted(changed)), file=sys.stderr)
        restore(changed)
        return 1
    print("guard: nothing outside agent/work/ changed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
