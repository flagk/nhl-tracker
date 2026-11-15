"""Policy immutability: the pipeline records the sha256 of config/policy.yaml on every run and halts HARD if it changed in a commit the owner did not author.

Honesty note: a git author field can be forged by anyone who can push. The real control is branch protection with CODEOWNERS covering config/policy.yaml and .github/** (a manual
setting, listed in OPERATIONS.md); this check detects the common failure (a bot, a workflow or an agent editing the policy) and a changed hash that nobody can account for.
"""
from __future__ import annotations

import os
from typing import Callable

from nhlbet.ops.checks import CheckResult, failed, passed


def owners() -> tuple[set[str], set[str]]:
    """(lowercase owner logins, lowercase owner emails) from the repository variables POLICY_OWNER_LOGINS / POLICY_OWNER_EMAILS: not stored in the policy file, so code cannot grant itself ownership."""
    split = lambda v: {x.strip().lower() for x in (v or "").split(",") if x.strip()}
    return split(os.environ.get("POLICY_OWNER_LOGINS")), split(os.environ.get("POLICY_OWNER_EMAILS"))


def last_policy_commit_github(repo: str, token: str | None, path: str = "config/policy.yaml", ref: str | None = None) -> dict | None:
    """{'login', 'email', 'sha'} of the newest commit touching the policy file, from the GitHub API (a shallow checkout cannot answer this). None when it cannot be determined."""
    import requests
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/commits", params={"path": path, "per_page": 1, **({"sha": ref} if ref else {})},
                         headers={"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}, timeout=20)
        r.raise_for_status()
        c = r.json()[0]
        return {"login": ((c.get("author") or {}).get("login") or "").lower(), "email": ((c["commit"].get("author") or {}).get("email") or "").lower(), "sha": c["sha"]}
    except Exception:
        return None


def check_policy(state_sha: str | None, current_sha: str, fetch_author: Callable[[], dict | None], owner_logins: set[str] | None = None, owner_emails: set[str] | None = None) -> tuple[CheckResult, str]:
    """(result, sha to record). First run (no recorded hash) records the current one. A changed hash passes only if the newest commit touching the file was authored by an owner."""
    if state_sha is None or state_sha == current_sha:
        return passed("policy_unchanged", "hard", "policy file matches the recorded hash" if state_sha else "first run: hash recorded"), current_sha
    logins, emails = (owner_logins, owner_emails) if owner_logins is not None else owners()
    a = fetch_author()
    if a and ((a.get("login") and a["login"] in logins) or (a.get("email") and a["email"] in emails)):
        return passed("policy_unchanged", "hard", f"policy changed in a commit by owner {a.get('login') or a.get('email')}: new hash recorded"), current_sha
    who = f"{a.get('login') or a.get('email')} ({a.get('sha', '')[:7]})" if a else "an author that could not be determined"
    return (failed("policy_unchanged", "hard", f"config/policy.yaml changed (hash {state_sha[:10]} -> {current_sha[:10]}) in a commit by {who}, who is not a listed owner",
                   "Review the diff of config/policy.yaml. If the change is yours, set the repository variable POLICY_OWNER_LOGINS to include your login if missing and re-run; otherwise revert it. "
                   "Then add this halt's id to autonomy.acknowledged_halts in config/policy.yaml."), state_sha)
