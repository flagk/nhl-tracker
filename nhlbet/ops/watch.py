"""Credential, quota and schedule watch: warnings only (they alert; they do not halt picks)."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from nhlbet.policy import policy_get


def _c(*k, default=None):
    return policy_get("autonomy", "credentials", *k, default=default)


def credential_events(now: datetime, remaining_credits: int | None = None) -> list[dict]:
    """Warning events (each with a stable ``key`` so it is sent once): the Claude OAuth token's expiry at 30/14/7 days, a missing issue date, the odds plan's renewal and low credits."""
    out = []
    d0 = now.date() if isinstance(now, datetime) else now
    issued = _c("claude_oauth_token_issued")
    if not issued:
        out.append({"type": "warning", "check": "claude_oauth_token", "key": f"claude_token_issue_date_missing:{d0:%G-W%V}",
                    "message": "The CLAUDE_CODE_OAUTH_TOKEN issue date is not recorded in config/policy.yaml (autonomy.credentials.claude_oauth_token_issued), so expiry warnings cannot work. `claude setup-token` tokens last one year."})
    else:
        exp = date.fromisoformat(str(issued)) + timedelta(days=int(_c("claude_oauth_token_valid_days", default=365)))
        left = (exp - d0).days
        for w in sorted(_c("warn_days_before_expiry", default=[30, 14, 7])):
            if left <= w:
                out.append({"type": "warning", "check": "claude_oauth_token", "key": f"claude_token_expiry_{w}:{issued}",
                            "message": f"The Claude OAuth token expires in {left} days ({exp}). Run `claude setup-token`, replace the CLAUDE_CODE_OAUTH_TOKEN secret, and update claude_oauth_token_issued in config/policy.yaml."})
                break
    renew_day = int(_c("odds_plan_renews_day_of_month", default=1))
    nxt = (d0.replace(day=28) + timedelta(days=4)).replace(day=min(renew_day, 28)) if d0.day >= renew_day else d0.replace(day=min(renew_day, 28))
    days_to_renewal = (nxt - d0).days
    if remaining_credits is not None:
        for lvl in sorted(_c("odds_credit_warn_levels", default=[100, 50, 25])):
            if remaining_credits <= lvl:
                out.append({"type": "warning", "check": "odds_credits", "key": f"odds_credits_{lvl}:{d0:%Y-%m}",
                            "message": f"{remaining_credits} odds credits left, {days_to_renewal} days until the plan renews on day {renew_day}. Player prices and review price refreshes stop first; moneylines last."})
                break
    return out


def keepalive_events(last_commit_age_days: float | None, last_keepalive_success_age_days: float | None, now: datetime) -> list[dict]:
    """GitHub switches off scheduled workflows in a public repository after 60 days without activity: alert well before that, and if the keepalive workflow itself has stopped succeeding."""
    out = []
    mc, ms = float(_c("keepalive_max_commit_age_days", default=50)), float(_c("keepalive_max_success_age_days", default=40))
    if last_commit_age_days is not None and last_commit_age_days > mc:
        out.append({"type": "warning", "check": "keepalive", "key": f"repo_quiet:{now:%G-W%V}", "message": f"The repository has had no commit for {last_commit_age_days:.0f} days; GitHub disables its schedules at 60. Push a commit or run the Keep scheduled workflows alive workflow."})
    if last_keepalive_success_age_days is not None and last_keepalive_success_age_days > ms:
        out.append({"type": "warning", "check": "keepalive", "key": f"keepalive_stale:{now:%G-W%V}", "message": f"The keepalive workflow has not succeeded for {last_keepalive_success_age_days:.0f} days (it runs monthly). Check Actions > Keep scheduled workflows alive."})
    if last_keepalive_success_age_days is None:
        out.append({"type": "warning", "check": "keepalive", "key": f"keepalive_never:{now:%G-W%V}", "message": "No successful run of the keepalive workflow was found. Check that Actions > Keep scheduled workflows alive exists and is enabled."})
    return out
