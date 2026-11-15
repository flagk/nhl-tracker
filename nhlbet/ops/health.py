"""The health gate that runs before every publish. Each check returns a CheckResult; any failure opens a halt (state machine) and the picks are withheld.

schema -> freshness -> row counts -> input validation -> value ranges -> cross-source agreement (NHL schedule vs odds feed) -> odds source -> quota -> log integrity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhlbet.hygiene import data_freshness, validate_inputs
from nhlbet.ops.checks import CheckResult, failed, passed
from nhlbet.policy import policy_get

REQUIRED_TABLES = {"games": {"game_id", "game_date", "home", "away", "home_score", "away_score", "home_win", "status", "game_type", "start_utc"},
                   "team_game": {"game_id", "team", "goals", "goals_against"}, "odds_snapshots": {"captured_at", "event_id", "book", "market", "outcome", "price", "game_id"},
                   "recommendations": {"run_id", "run_at", "game_id", "p_model", "p_market", "action"}}


def schema_check(store) -> CheckResult:
    problems = []
    for t, cols in REQUIRED_TABLES.items():
        have = {r[1] for r in store.conn.execute(f"PRAGMA table_info({t})")}
        if not have:
            problems.append(f"table {t} is missing")
        elif cols - have:
            problems.append(f"{t} lacks columns {sorted(cols - have)}")
    if problems:
        return failed("schema", "soft", "; ".join(problems), "The database cache is broken: run the backfill workflow, or delete the Actions cache so the next run rebuilds the database from the committed logs.")
    return passed("schema", "soft", "all required tables and columns present")


def freshness_check(store, date: str, refresh_error: str | None) -> CheckResult:
    ok, msg = data_freshness(store, date, refresh_error, int(policy_get("autonomy", "freshness", "max_data_age_days", default=3)))
    return passed("freshness", "soft", msg or "data fresh") if ok else failed("freshness", "soft", msg, "Wait: the NHL data source is unreachable or stale. The halt clears itself after the next healthy runs; if it persists, check https://api-web.nhle.com.")


def row_count_check(store) -> CheckResult:
    mn = int(policy_get("autonomy", "health", "min_completed_games", default=5000))
    n = int(store.df("SELECT COUNT(*) AS n FROM games WHERE home_score IS NOT NULL AND source = 'nhl_api'").n.iloc[0])
    if n < mn:
        return failed("row_counts", "soft", f"only {n:,} completed games in the database (minimum {mn:,}): the cache is missing or truncated",
                      "Run the Backfill workflow to rebuild the history, then the daily run resumes by itself.")
    return passed("row_counts", "soft", f"{n:,} completed games")


def input_validation_check(store) -> CheckResult:
    p = validate_inputs(store)
    return failed("input_validation", "soft", "; ".join(p), "Inspect the games table for the listed problems (duplicates, inconsistent results) and re-run the backfill.") if p else passed("input_validation", "soft", "inputs valid")


def range_check(payload: dict) -> CheckResult:
    """Every probability inside (p_min, 1 - p_min), every price inside the decimal range, nothing missing, in the games and ranking just built."""
    pmin = float(policy_get("autonomy", "health", "prob_min", default=0.001))
    lo, hi = float(policy_get("autonomy", "health", "odds_min_decimal", default=1.01)), float(policy_get("autonomy", "health", "odds_max_decimal", default=60.0))
    bad = []

    def prob(v, what):
        if v is None or not np.isfinite(v) or not (pmin < v < 1 - pmin):
            bad.append(f"{what}={v}")

    def price(v, what):
        if v is None or not np.isfinite(v) or not (lo <= v <= hi):
            bad.append(f"{what}={v}")
    for g in payload.get("games", []):
        prob(g.get("p_home"), f"p_home[{g.get('game_id')}]")
        for k, s in (g.get("sides") or {}).items():
            prob(s.get("p_model"), f"p_model[{g.get('game_id')},{k}]"); prob(s.get("p_market"), f"p_market[{g.get('game_id')},{k}]"); price(s.get("best_decimal"), f"price[{g.get('game_id')},{k}]")
    for r in payload.get("ranking", []):
        prob(r.get("p_model"), f"p_model[{r.get('pick')}]"); prob(r.get("p_market"), f"p_market[{r.get('pick')}]"); price(r.get("price"), f"price[{r.get('pick')}]")
    if bad:
        return failed("value_ranges", "soft", f"{len(bad)} values outside their allowed range, e.g. {', '.join(bad[:4])}", "A model or odds parse produced impossible numbers: inspect the daily log, then re-run; picks stay withheld meanwhile.")
    return passed("value_ranges", "soft", "all probabilities and prices in range")


def cross_source_check(store, date: str) -> CheckResult:
    """The NHL schedule and the odds feed must agree about today's slate: similar game counts, and almost every odds event linked to a scheduled game."""
    sched = int(store.df("SELECT COUNT(*) AS n FROM games WHERE game_date = ? AND game_type IN (2, 3) AND source = 'nhl_api'", [date]).n.iloc[0])
    last = store.df("SELECT MAX(captured_at) AS c FROM odds_snapshots WHERE market = 'h2h'").c.iloc[0]
    if not sched or last is None:
        return passed("cross_source", "soft", "no scheduled games or no odds captured: nothing to compare")
    ev = store.df("SELECT event_id, MAX(game_id) AS gid FROM odds_snapshots WHERE market = 'h2h' AND captured_at = ? GROUP BY event_id", [last])
    mx_gap, mx_unlinked = int(policy_get("autonomy", "health", "max_slate_vs_odds_game_gap", default=3)), float(policy_get("autonomy", "health", "max_unlinked_odds_share", default=0.35))
    today_odds = store.df("SELECT COUNT(DISTINCT s.event_id) AS n FROM odds_snapshots s JOIN games g ON g.game_id = s.game_id WHERE s.market = 'h2h' AND s.captured_at = ? AND g.game_date = ?", [last, date]).n.iloc[0]
    unlinked = float(ev.gid.isna().mean()) if len(ev) else 0.0
    if unlinked > mx_unlinked:
        return failed("cross_source", "soft", f"{unlinked:.0%} of the odds feed's events match no NHL game (limit {mx_unlinked:.0%}): team mapping or schedule mismatch",
                      "Check nhlbet/teams.py for a renamed or new club and the odds feed for unusual events; picks stay withheld until the sources agree.")
    if abs(int(today_odds) - sched) > mx_gap and int(today_odds) > 0:
        return failed("cross_source", "soft", f"the NHL schedule has {sched} games today but the odds feed prices {int(today_odds)} (allowed gap {mx_gap})",
                      "One of the two sources is incomplete: wait for the next run; if it persists compare the schedule with the odds feed by hand.")
    return passed("cross_source", "soft", f"{sched} scheduled games, {int(today_odds)} priced, {unlinked:.0%} unlinked")


def odds_source_check(odds_meta: dict | None, scheduled_games: int) -> CheckResult:
    if odds_meta is None or not odds_meta.get("enabled"):
        return passed("odds_source", "soft", "odds disabled for this run")
    if odds_meta.get("error") and scheduled_games:
        return failed("odds_source", "soft", f"the odds fetch failed ({odds_meta['error']}) with {scheduled_games} games scheduled", "Wait for the next run; if it persists check ODDS_API_KEY and https://the-odds-api.com status.")
    if odds_meta.get("stale") and scheduled_games:
        return failed("odds_source", "soft", "only stale odds are available", "Wait for the next run; if it persists check ODDS_API_KEY and the plan's credits.")
    return passed("odds_source", "soft", "odds fetched")


def quota_check(remaining: int | None) -> CheckResult:
    res = int(policy_get("odds", "reserve_credits", default=25))
    if remaining is not None and remaining < res:
        return failed("odds_quota", "soft", f"only {remaining} odds credits left (reserve {res}): the monthly plan is exhausted",
                      "Wait for the plan to renew on the 1st, or upgrade it at https://the-odds-api.com; picks stay withheld without fresh prices.")
    return passed("odds_quota", "soft", "credits available" if remaining is not None else "credit balance unknown")
