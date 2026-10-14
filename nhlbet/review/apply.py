"""Apply a validated review (or its absence) to the candidate picks. No LLM is involved from here on.

* vetoed picks leave the final list and go to the vetoed list with the reason; flagged picks stay, unchanged, with a visible warning;
  kept picks stay unchanged;
* ``assert_subset`` fails the run if a final pick is not exactly a candidate pick (same numbers, same stake);
* if the agent failed (error, timeout, max turns, invalid or missing output) every pick is published as UNREVIEWED, never as reviewed. The deterministic
  odds veto still applies, because it needs no agent;
* the logs are append-only JSONL keyed so that re-running the same review adds nothing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nhlbet.review.contract import CHECK_KEYS, FROZEN_FIELDS, build_candidates, candidate_id, pick_id
from nhlbet.review.odds_check import combine

STATE_NAME = "picks_final.json"
LOG_FINAL, LOG_VETOED = "final_picks.jsonl", "vetoed_picks.jsonl"


class ReviewIntegrityError(RuntimeError):
    """A final pick is not a candidate pick, or one of its numbers differs."""


def prepare_input(candidate: dict, wave_game_ids: set[int], facts: dict[str, dict], now: datetime) -> dict:
    """The batch the agent reviews: this wave's picks that code has not already vetoed, each with the odds fact computed in code."""
    picks = []
    for p in candidate["picks"]:
        if p["game_id"] not in wave_game_ids or facts.get(p["pick_id"], {}).get("floor") == "veto":
            continue
        picks.append({**p, "odds_movement_fact": facts.get(p["pick_id"], {}).get("summary", "unconfirmed: not computed")})
    gids = {str(p["game_id"]) for p in picks}
    inp = {"date": candidate["date"], "now_utc": now.isoformat(timespec="seconds"), "model_version": candidate.get("model_version"),
           "games": {k: v for k, v in candidate["games"].items() if k in gids}, "picks": picks}
    inp["candidate_id"] = candidate_id(inp)
    return inp


def assert_subset(candidate: dict, final: list[dict]) -> None:
    """Every final pick exists in the candidate file with identical frozen fields (probabilities, odds, line, stake...)."""
    by_id = {p["pick_id"]: p for p in candidate["picks"]}
    for p in final:
        c = by_id.get(p["pick_id"])
        if c is None:
            raise ReviewIntegrityError(f"final pick {p['pick_id']} is not in the candidate file")
        for k in FROZEN_FIELDS:
            if p.get(k) != c.get(k):
                raise ReviewIntegrityError(f"final pick {p['pick_id']}: {k} changed from {c.get(k)!r} to {p.get(k)!r}")
    ids = [p["pick_id"] for p in final]
    if len(ids) != len(set(ids)):
        raise ReviewIntegrityError("duplicate pick in the final list")


def _record(pick: dict, decision: str, by: str, status: str, agent: dict | None, fact: dict | None, run_id: str, now: datetime, err: str | None) -> dict:
    fact = fact or {}
    reason = (agent or {}).get("reason")
    if by == "code" and decision != "keep":
        reason = ("odds: " if decision == "veto" else "line: ") + fact.get("summary", "")
    if status == "UNREVIEWED" and by != "code":
        reason = "the review agent did not complete; this pick was not reviewed" + (f" ({err[:140]})" if err else "")
    checks = None
    if agent:
        checks = {**{k: agent["checks"][k] for k in CHECK_KEYS}, "odds_movement": fact.get("summary", "unconfirmed")}     # odds movement is always the code's fact
    rev = {"status": status, "decision": decision if (status == "REVIEWED" or decision != "keep") else None,
           "decided_by": by if (status == "REVIEWED" or decision != "keep") else None, "reason": reason, "confidence": (agent or {}).get("confidence"),
           "sources": (agent or {}).get("sources", []), "checks": checks, "odds_fact": fact or None, "run_id": run_id, "reviewed_at": now.isoformat(timespec="seconds"),
           "agent_reason": (agent or {}).get("reason") if by == "code" else None}
    if decision == "flag":
        rev["warning"] = reason
    if status == "UNREVIEWED" and decision != "veto":
        rev["warning"] = "UNREVIEWED: " + (reason or "")
    return {**{k: pick[k] for k in pick}, "review": rev}


def apply_review(candidate: dict, wave_game_ids: set[int], facts: dict[str, dict], review: dict | None, errors: list[str], run_id: str, now: datetime) -> dict:
    """Decisions for this wave's picks. ``review`` is a VALIDATED picks_review.json or None (agent failure; ``errors`` says why)."""
    ok = review is not None and not errors
    agent = {r["pick_id"]: r for r in review["reviews"]} if ok else {}
    final, vetoed = [], []
    for p in candidate["picks"]:
        if p["game_id"] not in wave_game_ids:
            continue
        fact = facts.get(p["pick_id"])
        a = agent.get(p["pick_id"])
        decision, by = combine(a["decision"] if a else None, fact)
        status = "REVIEWED" if (ok and a is not None) else "UNREVIEWED"
        if status == "UNREVIEWED" and by == "agent":
            by = "code"
        rec = _record(p, decision, by, status, a, fact, run_id, now, "; ".join(errors)[:200] if errors else None)
        (vetoed if decision == "veto" else final).append(rec)
    assert_subset(candidate, [{k: v for k, v in r.items() if k != "review"} for r in final + vetoed])
    return {"run_id": run_id, "date": candidate["date"], "status": "reviewed" if ok else "unreviewed", "errors": list(errors), "final": final, "vetoed": vetoed,
            "candidate_id": candidate["candidate_id"], "at": now.isoformat(timespec="seconds")}


# ---------------------------------------------------------------------------------------------- state (what the page shows) and logs
def load_state(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def merge_state(state: dict, result: dict) -> dict:
    """picks_final.json: the latest decision per pick for the day. A new day starts empty; re-running a pass replaces its own records."""
    if state.get("date") != result["date"]:
        state = {"date": result["date"], "picks": {}, "vetoed": {}, "passes": []}
    for r in result["final"] + result["vetoed"]:
        state["picks"].pop(r["pick_id"], None)
        state["vetoed"].pop(r["pick_id"], None)
    for r in result["final"]:
        state["picks"][r["pick_id"]] = r
    for r in result["vetoed"]:
        state["vetoed"][r["pick_id"]] = r
    state["passes"] = [x for x in state["passes"] if x["run_id"] != result["run_id"]] + [
        {"run_id": result["run_id"], "status": result["status"], "at": result["at"], "n_final": len(result["final"]), "n_vetoed": len(result["vetoed"]), "errors": result["errors"][:5]}]
    state["updated_at"] = result["at"]
    return state


def _key(row: dict) -> tuple:
    r = row["review"]
    return (r["run_id"], row["pick_id"], r["status"], r["decision"], r["decided_by"])


def append_log(path: Path, rows: list[dict], now: datetime) -> int:
    """Append rows whose key is not already in the file; never rewrite or delete a line. Returns the number added."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                seen.add(_key(json.loads(line)))
            except (ValueError, KeyError):
                continue
    new = [r for r in rows if _key(r) not in seen]
    if new:
        with path.open("a") as f:
            for r in new:
                f.write(json.dumps({**r, "date": r.get("date") or r["review"]["reviewed_at"][:10], "logged_at": now.isoformat(timespec="seconds")}, sort_keys=True, allow_nan=False) + "\n")
    return len(new)


def write_all(result: dict, root: str | Path, now: datetime, dry_run: bool = False) -> dict:
    """Update the day's state and append the logs. Dry run computes and returns the same numbers but writes nothing."""
    root = Path(root)
    state_path = root / "data" / "review" / STATE_NAME
    state = merge_state(load_state(state_path), result)
    if dry_run:
        return {"state": state, "added_final": 0, "added_vetoed": 0}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True, allow_nan=False))
    logs = root / "data" / "logs" / "review"
    day = result["date"]
    fin = [{**r, "date": day} for r in result["final"]]
    vet = [{**r, "date": day} for r in result["vetoed"]]
    return {"state": state, "added_final": append_log(logs / LOG_FINAL, fin, now), "added_vetoed": append_log(logs / LOG_VETOED, vet, now)}


# ---------------------------------------------------------------------------------------------- the page
def strip_review(payload: dict) -> dict:
    """Undo ``attach_to_payload`` so it can be applied again (the vetoed rows go back into the ranking)."""
    rows = list(payload.get("ranking", [])) + [r for r in payload.pop("vetoed", []) if "review" in r]
    for r in rows:
        r.pop("review", None)
    payload["ranking"] = sorted(rows, key=lambda r: r.get("rank", 10 ** 6))
    payload.pop("review", None)
    return payload


def _slim(rev: dict) -> dict:
    return {k: rev.get(k) for k in ("status", "decision", "decided_by", "reason", "confidence", "sources", "warning", "reviewed_at", "run_id", "checks")}


def attach_to_payload(payload: dict, state: dict | None) -> dict:
    """Mark every candidate pick on the page: kept / flagged (with a warning) / UNREVIEWED; move vetoed picks to ``payload['vetoed']``."""
    strip_review(payload)
    cand = {p["pick_id"] for p in build_candidates(payload)["picks"]}
    state = state if state and state.get("date") == payload.get("date") else {"picks": {}, "vetoed": {}, "passes": []}
    keep_rows, vetoed_rows = [], []
    for r in payload.get("ranking", []):
        leg = r.get("leg") or {}
        pid = pick_id(payload["date"], r["game_id"], leg.get("market", r.get("kind")), leg.get("point"), leg.get("side", ""), leg.get("player_id", 0))
        r["pick_id"] = pid
        if pid in state.get("vetoed", {}):
            r["review"] = _slim(state["vetoed"][pid]["review"])
            vetoed_rows.append(r)
            continue
        if pid in state.get("picks", {}):
            r["review"] = _slim(state["picks"][pid]["review"])
        elif pid in cand:
            r["review"] = {"status": "UNREVIEWED", "decision": None, "decided_by": None, "reason": "no review has run for this pick yet", "confidence": None, "sources": [],
                           "warning": "UNREVIEWED: no review has run for this pick yet", "reviewed_at": None, "run_id": None, "checks": None}
        keep_rows.append(r)
    payload["ranking"] = keep_rows
    payload["vetoed"] = vetoed_rows
    payload["review"] = {"enabled": True, "date": payload.get("date"), "passes": state.get("passes", []), "n_vetoed": len(vetoed_rows),
                         "n_unreviewed": sum(1 for r in keep_rows if (r.get("review") or {}).get("status") == "UNREVIEWED"),
                         "n_flagged": sum(1 for r in keep_rows if (r.get("review") or {}).get("decision") == "flag"),
                         "n_kept": sum(1 for r in keep_rows if (r.get("review") or {}).get("decision") == "keep")}
    return payload
