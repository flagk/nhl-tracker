"""The contract between the deterministic pipeline and the review agent.

``build_candidates`` turns the page's ranked picks into ``picks_candidate.json`` (one stable ``pick_id`` per pick). The agent receives a batch of
these as ``review_input.json`` and must answer with ``picks_review.json``; ``validate_review`` is the strict, separate gate that file has to pass.
Invalid, missing or partial output is an agent failure (see apply.py): it is never repaired or guessed at.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "agent" / "picks_review.schema.json"
MAX_CANDIDATES = 30                       # the agent reviews at most this many picks per day (best-ranked first)
MAX_SOURCE_AGE = timedelta(hours=24)      # a source retrieved longer ago than this cannot support a pre-game claim
CHECK_KEYS = ("confirmed_goalie", "lineup_changes", "injuries", "odds_movement")
# a pick is a candidate field-for-field; these never change between candidate and final (asserted in apply.py)
FROZEN_FIELDS = ("pick_id", "game_id", "market", "line", "side", "label", "odds", "book", "p_model", "p_market", "edge", "ev", "p_push", "kind", "player_id", "captured_at", "rank", "score", "stake")


def pick_id(date: str, game_id, market: str, line, side: str, player_id=0) -> str:
    """Stable across re-runs of the same day: the same bet on the same game always gets the same id."""
    key = f"{date}|{int(game_id)}|{market}|{'' if line is None else f'{float(line):g}'}|{side}|{int(player_id or 0)}"
    return "pk_" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _book(row: dict, game: dict | None) -> str | None:
    """Best-price book where the payload exposes it (the public site hides book names, and so does this file then)."""
    if not game:
        return None
    leg = row.get("leg") or {}
    if leg.get("market") == "h2h":
        s = (game.get("sides") or {}).get(leg.get("side"))
        return (s or {}).get("best_book")
    for a in game.get("alt") or []:
        if a.get("market") == leg.get("market") and a.get("side") == leg.get("side") and a.get("point") == leg.get("point"):
            return a.get("best_book")
    return None


def build_candidates(payload: dict, max_picks: int = MAX_CANDIDATES) -> dict:
    """``picks_candidate.json`` from a site payload: ranked picks whose raw edge (model minus no-vig market probability) clears the policy minimum."""
    date = payload["date"]
    min_edge = float((payload.get("policy") or {}).get("min_edge", 0.03))
    games = {int(g["game_id"]): g for g in payload.get("games", [])}
    picks, used = [], set()
    for r in payload.get("ranking", []):
        edge = float(r["p_model"]) - float(r["p_market"])
        leg = r.get("leg") or {}
        if edge < min_edge or r.get("price") in (None, 0):
            continue
        pid = pick_id(date, r["game_id"], leg.get("market", r.get("kind")), leg.get("point"), leg.get("side", ""), leg.get("player_id", 0))
        if pid in used:
            continue
        used.add(pid)
        g = games.get(int(r["game_id"]))
        picks.append({"pick_id": pid, "game_id": int(r["game_id"]), "market": leg.get("market", r.get("kind")), "line": leg.get("point"), "side": leg.get("side"),
                      "label": r["pick"], "odds": float(r["price"]), "book": _book(r, g), "p_model": float(r["p_model"]), "p_market": float(r["p_market"]),
                      "edge": edge, "ev": float(r["ev_raw"]), "p_push": float(r.get("p_push") or 0.0), "kind": r["kind"], "player_id": int(leg.get("player_id") or 0),
                      "captured_at": r.get("captured_at"), "rank": int(r["rank"]), "score": float(r["score"]), "stake": 10.0})
        if len(picks) >= max_picks:
            break
    gids = {p["game_id"] for p in picks}
    gfacts = {str(i): {"home": g["home"], "away": g["away"], "start_utc": g.get("start_utc"), "home_goalie": g.get("home_goalie"), "away_goalie": g.get("away_goalie"),
                       "goalie_status": g.get("goalie_status")} for i, g in games.items() if i in gids}
    body = {"schema_version": 1, "date": date, "run_type": payload.get("run_type"), "generated_at": payload.get("generated_at"),
            "model_version": (payload.get("model") or {}).get("version"), "min_edge": min_edge, "games": gfacts, "picks": picks}
    body["candidate_id"] = candidate_id(body)
    return body


def candidate_id(c: dict) -> str:
    """12 hex digits over the pick ids and their frozen numbers: changes whenever any pick or any of its numbers changes."""
    blob = json.dumps([[p["pick_id"], p["odds"], round(p["p_model"], 9), p["line"]] for p in c["picks"]], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _ts(s: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def validate_review(review, review_input: dict, now: datetime | None = None) -> list[str]:
    """Every reason ``review`` (the parsed picks_review.json) cannot be used; an empty list means it is valid.

    Schema first, then the rules a schema cannot express: exactly the ids that were asked for, each once; veto and flag explained; every claim sourced
    with a plausible retrieval time; nothing that is merely unconfirmed may be kept.
    """
    from jsonschema import Draft202012Validator, FormatChecker

    now = now or datetime.now(timezone.utc)
    errs = [f"schema: {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:160]}"
            for e in sorted(Draft202012Validator(load_schema(), format_checker=FormatChecker()).iter_errors(review), key=lambda e: list(map(str, e.absolute_path)))][:20]
    if errs:
        return errs
    if review["candidate_id"] != review_input["candidate_id"]:
        errs.append(f"candidate_id {review['candidate_id']} is not the batch that was sent ({review_input['candidate_id']})")
    asked = [p["pick_id"] for p in review_input["picks"]]
    got = [r["pick_id"] for r in review["reviews"]]
    dup = sorted({x for x in got if got.count(x) > 1})
    if dup:
        errs.append(f"duplicate pick_id: {dup}")
    if set(asked) - set(got):
        errs.append(f"missing reviews for {sorted(set(asked) - set(got))}")
    if set(got) - set(asked):
        errs.append(f"reviews for picks that were not asked about: {sorted(set(got) - set(asked))}")
    t = _ts(review["reviewed_at"])
    if t is None or t > now + timedelta(minutes=5):
        errs.append("reviewed_at is in the future or unreadable")
    for r in review["reviews"]:
        pid = r["pick_id"]
        urls = {s["url"] for s in r["sources"]}
        for s in r["sources"]:
            ts = _ts(s["retrieved_at"])
            if ts is None or ts > now + timedelta(minutes=5) or now - ts > MAX_SOURCE_AGE:
                errs.append(f"{pid}: source {s['url'][:60]} has an implausible retrieval time {s['retrieved_at']}")
        found = {}
        for k in CHECK_KEYS:
            c = r["checks"][k]
            if c == "unconfirmed":
                continue
            found[k] = c
            if k != "odds_movement" and not c["source_urls"]:
                errs.append(f"{pid}: check {k} has a finding but no source")
            for u in c["source_urls"]:
                if u not in urls:
                    errs.append(f"{pid}: check {k} cites {u[:60]} which is not in sources")
        if r["decision"] in ("flag", "veto") and len(r["reason"].split()) < 3:
            errs.append(f"{pid}: {r['decision']} needs a real reason")
        if r["decision"] == "veto":
            if not r["sources"]:
                errs.append(f"{pid}: a veto needs at least one source")
            if not [k for k in found if k != "odds_movement"]:
                errs.append(f"{pid}: a veto must rest on a confirmed finding (goalie, lineup or injury), not on unconfirmed checks or odds")
        if r["decision"] == "keep":
            unconf = [k for k in CHECK_KEYS if k != "odds_movement" and r["checks"][k] == "unconfirmed"]
            if unconf:
                errs.append(f"{pid}: cannot keep a pick while {unconf} are unconfirmed: that is a flag")
    return errs
