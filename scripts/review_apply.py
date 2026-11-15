"""Step 3 of the pre-game review: validate the agent's file, apply it deterministically, log it, rebuild the page. No LLM.

    python scripts/review_apply.py --agent-outcome success|failure|skipped [--dry-run] [--execution-file path]

Exit codes: 0 reviewed (or nothing needed an agent); 3 the review FAILED and the picks were published as UNREVIEWED (so the workflow shows red and GitHub
sends its failure notification); 4 an integrity assertion failed (a final pick differs from its candidate): nothing is written; 5 decisions were logged but the page could not be rebuilt.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.review.apply import ReviewIntegrityError, apply_review, attach_to_payload, load_state, write_all  # noqa: E402
from nhlbet.review.contract import validate_review  # noqa: E402


def load_review(work: Path, outcome: str, needed: bool, inp: dict, now: datetime) -> tuple[dict | None, list[str], str | None]:
    """(validated review or None, errors, raw text). Anything but a clean success with a valid file is an agent failure."""
    if not needed:
        return {"candidate_id": inp["candidate_id"], "reviewed_at": now.isoformat(), "reviews": []}, [], None
    errors: list[str] = []
    if outcome != "success":
        errors.append(f"the agent step ended with outcome '{outcome}' (error, timeout, max turns or tampering)")
    p = work / "picks_review.json"
    raw = p.read_text() if p.exists() else None
    if raw is None:
        errors.append("picks_review.json was not written")
    elif not errors:
        try:
            review = json.loads(raw)
        except ValueError as e:
            return None, [f"picks_review.json is not valid JSON: {e}"], raw
        errors = validate_review(review, inp, now)
        if not errors:
            return review, [], raw
    return None, errors, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--work", default="agent/work")
    ap.add_argument("--agent-outcome", default="success", choices=["success", "failure", "cancelled", "skipped"])
    ap.add_argument("--execution-file", help="the claude-code-action execution_file output (kept gzipped in review_logs/)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print everything, write nothing, publish nothing")
    ap.add_argument("--now", help="ISO time (tests)")
    a = ap.parse_args()
    root, work = Path(a.root), Path(a.root) / a.work
    now = datetime.fromisoformat(a.now) if a.now else datetime.now(timezone.utc)
    ctx = json.loads((work / "context.json").read_text())
    inp = json.loads((work / "review_input.json").read_text())
    cand = json.loads((root / "data" / "review" / "picks_candidate.json").read_text())
    if cand["candidate_id"] != ctx["candidate_id"]:
        print("candidate file changed since the review was prepared: refusing to apply", file=sys.stderr)
        return 4
    review, errors, raw = load_review(work, a.agent_outcome, bool(inp["picks"]), inp, now)
    try:
        res = apply_review(cand, set(ctx["wave_game_ids"]), ctx["facts"], review, errors, ctx["run_id"], now)
    except ReviewIntegrityError as e:
        print(f"INTEGRITY FAILURE: {e}", file=sys.stderr)
        return 4
    w = write_all(res, root, now, dry_run=a.dry_run)
    page_failed = False
    meta = {"run_id": res["run_id"], "date": res["date"], "status": res["status"], "errors": errors, "dry_run": a.dry_run, "reviewed_at": now.isoformat(timespec="seconds"),
            "model": os.environ.get("REVIEW_MODEL", ""), "action": os.environ.get("REVIEW_ACTION", ""), "agent_outcome": a.agent_outcome,
            "workflow_run": (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}" if os.environ.get("GITHUB_RUN_ID") else None),
            "candidate_id": cand["candidate_id"], "review_batch_id": inp["candidate_id"], "wave_start": ctx["wave_start"], "odds": ctx["odds"],
            "n_final": len(res["final"]), "n_vetoed": len(res["vetoed"]), "added_to_log": {"final": w["added_final"], "vetoed": w["added_vetoed"]},
            "sources": sorted({s["url"] for r in (review or {}).get("reviews", []) for s in r["sources"]}),
            "decisions": {r["pick_id"]: {"decision": r["review"]["decision"], "by": r["review"]["decided_by"], "status": r["review"]["status"]} for r in res["final"] + res["vetoed"]}}
    log = {"meta": meta, "input": inp, "facts": ctx["facts"], "agent_output_raw": raw}
    if a.dry_run:
        (work / "dry_run_result.json").write_text(json.dumps({"meta": meta, "final": res["final"], "vetoed": res["vetoed"]}, indent=1, default=str))
    else:
        d = root / "review_logs"
        d.mkdir(exist_ok=True)
        (d / f"{res['run_id']}.json").write_text(json.dumps(log, indent=1, sort_keys=True, allow_nan=False))       # same wave -> same file: a re-run replaces it, never adds a second
        if a.execution_file and Path(a.execution_file).exists():
            with open(a.execution_file, "rb") as src, gzip.open(d / f"{res['run_id']}.execution.json.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        pj = root / "site" / "picks.json"
        if pj.exists():
            try:
                from nhlbet.site.build import build_site
                payload = json.loads(pj.read_text())
                if payload.get("date") == res["date"]:
                    build_site(attach_to_payload(payload, w["state"]), root / "site")
            except Exception as e:                      # the decisions are already logged; say loudly that the page was not rebuilt
                print(f"PAGE NOT REBUILT: {e!r}", file=sys.stderr)
                page_failed = True
    print(json.dumps({k: meta[k] for k in ("run_id", "status", "n_final", "n_vetoed", "added_to_log", "errors", "dry_run")}, indent=1))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"### Pre-game review `{res['run_id']}`{' (DRY RUN)' if a.dry_run else ''}: **{res['status'].upper()}**\n\n"
                    f"{len(res['final'])} picks kept/flagged/unreviewed, {len(res['vetoed'])} vetoed. " + (f"\n\nFailure: {'; '.join(errors)[:600]}\n" if errors else "") + "\n")
    return 3 if res["status"] == "unreviewed" else 5 if page_failed else 0


if __name__ == "__main__":
    sys.exit(main())
