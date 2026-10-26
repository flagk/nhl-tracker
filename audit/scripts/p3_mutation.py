"""Phase 3: mutation testing of the leakage defences.

Each mutation re-introduces one specific leak into a COPY of the package (the repository is never modified), then the leakage test subset runs against the copy.
A mutation is KILLED when at least one test fails, SURVIVED when every test still passes: a survivor is a hole in the test suite and is reported as such.

    python audit/scripts/p3_mutation.py [--only M1,M2] [--out audit/results/mutation.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["tests/test_no_leakage.py", "tests/test_validity.py", "tests/test_audit_leakage.py", "tests/test_pit.py", "tests/test_audit_phase1.py", "tests/test_odds.py",
         "tests/test_data_integrity.py"]


def sub(old: str, new: str, count: int = 1):
    def f(src: str) -> str:
        if old not in src:
            raise SystemExit(f"mutation anchor not found: {old[:70]!r}")
        return src.replace(old, new, count)
    return f


def fold_before_emit(src: str) -> str:
    """Fold each date's results into the state BEFORE emitting that date's features."""
    emit = "            # ---- 1) emit features for every game on this date, from pre-date state only ----\n"
    fold_call = "                self._update(g, day, season, teams, goalies, elo, snapshot, league, season_pts,"
    if emit not in src or fold_call not in src:
        raise SystemExit("fold anchors not found")
    pre = ("            for g in day_games.itertuples(index=False):\n                if g.home_score != g.home_score or g.home_score is None:\n                    continue\n"
           "                self._update(g, day, season, teams, goalies, elo, snapshot, league, season_pts, _AdvView(adv_lookup, adv_asof, self._clock),\n"
           "                             starter_of, skaters_of, appear, pending)\n")
    return src.replace(emit, pre + emit, 1)


PEEK = ("        recs = ts.recs + ([Rec(day, season, is_home, opp, g.home, int(g.home_score if is_home else g.away_score), int(g.away_score if is_home else g.home_score), False, 1500.0, {}, {}, None)]"
        " if g.home_score == g.home_score and g.home_score is not None else [])\n")

MUTATIONS = {
    "M0": ("(control)", "features/builder.py", "CONTROL: no leak introduced (must SURVIVE, otherwise the harness itself breaks the tests)", lambda src: src + "\n# control\n"),
    "M1": ("builder.py", "features/builder.py", "results of a date folded into the state before that date's features are emitted", fold_before_emit),
    "M2": ("builder.py", "features/builder.py", "a game's own result included in its rolling window", sub("        recs = ts.recs\n", PEEK)),
    "M3": ("builder.py", "features/builder.py", "default goalie mode switched to 'actual' (post-game starter identity)", sub('goalie_mode: str = "probable"', 'goalie_mode: str = "actual"')),
    "M4": ("market.py", "features/market.py", "market features use snapshots after the decision time (closing line leaks in)", sub("s = s[s.captured_at <= s.decision].sort_values(\"captured_at\")", "s = s.sort_values(\"captured_at\")")),
    "M5": ("splits.py", "splits.py", "walk-forward training window includes the test block", sub("train = np.flatnonzero((d < start).to_numpy())", "train = np.flatnonzero((d < stop).to_numpy())")),
    "M6": ("calibration.py", "models/calibration.py", "online calibrator fitted on the block it is applied to", sub("past = dates < dates[block == b].min()", "past = dates <= dates[block == b].max()")),
    "M7": ("availability.py", "validity/availability.py", "availability guard disabled", sub("    cols = sorted(set(columns))\n", "    return\n    cols = sorted(set(columns))\n")),
    "M8": ("null.py", "validity/null.py", "outcome shuffle is a no-op (the null world is not null)", sub("        perm = rng.permutation(idx)", "        perm = idx")),
    "M9": ("goalies.py", "data/goalies.py", "goalie confirmations ignore the decision time", sub("    if as_of is not None:\n        conf = conf[", "    if False:\n        conf = conf[")),
    "M10": ("asof.py", "pit/asof.py", "point-in-time resolver ignores knowledge time", sub("            if kt <= t:", "            if True:")),
    "M11": ("snapshots.py", "pit/snapshots.py", "snapshot as_of returns versions fetched after the decision time", sub("ok = [v for v in self.versions(kind, key) if v.knowledge_time <= t]", "ok = [v for v in self.versions(kind, key)]")),
    "M12": ("walkforward.py", "models/walkforward.py", "walk-forward row order left to the input order (unstable ties)", sub(', kind="stable").copy()', ').copy()')),
    "M13": ("builder.py", "features/builder.py", "league goalie-start counters updated with the current game before it is emitted", sub('        pending[(g.game_id, team)] = (primary, b2b_after_primary)\n', '        pending[(g.game_id, team)] = (primary, b2b_after_primary)\n        league["rest_n"] += 1; league["rest_pri"] += int(starter_of.get((g.game_id, team)) == primary)\n')),
}


def run(mid: str, desc: str, rel: str, fn, keep: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "nhlbet"
        shutil.copytree(ROOT / "nhlbet", pkg, ignore=shutil.ignore_patterns("__pycache__"))
        for extra in ("config", "reports", "state", "data/models", "data/logs"):         # modules locate these relative to the package: without them tests fail for the wrong reason
            if (ROOT / extra).exists():
                shutil.copytree(ROOT / extra, Path(d) / extra, ignore=shutil.ignore_patterns("__pycache__", "*.joblib"))
        f = pkg / rel
        before = f.read_text()
        after = fn(before)
        if after == before:
            return {"id": mid, "description": desc, "status": "INVALID", "detail": "mutation changed nothing"}
        f.write_text(after)
        # the package is installed in editable mode, and its import finder outranks PYTHONPATH: drop it so the tests import the mutated copy
        launcher = ("import sys; sys.meta_path = [f for f in sys.meta_path if 'editable' not in (getattr(f, '__name__', '') + type(f).__name__).lower()]; "
                    f"sys.path.insert(0, {d!r}); import nhlbet, pytest; assert nhlbet.__file__.startswith({d!r}), nhlbet.__file__; "
                    f"sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', *{TESTS!r}]))")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run([sys.executable, "-c", launcher], cwd=ROOT, env=env, capture_output=True, text=True)
        if "AssertionError" in p.stderr and "nhlbet/__init__" in p.stderr:
            return {"id": mid, "description": desc, "status": "INVALID", "detail": "tests did not import the mutated copy"}
        failed = [ln.split(" - ")[0] for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        status = "KILLED" if p.returncode != 0 else "SURVIVED"
        return {"id": mid, "description": desc, "status": status, "n_failing_tests": len(failed), "killed_by": failed[:6], "pytest_tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(ROOT / "audit" / "results" / "mutation.json"))
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    res = []
    for mid, (_, rel, desc, fn) in MUTATIONS.items():
        if only and mid not in only:
            continue
        r = run(mid, desc, rel, fn)
        print(mid, r["status"], "-", desc, flush=True)
        res.append(r)
    control = next((r for r in res if r["id"] == "M0"), None)
    if control is not None and control["status"] != "SURVIVED":
        print("HARNESS BROKEN: the unmutated control fails the tests", control)
        res = [dict(r, status="INVALID" if r["id"] != "M0" else r["status"]) for r in res]
    out = {"mutants": res, "killed": sum(r["status"] == "KILLED" for r in res if r["id"] != "M0"), "control": (control or {}).get("status"), "survived": [r["id"] for r in res if r["status"] == "SURVIVED" and r["id"] != "M0"], "invalid": [r["id"] for r in res if r["status"] == "INVALID"], "total": len([r for r in res if r["id"] != "M0"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "mutants"}))
    return 1 if out["survived"] or out["invalid"] or out["control"] != "SURVIVED" else 0


if __name__ == "__main__":
    sys.exit(main())
