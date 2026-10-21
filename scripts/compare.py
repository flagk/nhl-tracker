"""Phase 4: model comparison on the research window, then the single holdout evaluation, then the (informational) market-blend check.

    python scripts/compare.py [--db data/nhl.db]

The decision rule is in reports/model_comparison_protocol.md and was committed before any result. This script is one of the few allowed to read the holdout lock; the
holdout is evaluated exactly once (a second run stops with HoldoutViolation).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nhlbet.report.model_selection import compare_candidates, score_models, select_model  # noqa: E402
from nhlbet.validity import holdout  # noqa: E402

REFERENCE = "stack__online"
CANDIDATES = ["home_rate", "elo", "elo__online", "logistic__online", "rf__online", "lgbm__online", "xgb__online", "wavg__online", "stack__platt"]

ap = argparse.ArgumentParser()
ap.add_argument("--wf", default="reports/walkforward_predictions.csv")
ap.add_argument("--db", default=None, help="database with live quotes (for the informational market-blend check)")
ap.add_argument("--out", default="reports")
ap.add_argument("--eval-path", default=None, help="where the once-only holdout result is recorded (default reports/holdout_eval.json); a scratch path gives a dry run")
a = ap.parse_args()
out = Path(a.out)
W = pd.read_csv(a.wf, index_col=0, parse_dates=["game_date"]).dropna(subset=[REFERENCE])
research = holdout.research_view(W)
holdout.assert_research_only(research.game_date, "model comparison")
lock = holdout.load_lock()
tab = compare_candidates(research, REFERENCE, CANDIDATES)
chosen = select_model(tab)
res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reference": REFERENCE, "chosen": chosen, "research_games": int(len(research)),
       "research_from": str(research.game_date.min().date()), "research_to": str(research.game_date.max().date()), "table": tab.to_dict("records")}
md = ["# Phase 4: model comparison (research window only)", "",
      f"Protocol: `reports/model_comparison_protocol.md` (committed before this result). {len(research):,} games, {res['research_from']} to {res['research_to']}; the holdout ({lock['start']} to {lock['end']}) was not read here.", "",
      f"Erratum to the protocol: it counted 1,973 research games; the production output `{REFERENCE}` only exists after the online calibrator's burn-in, so every candidate is compared on the {len(research):,} games where it exists.", "",
      "Gain = reference log loss minus candidate log loss (positive = candidate better). Two one-sided tests per candidate (date-clustered bootstrap, Diebold-Mariano with HAC variance), Holm-corrected across candidates; a candidate qualifies only if BOTH adjusted p-values are below 0.05.", "",
      tab.round(5).to_markdown(index=False), "", f"**Decision: {'keep the reference ' + REFERENCE if chosen == REFERENCE else 'adopt ' + chosen}.**", ""]
(out / "model_comparison_research.json").write_text(json.dumps(res, indent=1, default=float))


def holdout_fn(lk):
    H = W[holdout.in_holdout(W.game_date, lk)]
    names = list(dict.fromkeys([chosen, REFERENCE, "elo", "home_rate"]))
    t = score_models(H, names)
    r = research
    return {"n": int(len(H)), "from": str(H.game_date.min().date()), "to": str(H.game_date.max().date()), "selected": chosen, "reference": REFERENCE, "scores": t.to_dict("records"),
            "research_window_log_loss": {n: float(np.mean(-(r.y * np.log(np.clip(r[n], 1e-6, 1)) + (1 - r.y) * np.log(np.clip(1 - r[n], 1e-6, 1))))) for n in names}}


hres = holdout.evaluate_once(holdout_fn, path=Path(a.eval_path)) if a.eval_path else holdout.evaluate_once(holdout_fn)            # raises HoldoutViolation if it was already evaluated
ht = pd.DataFrame(hres["scores"])
md += ["## The single holdout evaluation", "", f"{hres['n']} games, {hres['from']} to {hres['to']}; evaluated once at {datetime.now(timezone.utc).isoformat(timespec='seconds')} (`reports/holdout_eval.json`). "
       "These games were examined in earlier reports, so this is a discipline check, not a pristine sample (see `reports/phase1_leakage.md`). "
       "Disclosure: before this recorded run, a local dry run of this same script (writing to a scratch path, not to `reports/holdout_eval.json`) printed these numbers. The selection above depends only on the research window, so the dry run could not have changed it, but the holdout was in fact seen once before it was recorded once.", "", ht.round(5).to_markdown(index=False), "",
       "Research-window log loss of the same models, for comparison (a large gap would indicate overfitting to the research window):", "",
       pd.DataFrame({"model": list(hres["research_window_log_loss"]), "research_log_loss": list(hres["research_window_log_loss"].values())}).round(5).to_markdown(index=False), ""]
if a.db and Path(a.db).exists():
    from nhlbet.data.store import Store
    from nhlbet.report import evaluation as ev
    d = ev.live_game_frame(Store(a.db), "proportional")
    if len(d) >= 20:
        b = ev.blend_scores(d)
        md += ["## Shrinking the model toward the market (live games, informational: no historical prices exist, so nothing here is a selection)", "",
               f"{len(d)} finished live games. p = (1-w) * market at decision + w * model; the market is the proportional no-vig probability.", "", b.round(4).to_markdown(index=False), ""]
        (out / "model_blend_live.json").write_text(json.dumps(b.to_dict("records"), indent=1, default=float))
(out / "model_comparison_phase4.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
