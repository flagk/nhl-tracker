"""Phase 3 on the REAL database (runs in GitHub Actions): canary matrix, time-shift placebo, adversarial validation, shuffled-market placebo.

Research window only: rows dated after RESEARCH_END are never read (that is the day before the locked half-season; the live record since 2026-09-30 is used for the market
placebo because nobody has tuned on it). Writes audit/results/phase3_real.json and .md. Exits 1 only on a broken check, never on an unflattering finding.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nhlbet.config import load_selected_features  # noqa: E402
from nhlbet.data.loaders import load_tables  # noqa: E402
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.features.builder import BuilderConfig, FeatureBuilder  # noqa: E402
from nhlbet.validity.canary import MODES, detect  # noqa: E402
from nhlbet.validity.placebo import adversarial_validation, shuffled_market_placebo, time_shift_placebo  # noqa: E402

RESEARCH_END = pd.Timestamp("2026-01-04")          # the day before the locked half-season (2026-01-05); nothing later is read
CANARY_FROM = pd.Timestamp("2024-07-01")           # canary matrix runs on the last two research seasons (each detector rebuilds the features several times)


def clip(tables, lo=None, hi=None):
    g = tables["games"]
    keep = set(g.loc[((g.game_date >= lo) if lo is not None else True) & (g.game_date <= hi), "game_id"])
    return {k: (v[v.game_id.isin(keep)] if "game_id" in v else v) for k, v in tables.items()}


def main() -> int:
    store = Store(str(ROOT / "data" / "nhl.db"))
    tables = load_tables(store)
    n = int(tables["games"].home_score.notna().sum())
    if n < 5000:
        sys.exit(f"only {n} completed games in the database: restore the cache first")
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "completed_games": n, "research_end": str(RESEARCH_END.date())}
    feats = load_selected_features()
    research = clip(tables, hi=RESEARCH_END)
    # 1. canary matrix
    window = clip(tables, lo=CANARY_FROM, hi=RESEARCH_END)
    cut = pd.Timestamp("2025-06-30")
    res["canary"] = {"window": [str(CANARY_FROM.date()), str(RESEARCH_END.date())], "games": int(window["games"].home_score.notna().sum()), "features_used": feats[:12]}
    res["canary"]["results"] = {str(m): detect(window, m, cut, features=feats[:12]) for m in (None,) + MODES}
    # 2. time-shift placebo
    F = FeatureBuilder(BuilderConfig()).build(research)
    F = F[(F.game_type == 2) & F.home_win.notna()]
    res["time_shift"] = {f: time_shift_placebo(F, f).round(4).to_dict("records") for f in ("gd_l10", "gd_ewm", "elo")}
    # 3. adversarial validation, research window only
    dfeats = [c for c in feats if c in F.columns]
    res["adversarial_validation"] = [adversarial_validation(F, dfeats, d) for d in ("2024-10-04", "2025-07-01")]
    # 4. shuffled-market placebo on the live record (decision-time market, after the locked block)
    from nhlbet.report.betlog import resolved
    d = resolved(store)
    d = d[d.p_market.notna() & d.p_model.notna() & d.home_win.notna()] if len(d) else d
    if len(d) >= 60:
        res["shuffled_market"] = {"games": int(len(d)), **shuffled_market_placebo(d.p_adj.fillna(d.p_model), d.p_market, d.home_win.astype(int), n_perm=1000)}
    else:
        res["shuffled_market"] = {"games": int(len(d)), "note": "too few resolved live games with a market price"}
    out = ROOT / "audit" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3_real.json").write_text(json.dumps(res, indent=1, default=float))
    ok = res["canary"]["results"]["None"]["caught"] is False and all(res["canary"]["results"][str(m)]["caught"] for m in MODES)
    print(json.dumps({"canary_matrix_ok": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
