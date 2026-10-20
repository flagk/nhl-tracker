"""Monte Carlo bankroll simulation -> reports/risk.md. At least 10,000 paths (policy) under three assumptions about the model's edge: its probabilities are true,
they are true only halfway to the market, the market is right (no skill). Reports the drawdown distribution, the probability of tripping the policy's drawdown kill switch,
and the risk of ruin.

    python scripts/risk_report.py [--db data/nhl.db]      # real bet template from the pick logs (needs the database)
    python scripts/risk_report.py --illustrative          # a synthetic template when no database is available
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from nhlbet.policy import policy_get  # noqa: E402
from nhlbet.risk.montecarlo import SCENARIOS, default_template, simulate, template_from_picks  # noqa: E402
from nhlbet.risk.policy import RiskConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--illustrative", action="store_true")
ap.add_argument("--out", default="reports")
ap.add_argument("--bankroll", type=float, default=1000.0)
a = ap.parse_args()
paths = int(policy_get("risk", "simulation", "paths", default=10000))
horizons = [int(h) for h in policy_get("risk", "simulation", "horizons", default=[200, 500])]
ruin = float(policy_get("risk", "simulation", "ruin_level", default=0.5))
kill = float(policy_get("risk", "kill_switch", "max_drawdown_pct", default=0.2))
cfg = RiskConfig(bankroll=a.bankroll)
source, t = "illustrative template (3% edge at 1.95, 1% of bankroll)", None
if not a.illustrative and Path(a.db).exists():
    from nhlbet.data.store import Store
    from nhlbet.report import evaluation as ev
    d = ev.pick_frame(Store(a.db))
    m = d[(d.group == "model") & (d.market == "h2h") & d.settled].rename(columns={"p_dec_proportional": "p_market_np"})
    if len(m):
        m = m.assign(p_market=m.p_market_np.fillna(m.p_market))
        t = template_from_picks(m, cfg)
        if len(t) >= 30:
            source = f"{len(t)} real moneyline picks that clear the policy's edge floor, staked as the live policy would"
        else:
            t = None
if t is None:
    t = default_template(0.03, 1.95, 0.01)
    if "illustrative" not in source:
        source += " (fewer than 30 real picks clear the policy's edge floor)"
rows = []
for h in horizons:
    r = simulate(t, paths, h, tuple(SCENARIOS.values()), a.bankroll, ruin, (0.1, 0.2, 0.3, 0.5), kill_dd=kill).reset_index()
    r.insert(0, "scenario", [k for k in SCENARIOS])
    rows.append(r)
res = pd.concat(rows)
cols = ["scenario", "bets", "median_final", "p05_final", "p95_final", "p_loss", f"p_ruin_{int(ruin * 100)}pct", "median_maxdd", "p90_maxdd", "p99_maxdd", "p_kill_switch"]
md = ["# Risk: Monte Carlo bankroll simulation", "",
      f"Template: {source}. {paths:,} paths per scenario, bankroll ${a.bankroll:,.0f}, quarter Kelly with a {cfg.max_bet_pct:.0%} per-bet cap (policy `staking`). The kill switch trips at a {kill:.0%} drawdown "
      f"from the high-water mark (policy `risk.kill_switch`); ruin = the bankroll ever at or below {ruin:.0%} of its start.", "",
      "Scenarios: *model true* (the model's probabilities are the real ones), *shrunk halfway* (the real probability is midway between the model's and the market's no-vig one), "
      "*market true* (the model has no skill: the only thing that happens to the bankroll is the bookmaker's margin). The live evidence points at the last one "
      "(`reports/evaluation.md`), so the third row is the one to plan around. This answers \"what could happen if the policy were followed\", not \"what will happen\". Research and education only.", "",
      res[cols].round(3).to_markdown(index=False), ""]
Path(a.out).mkdir(exist_ok=True)
Path(a.out, "risk.md").write_text("\n".join(md) + "\n")
res.to_csv(Path(a.out) / "risk.csv", index=False)
print("\n".join(md))
