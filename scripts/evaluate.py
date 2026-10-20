"""Honest evaluation of every pick -> reports/evaluation.md and reports/evaluation.json. One command, reproducible from the database and the committed logs:

    python scripts/evaluate.py [--db data/nhl.db]

Primary no-vig method and every threshold come from config/policy.yaml. Every figure is computed here; nothing is copied from another report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nhlbet.data.store import Store  # noqa: E402
from nhlbet.policy import policy_get, policy_hash  # noqa: E402
from nhlbet.report import evaluation as ev  # noqa: E402
from nhlbet.report.validation import required_n  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/nhl.db")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
store = Store(a.db)
primary = policy_get("evaluation", "devig", "primary", default="proportional")
others = list(policy_get("evaluation", "devig", "sensitivity", default=["power", "shin"]))
mn, mnc = int(policy_get("evaluation", "min_picks_per_market", default=100)), int(policy_get("evaluation", "min_clv_picks", default=30))
d = ev.pick_frame(store)
if d.empty:
    sys.exit("no settled paper picks in the database: restore the cache first")
off = ev.official_picks(store)
res: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "policy_sha256": policy_hash(), "primary_devig": primary, "sensitivity_devig": others,
             "min_picks_per_market": mn, "min_clv_picks": mnc, "settled_model_picks": int(((d.group == "model") & d.settled).sum()), "data_through": str(d.date.max())}
try:
    res["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    res["commit"] = None
tables = {"model": ev.market_table(d, primary, "model"), "control": ev.market_table(d, primary, "control")}
if len(off):
    tables["official"] = ev.market_table(off, primary, "official")
res["markets"] = {k: v.to_dict("records") for k, v in tables.items()}
res["splits"] = ev.split_cells(d, primary, "model").to_dict("records")
res["edge_thresholds_all_model_picks"] = ev.edge_threshold_table(d, primary).to_dict("records")
res["edge_thresholds_moneyline"] = ev.edge_threshold_table(d, primary, market="h2h").to_dict("records")
res["method_sensitivity"] = ev.method_sensitivity(d).to_dict("records")
res["baselines"] = ev.baselines(store, d, primary)
res["game_scoring"] = {m: ev.score_games(store, m) for m in [primary] + others}
# how many more bets would the OBSERVED closing-line value need to be detected at 95% confidence (one-sided, 80% power)?
need = {}
for mk, x in d[(d.group == "model") & d.settled].groupby("market"):
    c = x[f"clv_pts_{primary}"].dropna() / 100.0
    if len(c) >= 5:
        mean, sd = float(c.mean()), float(c.std(ddof=1))
        n_needed = required_n(mean, sd) if mean > 0 and sd > 0 else None
        need[mk] = {"clv_n": int(len(c)), "mean_clv_prob": mean, "sd": sd, "bets_needed": n_needed, "more_bets_needed": (max(0, n_needed - len(c)) if n_needed else None),
                    "calculation": ("n = ((z_0.95 + z_0.80) * sd / mean)^2 = ((1.645 + 0.842) * %.4f / %.4f)^2 = %s" % (sd, mean, n_needed)) if n_needed else "observed CLV is not positive: no sample size detects an effect of zero or less"}
res["clv_bets_needed"] = need
Path(a.out).mkdir(exist_ok=True)
Path(a.out, "evaluation.json").write_text(json.dumps(res, indent=1, default=float))


def fmt(t: pd.DataFrame, cols: list[str], pct: list[str] = (), digits: int = 3) -> str:
    t = t[cols].copy()
    for c in pct:
        t[c] = (t[c] * 100).round(1).astype(str) + "%"
    return t.round(digits).to_markdown(index=False)


md = ["# Evaluation (decision-time prices; closing-line value is the primary evidence)", "",
      f"Generated {res['generated_at']} at commit `{res['commit']}`, policy sha256 `{res['policy_sha256'][:12]}`, data through {res['data_through']}. Reproduce with `python scripts/evaluate.py`.", "",
      f"Primary no-vig method: **{primary}**; sensitivity: {', '.join(others)}. Production edges use Shin. Intervals resample whole game dates ({policy_get('evaluation', 'bootstrap_resamples')} resamples). "
      f"Significance is one-sided against zero and corrected with **{policy_get('evaluation', 'multiple_comparisons')}** across markets, and across all split cells with at least {mnc} bets. "
      f"Below {mn} bets a market's ROI is labelled insufficient evidence; below {mnc} CLV observations, likewise. Research and education only; nothing here is betting advice.", ""]
for name, t in tables.items():
    if t.empty:
        continue
    md += [f"## Per market: {name} picks", "",
           fmt(t, ["market", "picks", "pushes_or_void", "win_rate", "roi", "roi_lo", "roi_hi", "p_roi_adj", "verdict_roi"], ["win_rate", "roi", "roi_lo", "roi_hi"]), "",
           "CLV = margin-free move of the closing no-vig probability toward the pick (probability points; log-odds in the json). Player props have no closing capture.", "",
           fmt(t, ["market", "clv_n", "clv_pts", "clv_lo", "clv_hi", "clv_logodds", "p_clv_adj", "verdict_clv", "roi_at_closing_price"], [], 4), ""]
md += ["## Baselines (moneyline, totals and puck-line picks, same games)", "", "```json", json.dumps(res["baselines"], indent=1, default=float), "```", "",
       "## Model against the no-vig market on the same finished games", ""]
for m, s in res["game_scoring"].items():
    if s.get("n", 0) >= 20:
        md += [f"**{m}** (n={s['n']}): log loss model {s['model']['log_loss']:.4f}, market at decision {s['market_at_decision']['log_loss']:.4f}, market close {s['market_close']['log_loss']:.4f}; "
               f"Brier {s['model']['brier']:.4f} / {s['market_at_decision']['brier']:.4f} / {s['market_close']['brier']:.4f}. Model's log-loss gain over the closing market: "
               f"{s['model_vs_market_close_log_loss_gain']['mean_gain']:+.4f} (95% interval {s['model_vs_market_close_log_loss_gain']['ci'][0]:+.4f} to {s['model_vs_market_close_log_loss_gain']['ci'][1]:+.4f}).", ""]
        if m == primary:
            md += ["Reliability by model-probability quintile:", "", pd.DataFrame(s["reliability"]).round(3).to_markdown(index=False), ""]
md += ["## Edge-threshold sensitivity (all model picks, then moneyline only)", "", fmt(pd.DataFrame(res["edge_thresholds_all_model_picks"]), ["min_edge", "picks", "roi", "roi_lo", "roi_hi", "clv_n", "clv_pts"], ["roi", "roi_lo", "roi_hi"], 4), "",
       fmt(pd.DataFrame(res["edge_thresholds_moneyline"]), ["min_edge", "picks", "roi", "roi_lo", "roi_hi", "clv_n", "clv_pts"], ["roi", "roi_lo", "roi_hi"], 4), "",
       "## CLV under each no-vig method (is the result robust to the method?)", "", pd.DataFrame(res["method_sensitivity"]).round(4).to_markdown(index=False), "",
       "## Splits (one correction family over every cell with enough bets)", "", fmt(pd.DataFrame(res["splits"]), ["split", "cell", "n", "roi", "roi_lo", "roi_hi", "p_roi_adj", "clv_n", "clv_pts", "p_clv_adj", "label"], ["roi", "roi_lo", "roi_hi"], 3), "",
       "## How many bets would the observed CLV need?", ""]
for mk, v in need.items():
    md.append(f"- **{mk}**: {v['clv_n']} CLV observations, mean {v['mean_clv_prob'] * 100:+.2f} probability points; {v['calculation']}" + (f"; about {v['more_bets_needed']:,} more bets" if v.get("more_bets_needed") is not None else ""))
Path(a.out, "evaluation.md").write_text("\n".join(md) + "\n")
print("\n".join(md)[:6000])
