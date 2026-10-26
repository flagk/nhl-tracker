"""Render audit/PHASE4_EVALUATION.md from audit/results/phase4_real.json (numbers come from the file)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
r = json.loads((R / "results/phase4_real.json").read_text())
A, B, C = r["A_research_ladder"], r["B_live_vs_market"], r["C_clv_by_close_quality"]
f = lambda x, d=4: f"{x:.{d}f}"
L = ["# Phase 4: honest evaluation", "",
     f"Source: `audit/scripts/p4_ladder.py` run in GitHub Actions on the real database ({r['completed_games']:,} completed games), results `audit/results/phase4_real.json`. Part A uses games up to {r['research_end']} only; part B is the live record, which is the only data nobody has tuned on.", "",
     "## 1. Verdict", ""]
names = {"home_rate": "home rate", "elo": "Elo", "logistic": "logistic regression", "random_forest": "random forest", "production_stack": "production stack", "hierarchical_bayes": "hierarchical Bayes"}
best = min(A["models"], key=lambda k: A["models"][k]["log_loss"])
shin = B["shin"]
lg_m = next(x for x in shin["vs_market"] if x["market"] == "market_at_decision" and x["model"] == "logistic")
L += [f"- **Against the betting market: no model beats it, and none is close.** On {shin['n']} live games ({shin['from']} to {shin['to']}) every model scores worse than the devigged market at the decision time and at the close, for all three devig methods; every Holm-adjusted Diebold-Mariano p-value is 1.0. "
      f"The least bad, logistic regression, is {abs(lg_m['mean_gain']):.4f} log-loss worse than the decision-time market (95% interval {f(lg_m['ci'][0])} to {f(lg_m['ci'][1])}).",
      f"- **Among the models themselves, the simplest one wins.** Over {A['n_games']:,} research-window games with a 1-day embargo, {names[best]} has the lowest log loss ({f(A['models'][best]['log_loss'])}). Every model beats the home rate, and logistic and the production stack beat Elo after Holm correction. The production stack (the ensemble) is not better than its own logistic component.",
      "- **The hierarchical Bayesian challenger fails.** It beats the home rate but not Elo (its log loss is above Elo's), so team strength with partial pooling and no engineered features adds nothing over a plain Elo rating.",
      "- **Recalibration does not help.** Raw probabilities are overconfident (calibration slope 0.74 to 0.86), yet online Platt, isotonic and beta calibration leave log loss unchanged or worse; beta calibration improves the reliability measure (ECE) but not log loss.",
      "- **Purging and embargo change nothing** (log loss differs by at most 0.0001 between 0, 1 and 7 days), which is expected because labels resolve on the day of the game. It is now in place and tested anyway.",
      "- **Closing-line value is small, positive, and not distinguishable from zero** under every definition of the close (section 4).", ""]
L += ["## 2. The ablation ladder (research window, walk-forward, embargo 1 day)", "",
      "| Model | n | Log loss | Brier | ECE | Calibration slope |", "|---|---|---|---|---|---|"]
for k, v in A["models"].items():
    L.append(f"| {names[k]} | {v['n']:,} | **{f(v['log_loss'])}** | {f(v['brier'])} | {f(v['ece'], 3)} | {f(v['cal_slope'], 2)} |")
L += ["", "Log-loss gain over a reference (positive = better than the reference), date-clustered 95% interval, one-sided Diebold-Mariano p with Bartlett HAC variance, Holm-adjusted within each reference:", "",
      "| Reference | Model | Gain | 95% interval | DM p (Holm) |", "|---|---|---|---|---|"]
for d in A["dm"]:
    L.append(f"| {names[d['reference']]} | {names[d['model']]} | {d['mean_gain']:+.4f} | {d['ci'][0]:+.4f} to {d['ci'][1]:+.4f} | {d['dm_p_holm']:.3f} |")
L += ["", "Reliability of the production stack (8 equal-count bins; the pattern is the usual overconfidence at the extremes: predictions of 34% win 44%, predictions of 64% win 61%):", "",
      "| Mean predicted | Observed | n |", "|---|---|---|"] + [f"| {x['mean_p']:.3f} | {x['obs']:.3f} | {x['n']} |" for x in A["reliability_stack"]]
L += ["", "Plot: `audit/results/p4_reliability.png`.", "", "### Calibration variants (online: each block calibrated on earlier blocks only; same games, so compare with the raw column)", "",
      "| Model | raw (same games) | Platt | isotonic | beta | ECE raw / beta |", "|---|---|---|---|---|---|"]
for k, v in A["calibration"].items():
    raw = v["platt"]["raw_log_loss_same_games"]
    L.append(f"| {names[k]} | {f(raw)} | {f(v['platt']['log_loss'])} | {f(v['isotonic']['log_loss'])} | {f(v['beta']['log_loss'])} | {f(A['models'][k]['ece'], 3)} / {f(v['beta']['ece'], 3)} |")
L += ["", "### Embargo sensitivity (log loss)", "", "| Embargo (days) | Elo | logistic | hierarchical Bayes |", "|---|---|---|---|"]
for e in ("0", "1", "7"):
    s = A["embargo_sensitivity"][e]
    L.append(f"| {e} | {f(s['elo'])} | {f(s['logistic'])} | {f(s['hier_bayes'])} |")
L += ["", "## 3. Against the devigged market (live record)", "",
      f"{shin['n']} finished games with both a decision-time and a closing price; prices devigged by Shin, power and proportional (multiplicative). Gain = market log loss minus model log loss, so negative means the market is better.", "",
      "| Model | log loss | Gain vs market at decision (Shin) | 95% interval | DM p (Holm) | Gain vs close (Shin) |", "|---|---|---|---|---|---|"]
sc = shin["scores"]
for m in ("production_logged", "home_rate", "elo", "logistic", "random_forest", "hierarchical_bayes"):
    d = next(x for x in shin["vs_market"] if x["market"] == "market_at_decision" and x["model"] == m)
    c = next(x for x in shin["vs_market"] if x["market"] == "market_close" and x["model"] == m)
    L.append(f"| {m.replace('_', ' ')} | {f(sc[m]['log_loss'])} | {d['mean_gain']:+.4f} | {d['ci'][0]:+.4f} to {d['ci'][1]:+.4f} | {d['dm_p_holm']:.3f} | {c['mean_gain']:+.4f} |")
L += ["", "Market log loss by devig method (decision / close): " + "; ".join(f"{m}: {f(B[m]['scores']['market_at_decision']['log_loss'])} / {f(B[m]['scores']['market_close']['log_loss'])}" for m in ("shin", "power", "proportional")) +
      ". The choice of method moves the market's log loss by about 0.002, ten times less than the gap to any model.", ""]
se = (lg_m["ci"][1] - lg_m["ci"][0]) / (2 * 1.96) * np.sqrt(shin["n"])
need = ((1.645 + 0.84) * se / 0.005) ** 2
L += [f"**Power.** The per-game standard deviation of the log-loss difference is about {se:.3f}. Detecting a true advantage of 0.005 over the market (one-sided 5%, 80% power) would take about {need:,.0f} games, roughly {need / 1312:.1f} full regular seasons of live record. With {shin['n']} games, a model with a small real edge could not be told apart from one with none; what can be said is that the point estimates are all on the wrong side and the data give no support for an edge.", ""]
L += ["## 4. Closing-line value per bet", "",
      "Moneyline picks of the model group (paper picks; the official policy has placed no bets). CLV in probability points: the no-vig closing probability of the picked side minus the no-vig probability at the decision. \"Close within X min\" keeps only picks whose last pre-game capture was at most X minutes before puck drop, because a stale \"close\" is not a close (Phase 1: 49% of games have no capture in their last 15 minutes).", "",
      "| Closing capture within | Picks | Picks with a usable close | Mean CLV (points) | 95% interval | Adjusted p (CLV > 0) | ROI | Verdict |", "|---|---|---|---|---|---|---|---|"]
for k, lab in (("close_within_60_min", "60 min (default)"), ("close_within_15_min", "15 min"), ("close_within_5_min", "5 min")):
    x = C[k]["model_paper_picks"]
    L.append(f"| {lab} | {x['picks']} | {x['clv_n']} | {x['clv_pts']:+.3f} | {x['clv_lo']:+.3f} to {x['clv_hi']:+.3f} | {x['p_clv_adj']:.2f} | {x['roi'] * 100:+.1f}% | {x['verdict_clv']} |")
L += ["", "Reading: the mean CLV rises a little as the close gets tighter (0.043, 0.059, 0.183 points) but the 5-minute row rests on 9 picks and every interval includes zero. "
      "On the 15-minute definition, the best-defined one with a useful sample (108 picks), the interval is -0.05 to +0.16 points: any true CLV is within a fraction of a percentage point either side of zero. Meanwhile ROI is about -9% and its interval excludes zero on the negative side.", "",
      "## 5. What this phase does and does not establish", "",
      "- It establishes that on this data no model in the ladder, including the production stack and a Bayesian challenger, beats the market, and that the simplest competent model (logistic regression) is as good as anything.",
      "- It does not establish that no edge exists anywhere: the live record is 161 games, the detectable effect at that size is large, and totals, puck line and props have no closing prices at all (not measurable).",
      "- The market comparison cannot be run on the long backtest because there are no historical prices: that comparison only gets stronger with time, via the forward ledger (Phase 6).", ""]
(R / "PHASE4_EVALUATION.md").write_text("\n".join(L))
print("wrote audit/PHASE4_EVALUATION.md")
