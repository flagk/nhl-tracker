"""Render audit/PHASE5_OVERFITTING_RISK.md from audit/results/phase5_real.json and audit/data/trial_register.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
r = json.loads((R / "results/phase5_real.json").read_text())
reg = list(csv.DictReader((R / "data/trial_register.csv").open()))
pct = lambda x: f"{x * 100:.0f}%"
L = ["# Phase 5: overfitting and risk accounting", "",
     "Source: `audit/scripts/p5_trials.py` (trial register), `audit/scripts/p5_overfit.py` run in GitHub Actions on the real database; statistics in `nhlbet/stats/overfit.py` and `nhlbet/risk/uncertainty.py`, both unit-tested with positive and negative controls (noise gives PBO near 0.5 and no reality-check rejection; a planted edge is found). Live record only, decision-time prices only.", ""]
M, P, D1, D2, RC = r["matrix"], r["pbo"], r["deflated_sharpe_matrix_trials"], r["deflated_sharpe_register_trials"], r["reality_check"]
L += ["## 1. Verdict", "",
      f"- **There is no evidence that any strategy ever tried has an edge once the number of tries is accounted for.** White's reality check p = {RC['p_white_reality_check']:.2f} and Hansen's SPA p = {RC['p_hansen_spa']:.2f} across {M['strategies']} paper strategies: the best one is what the best of {M['strategies']} strategies with no edge would look like.",
      f"- The best in-sample strategy is **{r['best_strategy_in_sample']['name']}**, a control that bets the home team in every game. Strategies with a positive mean: {', '.join(r['strategies_with_positive_mean'])} (mostly controls and prop variants). Deflated for {M['strategies']} trials its deflated Sharpe ratio is {D1['dsr']:.3f}, and for the {r['trial_register_total']} trials in the register {D2['dsr']:.1e}: nowhere near the 0.95 that would count as evidence.",
      f"- Probability of backtest overfitting: **{P['pbo']:.2f}** ({P['variants']} variants, {P['periods']} days, {P['combinations']} train/test splits). Above 0.5 means the strategy that looked best in one half tended to rank below the median in the other half. It is computed from only {M['days']} days, so it is a weak statement, but it points the wrong way for any claim of edge.",
      "- **Bankroll simulation:** if the policy were followed with stakes sized on the model's claims, the result depends almost entirely on how much of the model's disagreement with the market is real (theta), and the live record cannot pin that down (section 4). Under no skill the bankroll bleeds the margin and the 20% drawdown kill switch trips; under the estimate from the live record it is worse.", ""]
L += ["## 2. Trial register: every variant that could have shaped a result", "", "| Family | Variants | Source |", "|---|---|---|"]
for x in reg:
    L.append(f"| {x['family']} | {x['variants']} | {x['source']} |")
L += ["", "The total is a **lower bound**: manual experiments that left no trace in the repository are not counted. It is the number of trials used in the deflated Sharpe ratio. Grid points, feature candidates and settings are not independent bets, so the true effective number of trials is smaller than the sum, but the deflated Sharpe ratio is far below 0.95 for any count above a few dozen.", ""]
L += ["## 3. Overfitting statistics on the paper-strategy returns", "",
      f"Matrix: {M['strategies']} strategies by {M['days']} days (daily return per unit staked; days without a bet count as zero); {M['dropped_for_fewer_than_10_betting_days']} strategies dropped for fewer than 10 betting days.", "",
      "| Statistic | Value | Reading |", "|---|---|---|",
      f"| Probability of backtest overfitting (CSCV, 12 blocks) | {P['pbo']:.2f} | above 0.5: the in-sample winner tends to disappoint |",
      f"| Best strategy's Sharpe (per day) | {D1['sharpe_per_period']:.2f} | skew {D1['skew']:.1f}, kurtosis {D1['kurtosis']:.1f} over {D1['periods']} days |",
      f"| Probabilistic Sharpe vs zero | {D1['psr_vs_zero']:.3f} | what a naive reading would claim |",
      f"| Deflated Sharpe, {D1['n_trials']} strategies | {D1['dsr']:.4f} | expected maximum Sharpe under no skill: {D1['benchmark_sharpe_from_trials']:.2f} |",
      f"| Deflated Sharpe, {D2['n_trials']} register trials | {D2['dsr']:.1e} | expected maximum Sharpe under no skill: {D2['benchmark_sharpe_from_trials']:.2f} |",
      f"| White's reality check p | {RC['p_white_reality_check']:.2f} | H0: no strategy has a positive expected return |",
      f"| Hansen's SPA p | {RC['p_hansen_spa']:.2f} | studentised, consistent centring |", "",
      f"The headline Sharpe of the best strategy looks passable on its own (probabilistic Sharpe {D1['psr_vs_zero']:.2f}); deflation for the number of tries removes all of it. {M['days']} days is a short window, and these results will be rerun as the forward record grows.", ""]
L += ["## 4. Bankroll simulation under parameter uncertainty", "",
      "True win probability of a bet: p_true = p_market + theta x (p_model - p_market), where theta is how much of the model's disagreement with the market is real. theta is estimated by a date-clustered regression on the live games with a market price; each simulated path draws its own theta from the estimate's sampling distribution. "
      "Bets come from the paper record at the logged decision-time best price, with a 2% price haircut (the best price across books is not always obtainable at your own book; 0% is in the JSON). Stakes: Kelly on the model's claim x Kelly fraction, capped at the policy's 2% per bet; 10,000 paths; $1,000 start; ruin = at or below $500; the policy kill switch trips at a 20% drawdown.", ""]
for tname, t in r["templates"].items():
    if "mean_decimal" not in t:
        L += [f"### {tname}", "", str(t), ""]
        continue
    th = r["theta"][tname]
    small = [f"**Caution: only {t['bets']} settled bets, resampled with replacement, so these paths are indicative of stake size and drawdown depth, not of outcomes. Theta here is measured on the shrunk scale (the policy shrinks claims toward the market, so p_adj - p_market is small and theta is large in magnitude and very uncertain); it is not comparable with the other template's theta.**", ""] if t["bets"] < 30 else []
    L += [f"### Template: {tname}", "", *small,
          f"{t['bets']} bets, mean decimal {t['mean_decimal']:.2f}, mean claimed edge {t['mean_claimed_edge_pts']:.1f} points, {pct(t['share_with_positive_claimed_edge'])} with a positive claimed edge. **theta = {th['theta']:.2f}, standard error {th['se']:.2f}, 95% interval {th['ci95'][0]:.2f} to {th['ci95'][1]:.2f}** over {th['n']} games: the interval contains 0 (no skill) and 1 (exactly right).", "",
          "| theta scenario | Kelly | bets | median final | 5th to 95th pct | P(end below start) | P(ruin) | P(drawdown at least 20%) | median / 95th pct max drawdown |", "|---|---|---|---|---|---|---|---|---|"]
    for sm in r["simulations"]:
        if sm["template"] == tname and sm["price_haircut"] == 0.02:
            L.append(f"| {sm['theta_scenario']} | {sm['kelly_fraction']} | {sm['horizon']} | ${sm['median_final']:,.0f} | ${sm['p05_final']:,.0f} to ${sm['p95_final']:,.0f} | {pct(sm['p_loss'])} | {pct(sm['p_ruin'])} | {pct(sm['p_drawdown_20'])} | {pct(sm['median_max_drawdown'])} / {pct(sm['p95_max_drawdown'])} |")
    L.append("")
L += ["Reading:", "- The aggressive template (any model pick, unshrunk claims) overstates what the actual policy would do: the policy stakes on shrunk probabilities with a 3% edge floor, and in fact has placed no bets in the live record (at a 1% floor it would have placed 1, and with no shrinkage none). The policy-like template (the policy's own rules with only the early-season guard removed; the guard blocks nearly everything in the first weeks of a season) is the relevant one; compare it with `reports/risk.md` from the hardening branch, which used a different, smaller template and reported a 0.7% kill-switch probability under no skill.",
      "- With no skill every template loses roughly the margin; whether the 20% drawdown kill switch trips depends on stake size, which is why the two templates differ so much.",
      "- The \"estimated\" scenario is worse than no skill when the point estimate of theta is negative (the model's disagreements with the market lost money on average), with a standard error so large that the interval spans everything from harmful to perfect.",
      "- Only the optimistic prior, which the live record neither supports nor excludes, produces profitable paths.",
      "- The policy-like template shows the practical consequence of the policy's caution: because stakes are sized on claims shrunk toward the market, drawdowns stay shallow even under a negative theta; the 20% kill switch is a remote event at quarter Kelly and the risk that matters is the slow loss of the margin, not ruin.",
      "- Quarter Kelly (the policy) has shallower drawdowns than half Kelly in every row; nothing here argues for more aggressive staking.", "",
      "## 5. Limits of this phase", "",
      f"- The strategy matrix covers {M['days']} days of paper trading (since 2026-09-30). PBO and the reality check need far more periods to mean much; they are reported because the instructions require them and because their direction is consistent with every other test, not because they are conclusive.",
      "- theta is estimated on all games with a price, but bets are chosen where the model disagrees most; if skill differs between large and small disagreements, the simulation is off in either direction.",
      "- Prices are the best across books at the decision-time capture. Per-book obtainability is not recorded (raw quotes live only in the cache, and the haircut stands in for it). The official policy has placed no bets, so there are no fills to check prices against.",
      "- The simulation resamples bets independently; correlated same-night exposure is limited by the policy's caps but not modelled here."]
(R / "PHASE5_OVERFITTING_RISK.md").write_text("\n".join(L))
print("wrote")
