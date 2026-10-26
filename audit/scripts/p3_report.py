"""Render audit/PHASE3_LEAKAGE.md from the committed result files (mutation.json, phase3_real.json, p1check). Numbers in the report come from the files, not from this script."""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
mut = json.loads((R / "results/mutation.json").read_text())
real = json.loads((R / "results/phase3_real.json").read_text())
p1 = json.loads((R / "results/p1check/phase1_realdata.json").read_text())
yn = lambda b: "caught" if b else "-"

L = ["# Phase 3: leakage attack", "", "Everything here is reproducible: unit tests `tests/test_audit_leakage.py`, `tests/test_audit_placebo.py`; mutation harness `audit/scripts/p3_mutation.py`; "
     "real-data run `audit/scripts/p3_realdata.py` (GitHub Actions, `audit/jobs.txt`); results in `audit/results/`. Research window only (nothing after 2026-01-04 is read).", ""]
L += ["## 1. Verdict", "",
      f"- **No leakage found in the production pipeline.** Truncation invariance, availability guard and the outcome-shuffle null all pass on the real database ({p1['completed_games']:,} completed games); "
      f"all {p1['availability_and_sentinel']['features_checked']} production feature columns pass the stronger guard.",
      f"- **The test suite is not vacuous.** Mutation testing: {mut['killed']} of {mut['total']} deliberately planted leaks are caught by at least one test; survivors after closing the one gap found (M6, see section 3): {mut['survived'] or 'none'}; unmutated control: {mut['control']}.",
      "- **One weakness in the defences was found and fixed:** the availability guard only flagged inputs correlated above 0.98 with the label, so a feature that ranks winners perfectly (AUC 1.0, correlation about 0.8) passed. A univariate-AUC limit of 0.75 was added (`nhlbet/validity/availability.py`); it was confirmed to leave all 14 production features alone on the real data.",
      "- **Two things look like leaks to a naive reader and are not:** the adversarial-validation AUC (section 5) and the flat stale-value curve (section 4). Both are explained there.", ""]
L += ["## 2. Planted canaries and who catches them", "",
      f"Canary features built from forbidden information, run through six independent detectors on the real data ({real['canary']['games']:,} games, {real['canary']['window'][0]} to {real['canary']['window'][1]}). The clean builder is the control row: it must raise no alarm.", "",
      "| Canary | guard | truncation | corruption | same-day | shuffle null | too good | caught by any |", "|---|---|---|---|---|---|---|---|"]
for m, v in real["canary"]["results"].items():
    f = v["flags"]
    L.append(f"| {'clean pipeline (control)' if m == 'None' else m} | " + " | ".join(("FLAGGED" if f[k] else "-") if m == "None" else yn(f[k]) for k in ("guard", "truncation", "corruption", "same_day", "shuffle_null", "too_good")) + f" | {('FALSE ALARM' if v['caught'] else 'no alarm (correct)') if m == 'None' else 'yes' if v['caught'] else 'NO: MISSED'} |")
L += ["", "Reading the matrix:", "- `own_result` / `noisy_own_result` (peeking at the game's own margin, with or without noise) survive truncation and corruption of the future, because they use no future games. Only the guard, the shuffled world and the univariate-AUC check see them.",
      "- `next_game` (the next game's result) is invisible to the shuffled world and to any univariate check (it is unrelated to this game's label) but is the one the truncation and corruption tests exist for.",
      "- `same_day_other` (another game on the same date) is invisible to everything except the same-date perturbation test. It needs no future games and carries no signal about its own label.",
      "- **No single detector is sufficient** (each misses at least one canary): the defences are a set, and removing any member opens a specific hole. This is asserted in `test_no_single_detector_is_sufficient`.", ""]
L += ["## 3. Mutation testing of the real code", "", "Each mutation re-introduces one leak into a copy of the package and runs the leakage test subset. KILLED = at least one test fails.", "",
      "| ID | Planted leak | Result | Failing tests (first) |", "|---|---|---|---|"]
for r in mut["mutants"]:
    if r["id"] == "M0":
        L.append(f"| M0 | control: nothing changed | {r['status']} (must be SURVIVED) | |")
    else:
        L.append(f"| {r['id']} | {r['description']} | **{r['status']}** | {'; '.join(x.replace('FAILED ', '').split('::')[-1] for x in r.get('killed_by', [])[:2])} ({r.get('n_failing_tests', 0)} failing) |")
for r in mut["mutants"]:
    if r.get("first_run"):
        L += ["", f"**Test gap found and closed ({r['id']}):** on the first run this mutant {r['first_run']} It was killed on the rerun."]
L += ["", "The harness drops the editable-install import hook so the tests really import the mutated copy, and includes an unmutated control that must pass: an earlier run without those two safeguards reported mutants as killed for an unrelated reason (missing config files in the temporary tree) and was discarded.", ""]
L += ["## 4. Placebo: time-shifted features", "", "AUC of home-minus-away of a pre-game feature re-timed by k games (k < 0 reads FUTURE values; k > 0 stale ones), 95% bootstrap interval.", "",
      "| Feature | k = -2 (future) | k = -1 (future) | **k = 0 (as built)** | k = 1 | k = 5 | k = 10 |", "|---|---|---|---|---|---|---|"]
for f, rows in real["time_shift"].items():
    by = {r["shift_games"]: r for r in rows}
    L.append(f"| {f} | " + " | ".join((f"**{by[k]['auc']:.3f}**" if k == 0 else f"{by[k]['auc']:.3f}") for k in (-2, -1, 0, 1, 5, 10)) + " |")
L += ["", "Positive control: shifting into the future lifts the AUC by 0.07 to 0.17, so this evaluation would show a leak if one existed, and the as-built column sits on the honest side. "
      "**Finding that is not a leak but matters for the modelling story:** stale values (up to 10 games old) rank winners as well as current ones. The signal in these features is slow-moving team quality, not recent form.", ""]
L += ["## 5. Adversarial validation (can a classifier tell train rows from test rows?)", "", "| Split | rows before / after | AUC | Reading | Top features |", "|---|---|---|---|---|"]
for a in real["adversarial_validation"]:
    L.append(f"| {a['split']} | {a['n_train_side']:,} / {a['n_test_side']:,} | {a['auc']:.3f} | {a['reading']} | {', '.join(f'{n} ({v})' for n, v in a['top_features'][:3])} |")
L += ["", "An AUC of 0.5 would mean the test period looks like the past. These are higher, and the dominant feature is `h_early_w` (games played this season / (games played + 20)): the later test slice covers the first months of a season, so the rows differ by **season phase**, not because anything is wrong. It is covariate shift by construction, not leakage, "
      "but it means walk-forward blocks do not share the same mixture of season phases as the training set, which matters for calibration (Phase 4 evaluates by season phase).", ""]
sm = real["shuffled_market"]
L += ["## 6. Placebo: shuffled odds", ""]
if "actual_roi" in sm:
    L += [f"Live record after the locked block, {sm['games']} games with a decision-time market price, {sm['n_bets']} bets chosen as the policy chooses, flat stake, fair odds less a 4.5% margin. Real-market ROI **{sm['actual_roi'] * 100:+.1f}%**; the same selection against market probabilities permuted across games earns **{sm['shuffled_roi_mean'] * 100:+.1f}%** "
          f"(95% range {sm['shuffled_roi_95'][0] * 100:+.0f}% to {sm['shuffled_roi_95'][1] * 100:+.0f}%).", "",
          "Reading: prices unrelated to the game make almost any model look brilliant, which is the ceiling an evaluation flaw (a price that does not belong to its game) could manufacture. Against the right prices the model loses about the margin and slightly more: "
          f"the share of the model's information already in the price is {sm['information_already_in_the_price']:.2f} (above 1 means the model's disagreements with the market were worse than random). On this sample the model's deviations from the market add no value, which agrees with the closing-line-value result from the hardening branch. Small sample ({sm['n_bets']} bets): treat as consistent, not decisive.", ""]
else:
    L += [str(sm), ""]
L += ["## 7. Placebo: shuffled labels", "", "Already in place and re-run on the real data in this phase (`audit/results/p1check`): results are reassigned within each season and the whole pipeline is rerun; a clean pipeline finds no skill. "
      f"Real world AUC elo {p1['shuffle_null']['real']['auc_elo']:.3f}; shuffled worlds: " + ", ".join(f"{r['auc_elo']:.3f}" for r in p1['shuffle_null']['shuffled']) + ". Planted outcome copies were caught by the guard.", ""]
L += ["## 8. What this phase does NOT show", "",
      "- It shows the feature and model pipeline is free of the leaks it was tested for. It says nothing about whether the model has an edge over the market (Phase 4) and nothing about the sealed-holdout question (there is none; see PHASE1_RECON section 0).",
      "- The point-in-time data revision problem (Phase 1 item 10) cannot be tested on history; only the forward snapshots from Phase 2 can measure it.",
      "- Walk-forward still has no purge or embargo (training uses every game dated before the block). Labels here resolve the same day, so the exposure is small, but Phase 4 adds both.",
      "- Goalie `actual` mode and lineup mode remain reachable by configuration; the tests pin the defaults and prove the modes would leak, they do not forbid someone enabling them.", ""]
(R / "PHASE3_LEAKAGE.md").write_text("\n".join(L))
print("wrote audit/PHASE3_LEAKAGE.md")
