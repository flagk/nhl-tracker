# Phase 1 checks on the real database

Run 2026-10-26T02:17:28+00:00; 10,694 completed games; features: d_elo, d_xg_share, d_cf_pct, h_b2b, d_xga_pg, h_early_w, d_g_sv_l5, h_stakes, d_pen_drawn_pg, d_games_last_7d, d_ga_season_shrunk, a_b2b, h_homestand_n, d_g_sv_season.

**Truncation invariance** (data cut at 2025-03-01): 959 games compared; max difference in any model, the stack or any calibrator 0.00e+00; weighted average 0.00e+00 (solver noise). **PASS**

**Availability guard**: 14 features clean = True; planted outcome copies caught: {'outcome_copy': True, 'outcome_complement': True, 'noisy_outcome': True}. **PASS**

**Outcome-shuffle null** (results reassigned within each season; a clean pipeline finds no skill):

- real world: 2,803 out-of-sample games, AUC elo 0.584, logistic 0.598
- shuffled seed 1: AUC elo 0.497, logistic 0.481, log-loss gain over the base rate -0.0017, ROI against a fair market with a 4.5% margin -4.2%
- shuffled seed 2: AUC elo 0.496, logistic 0.494, log-loss gain over the base rate -0.0016, ROI against a fair market with a 4.5% margin -4.9%
- shuffled seed 3: AUC elo 0.497, logistic 0.486, log-loss gain over the base rate -0.0015, ROI against a fair market with a 4.5% margin -3.4%
- **PASS**

**Settlement sensitivity** (does the shootout rule matter for totals paper bets?): {'totals_bets_so_adds_goal': 634, 'totals_bets_so_excluded': 634, 'bets_whose_result_changes': 20, 'bets_that_become_pushes_or_leave': 0, 'profit_so_adds_goal': -1020.5, 'profit_so_excluded': -794.5999999999999}
