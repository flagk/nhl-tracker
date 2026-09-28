# Bet sizing and risk management

`nhlbet/risk/` - configuration in `RiskConfig` (all defaults are conservative and configurable).

| Rule | Default | Where |
|---|---|---|
| Stake = fractional Kelly of the **shrunk** probability at the **best available price** | quarter Kelly | `kelly.py`, `policy.py` |
| Max per bet | 2% of bankroll | `max_bet_pct` |
| Max total exposure per day (stakes scaled down proportionally, rounded down) | 5% | `max_daily_exposure_pct` |
| Max bets per day (highest EV kept) | 5 | `max_bets_per_day` |
| Minimum **raw** edge (model prob - no-vig market prob) | 3% (4% if the starting goalie is unconfirmed) | `min_edge` |
| Minimum EV after shrinkage | 1% per $1 | `min_ev` |
| Minimum model win probability of the chosen side (no long-shot noise) / max price | 40% / 4.0 | `min_prob`, `max_decimal` |
| Model-vs-market gap treated as *model error*, not edge | > 12 points | `max_disagreement` |
| Refuse to bet on stale odds; suspend all bets when the drift monitor says ALERT | on | `allow_stale_odds`, `model_status` |
| Minimum stake | $1 | `min_stake` |

**"No bet" is a first-class outcome** with a plain-language reason (`Recommendation.explain()`); nothing forces a pick.

## Shrinking toward the market
`p_adj = p_market + w(d) * (p_model - p_market)`, `w(d) = w0 / (1 + (|d|/d0)^2)`, `w0 = 0.5`, `d0 = 6` points.
Trust in the model falls as it disagrees more with the line - the adjusted edge peaks near a 6-point gap and then *falls*
(tested), because a 15-point gap is far more likely a stale feature or missing injury than a real edge. Screens use the raw
edge; **sizing** uses the shrunk probability. `w0` is baseline humility, not an estimate: once a few hundred games have
model probability + market price + result, `risk.shrink.estimate_trust` fits the blend that maximises out-of-sample
likelihood and should replace it.

## Bankroll tracking and Monte Carlo
`bankroll.py`: equity curve, drawdown, longest underwater streak, ROI/win rate from the bet log.
`montecarlo.py` / `scripts/risk_sim.py`: simulates paths under an explicit assumption about how much of the claimed edge
is real (`skill` 0 to 1). Illustrative result (3% claimed edge at ~1.95 odds, 1% stakes, 500 bets, 5,000 paths):

| Real edge share | Median final of $1,000 | Chance of finishing down | Chance of a >=30% drawdown | Chance bankroll ever <= 50% |
|---|---|---|---|---|
| 0% (no edge, vig only) | $784 | 87% | 55% | 2.5% |
| 50% | $916 | 66% | 30% | 0.6% |
| 100% | $1,052 | 40% | 14% | 0.0% |

Even a *real* 3% edge leaves a 2-in-5 chance of being down after 500 bets, and a coin-flip chance of a 20% drawdown. This is
why stakes are small and why CLV, not short-run profit, is the metric to trust.
