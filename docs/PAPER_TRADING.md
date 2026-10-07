# Paper trading ("shadow bettors")

**Fake money, real games.** On every slate, besides the live recommendations, `nhlbet/risk/shadow.py` runs several alternative
strategies with pretend stakes. They are logged before the games (`data/logs/shadow_bets/`), settled against results and closing
lines automatically, and shown in the daily report and on the picks page.

## What it is for - and what it is not
- **Not for training.** The model learns from game *results*, and it already retrains on every finished game whether or not a bet was
  placed. Feeding it its own picks would be circular (it would learn from its own past choices).
- **For measurement.** The live policy is deliberately selective, so it produces few bets and takes months to say anything. Five
  strategies on the same games produce far more observations of profit and closing-line value, so design questions get data answers.

| Strategy | Question it answers |
|---|---|
| `market_favorite` (**control**) | What does "no skill" cost? Flat bets on the market favourite lose roughly the bookmaker's margin (~4%). Everything is judged against this. |
| `flat_model_side` | Does betting every game where the model sees any positive edge make money, or only the selective bets? |
| `every_game` | Flat 1% on the model's preferred side of EVERY game with fresh odds, even at a negative edge. Today's pretend stakes are listed in the daily report ("Today's fake bets"). Expect it to lose about the bookmaker margin unless the model beats the market; that is the point of measuring it. |
| `edge_1pct` | Is the 3% minimum edge too strict (or too loose)? |
| `no_shrink` | Is shrinking the model toward the market helping or hurting? |
| `no_guard` | Does the early-season guard (no bets until each team has 10 games) actually protect us? |

## How to read it
Judge by **CLV** and by beating the control, not by early ROI: a 40-bet ROI is mostly luck (the 95% interval is shown). When a strategy
has a few hundred settled bets and a clearly positive CLV vs the control, that is evidence to change a threshold in `RiskConfig`;
until then the live policy stays as is. Changing thresholds because one paper strategy had a lucky month would be exactly the
overfitting this project tries to avoid.

## Extending
Add a `Strategy(...)` to `STRATEGIES` (a `RiskConfig` override, or a new `kind`). Tests in `tests/test_shadow.py` include a property test that the
control loses about the margin on a fair market. Stakes are always pretend; nothing here places a bet.

## Totals and puck-line strategies
`totals_edge`, `puckline_edge`, `every_total`, `every_puckline` and the control `always_over` use the goals model (see [MARKETS.md](MARKETS.md)).

**Bet-type variety.** To learn which kinds of bet (if any) the model is good at, there are also strategies that each bet on one specific kind, at a 2%+ edge:
`under_edge`, `over_edge`, `puckline_dog_edge` (+1.5 side), `puckline_fav_edge` (-1.5 side), `underdog_ml` (moneyline underdogs) and, for player shots,
`sog_over_edge` and `sog_under_edge`. No-skill controls (flat $10) give each a baseline: `always_under`, `puckline_dog_control`, `underdog_ml_control`, `home_ml_control`.
They settle on the score (totals exclude the shootout goal, the puck line uses the official margin; pushes refund the stake), and have no CLV because
closing snapshots only fetch moneylines.


## Stake sizes ($5 to $30)
Pretend stakes scale with the model's conviction so the data can answer "do the bets it is surest about do better?":
- edge-based strategies: a 3-point edge is about $12, a 10-point edge or more is $30;
- "bet every game" strategies: 50% model probability is $5, 75% or more is $30;
- the live-policy copies (`no_guard`, `edge_1pct`, `no_shrink`) keep their Kelly-based sizing;
- the no-skill controls (`market_favorite`, `always_over`, `always_under`, `puckline_dog_control`, `underdog_ml_control`, `home_ml_control`) always stake $10, so their ROI is a clean baseline.
Compare strategies on ROI (profit per dollar staked), not total profit, because stake sizes differ.

## Player-prop strategies
`sog_edge` and the control `sog_over_control` (see [MARKETS.md](MARKETS.md)) bet on players' shots on goal. A player can have several bets per game, so these live in their own table (`prop_bets`) and settle from the player's actual shots; a skater who did not play voids the bet (stake refunded, not counted).
