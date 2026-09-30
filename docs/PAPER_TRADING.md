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
