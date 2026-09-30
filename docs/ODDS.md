# Live odds integration

## Setup
1. Create a free key at <https://the-odds-api.com> (free tier: 500 credits/month).
2. Locally: `export ODDS_API_KEY=...` (or put it in an untracked `.env`). In GitHub: *Settings -> Secrets and variables ->
   Actions -> New repository secret* named `ODDS_API_KEY`. The key is read **only** from the environment; it is never
   written to the cache, logs, exceptions or the repo (tested in `tests/test_odds.py`).
3. `python scripts/fetch_odds.py` stores a timestamped snapshot; `--markets h2h spreads totals` also stores puck line/totals.

## Quota plan
One call costs `markets x regions` credits. Moneyline-only, one region = **1 credit/call** -> ~16 calls/day on the free
tier. Suggested schedule: a morning call (opening reference), one mid-day, one ~2 h before the first puck drop (decision),
and 1-2 calls around game time (closing line for CLV). Puck line/totals triple the cost; the model bets moneylines only.

Behaviour on trouble (all covered by tests): responses are cached for 10 min; retries with back-off on network errors/5xx/429;
a low-quota reserve stops calls before credits run out; on failure the last snapshot is served **flagged as stale**
(<= 6 h old) and the daily report says so; with nothing usable the answer is "no odds -> no bet".

## What is stored (`odds_snapshots`, append-only)
`captured_at, event_id, book, market, outcome, point, price (decimal), book_updated, game_id`. Because nothing is
overwritten, the *opening* line is the first snapshot and the *closing* line is the last one before puck drop.

## Math (`nhlbet/odds/math.py`)
- implied probability = 1/decimal (includes vig); overround = sum of implied - 1.
- vig removal: **Shin** (default; corrects favourite-longshot bias) or proportional; consensus market probability =
  mean of per-book no-vig probabilities. Books whose own line is >90 min older than the snapshot are dropped so a
  stale price cannot masquerade as the "best price".
- best price per side across books; **edge = model prob - no-vig market prob**; **EV per $1 = p * (dec - 1) - (1 - p)** at the best price.
- **CLV** = bet decimal x closing no-vig probability - 1 (positive = beat the close). See `nhlbet/odds/clv.py`.

## Market features
`nhlbet/features/market.py` may use the opening line and movement up to *decision time* (default 90 min before puck drop),
never the closing line; tests assert later snapshots are ignored. Market features are **not** in the default model.

## Limits
- The Odds API's historical endpoint is a paid feature and history before this project started collecting is not available,
  so backtest ROI cannot use real lines until enough snapshots accumulate. Until then ROI is reported as scenarios only.
- Not verified against the live API from the dev sandbox (network policy); the request/response handling follows the public v4 docs
  and is exercised with faithful fakes. First live run: `python scripts/fetch_odds.py` and inspect `odds_fetch_log`.
