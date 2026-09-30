# Picks page (`site/index.html`)

Each daily run writes a self-contained page: `site/index.html` (data embedded, no server, no API key, works from disk) and
`site/picks.json`. Download the **picks-site** artifact from the workflow run, or `git pull` and open `site/index.html`.
(Kept private by design: it contains bookmaker prices from The Odds API, whose terms restrict redistribution.)

## What it does
- **Your units.** Set what 1 unit is worth, your bankroll, the staking style (quarter Kelly, half Kelly, or flat 1 unit), the per-bet
  cap and the daily cap. Stakes are recomputed instantly in your browser and shown in dollars and units. Settings are remembered in
  that browser only.
- **Top 5.** Up to five games with a positive expected value, recommended bets first. Each is tagged **RECOMMENDED** (clears every
  rule, has a stake) or **LEAN - not recommended** (shown for information, with the exact reason it failed). If nothing qualifies the
  page says so; five slots are never filled with bets the rules reject.
- **Parlay calculator.** Tick legs on the pick cards. Only legs with a positive edge, one per game (no same-game parlays), up to 4 legs.
  It shows the combined win probability, fair odds, the odds offered by the best *single* book that carries every leg, the parlay's EV,
  and the EV of placing the same legs as separate bets. A parlay stake (capped at 0.5 units / 0.5% of bankroll) is only suggested when
  every leg is a recommended bet and the EV is positive. Parlays have higher variance: the page shows how rarely they win.
- **All games**, **track record** (ROI, CLV, model vs market log loss) and the disclaimer on every view.

## Guarantees (tested)
- The stakes and selections the browser computes equal the Python policy's for 80 random slates (`tests/test_site.py`, run under Node),
  so the page and the daily report cannot disagree.
- Data is embedded so it cannot break out of its `<script>` block; the UI only writes text nodes (no `innerHTML`).
- Loaded in headless Chromium during development: no JavaScript errors; unit changes update stakes; parlay legs compute.
