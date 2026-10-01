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

## Price in your app
Each pick (and each totals / puck-line row) has a "Price in your app" box. Type the price your own sportsbook app shows (American or decimal) and the page
recalculates expected value, and for moneyline picks the suggested stake, at that price; it also shows the minimum price at which the bet is worth taking.
Nothing per-book is published (the odds feed's terms and the public repo rule that out), so the page cannot know which app you use. It works from the model
probability and the price you type, and remembers your entries in this browser.

## Scheduling (why there are so many cron lines)
GitHub's scheduled triggers are best-effort and have arrived 3-6 hours late on this repository (a "late" run fired after puck drop; closing-line snapshots meant
for 22:45 UTC fired at 01:35). So the daily and closing-line workflows are triggered often and `scripts/gate.py` (standard library only, runs before anything is
installed) decides whether *now* is a useful moment: one morning run per game day; one late run 20 minutes to 3 hours before the first unstarted game; a closing
snapshot only when a game starts within 35 minutes and no live capture happened in the last 20. Triggers that arrive too late exit immediately and spend no API
credits. Manual dispatches always run.
