# Settlement rules

How a bet is graded decides every ROI and every probability the model is judged against, so the rules live in one place: `config/policy.yaml` (`settlement:`), read through
`nhlbet/settle.py`. The model's goal distribution (`models/goals.py`), the paper-bet settlement (`report/betlog.py`) and the exports all use the same switch, so they cannot disagree.

## Rule sets

| Rule | `so_adds_goal` (policy default) | `so_excluded` (what the repo assumed before Phase 1) |
|---|---|---|
| Moneyline | final result including overtime and shootout | same |
| Puck line | official margin: an OT or shootout winner wins by exactly one goal | same |
| Game total | regulation + overtime goals **+ one goal if the game goes to a shootout** (the official score already includes it) | shootout goal excluded |
| Whole-number line landing exactly | push: stake refunded | same |
| Quarter lines | stake split across the two adjacent lines | same |
| Player props | box-score totals: overtime counts, shootout never does; a player who did not play voids the bet | same |
| Team totals, both teams to score | regulation + overtime only (the shootout goal is never counted; some books offer a 60-minute-only variant) | same |
| Postponed or cancelled game, or a game moved to another date than the pick's | void, stake refunded | same |

## What is verified and what is not

The books' own rules pages could not be read from the build environment (the domains are blocked), so nothing below is verified against a primary source; every book in the policy
file carries `verified: false`, and the Phase 7 health gate surfaces that flag.

Evidence used (search-result snippets, not full pages):
- Playat Gila house rules (via search snippet): "the winner of the shootout will have one (1) goal added to its score and one goal will be added to the game total".
- bet365 rules (via the same search): the same treatment, except for markets that exclude overtime/shootout.
- FanDuel guide (via search): overtime goals count toward the game total; FanDuel's team totals are regulation-only (reported by Action Network).
- DraftKings House Rules (Massachusetts filing, 2023): a total exactly equal to the line is void; DraftKings offers team totals both with and without overtime and shootout.
- Industry explainers (Action Network, Sigma): puck-line and moneyline settle on the official result including overtime and shootout, so a +1.5 underdog wins as soon as a game goes past regulation.

Why the default changed: the repo excluded the shootout goal from totals. Two independent house-rule snippets say the opposite is the common convention, and the prompt that started this
hardening says the same. Under `so_excluded` a shootout game (about one game in eight to ten) is graded one goal lower than most books would grade it, and the model's totals distribution
was fitted to that lower number. Treat the totals market as RESEARCH ONLY until each book's rule is confirmed by a person who can read its rules page. The sensitivity of the paper-bet record to
the rule is measured in `reports/phase1_realdata.md` (`settlement_sensitivity`).

## What must be done by hand
For each book you place bets at, read its NHL rules page for: shootout goal in game totals, team totals basis, goalie-prop treatment when the starter is pulled, voiding of props for players who do not dress.
Set `verified: true` and the right `ruleset` for that book in `config/policy.yaml` (only you edit that file).
