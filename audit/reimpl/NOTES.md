1. Power de-vig: spec gives no bracket/tolerance; bisection on c in [1, doubling upper bound], 1e-12 width, then renormalise.
2. Shin: if sum(p(z)) at the midpoint > 1 move z up (sum is decreasing); final z = bracket midpoint; renormalise.
3. The s <= 1+1e-12 early return applies to all methods including unknown method names; unknown method otherwise raises ValueError.
4. Elo: first date has no previous season, so no regression is applied there. Season change is detected from the first game of each date. Unseen teams use 1500 (also inside snapshot).
5. Elo ratings of teams not playing on a date are unchanged; regression applies to all stored teams.
6. decide: "usable quote" with None odds is skipped; stale check (block 4) comes after quote check; order only matters for nothing observable (all give NO_BET).
7. decide returns an extra key "ev" (None for NO_BET) so decide_slate can rank; NO_BET has side None, odds None, stake 0.0.
8. cfg overrides merge over defaults; cfg may also carry max_game_pct, max_daily_pct, max_bets for the slate (spec lists them only for D; max_bet_pct is shared with C).
9. Tie between sides on ev: home wins (strict > needed to switch to away).
10. Slate: per-bet cap uses down(bankroll*max_bet_pct) applied after decide (which already caps via pct, so it rarely triggers). Ranking uses stable sort on -ev. Game-cap scaling is applied only to kept bets, daily total computed after game scaling. Dropped bets get side None, odds None, stake 0.0. Surviving bets keep side/odds/ev; no game_id key is added to outputs.
11. ctx missing keys treated as falsy (goalie_confirmed missing => unconfirmed threshold; odds_stale missing => not stale); missing games_played_min => None.
12. Spec says "round" ordinary: Python round() used on the float (banker's rounding only on exact binary ties).
