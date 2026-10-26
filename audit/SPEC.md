# Specification: decision layer of the NHL moneyline pipeline

This document is written so that someone who has never seen the code can reimplement the parts of the pipeline below. Where it is silent, say so in your notes rather than guessing silently. All probabilities are in [0, 1]; odds are decimal odds (a 2.00 price returns 2 for 1 staked).

Scope: (A) removing the bookmaker margin, (B) the Elo rating model, (C) the bet/no-bet decision and stake for one game, (D) the portfolio caps over a day's slate. The model that produces the home-win probability for a game is out of scope: it is an input.

## A. De-vig (remove the margin from a set of decimal odds)

Let q_i = 1 / odds_i be the implied probabilities and s = sum(q_i) the booksum. If s <= 1 + 1e-12 return q_i / s for every method.

* proportional: p_i = q_i / s.
* power: find the exponent c >= 1 such that sum(q_i ^ c) = 1; p_i = q_i ^ c (renormalise to sum 1 at the end).
* shin: find z in [0, 0.999] such that sum_i p_i(z) = 1, where p_i(z) = ( sqrt(z^2 + 4 (1 - z) q_i^2 / s) - z ) / ( 2 (1 - z) ); p_i = p_i(z) (renormalise to sum 1 at the end). The sum is decreasing in z; use bisection to a width of 1e-12.

Two-way markets (home, away) are the use case; the methods work for any number of outcomes.

## B. Elo rating model

State: a rating per team, default 1500 for a team not yet seen. Parameters: K = 6, home advantage H = 35, regression R = 0.30, start 1500.

* Expected home win probability from ratings rh (home) and ra (away): 1 / (1 + 10 ^ ( -(rh + H - ra) / 400 )).
* Update after a game with final goals hg (home) and ag (away), given the PRE-game ratings rh and ra used for that game: exp = expected home win probability; margin = |hg - ag|; won = hg > ag; diff = (rh + H - ra) if won else -(rh + H - ra); mult = ln(margin + 1) * (2.2 / (diff * 0.001 + 2.2)); delta = K * mult * ((1 if won else 0) - exp); home rating becomes rh + delta; away rating becomes ra - delta. (No game has equal scores.)
* New season: every stored rating becomes (1 - R) * rating + R * 1500.
* Timing rule: games are processed one calendar date at a time, in date order. When the season number changes from the previous date's season, the new-season regression is applied first, before anything else on the new date. Then a snapshot of all ratings is taken. The pre-game win probability of EVERY game on that date is computed from that snapshot, and only after all of that date's probabilities are computed are that date's results folded in, each update using the snapshot ratings of its two teams. A team plays at most once per date.

Output wanted: for every game in input order, the pre-game home win probability.

## C. The decision for one game

Inputs: model probability that the home team wins (p_home); a list of bookmaker quotes (home odds, away odds per book); context: model_status ("OK", "WARN" or "ALERT"), odds_stale (bool), goalie_confirmed (bool), games_played_min (the smaller number of games played this season by the two teams, or None); bankroll; configuration below.

Configuration (defaults): kelly_fraction 0.25, max_bet_pct 0.02, min_edge 0.03, min_edge_unconfirmed_goalie 0.04, min_ev 0.01, min_prob 0.40, max_decimal 4.0, max_disagreement 0.12, trust_w0 0.5, trust_d0 0.06, min_stake 1.0, min_games_played 10, allow_stale_odds false.

Step 1, game-level blocks, checked in this order, any of which means NO_BET:
1. model_status == "ALERT".
2. games_played_min is not None and games_played_min < min_games_played.
3. no usable quotes. A quote is usable if both odds are present and both > 1. If none is usable there are no quotes.
4. odds_stale is true and allow_stale_odds is false.

Step 2, market probability and best prices. For each usable book compute the de-vigged home probability with the Shin method on [home odds, away odds] and take element 0. The market home probability is the arithmetic mean of these over the usable books; the market away probability is 1 minus it. The best price for each side is the maximum odds for that side over the usable books.

Step 3, for each side (home, away): model probability p_model = p_home (home) or 1 - p_home (away); market probability p_mkt; edge = p_model - p_mkt; best odds d. Screens, each producing a failure:
* edge < need, where need = min_edge if goalie_confirmed else min_edge_unconfirmed_goalie;
* p_model < min_prob;
* d > max_decimal;
* |edge| > max_disagreement.
Shrunk probability: disagreement x = p_model - p_mkt; weight w = trust_w0 / (1 + (|x| / trust_d0)^2); p_adj = clip(p_mkt + w * x, 0.01, 0.99). Expected value ev = p_adj * d - 1. If the side has no failure so far and ev < min_ev, that is a failure.

Step 4: sides without any failure pass. If none passes: NO_BET. Otherwise take the passing side with the larger ev (home wins ties by being listed first). Full Kelly fraction f = max(0, ((d - 1) * p_adj - (1 - p_adj)) / (d - 1)) using the SHRUNK probability. pct = min(kelly_fraction * f, max_bet_pct). Stake = round(bankroll * pct, 2) (round to two decimals, ordinary rounding of the float). If the stake < min_stake: NO_BET. Otherwise BET with that side, odds d and stake; the bet's ev (used for ranking in D) is the ev above.

## D. Portfolio caps over a slate

Input: the BET decisions of a slate in input order, each with game id, stake, ev; bankroll; max_bet_pct 0.02; max_game_pct 0.03; max_daily_pct 0.05; max_bets 5; min_stake 1.0. down(x) = floor(x * 100 + 1e-9) / 100.
1. Per-bet cap: if stake > bankroll * max_bet_pct, stake = down(bankroll * max_bet_pct).
2. Keep at most max_bets bets: rank by ev descending (ties keep input order); the rest are dropped.
3. Per game (bets sharing a game id): total = sum of stakes; if total > bankroll * max_game_pct > 0, every stake in that game becomes down(stake * cap / total) with cap = bankroll * max_game_pct.
4. Daily: total over all kept bets; if total > bankroll * max_daily_pct > 0, every stake becomes down(stake * (bankroll * max_daily_pct) / total).
5. Any kept bet with stake < min_stake after the above is dropped.
Dropped bets become NO_BET with stake 0.

## Interface to implement (Python 3.11, numpy allowed, nothing else)

```
devig(decimals: list[float], method: str) -> list[float]
elo_probs(games: list[tuple], k=6.0, home_adv=35.0, regress=0.30) -> list[float]
    # games: (season: int, date: str 'YYYY-MM-DD', home: str, away: str, home_goals: int, away_goals: int), sorted by date; returns pre-game home win probability per game
decide(p_home: float, quotes: list[tuple[float, float]], ctx: dict, bankroll: float, cfg: dict | None = None) -> dict
    # quotes: (home_odds, away_odds) per book; returns {"action": "BET"|"NO_BET", "side": "home"|"away"|None, "stake": float, "odds": float|None}
decide_slate(games: list[dict], bankroll: float, cfg: dict | None = None) -> list[dict]
    # games: {"game_id": int, "p_home": float, "quotes": [...], "ctx": {...}}; applies decide to each and then the portfolio caps; same output format per game, in input order
```
