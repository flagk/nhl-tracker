/* Pure, DOM-free logic for the picks page. Mirrors nhlbet/risk/policy.py + parlay.py and is parity-tested against them
 * with Node (tests/test_site.py). Stakes are recomputed in the browser from the user's unit/bankroll/risk settings. */
(function (root) {
  "use strict";

  function kelly(p, d) {
    var b = d - 1;
    return Math.max(0, (b * p - (1 - p)) / b);
  }
  function round2(x) { return Math.round(x * 100) / 100; }
  function floor2(x) { return Math.floor(x * 100 + 1e-9) / 100; }

  /* settings: {bankroll, unit, mode: 'quarter'|'half'|'flat', maxPct, dailyPct, maxBets, minStake} */
  function kellyMultiplier(mode) { return mode === "half" ? 0.5 : 0.25; }

  function rawStake(side, s) {
    var pct;
    if (s.mode === "flat") pct = Math.min(s.unit / s.bankroll, s.maxPct);
    else pct = Math.min(kellyMultiplier(s.mode) * side.kelly_full, s.maxPct);
    return round2(s.bankroll * pct);
  }

  /* games: payload.games. Returns {gameId: {side, stake, units, scaled}} for qualifying games, after portfolio caps. */
  function allocate(games, s) {
    var cands = [];
    games.forEach(function (g) {
      var best = null;
      ["home", "away"].forEach(function (k) {
        var sd = g.sides[k];
        if (sd && sd.qualifies && (best === null || sd.ev > best.sd.ev)) best = { k: k, sd: sd };
      });
      if (!best) return;
      var st = rawStake(best.sd, s);
      if (st < s.minStake) return;
      cands.push({ id: g.game_id, side: best.k, ev: best.sd.ev, stake: st, scaled: 1 });
    });
    cands.sort(function (a, b) { return b.ev - a.ev; });
    cands = cands.slice(0, s.maxBets);
    var total = cands.reduce(function (t, c) { return t + c.stake; }, 0);
    var cap = s.bankroll * s.dailyPct;
    if (total > cap && cap > 0) {
      var scale = cap / total;
      cands.forEach(function (c) { c.stake = floor2(c.stake * scale); c.scaled = scale; });
      cands = cands.filter(function (c) { return c.stake >= s.minStake; });
    }
    var out = {};
    cands.forEach(function (c) { out[c.id] = { side: c.side, stake: c.stake, units: s.unit > 0 ? c.stake / s.unit : 0, scaled: c.scaled }; });
    return out;
  }

  function prodOf(a) { return a.reduce(function (t, x) { return t * x; }, 1); }

  /* legs: [{game_id, p, best_decimal, books: {book: decimal}}] */
  function parlay(legs, maxLegs) {
    maxLegs = maxLegs || 4;
    if (legs.length < 2) throw new Error("a parlay needs at least 2 legs");
    if (legs.length > maxLegs) throw new Error("at most " + maxLegs + " legs");
    var ids = legs.map(function (l) { return l.game_id; });
    if (new Set(ids).size !== ids.length) throw new Error("legs must come from different games");
    var p = prodOf(legs.map(function (l) { return l.p; }));
    var common = Object.keys(legs[0].books).filter(function (b) { return legs.every(function (l) { return b in l.books; }); }).sort();
    var offered = 0, book = null;
    common.forEach(function (b) {
      var o = prodOf(legs.map(function (l) { return l.books[b]; }));
      if (o > offered) { offered = o; book = b; }
    });
    if (book === null) offered = prodOf(legs.map(function (l) { return l.best_decimal; }));
    var singles = legs.reduce(function (t, l) { return t + l.p * l.best_decimal - 1; }, 0) / legs.length;
    var ev = p * offered - 1;
    return { legs: legs.length, p: p, fair_decimal: 1 / p, offered_decimal: offered, book: book, single_book_available: book !== null,
             ev: ev, singles_ev: singles, better_as_singles: singles >= ev };
  }

  /* ---- "price in MY app": everything below depends only on the model probability and the price the user types, never on other books ---- */
  function toDecimal(odds, format) {
    var x = Number(odds);
    if (!isFinite(x)) return null;
    if (format === "american") return (x === 0 || (x > -100 && x < 100)) ? null : (x > 0 ? 1 + x / 100 : 1 + 100 / -x);
    return x > 1 ? x : null;
  }
  /* pWin / pPush: probability the bet wins / pushes (stake refunded). EV per $1 staked. */
  function evAt(pWin, pPush, d) { return pWin * (d - 1) - (1 - pWin - (pPush || 0)); }
  /* the lowest decimal price at which the bet has non-negative expected value */
  function minDecimal(pWin, pPush) { return pWin > 0 ? (1 - (pPush || 0)) / pWin : Infinity; }
  /* stake for a moneyline pick at the user's own price, same sizing rules as rawStake (Kelly multiplier, per-bet cap, or flat unit) */
  function stakeAt(pAdj, d, s) { return rawStake({ kelly_full: kelly(pAdj, d) }, s); }

  var api = { kelly: kelly, allocate: allocate, rawStake: rawStake, parlay: parlay, toDecimal: toDecimal, evAt: evAt, minDecimal: minDecimal, stakeAt: stakeAt };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NHLCore = api;
})(typeof window !== "undefined" ? window : this);
