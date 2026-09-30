/* Pure bet-tracking maths shared by the browser page and the Node tests. No DOM here. */
(function (root) {
  "use strict";
  function toDecimal(odds, format) {
    var x = Number(odds);
    if (!isFinite(x)) return null;
    if (format === "american") {
      if (x === 0 || (x > -100 && x < 100)) return null;
      return x > 0 ? 1 + x / 100 : 1 + 100 / -x;
    }
    return x > 1 ? x : null;
  }
  function settle(bet) {                      // bet: {stake, decimal, result: pending|won|lost|push}
    var s = Number(bet.stake), d = Number(bet.decimal);
    if (bet.result === "won") return s * (d - 1);
    if (bet.result === "lost") return -s;
    if (bet.result === "push") return 0;
    return null;
  }
  function summarize(bets) {
    var r = { n: bets.length, pending: 0, settled: 0, won: 0, lost: 0, push: 0, staked: 0, profit: 0, roi: null, hit: null, open_stake: 0 };
    bets.forEach(function (b) {
      var p = settle(b);
      if (p === null) { r.pending++; r.open_stake += Number(b.stake); return; }
      r.settled++; r[b.result]++; r.profit += p;
      if (b.result !== "push") r.staked += Number(b.stake);
    });
    if (r.staked > 0) r.roi = r.profit / r.staked;
    if (r.won + r.lost > 0) r.hit = r.won / (r.won + r.lost);
    return r;
  }
  function cumulative(bets) {                 // settled bets in date order -> [{date, profit, cum}]
    var out = [], cum = 0;
    bets.filter(function (b) { return settle(b) !== null; })
        .sort(function (a, b) { return a.date < b.date ? -1 : a.date > b.date ? 1 : 0; })
        .forEach(function (b) { cum += settle(b); out.push({ date: b.date, profit: settle(b), cum: cum }); });
    return out;
  }
  var api = { toDecimal: toDecimal, settle: settle, summarize: summarize, cumulative: cumulative };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BetsCore = api;
})(typeof window !== "undefined" ? window : this);
