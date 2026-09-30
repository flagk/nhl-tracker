(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent);
  var C = window.NHLCore;
  var app = document.getElementById("app");
  var P = D.policy;

  /* ---------- settings (remembered per browser; storage may be unavailable) ---------- */
  var defaults = { bankroll: P.bankroll, unit: Math.max(1, Math.round(P.bankroll / 100)), mode: "quarter", maxPct: P.max_bet_pct * 100, dailyPct: P.max_daily_exposure_pct * 100 };
  var S = Object.assign({}, defaults);
  try { Object.assign(S, JSON.parse(localStorage.getItem("nhlSettings") || "{}")); } catch (e) { /* ignore */ }
  var legs = {};   // game_id -> side

  function save() { try { localStorage.setItem("nhlSettings", JSON.stringify(S)); } catch (e) { /* ignore */ } }
  function num(x, d) { var v = parseFloat(x); return isFinite(v) && v >= 0 ? v : d; }
  function cfg() {
    return { bankroll: Math.max(1, num(S.bankroll, defaults.bankroll)), unit: Math.max(0.01, num(S.unit, defaults.unit)), mode: S.mode,
             maxPct: Math.min(num(S.maxPct, 2), 25) / 100, dailyPct: Math.min(num(S.dailyPct, 5), 100) / 100, maxBets: P.max_bets_per_day, minStake: P.min_stake };
  }

  /* ---------- helpers ---------- */
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "class") n.className = attrs[k]; else if (k === "text") n.textContent = attrs[k];
      else if (k.slice(0, 2) === "on") n.addEventListener(k.slice(2), attrs[k]); else n.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function pct(x, d) { return x === null || x === undefined ? "–" : (x * 100).toFixed(d === undefined ? 1 : d) + "%"; }
  function spct(x) { return x === null || x === undefined ? "–" : (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%"; }
  function usd(x) { return "$" + x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
  function american(d) { return d >= 2 ? "+" + Math.round((d - 1) * 100) : String(Math.round(-100 / (d - 1))); }
  function price(d) { return d.toFixed(2) + " (" + american(d) + ")"; }
  function when(iso) {
    if (!iso) return "";
    var t = new Date(iso); return isNaN(t) ? "" : t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  function bestSide(g) {
    var b = null;
    ["home", "away"].forEach(function (k) { var s = g.sides[k]; if (s && (b === null || s.ev > g.sides[b].ev)) b = k; });
    return b;
  }
  function metric(k, v) { return el("div", { class: "metric" }, [el("div", { class: "v", text: v }), el("div", { class: "k", text: k })]); }

  /* ---------- sections ---------- */
  function header() {
    var st = D.model.status, cls = st === "OK" ? "ok" : st === "ALERT" ? "alert" : "warn";
    var kids = [
      el("h1", { text: "NHL picks · " + D.date }),
      el("div", { class: "sub" }, ["Model ", el("b", { text: D.model.version || "?" }), " · health ", el("span", { class: "pill " + cls, text: st }),
        " · run: " + D.run_type + " · built " + new Date(D.generated_at).toLocaleString()]),
      el("div", { class: "banner disc", text: D.disclaimer })
    ];
    if (st === "ALERT") kids.push(el("div", { class: "banner bad", text: "Model health is ALERT (" + (D.model.reasons.join("; ") || "performance drift") + "). Recommendations are suspended." }));
    else if (st === "WARN") kids.push(el("div", { class: "banner info", text: "Model health WARN: " + (D.model.reasons.join("; ") || "see report") + ". Treat picks with extra scepticism." }));
    if (D.odds && D.odds.stale) kids.push(el("div", { class: "banner bad", text: "Odds are STALE (the odds API was unreachable). No bets are recommended on stale prices." }));
    if (D.odds && D.odds.enabled === false) kids.push(el("div", { class: "banner info", text: "Odds are disabled (no ODDS_API_KEY), so every game is 'no odds → no bet'." }));
    (D.notes || []).forEach(function (n) { kids.push(el("div", { class: "banner info", text: n })); });
    return el("div", {}, kids);
  }

  function field(label, input) { return el("div", {}, [el("label", { text: label }), input]); }
  function settingsPanel() {
    function inp(key, step) {
      return el("input", { type: "number", min: "0", step: step || "any", value: String(S[key]), oninput: function (e) { S[key] = e.target.value; save(); render(true); } });
    }
    var mode = el("select", { onchange: function (e) { S.mode = e.target.value; save(); render(true); } }, [
      el("option", { value: "quarter", text: "Quarter Kelly (default)" }), el("option", { value: "half", text: "Half Kelly (more aggressive)" }),
      el("option", { value: "flat", text: "Flat: 1 unit" })]);
    mode.value = S.mode;
    return el("div", { class: "card", id: "settings" }, [
      el("div", { class: "row" }, [el("b", { text: "Your betting units" }), el("span", { class: "muted", text: "saved in this browser only" })]),
      el("div", { class: "grid", style: "margin-top:8px" }, [
        field("1 unit = $", inp("unit")), field("Bankroll $", inp("bankroll")), field("Staking", mode),
        field("Max per bet (% bankroll)", inp("maxPct")), field("Max per day (% bankroll)", inp("dailyPct"))]),
      el("div", { class: "sub", style: "margin-top:8px", text: "A bet only appears if the model's edge over the market's no-vig price is at least " + pct(P.min_edge, 0) +
        " after safety checks; stakes shrink the model toward the market and are capped. Zero picks is a normal day." })
    ]);
  }

  function pickCard(g, alloc) {
    var k = bestSide(g), s = g.sides[k], a = alloc[g.game_id];
    var rec = !!a && a.side === k;
    var head = el("div", { class: "row" }, [
      el("div", {}, [el("span", { class: "big", text: s.team + " " }), el("span", { class: "muted", text: (k === "home" ? "vs " + g.away : "@ " + g.home) + (g.start_utc ? " · " + when(g.start_utc) : "") })]),
      el("span", { class: "pill " + (rec ? "rec" : "lean"), text: rec ? "RECOMMENDED" : "LEAN – not recommended" })]);
    var m = [metric("Model win", pct(s.p_model)), metric("Market (no-vig)", pct(s.p_market)), metric("Edge", spct(s.edge)),
             metric("EV per $1", spct(s.ev)), metric("Best price", price(s.best_decimal) + " @ " + s.best_book)];
    if (rec) m.push(metric("Stake", usd(a.stake) + " · " + a.units.toFixed(1) + " u"));
    var why = rec ? "Clears every check." + (a.scaled < 1 ? " Stake scaled down to respect your daily cap." : "") : (s.fails.length ? "Why not: " + s.fails.join("; ") + "." : "Below the recommendation rules.");
    var goalies = g.goalie_status === "unknown" ? "" : "Goalies: " + g.away_goalie + " (away) / " + g.home_goalie + " (home), " + g.goalie_status + ". ";
    var box = [head, el("div", { class: "metrics" }, m), el("div", { class: "sub", text: goalies + why })];
    if (g.notes && g.notes.length) box.push(el("ul", { class: "sub" }, g.notes.map(function (n) { return el("li", { text: n }); })));
    if (s.edge > 0) {
      var cb = el("input", { type: "checkbox", onchange: function (e) { if (e.target.checked) legs[g.game_id] = k; else delete legs[g.game_id]; renderParlay(); } });
      cb.checked = legs[g.game_id] === k;
      box.push(el("label", { class: "check" }, [cb, "Add to parlay calculator"]));
    }
    return el("div", { class: "card" }, box);
  }

  function topPicks(alloc) {
    var ranked = D.games.filter(function (g) { var k = bestSide(g); return k && g.sides[k].ev > 0; })
      .sort(function (a, b) { return (isRec(b, alloc) - isRec(a, alloc)) || (g_ev(b) - g_ev(a)); }).slice(0, 5);
    var nrec = Object.keys(alloc).length;
    var kids = [el("h2", { text: "Top picks" }),
      el("div", { class: "sub", text: nrec ? nrec + " recommended bet(s) today; other entries are leans shown for information only." : "No recommended bets today. Below are the best leans by expected value, for information only." })];
    if (!D.games.length) kids.push(el("div", { class: "card", text: "No regular-season games to show for this date." }));
    else if (!ranked.length) kids.push(el("div", { class: "card", text: "No game has a positive expected value against the current odds." }));
    ranked.forEach(function (g) { kids.push(pickCard(g, alloc)); });
    return el("div", {}, kids);
  }
  function g_ev(g) { return g.sides[bestSide(g)].ev; }
  function isRec(g, alloc) { var a = alloc[g.game_id]; return a && a.side === bestSide(g) ? 1 : 0; }

  var parlayBox = el("div", { id: "parlay" });
  function renderParlay() {
    var sel = Object.keys(legs).map(function (id) {
      var g = D.games.filter(function (x) { return String(x.game_id) === String(id); })[0]; var s = g.sides[legs[id]];
      return { g: g, s: s, leg: { game_id: g.game_id, p: s.p_adj, best_decimal: s.best_decimal, books: s.books } };
    });
    var c = cfg(), kids = [el("h2", { text: "Parlay calculator" }),
      el("div", { class: "sub", text: "Parlays multiply every leg's bookmaker margin, so they are usually worse than betting the same picks separately. Only legs with a positive edge can be added, each from a different game (no same-game parlays), up to " + P.parlay_max_legs + " legs. Tick 'Add to parlay calculator' on a pick above." })];
    if (sel.length < 2) { kids.push(el("div", { class: "card", text: sel.length ? "Add at least one more leg." : "No legs selected." })); }
    else if (sel.length > P.parlay_max_legs) { kids.push(el("div", { class: "banner bad", text: "Too many legs (max " + P.parlay_max_legs + ")." })); }
    else {
      var r = C.parlay(sel.map(function (x) { return x.leg; }), P.parlay_max_legs);
      var allRec = sel.every(function (x) { return x.s.qualifies; });
      var stake = Math.min(0.5 * c.unit, P.parlay_max_pct * c.bankroll);
      var verdict;
      if (r.ev <= 0) verdict = ["bad", "Negative expected value: the parlay is expected to lose money on the model's numbers."];
      else if (r.better_as_singles) verdict = ["warn", "Betting these legs separately has a higher expected value per dollar than the parlay (" + spct(r.singles_ev) + " vs " + spct(r.ev) + ")."];
      else verdict = ["good", "Positive expected value, and higher per dollar than betting the legs separately - but it only wins " + pct(r.p, 0) + " of the time, so results swing a lot."];
      kids.push(el("div", { class: "card" }, [
        el("div", { class: "sub", text: sel.map(function (x) { return x.s.team + " @ " + price(x.s.best_decimal); }).join("  +  ") }),
        el("div", { class: "metrics" }, [metric("Combined win probability", pct(r.p, 2)), metric("Fair odds", price(r.fair_decimal)),
          metric("Offered odds", price(r.offered_decimal) + (r.book ? " @ " + r.book : "")), metric("Parlay EV per $1", spct(r.ev)), metric("Same legs as singles (avg EV)", spct(r.singles_ev))]),
        el("div", { class: verdict[0], text: verdict[1] }),
        el("div", { class: "sub", text: !r.single_book_available ? "No single book carries every leg at the listed prices, so the offered odds are only a reference." :
          "Probabilities assume the legs are independent." }),
        el("div", { class: "sub", text: r.ev > 0 && allRec && r.single_book_available ? "If you still want it: stake no more than " + usd(stake) + " (" + (stake / c.unit).toFixed(1) + " u)." :
          "No parlay stake suggested (" + (r.ev <= 0 ? "negative EV" : !allRec ? "a leg is not a recommended bet" : "no single book has all legs") + ")." })]));
    }
    parlayBox.replaceChildren.apply(parlayBox, kids);
  }

  function allGames(alloc) {
    var rows = D.games.map(function (g) {
      var k = bestSide(g), s = k ? g.sides[k] : null, a = alloc[g.game_id];
      return el("tr", {}, [el("td", { text: g.away + " @ " + g.home + (g.start_utc ? " (" + when(g.start_utc) + ")" : "") }), el("td", { text: pct(g.p_home) + " home" }),
        el("td", { text: g.sides.home ? pct(g.sides.home.p_market) + " home" : "no odds" }),
        el("td", { text: s ? s.team + " " + spct(s.edge) : "–" }), el("td", { text: a ? usd(a.stake) + " · " + a.units.toFixed(1) + " u" : g.blocked || (s && s.fails[0]) || "no bet" })]);
    });
    return el("div", {}, [el("h2", { text: "All games" }), el("div", { class: "card scroll" }, [el("table", {}, [
      el("thead", {}, [el("tr", {}, ["Game", "Model", "Market (no-vig)", "Best side · edge", "Decision"].map(function (h) { return el("th", { text: h }); }))]), el("tbody", {}, rows)])])]);
  }

  function track() {
    var t = D.track, kids = [el("h2", { text: "Track record" })];
    if (!t || !t.resolved_games) return el("div", {}, kids.concat([el("div", { class: "card muted", text: "No resolved recommendations yet. Every recommendation is logged before its game, so this fills in automatically. Judge the model by closing-line value and log loss against the market, not by early profit." })]));
    var m = [metric("Resolved games", String(t.resolved_games))];
    if (t.bets && t.bets.n) { m.push(metric("Bets", String(t.bets.n)), metric("ROI", spct(t.bets.roi)), metric("Win rate", pct(t.bets.win_rate))); }
    if (t.clv && t.clv.n >= 5) m.push(metric("Avg CLV per $1", spct(t.clv.mean)), metric("Beat the close", pct(t.clv.beat_close, 0)));
    if (t.model) m.push(metric("Model log loss", t.model.log_loss.toFixed(4)));
    if (t.vs_market) m.push(metric("Market-close log loss", t.vs_market.market_close_log_loss.toFixed(4)));
    kids.push(el("div", { class: "card" }, [el("div", { class: "metrics" }, m),
      el("div", { class: "sub", text: (t.bets && t.bets.n < 500 ? "Fewer than 500 bets cannot separate skill from luck (about 6,900 are needed to detect a true 3% ROI). " : "") + "Coin-flip log loss is 0.6931." })]));
    return el("div", {}, kids);
  }

  function footer() { return el("div", { class: "banner disc", style: "margin-top:24px", text: D.disclaimer }); }

  var dynamic = el("div", { id: "dynamic" });
  function render(keepFocus) {
    var alloc = C.allocate(D.games, cfg());
    dynamic.replaceChildren(topPicks(alloc), parlayBox, allGames(alloc), track());
    renderParlay();
  }
  app.replaceChildren(header(), settingsPanel(), dynamic, footer());
  render();
})();
