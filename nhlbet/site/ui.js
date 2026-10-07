(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent);
  var C = window.NHLCore, CH = window.NHLChrome, el = CH.make;
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
  function kvItem(k, v, cls) { return el("div", {}, [el("span", { class: "k", text: k }), el("span", { class: "v " + (cls || ""), text: v })]); }
  function metric(k, v) { return el("div", { class: "metric" }, [el("div", { class: "v", text: v }), el("div", { class: "k", text: k })]); }
  function section(id, title, count, kids) {
    var h = el("h2", {}, [title, count ? el("span", { class: "count", text: count }) : null]);
    return el("section", { class: "section", id: id }, [h].concat(kids));
  }
  function table(head, rows, numCols) {
    var ths = head.map(function (h, i) { return el("th", { class: numCols && numCols.indexOf(i) >= 0 ? "num" : "", text: h }); });
    rows.forEach(function (r) { Array.prototype.forEach.call(r.children, function (td, i) { if (head[i]) td.setAttribute("data-label", head[i]); }); });   // phones show each row as a card
    return el("div", { class: "card flat scroll" }, [el("table", { class: "rtable" }, [el("thead", {}, [el("tr", {}, ths)]), el("tbody", {}, rows)])]);
  }
  /* model vs market on one 0-100% bar: blue fill = model, black tick = market, green/red segment = the gap between them */
  function probBar(model, market, teamLabel) {
    var lo = Math.min(model, market), hi = Math.max(model, market);
    var track = el("div", { class: "pbar-track" }, [
      el("div", { class: "pbar-fill", style: "width:" + (model * 100).toFixed(2) + "%" }),
      el("div", { class: "pbar-gap " + (model >= market ? "up" : "down"), style: "left:" + (lo * 100).toFixed(2) + "%;width:" + ((hi - lo) * 100).toFixed(2) + "%" }),
      el("div", { class: "pbar-mkt", style: "left:" + (market * 100).toFixed(2) + "%" })]);
    return el("div", { class: "pbar", role: "img", "aria-label": teamLabel + " win probability: model " + pct(model) + ", market " + pct(market) }, [
      track, el("div", { class: "pbar-labels" }, [el("span", {}, ["Model ", el("b", { text: pct(model) })]), el("span", {}, ["Market ", el("b", { text: pct(market) })]),
        el("span", { class: model >= market ? "good" : "bad", text: (model >= market ? "+" : "") + ((model - market) * 100).toFixed(1) + " pts" })])]);
  }

  /* ---------- "price in my app": the user types what their own sportsbook shows (no bookmaker data is published) ---------- */
  var MY = {};
  try { MY = JSON.parse(localStorage.getItem("nhlMyPrices") || "{}"); } catch (e) { MY = {}; }
  function saveMy() { try { localStorage.setItem("nhlMyPrices", JSON.stringify(MY)); } catch (e) { /* ignore */ } }
  function myKey(id) { return D.date + ":" + id; }
  function priceBox(id, onChange, withLabel) {
    var st = MY[myKey(id)] || { fmt: "american", v: "" };
    var fmt = el("select", { "aria-label": "Odds format" }, [el("option", { value: "american", text: "American" }), el("option", { value: "decimal", text: "Decimal" })]);
    fmt.value = st.fmt;
    var inp = el("input", { type: "number", step: "any", "aria-label": "Price in your app", placeholder: st.fmt === "american" ? "e.g. -115" : "e.g. 1.87", value: String(st.v) });
    function fire() { MY[myKey(id)] = { fmt: fmt.value, v: inp.value }; saveMy(); inp.placeholder = fmt.value === "american" ? "e.g. -115" : "e.g. 1.87"; onChange(C.toDecimal(inp.value, fmt.value)); }
    inp.addEventListener("input", fire); fmt.addEventListener("change", fire);
    var kids = withLabel === false ? [fmt, inp] : [el("label", { text: "Price in your app" }), fmt, inp];
    return { box: el("div", { class: "mine" }, kids), current: function () { return C.toDecimal(inp.value, fmt.value); } };
  }

  /* ---------- top of page ---------- */
  function heading() {
    var st = D.model.status, cls = st === "OK" ? "ok" : st === "ALERT" ? "alert" : "warn";
    var kids = [el("div", { class: "pagehead" }, [
      el("div", { class: "stack" }, [el("h1", { text: "Today's picks" }),
        el("div", { class: "when", text: D.date + " · " + D.run_type + " run · built " + new Date(D.generated_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) })]),
      el("div", { class: "note" }, ["Model ", el("b", { text: D.model.version || "?" }), " · health ", el("span", { class: "pill " + cls, text: st })])])];
    if (st === "ALERT") kids.push(el("div", { class: "banner bad", text: "Model health is ALERT (" + (D.model.reasons.join("; ") || "performance drift") + "). Recommendations are suspended." }));
    else if (st === "WARN") kids.push(el("div", { class: "banner", text: "Model health WARN: " + (D.model.reasons.join("; ") || "see report") + ". Treat picks with extra scepticism." }));
    if (D.odds && D.odds.stale) kids.push(el("div", { class: "banner bad", text: "Odds are STALE (the odds API was unreachable). No bets are recommended on stale prices." }));
    if (D.odds && D.odds.enabled === false) kids.push(el("div", { class: "banner", text: "Odds are disabled (no ODDS_API_KEY), so every game is 'no odds → no bet'." }));
    (D.notes || []).forEach(function (n) { kids.push(el("div", { class: "banner", text: n })); });
    kids.push(el("div", { class: "note", text: "Research and education only. Not betting advice, and no model guarantees profit. Only stake money you can afford to lose." }));
    return el("div", { class: "stack", style: "gap:.9rem" }, kids);
  }
  function chips() {
    var items = [["sec-picks", "Top picks"], ["sec-games", "All games"], ["sec-stats", "Game stats"], ["sec-other", "Totals & puck line"], ["sec-props", "Player props"], ["sec-parlay", "Parlay"], ["sec-inputs", "Model inputs"], ["sec-track", "Track record"], ["sec-paper", "Fake bets"]];
    return el("nav", { class: "chips", "aria-label": "Sections" }, items.map(function (i) { return el("a", { class: "chip", href: "#" + i[0], text: i[1] }); }));
  }
  var summaryLine = el("span", { class: "muted" });
  function updateSummary() { var c = cfg(); summaryLine.textContent = "1 unit = " + usd(c.unit) + " · " + (S.mode === "flat" ? "flat 1 unit" : S.mode === "half" ? "half Kelly" : "quarter Kelly") + " · max " + (c.maxPct * 100).toFixed(1) + "% per bet"; }
  function settingsPanel() {
    function field(label, input) { return el("div", {}, [el("label", { text: label }), input]); }
    function inp(key, step) {
      return el("input", { type: "number", min: "0", step: step || "any", value: String(S[key]), "aria-label": key, oninput: function (e) { S[key] = e.target.value; save(); updateSummary(); render(); } });
    }
    var mode = el("select", { "aria-label": "Staking mode", onchange: function (e) { S.mode = e.target.value; save(); updateSummary(); render(); } }, [
      el("option", { value: "quarter", text: "Quarter Kelly (default)" }), el("option", { value: "half", text: "Half Kelly (more aggressive)" }), el("option", { value: "flat", text: "Flat: 1 unit" })]);
    mode.value = S.mode;
    updateSummary();
    var d = el("details", { class: "drawer", id: "settings" }, [
      el("summary", {}, [el("span", {}, ["Your betting units: ", summaryLine])]),
      el("div", { class: "drawer-body" }, [
        el("div", { class: "grid" }, [field("1 unit = $", inp("unit")), field("Bankroll $", inp("bankroll")), field("Staking", mode), field("Max per bet (% bankroll)", inp("maxPct")), field("Max per day (% bankroll)", inp("dailyPct"))]),
        el("div", { class: "note", text: "Saved in this browser only. A bet appears only if the model's edge over the market's no-vig price is at least " + pct(P.min_edge, 0) + " after safety checks; stakes shrink the model toward the market and are capped. Zero picks is a normal day." })])]);
    return d;
  }
  function statusBar(alloc) {
    var recs = Object.keys(alloc), stake = recs.reduce(function (t, k) { return t + alloc[k].stake; }, 0), c = cfg();
    var stat = function (k, v) { return el("div", { class: "stat" }, [el("span", { class: "k", text: k }), el("span", { class: "v", text: v })]); };
    return el("div", { class: "statusbar" }, [stat("Games", String(D.games.length)), stat("Recommended bets", String(recs.length)),
      stat("Total stake", recs.length ? usd(stake) : "$0.00"), stat("Of bankroll", (stake / c.bankroll * 100).toFixed(2) + "%")]);
  }

  /* ---------- sections ---------- */
  function pickCard(g, alloc) {
    var k = bestSide(g), s = g.sides[k], a = alloc[g.game_id];
    var rec = !!a && a.side === k;
    var head = el("div", { class: "pick-top" }, [
      el("div", {}, [el("div", { class: "pick-team", text: s.team }), el("div", { class: "pick-sub", text: (k === "home" ? "vs " + g.away : "@ " + g.home) + (g.start_utc ? " · " + when(g.start_utc) : "") })]),
      el("span", { class: "pill " + (rec ? "rec" : "lean"), text: rec ? "Recommended" : "Lean" })]);
    var kv = [kvItem("Edge", spct(s.edge), s.edge >= 0 ? "good" : "bad"), kvItem("EV per $1", spct(s.ev), s.ev >= 0 ? "good" : "bad"),
              kvItem("Best price", price(s.best_decimal) + (s.best_book ? " @ " + s.best_book : ""))];
    if (rec) kv.push(kvItem("Stake", usd(a.stake) + " · " + a.units.toFixed(1) + " u"));
    var why = rec ? "Clears every check." + (a.scaled < 1 ? " Stake scaled down to respect your daily cap." : "") : (s.fails.length ? "Why not: " + s.fails.join("; ") + "." : "Below the recommendation rules.");
    var goalies = g.goalie_status === "unknown" ? "" : "Goalies: " + g.away_goalie + " (away) / " + g.home_goalie + " (home), " + g.goalie_status + ". ";
    var box = [head, probBar(s.p_model, s.p_market, s.team), el("div", { class: "kv" }, kv)];
    var whyKids = [goalies + why];
    if (g.notes && g.notes.length) whyKids.push(el("ul", {}, g.notes.map(function (n) { return el("li", { text: n }); })));
    box.push(el("p", { class: "why" }, whyKids));
    box.push(el("a", { class: "more", href: "#game-" + g.game_id, text: "See all the stats behind this game", onclick: function () { openGame(g.game_id); } }));
    var minD = C.minDecimal(s.p_adj, 0);
    box.push(el("div", { class: "note", text: "Only worth betting at " + price(minD) + " or better (the model's break-even price after shrinking toward the market). Compare it with the price in your own app." }));
    var out = el("div", { class: "mine-out" });
    function showMine(d) {
      if (!d) { out.textContent = ""; return; }
      var ev = C.evAt(s.p_adj, 0, d), c = cfg();
      var msg = "At " + price(d) + " in your app: expected value " + spct(ev) + " per $1";
      if (ev <= 0) msg += ". Not worth betting at this price.";
      else if (!s.qualifies) msg += ". Positive at this price, but this pick still fails the checks above, so no stake is suggested.";
      else { var st = C.stakeAt(s.p_adj, d, c); msg += ". Suggested stake at this price: " + usd(st) + " (" + (st / c.unit).toFixed(1) + " u)."; }
      out.textContent = msg;
    }
    var mine = priceBox(g.game_id + ":" + k, showMine);
    mine.box.appendChild(out); box.push(mine.box); showMine(mine.current());
    if (s.edge > 0) {
      var cb = el("input", { type: "checkbox", id: "leg-" + g.game_id, onchange: function (e) { if (e.target.checked) legs[g.game_id] = k; else delete legs[g.game_id]; renderParlay(); } });
      cb.checked = legs[g.game_id] === k;
      box.push(el("label", { class: "check", for: "leg-" + g.game_id }, [cb, "Add to parlay calculator"]));
    }
    return el("article", { class: "card pick" + (rec ? " is-rec" : "") }, box);
  }
  function topPicks(alloc) {
    var ranked = D.games.filter(function (g) { var k = bestSide(g); return k && g.sides[k].ev > 0; })
      .sort(function (a, b) { return (isRec(b, alloc) - isRec(a, alloc)) || (g_ev(b) - g_ev(a)); }).slice(0, 5);
    var nrec = Object.keys(alloc).length;
    var kids = [el("p", { class: "lead", text: nrec ? nrec + " recommended bet(s) today. The other entries are leans, shown for information only." : "No recommended bets today. Below are the best leans by expected value, for information only." })];
    if (!D.games.length) kids.push(el("div", { class: "card", text: "No regular-season games to show for this date." }));
    else if (!ranked.length) kids.push(el("div", { class: "card", text: "No game has a positive expected value against the current odds." }));
    else kids.push(el("div", { class: "picks" }, ranked.map(function (g) { return pickCard(g, alloc); })));
    return section("sec-picks", "Top picks", ranked.length ? "best " + ranked.length + " by expected value" : "", kids);
  }
  function g_ev(g) { return g.sides[bestSide(g)].ev; }
  function isRec(g, alloc) { var a = alloc[g.game_id]; return a && a.side === bestSide(g) ? 1 : 0; }

  function allGames(alloc) {
    var rows = D.games.map(function (g) {
      var k = bestSide(g), s = k ? g.sides[k] : null, a = alloc[g.game_id];
      var dec = a ? el("span", { class: "pill rec", text: usd(a.stake) + " · " + a.units.toFixed(1) + " u" })
        : el("div", { class: "reason" }, [el("span", { class: "pill mute", text: "No bet" }), el("div", { class: "muted", text: g.blocked || (s && s.fails[0]) || "" })]);
      var edge = s ? el("span", { class: "edge-cell " + (s.edge >= 0 ? "good" : "bad") }, [el("span", { class: "dot " + (s.edge >= 0 ? "up" : "down") }), s.team + " " + spct(s.edge)]) : "–";
      return el("tr", {}, [el("td", {}, [el("b", { text: g.away + " @ " + g.home }), g.start_utc ? el("span", { class: "muted", text: " " + when(g.start_utc) }) : null]),
        el("td", { class: "num", text: pct(g.p_home) }), el("td", { class: "num", text: g.sides.home ? pct(g.sides.home.p_market) : "no odds" }), el("td", {}, [edge]), el("td", {}, [dec])]);
    });
    return section("sec-games", "All games", String(D.games.length), [table(["Game", "Model (home)", "Market (home)", "Best side · edge", "Decision"], rows, [1, 2])]);
  }

  /* ---------- the numbers behind each game ---------- */
  var INPUT_KEYS = {};
  ((D.model && D.model.inputs) || []).forEach(function (i) { INPUT_KEYS[i.feature.slice(2)] = true; });   // base names (elo, cf_pct, b2b, ...) the model really uses
  function driversBlock(g) {
    var ds = g.drivers || [];
    if (!ds.length) return null;
    var mx = Math.max.apply(null, ds.map(function (d) { return Math.abs(d.value); })) || 1;
    var rows = ds.map(function (d) {
      var w = Math.abs(d.value) / mx * 50, home = d.value >= 0;
      return el("div", { class: "drow" }, [el("div", { class: "dlabel", text: d.label }),
        el("div", { class: "dtrack", role: "img", "aria-label": d.label + ": pulls toward the " + (home ? "home" : "away") + " team" }, [
          el("div", { class: "dbar " + (home ? "home" : "away"), style: (home ? "left:50%;" : "right:50%;") + "width:" + w.toFixed(1) + "%" })]),
        el("div", { class: "dval", text: (home ? g.home : g.away) + " +" + (Math.abs(d.value) * 25).toFixed(1) + " pts" })]);
    });
    return el("div", { class: "stack" }, [el("h3", { text: "What pulls hardest on the prediction" }),
      el("div", { class: "note", text: "From the model's linear component (one of five models blended into the final number), so it shows direction and rough size (about how many win-probability points each input is worth), not the exact total. Bars to the right favour " + g.home + ", to the left " + g.away + "." }),
      el("div", { class: "drivers" }, [el("div", { class: "drow dhead" }, [el("span", {}), el("span", { class: "dends" }, [el("span", { text: "← " + g.away }), el("span", { text: g.home + " →" })]), el("span", {})])].concat(rows))]);
  }
  function statsBlock(g) {
    var groups = {}, order = [];
    (g.stats || []).forEach(function (r) { if (!groups[r.group]) { groups[r.group] = []; order.push(r.group); } groups[r.group].push(r); });
    if (!order.length) return el("div", { class: "note", text: "No team statistics are available for this game yet." });
    var wanted = ["Strength", "Puck possession and chances", "Goaltending", "Special teams", "Schedule and travel", "Luck check", "Context"];
    order.sort(function (a, b) { return wanted.indexOf(a) - wanted.indexOf(b); });
    var kids = [el("h3", { text: "Team stats side by side" }), el("div", { class: "note", text: "Everything is computed from games before this one. The better number is highlighted; ★ marks inputs the model actually uses. Luck-check rows are shown for context, not as quality scores." })];
    order.forEach(function (name) {
      kids.push(el("div", { class: "sgroup", text: name }));
      groups[name].forEach(function (r) {
        kids.push(el("div", { class: "srow", title: r.tip }, [
          el("div", { class: "sval away" + (r.better === "away" ? " better" : ""), text: r.away_s }),
          el("div", { class: "slabel" }, [r.label, INPUT_KEYS[r.key] ? el("span", { class: "star", title: "The model uses this", text: " ★" }) : null]),
          el("div", { class: "sval home" + (r.better === "home" ? " better" : ""), text: r.home_s })]));
      });
    });
    return el("div", { class: "stack", style: "gap:.3rem" }, kids);
  }
  function gameDetail(g) {
    var head = el("summary", {}, [
      el("span", { class: "gd-title" }, [el("b", { text: g.away + " @ " + g.home }), g.start_utc ? el("span", { class: "muted", text: "  " + when(g.start_utc) }) : null]),
      el("span", { class: "muted", text: "model " + pct(g.p_home, 0) + " home" + (g.sides.home ? " · market " + pct(g.sides.home.p_market, 0) : "") })]);
    var gl = g.goalie_status === "unknown" || (!g.home_goalie && !g.away_goalie) ? null :
      el("div", { class: "note", text: "Expected starters: " + g.away + " " + (g.away_goalie || "unknown") + ", " + g.home + " " + (g.home_goalie || "unknown") + " (" + g.goalie_status + "). Home ice is built into the model for every game." });
    var body = el("div", { class: "drawer-body" }, [gl, driversBlock(g), statsBlock(g)]);
    return el("details", { class: "drawer gamebox", id: "game-" + g.game_id }, [head, body]);
  }
  function openGame(id) {
    var d = document.getElementById("game-" + id);
    if (d) { d.open = true; setTimeout(function () { d.scrollIntoView({ behavior: "smooth", block: "start" }); }, 0); }
  }
  function gameStats() {
    var kids = [el("p", { class: "lead", text: "Open a game to see the numbers the model is working from: both teams' strength, shot-attempt (Corsi) and expected-goals shares, goaltending, special teams, rest and travel, plus which inputs pull hardest on this prediction. Nothing here is seen after puck drop; every number is as of before the game." })];
    if (!D.games.length) kids.push(el("div", { class: "card muted", text: "No games to show." }));
    else kids.push(el("div", { class: "stack", style: "gap:.5rem" }, D.games.map(gameDetail)));
    return section("sec-stats", "Game stats", D.games.length ? String(D.games.length) : "", kids);
  }
  function modelInputs() {
    var inputs = (D.model && D.model.inputs) || [];
    var kids = [el("p", { class: "lead", text: "These are the inputs the model uses, after testing many more and dropping the ones that did not help on games it had not seen. The bar shows how much each input moves predictions on average. Injuries, line combinations and betting-market movement are not inputs." })];
    if (!inputs.length) return section("sec-inputs", "What the model looks at", "", kids.concat([el("div", { class: "card muted", text: "Input details appear after the next model build." })]));
    var mx = Math.max.apply(null, inputs.map(function (i) { return i.share || 0; })) || 1;
    var rows = inputs.map(function (i) {
      var has = i.share !== null && i.share !== undefined;
      return el("div", { class: "irow" }, [el("div", { class: "ilabel" }, [el("b", { text: i.label }), i.group ? el("span", { class: "pill mute", text: i.group }) : null]),
        has ? el("div", { class: "dtrack" }, [el("div", { class: "dbar home", style: "left:0;width:" + (i.share / mx * 100).toFixed(1) + "%" })]) : el("div", {}), el("div", { class: "dval", text: has ? (i.share * 100).toFixed(0) + "%" : "" })]);
    });
    kids.push(el("div", { class: "card stack" }, rows));
    if (inputs[0].share !== null && inputs[0].share !== undefined) kids.push(el("div", { class: "note", text: "Importance is each input's share of the average absolute SHAP value measured on out-of-sample games (see docs/FEATURES.md). Team strength usually dominates and the rest add small refinements, which is typical for hockey." }));
    return section("sec-inputs", "What the model looks at", String(inputs.length) + " inputs", kids);
  }

  /* ---------- player props (shots on goal over/under) ---------- */
  var showAllProps = false;
  function noun(q) { return q.stat === "points" ? "points" : "shots"; }
  function hv(x) { return x.val !== undefined ? x.val : x.sog; }
  function historyStrip(q) {
    var h = q.history || [];
    if (!h.length) return null;
    var mx = Math.max(q.point + 1, Math.max.apply(null, h.map(hv)));
    var bars = h.map(function (x) {
      return el("div", { class: "hbar " + (hv(x) > q.point ? "over" : "under"), title: x.date + " vs " + x.opp + ": " + hv(x) + " " + noun(q), style: "height:" + Math.max(6, hv(x) / mx * 100).toFixed(0) + "%" }, [el("span", { text: String(hv(x)) })]);
    });
    var line = el("div", { class: "hline", style: "bottom:" + (q.point / mx * 100).toFixed(0) + "%" }, [el("span", { text: String(q.point) })]);
    return el("div", { class: "hist", role: "img", "aria-label": noun(q) + " in the last " + h.length + " games: " + h.map(hv).join(", ") + ". Line " + q.point }, [line].concat(bars));
  }
  function propCard(g, q) {
    var side = q.take || q.best_side, over = side === "over";
    var pSide = over ? q.p_over : 1 - q.p_over, mSide = over ? q.p_over_market : 1 - q.p_over_market, ev = over ? q.ev_over : q.ev_under, dec = over ? q.over_price : q.under_price;
    var head = el("div", { class: "pick-top" }, [
      el("div", {}, [el("div", { class: "pick-team", text: q.name }), el("div", { class: "pick-sub", text: q.team + " vs " + q.opp + " · " + (q.stat === "points" ? "points (goals + assists)" : "shots on goal") + ", line " + q.point })]),
      el("span", { class: "pill " + (q.take ? "lean" : "mute"), text: q.take ? "Take " + (over ? "Over " : "Under ") + q.point : "No edge" })]);
    var kv = el("div", { class: "kv" }, [kvItem("Model expects", q.lam.toFixed(2) + " " + noun(q)), kvItem((over ? "Over " : "Under ") + "price", price(dec)), kvItem("EV per $1", spct(ev), ev >= 0 ? "good" : "bad"),
      kvItem("Edge", spct(pSide - mSide), pSide >= mSide ? "good" : "bad")]);
    var hist = historyStrip(q);
    var facts = [];
    if (q.avg_l10 !== null && q.avg_l10 !== undefined) facts.push("last 10 avg " + q.avg_l10.toFixed(1));
    if (q.avg_season !== null && q.avg_season !== undefined) facts.push("season avg " + q.avg_season.toFixed(1));
    if (q.hit_l10 !== null && q.hit_l10 !== undefined) facts.push("over " + q.point + " in " + Math.round(q.hit_l10 * (q.history || []).length) + " of the last " + (q.history || []).length);
    if (q.hit_l20 !== null && q.hit_l20 !== undefined) facts.push(Math.round(q.hit_l20 * 100) + "% of his last 20");
    var out = el("div", { class: "mine-out" });
    function show(d) { out.textContent = d ? "At " + price(d) + " in your app: expected value " + spct(C.evAt(pSide, 0, d)) + " per $1" + (C.evAt(pSide, 0, d) > 0 ? "." : ". Not worth it at this price.") : ""; }
    var mine = priceBox(g.game_id + ":prop:" + q.player_id, show); mine.box.appendChild(out); show(mine.current());
    var kids = [head, probBar(pSide, mSide, (over ? "Over " : "Under ") + q.point), kv];
    if (hist) kids.push(el("div", { class: "stack", style: "gap:.2rem" }, [el("div", { class: "note", text: noun(q).charAt(0).toUpperCase() + noun(q).slice(1) + " in his last " + q.history.length + " games (bars above the line are overs)" }), hist]));
    kids.push(el("p", { class: "why", text: facts.join(" · ") + (facts.length ? ". " : "") + "Model blends his recent " + (q.stat === "points" ? "scoring" : "shot") + " rate (shrunk toward his position's average), tonight's opponent, home ice, ice time and rest. " + q.n_books + " book(s) quoted this line." }));
    kids.push(mine.box);
    return el("article", { class: "card pick" }, kids);
  }
  function playerProps() {
    var items = [];
    D.games.forEach(function (g) { (g.props || []).forEach(function (q) { items.push({ g: g, q: q }); }); });
    var kids = [el("p", { class: "lead", text: "Player shots-on-goal and points over/under lines, priced by separate models built from each player's own history, the opponent's shots allowed, home ice, ice time and rest. Experimental and paper-traded only: it has no track record against the market, so treat \"Take\" as a lean to check against your own app, not a recommendation. Only a few games a day are priced (each game costs an odds-API credit)." })];
    var pe = D.odds && D.odds.props && D.odds.props.error;
    if (pe) kids.push(el("div", { class: "banner bad", text: "Player-prop prices could not be fetched on the last run (" + pe + "). Your odds plan may not include player props." }));
    if (!items.length) return section("sec-props", "Player props", "experimental", kids.concat([el("div", { class: "card muted", text: "No player-prop prices yet today. They are fetched for the first few games to start, in the late run." })]));
    items.sort(function (a, b) { return Math.max(b.q.ev_over, b.q.ev_under) - Math.max(a.q.ev_over, a.q.ev_under); });
    var shown = showAllProps ? items : items.slice(0, 12);
    kids.push(el("div", { class: "picks" }, shown.map(function (x) { return propCard(x.g, x.q); })));
    if (items.length > 12) kids.push(el("div", {}, [el("button", { type: "button", class: "btn-ghost", text: showAllProps ? "Show the top 12 only" : "Show all " + items.length + " players", onclick: function () { showAllProps = !showAllProps; render(); } })]));
    return section("sec-props", "Player props", "shots & points · experimental", kids);
  }

  var showAllOther = false;
  function otherMarkets() {
    var best = {};
    D.games.forEach(function (g) { (g.alt || []).forEach(function (q) { var k = g.game_id + ":" + q.market; if (!best[k] || q.edge > best[k].q.edge) best[k] = { g: g, q: q }; }); });
    var rows = Object.keys(best).map(function (k) { return best[k]; });
    var kids = [el("p", { class: "lead", text: "A separate goals model prices the over/under and the puck line. It has no track record against the market yet, so these are not recommendations and no stake is suggested: they are paper-traded (see the Bets page) until real results support them. Edges of several points are more likely model error than opportunity. Each row shows the side the model likes more; type your app's price to see the expected value there." })];
    if (!rows.length) return section("sec-other", "Totals & puck line", "experimental", kids.concat([el("div", { class: "card muted", text: "No totals or puck-line odds are available for today's games yet." })]));
    rows.sort(function (a, b) { return b.q.edge - a.q.edge; });
    var shown = showAllOther ? rows : rows.slice(0, 8);
    var trs = shown.map(function (r) {
      var pw = r.q.p_model * (1 - (r.q.p_push || 0)), minD = C.minDecimal(pw, r.q.p_push);
      var evCell = el("td", { class: "num", text: "–" });
      var mine = priceBox(r.g.game_id + ":" + r.q.market + ":" + r.q.side, function (d) { evCell.textContent = d ? spct(C.evAt(pw, r.q.p_push, d)) : "–"; }, false);
      evCell.textContent = mine.current() ? spct(C.evAt(pw, r.q.p_push, mine.current())) : "–";
      return el("tr", {}, [el("td", { text: r.g.away + " @ " + r.g.home }), el("td", { text: r.q.market === "totals" ? "Total" : "Puck line" }), el("td", {}, [el("b", { text: r.q.label })]),
        el("td", { class: "num", text: pct(r.q.p_model) }), el("td", { class: "num", text: pct(r.q.p_market) }),
        el("td", { class: "num" }, [el("span", { class: "edge-cell " + (r.q.edge >= 0 ? "good" : "bad") }, [el("span", { class: "dot " + (r.q.edge >= 0 ? "up" : "down") }), spct(r.q.edge)])]),
        el("td", { class: "num", text: price(r.q.best_decimal) + (r.q.best_book ? " @ " + r.q.best_book : "") }), el("td", { class: "num", text: price(minD) }), el("td", {}, [mine.box]), evCell]);
    });
    kids.push(table(["Game", "Market", "Side", "Model", "Market (no-vig)", "Edge", "Best price", "Min price to bet", "Your price", "EV at your price"], trs, [3, 4, 5, 6, 7, 9]));
    if (rows.length > 8) kids.push(el("div", {}, [el("button", { type: "button", class: "btn-ghost", text: showAllOther ? "Show top 8 only" : "Show all " + rows.length + " rows", onclick: function () { showAllOther = !showAllOther; render(); } })]));
    return section("sec-other", "Totals & puck line", "experimental", kids);
  }

  var parlayBox = el("div", { class: "stack", id: "parlay" });
  function renderParlay() {
    var sel = Object.keys(legs).map(function (id) {
      var g = D.games.filter(function (x) { return String(x.game_id) === String(id); })[0]; var s = g.sides[legs[id]];
      return { g: g, s: s, leg: { game_id: g.game_id, p: s.p_adj, best_decimal: s.best_decimal, books: s.books } };
    });
    var c = cfg(), kids = [el("p", { class: "lead", text: "Parlays multiply every leg's bookmaker margin, so they are usually worse than betting the same picks separately. Only legs with a positive edge can be added, each from a different game (no same-game parlays), up to " + P.parlay_max_legs + " legs. Tick \"Add to parlay calculator\" on a pick above." })];
    if (sel.length < 2) { kids.push(el("div", { class: "card muted", text: sel.length ? "Add at least one more leg." : "No legs selected." })); }
    else if (sel.length > P.parlay_max_legs) { kids.push(el("div", { class: "banner bad", text: "Too many legs (max " + P.parlay_max_legs + ")." })); }
    else {
      var r = C.parlay(sel.map(function (x) { return x.leg; }), P.parlay_max_legs);
      var allRec = sel.every(function (x) { return x.s.qualifies; });
      var stake = Math.min(0.5 * c.unit, P.parlay_max_pct * c.bankroll);
      var verdict;
      if (r.ev <= 0) verdict = ["bad", "Negative expected value: the parlay is expected to lose money on the model's numbers."];
      else if (r.better_as_singles) verdict = ["", "Betting these legs separately has a higher expected value per dollar than the parlay (" + spct(r.singles_ev) + " vs " + spct(r.ev) + ")."];
      else verdict = ["good", "Positive expected value, and higher per dollar than betting the legs separately, but it only wins " + pct(r.p, 0) + " of the time, so results swing a lot."];
      kids.push(el("div", { class: "card stack" }, [
        el("div", { class: "note", text: sel.map(function (x) { return x.s.team + " @ " + price(x.s.best_decimal); }).join("  +  ") }),
        el("div", { class: "metrics" }, [metric("Combined win probability", pct(r.p, 2)), metric("Fair odds", price(r.fair_decimal)),
          metric("Offered odds", price(r.offered_decimal) + (r.book ? " @ " + r.book : "")), metric("Parlay EV per $1", spct(r.ev)), metric("Same legs as singles (avg EV)", spct(r.singles_ev))]),
        el("div", { class: "banner " + verdict[0], text: verdict[1] }),
        el("div", { class: "note", text: !r.single_book_available ? (D.public_safe ? "Public view: bookmaker names are hidden, so the offered odds multiply the best available price per leg and are only illustrative; check that one book carries every leg." :
          "No single book carries every leg at the listed prices, so the offered odds are only a reference.") : "Probabilities assume the legs are independent." }),
        el("div", { class: "note", text: r.ev > 0 && allRec && r.single_book_available ? "If you still want it: stake no more than " + usd(stake) + " (" + (stake / c.unit).toFixed(1) + " u)." :
          "No parlay stake suggested (" + (r.ev <= 0 ? "negative EV" : !allRec ? "a leg is not a recommended bet" : D.public_safe ? "bookmaker detail is hidden in the public view" : "no single book has all legs") + ")." })]));
    }
    parlayBox.replaceChildren.apply(parlayBox, kids);
  }

  function track() {
    var t = D.track, kids = [];
    if (!t || !t.resolved_games) return section("sec-track", "Track record", "", [el("div", { class: "card muted", text: "No resolved recommendations yet. Every recommendation is logged before its game, so this fills in automatically. Judge the model by closing-line value and log loss against the market, not by early profit." })]);
    var m = [metric("Resolved games", String(t.resolved_games))];
    if (t.bets && t.bets.n) { m.push(metric("Bets", String(t.bets.n)), metric("ROI", spct(t.bets.roi)), metric("Win rate", pct(t.bets.win_rate))); }
    if (t.clv && t.clv.n >= 5) m.push(metric("Avg CLV per $1", spct(t.clv.mean)), metric("Beat the close", pct(t.clv.beat_close, 0)));
    if (t.model) m.push(metric("Model log loss", t.model.log_loss.toFixed(4)));
    if (t.vs_market) m.push(metric("Market-close log loss", t.vs_market.market_close_log_loss.toFixed(4)));
    kids.push(el("div", { class: "card stack" }, [el("div", { class: "metrics" }, m),
      el("div", { class: "note", text: (t.bets && t.bets.n < 500 ? "Fewer than 500 bets cannot separate skill from luck (about 6,900 are needed to detect a true 3% ROI). " : "") + "Coin-flip log loss is 0.6931." })]));
    return section("sec-track", "Track record", "", kids);
  }

  function paper() {
    var kids = [el("p", { class: "lead", text: "Alternative strategies run on every slate with pretend stakes, to measure what works faster than the selective live policy can. market_favorite is a no-skill control: a strategy only means something if it beats it by more than the noise. Details on the Bets page." })];
    if (!D.paper || !D.paper.length) return section("sec-paper", "Fake bets", "paper trading", kids.concat([el("div", { class: "card muted", text: "No settled paper bets yet." })]));
    var rows = D.paper.map(function (r) {
      return el("tr", {}, [el("td", {}, [el("b", { text: r.strategy })]), el("td", { class: "num", text: String(r.bets) }), el("td", { class: "num " + (r.bets && r.roi < 0 ? "bad" : r.bets ? "good" : ""), text: r.bets ? spct(r.roi) : "–" }),
        el("td", { class: "num", text: r.roi_lo === null || r.roi_lo === undefined ? "–" : spct(r.roi_lo) + " to " + spct(r.roi_hi) }),
        el("td", { class: "num", text: r.bets ? pct(r.win_rate, 0) : "–" }), el("td", { class: "num", text: r.n_clv ? spct(r.avg_clv) : "–" })]);
    });
    kids.push(table(["Strategy", "Bets", "ROI", "95% CI", "Win rate", "Avg CLV"], rows, [1, 2, 3, 4, 5]));
    return section("sec-paper", "Fake bets", "paper trading", kids);
  }

  var dynamic = el("div", { class: "stack", style: "gap:1.5rem" });
  function render() {
    var alloc = C.allocate(D.games, cfg());
    dynamic.replaceChildren(statusBar(alloc), topPicks(alloc), allGames(alloc), gameStats(), otherMarkets(), playerProps(), section("sec-parlay", "Parlay calculator", "", [parlayBox]), modelInputs(), track(), paper());
    renderParlay();
  }
  document.body.insertBefore(CH.topbar("picks"), app);
  app.replaceChildren(heading(), chips(), settingsPanel(), dynamic, el("footer", { class: "footer" }, [el("div", { class: "banner disc", text: D.disclaimer }),
    el("div", {}, ["Past days: ", el("a", { href: (location.pathname.indexOf("/archive/") >= 0 ? "../" : "") + "history.html", text: "history and results" }), "."])]));
  render();
  if (location.hash.indexOf("#game-") === 0) openGame(location.hash.slice(6));
})();
