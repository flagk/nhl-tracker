(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent), C = window.BetsCore, CH = window.NHLChrome, KEY = "nhl-my-bets-v1";
  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { if (k === "text") e.textContent = attrs[k]; else if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]); });
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  var money = function (x) { return (x < 0 ? "-$" : "$") + Math.abs(x).toFixed(2); }, signed = function (x) { return (x >= 0 ? "+$" : "-$") + Math.abs(x).toFixed(2); };
  var pct = function (x) { return x === null || x === undefined ? "-" : (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%"; };
  function load() { try { var v = JSON.parse(localStorage.getItem(KEY) || "[]"); return Array.isArray(v) ? v : []; } catch (e) { return []; } }
  function save(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); return true; } catch (e) { return false; } }
  function metric(k, v, cls) { return el("div", { class: "metric" }, [el("div", { class: "v " + (cls || ""), text: v }), el("div", { class: "k", text: k })]); }
  function metrics(s) {
    return el("div", { class: "metrics" }, [metric("Bets", String(s.n)), metric("Pending", s.pending + (s.open_stake ? " (" + money(s.open_stake) + ")" : "")),
      metric("Record", s.won + "-" + s.lost + (s.push ? "-" + s.push : "")), metric("Staked", money(s.staked)),
      metric("Profit", signed(s.profit), s.profit >= 0 ? "good" : "bad"), metric("ROI", pct(s.roi), s.roi !== null && s.roi < 0 ? "bad" : "good")]);
  }
  function chart(points, label) {
    var W = 680, H = 210, L = 58, R = 16, T = 16, B = 26, ns = "http://www.w3.org/2000/svg";
    function node(tag, attrs, text) { var n = document.createElementNS(ns, tag); Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); }); if (text !== undefined) n.textContent = text; return n; }
    var svg = node("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", role: "img", "aria-label": label });
    if (points.length < 2) { svg.appendChild(node("text", { x: 12, y: 34, fill: "currentColor", "font-size": 14 }, "Needs at least two settled bets to draw a curve.")); return svg; }
    var ys = points.map(function (p) { return p.cum; }).concat([0]), lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), pad = (hi - lo || 1) * 0.08;
    lo -= pad; hi += pad;
    var X = function (i) { return L + (W - L - R) * i / (points.length - 1); }, Y = function (v) { return H - B - (H - T - B) * (v - lo) / (hi - lo); };
    var ticks = [lo + pad, 0, hi - pad].filter(function (v, i, a) { return a.indexOf(v) === i; });
    ticks.forEach(function (v) {
      svg.appendChild(node("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--line-strong)", "stroke-width": v === 0 ? 1.5 : 1, "stroke-dasharray": v === 0 ? "" : "3 4" }));
      svg.appendChild(node("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "currentColor" }, signed(v)));
    });
    var pts = points.map(function (p, i) { return X(i).toFixed(1) + "," + Y(p.cum).toFixed(1); });
    svg.appendChild(node("polygon", { points: X(0).toFixed(1) + "," + Y(0).toFixed(1) + " " + pts.join(" ") + " " + X(points.length - 1).toFixed(1) + "," + Y(0).toFixed(1), fill: "var(--blue)", "fill-opacity": 0.13 }));
    svg.appendChild(node("polyline", { points: pts.join(" "), fill: "none", stroke: "var(--blue)", "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    var last = points[points.length - 1];
    svg.appendChild(node("circle", { cx: X(points.length - 1), cy: Y(last.cum), r: 5, fill: last.cum >= 0 ? "var(--good)" : "var(--bad)", stroke: "var(--surface)", "stroke-width": 2 }));
    svg.appendChild(node("text", { x: L, y: H - 6, "font-size": 12, fill: "currentColor" }, points[0].date));
    svg.appendChild(node("text", { x: W - R, y: H - 6, "text-anchor": "end", "font-size": 12, fill: "currentColor" }, last.date));
    return svg;
  }
  function table(head, rows) {
    rows.forEach(function (r) { Array.prototype.forEach.call(r.children, function (td, i) { if (head[i]) td.setAttribute("data-label", head[i]); }); });   // phones show each row as a card
    return el("div", { class: "card flat scroll" }, [el("table", { class: "rtable" }, [el("thead", {}, [el("tr", {}, head.map(function (h) { return el("th", { text: h }); }))]),
      el("tbody", {}, rows.length ? rows : [el("tr", {}, [el("td", { colspan: String(head.length), class: "muted", text: "Nothing here yet." })])])])]);
  }
  function sec(title, count, kids) { return el("section", { class: "section" }, [el("h2", {}, [title, count ? el("span", { class: "count", text: count }) : null])].concat(kids)); }
  var tr = function (cells) { return el("tr", {}, cells.map(function (c) { return el("td", {}, [c instanceof Node ? c : document.createTextNode(String(c))]); })); };

  /* ---------- my bets (stored only in this browser) ---------- */
  function myBets() {
    var bets = load(), box = el("div", { class: "stack", style: "gap:1.5rem" });
    function rerender() { var p = box.parentNode; var n = myBets(); p.replaceChild(n, box); }
    var games = D.games || [];
    var f = { date: el("input", { type: "date", value: D.date }), game: el("select", {}), pick: el("input", { type: "text", placeholder: "e.g. BOS moneyline" }),
      fmt: el("select", {}, [el("option", { value: "american", text: "American (-110)" }), el("option", { value: "decimal", text: "Decimal (1.91)" })]),
      odds: el("input", { type: "number", step: "any", placeholder: "-110" }), stake: el("input", { type: "number", step: "0.01", min: "0", placeholder: "10" }),
      note: el("input", { type: "text", placeholder: "optional note" }),
      type: el("select", {}, ["Moneyline", "Puck line", "Total (over/under)", "Parlay", "Player prop", "Other"].map(function (t) { return el("option", { value: t, text: t }); })) };
    f.game.appendChild(el("option", { value: "", text: "(custom bet)" }));
    games.forEach(function (g, i) { f.game.appendChild(el("option", { value: String(i), text: g.away + " @ " + g.home })); });
    var HINTS = { "Moneyline": "e.g. BOS moneyline", "Puck line": "e.g. BOS -1.5", "Total (over/under)": "e.g. Over 6.5", "Parlay": "e.g. BOS ML + TOR ML + Over 6.5", "Player prop": "e.g. McDavid over 1.5 points", "Other": "describe the bet" };
    function prefill() {                    // fill pick/odds from today's prices for the chosen game and bet type (the model's preferred side; you can overwrite)
      f.pick.placeholder = HINTS[f.type.value] || "";
      var g = games[Number(f.game.value)]; if (!g) return;
      var t = f.type.value, cand = null;
      if (t === "Moneyline") { var side = g.model_side; if (side && g.sides[side]) cand = { label: g.sides[side].team + " moneyline", dec: g.sides[side].best_decimal }; }
      else if (t === "Puck line" || t === "Total (over/under)") {
        var mk = t === "Puck line" ? "spreads" : "totals", qs = (g.alt || []).filter(function (q) { return q.market === mk; }).sort(function (a, b) { return b.p_model - a.p_model; });
        if (qs.length) cand = { label: qs[0].label, dec: qs[0].best_decimal };
      }
      if (cand) { f.pick.value = cand.label; f.fmt.value = "decimal"; f.odds.value = cand.dec.toFixed(2); }
    }
    f.game.addEventListener("change", prefill); f.type.addEventListener("change", prefill);
    var err = el("div", { class: "banner bad", style: "display:none" });
    var add = el("button", { class: "btn-primary", type: "button", text: "Add bet" });
    add.addEventListener("click", function () {
      var dec = C.toDecimal(f.odds.value, f.fmt.value), stake = Number(f.stake.value);
      if (!f.pick.value.trim() || dec === null || !(stake > 0)) { err.textContent = "Enter a pick, valid odds and a stake above zero."; err.style.display = ""; return; }
      var g = games[Number(f.game.value)];
      bets.push({ id: Date.now() + "-" + Math.random().toString(36).slice(2, 7), date: f.date.value || D.date, type: f.type.value, game: g ? g.away + " @ " + g.home : "", pick: f.pick.value.trim(), decimal: dec, stake: stake, result: "pending", note: f.note.value.trim() });
      if (!save(bets)) { err.textContent = "Could not save in this browser (private mode?). Use Export to keep your bets."; err.style.display = ""; return; }
      rerender();
    });
    var form = el("div", { class: "card" }, [el("div", { class: "grid" }, [
      el("div", {}, [el("label", { text: "Date" }), f.date]), el("div", {}, [el("label", { text: "Bet type" }), f.type]), el("div", {}, [el("label", { text: "Today's game (optional)" }), f.game]),
      el("div", {}, [el("label", { text: "Pick / line" }), f.pick]),
      el("div", {}, [el("label", { text: "Odds format" }), f.fmt]), el("div", {}, [el("label", { text: "Odds" }), f.odds]), el("div", {}, [el("label", { text: "Stake ($)" }), f.stake]),
      el("div", {}, [el("label", { text: "Note" }), f.note])]), err, el("div", { style: "margin-top:10px" }, [add])]);
    var s = C.summarize(bets), rows = bets.slice().sort(function (a, b) { return a.date < b.date ? 1 : a.date > b.date ? -1 : 0; }).map(function (b) {
      var sel = el("select", {}, ["pending", "won", "lost", "push"].map(function (r) { var o = el("option", { value: r, text: r }); if (r === b.result) o.selected = true; return o; }));
      sel.addEventListener("change", function () { b.result = sel.value; save(bets); rerender(); });
      var del = el("button", { text: "Delete" }); del.addEventListener("click", function () { bets = bets.filter(function (x) { return x.id !== b.id; }); save(bets); rerender(); });
      var p = C.settle(b);
      return tr([b.date, b.type || "Moneyline", b.game || "-", b.pick, b.decimal.toFixed(2), money(b.stake), sel, p === null ? "-" : signed(p), del]);
    });
    var exp = el("button", { text: "Export JSON" });
    exp.addEventListener("click", function () { var a = el("a", { href: "data:application/json;charset=utf-8," + encodeURIComponent(JSON.stringify(bets, null, 1)), download: "my-bets.json" }); document.body.appendChild(a); a.click(); a.remove(); });
    var imp = el("input", { type: "file", accept: "application/json" });
    imp.addEventListener("change", function () {
      var fr = new FileReader();
      fr.onload = function () {
        try {
          var v = JSON.parse(fr.result), ok = Array.isArray(v) ? v.filter(function (b) { return b && typeof b.pick === "string" && b.decimal > 1 && b.stake > 0 && ["pending", "won", "lost", "push"].indexOf(b.result) >= 0; }) : [];
          var have = {}; bets.forEach(function (b) { have[b.id] = 1; });
          ok.forEach(function (b) { if (!b.id || have[b.id]) b.id = Date.now() + "-" + Math.random().toString(36).slice(2, 7); bets.push({ id: String(b.id), date: String(b.date || D.date).slice(0, 10), type: String(b.type || "Moneyline"), game: String(b.game || ""), pick: b.pick, decimal: Number(b.decimal), stake: Number(b.stake), result: b.result, note: String(b.note || "") }); });
          save(bets); rerender();
        } catch (e) { err.textContent = "That file is not a valid export."; err.style.display = ""; }
      };
      fr.readAsText(imp.files[0]);
    });
    var storeNote = el("div", { class: "banner", text: "Your bets are stored only in this browser (localStorage). They are never sent anywhere or written to the repository. Use Export to back them up or move them to another device." });
    box.appendChild(sec("Log a bet", "", [storeNote, form]));
    box.appendChild(sec("My results", "", [el("div", { class: "card stack" }, [metrics(s), chart(C.cumulative(bets), "My cumulative profit")])]));
    box.appendChild(sec("My bets", String(bets.length), [table(["Date", "Type", "Game", "Pick", "Price", "Stake", "Result", "Profit", ""], rows),
      el("div", { class: "row", style: "justify-content:flex-start;align-items:center;gap:.75rem" }, [exp, el("label", { text: "Import JSON", style: "margin:0" }), imp])]));
    return box;
  }

  /* ---------- fake (paper) bets ---------- */
  function fakeBets() {
    var box = el("div", { class: "stack", style: "gap:1.5rem" }), names = (D.paper_strategies || []).map(function (s) { return s.name; }), cur = names.indexOf("every_game") >= 0 ? "every_game" : names[0];
    var sel = el("select", { id: "strat" }, (D.paper_strategies || []).map(function (s) { var o = el("option", { value: s.name, text: s.name }); if (s.name === cur) o.selected = true; return o; }));
    var body = el("div", { class: "stack", style: "gap:1.5rem" });
    function draw() {
      body.textContent = "";
      var st = (D.paper_strategies || []).filter(function (s) { return s.name === sel.value; })[0];
      if (!st) { body.appendChild(el("div", { class: "banner info", text: "No paper bets have been logged yet. They appear after the next daily run." })); return; }
      var all = D.paper_bets.filter(function (b) { return b.strategy === sel.value; }), sm = C.summarize(all);
      body.appendChild(el("div", { class: "banner info", text: st.description }));
      body.appendChild(el("div", { class: "card stack" }, [metrics(sm), chart(C.cumulative(all), "Cumulative fake profit")]));
      var pend = all.filter(function (b) { return b.result === "pending"; });
      body.appendChild(sec("Open fake bets", pend.length + " · " + money(sm.open_stake) + " pretend stake", [table(["Date", "Type", "Game", "Pretend pick", "Price", "Pretend stake"], pend.map(function (b) { return tr([b.date, b.type, b.game, b.pick, b.decimal.toFixed(2), money(b.stake)]); }))]));
      var done = all.filter(function (b) { return b.result !== "pending"; }).sort(function (a, b) { return a.date < b.date ? 1 : a.date > b.date ? -1 : 0; }).slice(0, 200);
      body.appendChild(sec("Settled fake bets", "latest 200", [table(["Date", "Type", "Game", "Pretend pick", "Price", "Stake", "Result", "Profit"], done.map(function (b) { return tr([b.date, b.type, b.game, b.pick, b.decimal.toFixed(2), money(b.stake), b.result, signed(C.settle(b))]); }))]));
    }
    sel.addEventListener("change", draw);
    var rows = (D.paper_strategies || []).map(function (s) { var x = C.summarize(D.paper_bets.filter(function (b) { return b.strategy === s.name; })); var pr = el("span", { class: x.profit > 0 ? "good" : x.profit < 0 ? "bad" : "muted", text: signed(x.profit) }); return { n: x.n, row: tr([el("b", { text: s.name }), x.n, x.pending, x.settled, pr, pct(x.roi), x.hit === null ? "-" : (x.hit * 100).toFixed(0) + "%"]) }; })
      .sort(function (a, b) { return b.n - a.n; }).map(function (o) { return o.row; });
    box.appendChild(el("div", { class: "banner info", text: "Fake money for measurement only. Every strategy stakes 1% of a $" + D.bankroll.toFixed(0) + " pretend bankroll per bet (the live-policy copies use their own sizing). The no-skill control is market_favorite; every_game, every_total and every_puckline bet the model's side of every game in that market, even at a negative edge (totals and puck line are experimental)." }));
    box.appendChild(sec("All strategies", String(rows.length), [table(["Strategy", "Bets", "Open", "Settled", "Profit", "ROI", "Hit rate"], rows)]));
    box.appendChild(sec("Strategy detail", "", [el("div", { style: "max-width:18rem" }, [el("label", { text: "Strategy", for: "strat" }), sel])]));
    box.appendChild(body); draw();
    return box;
  }

  function main() {
    var app = document.getElementById("app"), view = el("div", {}), tab = "mine";
    function show() { view.textContent = ""; view.appendChild(tab === "mine" ? myBets() : fakeBets()); b1.className = tab === "mine" ? "on" : ""; b2.className = tab === "fake" ? "on" : ""; }
    var b1 = el("button", { type: "button", text: "My bets" }), b2 = el("button", { type: "button", text: "Fake bets" });
    b1.addEventListener("click", function () { tab = "mine"; show(); }); b2.addEventListener("click", function () { tab = "fake"; show(); });
    document.body.insertBefore(CH.topbar("bets"), app);
    app.appendChild(el("div", { class: "pagehead" }, [el("div", { class: "stack" }, [el("h1", { text: "Bets" }),
      el("div", { class: "when", text: "Your own bets, and the pretend-money bets the model places on every game · built " + new Date(D.generated_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) })]),
      el("div", { class: "tabs", role: "group", "aria-label": "View" }, [b1, b2])]));
    app.appendChild(view); show();
    app.appendChild(el("footer", { class: "footer" }, [el("div", { class: "banner disc", text: D.disclaimer })]));
  }
  main();
})();
