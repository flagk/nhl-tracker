(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent), C = window.BetsCore, KEY = "nhl-my-bets-v1";
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
    var W = 640, H = 160, P = 28, ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("width", "100%"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", label);
    if (points.length < 2) { var t = document.createElementNS(ns, "text"); t.setAttribute("x", 10); t.setAttribute("y", 30); t.setAttribute("fill", "currentColor"); t.textContent = "Needs at least two settled bets to draw a curve."; svg.appendChild(t); return svg; }
    var ys = points.map(function (p) { return p.cum; }).concat([0]), lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), span = hi - lo || 1;
    var X = function (i) { return P + (W - 2 * P) * i / (points.length - 1); }, Y = function (v) { return H - P - (H - 2 * P) * (v - lo) / span; };
    var z = document.createElementNS(ns, "line"); [["x1", P], ["x2", W - P], ["y1", Y(0)], ["y2", Y(0)], ["stroke", "currentColor"], ["stroke-opacity", ".3"], ["stroke-dasharray", "4 4"]].forEach(function (a) { z.setAttribute(a[0], a[1]); }); svg.appendChild(z);
    var pl = document.createElementNS(ns, "polyline"); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", "var(--accent)"); pl.setAttribute("stroke-width", "2");
    pl.setAttribute("points", points.map(function (p, i) { return X(i).toFixed(1) + "," + Y(p.cum).toFixed(1); }).join(" ")); svg.appendChild(pl);
    [[hi, P - 4], [lo, H - P + 14]].forEach(function (a) { var t = document.createElementNS(ns, "text"); t.setAttribute("x", 4); t.setAttribute("y", a[1] + (a[0] === hi ? 8 : 0)); t.setAttribute("font-size", "11"); t.setAttribute("fill", "currentColor"); t.textContent = signed(a[0]); svg.appendChild(t); });
    return svg;
  }
  function table(head, rows) {
    return el("div", { class: "scroll" }, [el("table", {}, [el("thead", {}, [el("tr", {}, head.map(function (h) { return el("th", { text: h }); }))]),
      el("tbody", {}, rows.length ? rows : [el("tr", {}, [el("td", { colspan: String(head.length), class: "muted", text: "Nothing here yet." })])])])]);
  }
  var tr = function (cells) { return el("tr", {}, cells.map(function (c) { return el("td", {}, [c instanceof Node ? c : document.createTextNode(String(c))]); })); };

  /* ---------- my bets (stored only in this browser) ---------- */
  function myBets() {
    var bets = load(), box = el("div", {});
    function rerender() { var p = box.parentNode; var n = myBets(); p.replaceChild(n, box); }
    var games = D.games || [];
    var f = { date: el("input", { type: "date", value: D.date }), game: el("select", {}), pick: el("input", { type: "text", placeholder: "e.g. BOS moneyline" }),
      fmt: el("select", {}, [el("option", { value: "american", text: "American (-110)" }), el("option", { value: "decimal", text: "Decimal (1.91)" })]),
      odds: el("input", { type: "number", step: "any", placeholder: "-110" }), stake: el("input", { type: "number", step: "0.01", min: "0", placeholder: "10" }),
      note: el("input", { type: "text", placeholder: "optional note" }) };
    f.game.appendChild(el("option", { value: "", text: "(custom bet)" }));
    games.forEach(function (g, i) { f.game.appendChild(el("option", { value: String(i), text: g.away + " @ " + g.home })); });
    f.game.addEventListener("change", function () {
      var g = games[Number(f.game.value)]; if (!g) return;
      var side = g.model_side; if (side && g.sides[side]) { f.pick.value = g.sides[side].team + " moneyline"; f.fmt.value = "decimal"; f.odds.value = g.sides[side].best_decimal.toFixed(2); }
    });
    var err = el("div", { class: "banner bad", style: "display:none" });
    var add = el("button", { text: "Add bet" });
    add.addEventListener("click", function () {
      var dec = C.toDecimal(f.odds.value, f.fmt.value), stake = Number(f.stake.value);
      if (!f.pick.value.trim() || dec === null || !(stake > 0)) { err.textContent = "Enter a pick, valid odds and a stake above zero."; err.style.display = ""; return; }
      var g = games[Number(f.game.value)];
      bets.push({ id: Date.now() + "-" + Math.random().toString(36).slice(2, 7), date: f.date.value || D.date, game: g ? g.away + " @ " + g.home : "", pick: f.pick.value.trim(), decimal: dec, stake: stake, result: "pending", note: f.note.value.trim() });
      if (!save(bets)) { err.textContent = "Could not save in this browser (private mode?). Use Export to keep your bets."; err.style.display = ""; return; }
      rerender();
    });
    var form = el("div", { class: "card" }, [el("div", { class: "grid" }, [
      el("div", {}, [el("label", { text: "Date" }), f.date]), el("div", {}, [el("label", { text: "Today's game (optional)" }), f.game]), el("div", {}, [el("label", { text: "Pick" }), f.pick]),
      el("div", {}, [el("label", { text: "Odds format" }), f.fmt]), el("div", {}, [el("label", { text: "Odds" }), f.odds]), el("div", {}, [el("label", { text: "Stake ($)" }), f.stake]),
      el("div", {}, [el("label", { text: "Note" }), f.note])]), err, el("div", { style: "margin-top:10px" }, [add])]);
    var s = C.summarize(bets), rows = bets.slice().sort(function (a, b) { return a.date < b.date ? 1 : a.date > b.date ? -1 : 0; }).map(function (b) {
      var sel = el("select", {}, ["pending", "won", "lost", "push"].map(function (r) { var o = el("option", { value: r, text: r }); if (r === b.result) o.selected = true; return o; }));
      sel.addEventListener("change", function () { b.result = sel.value; save(bets); rerender(); });
      var del = el("button", { text: "Delete" }); del.addEventListener("click", function () { bets = bets.filter(function (x) { return x.id !== b.id; }); save(bets); rerender(); });
      var p = C.settle(b);
      return tr([b.date, b.game || "-", b.pick, b.decimal.toFixed(2), money(b.stake), sel, p === null ? "-" : signed(p), del]);
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
          ok.forEach(function (b) { if (!b.id || have[b.id]) b.id = Date.now() + "-" + Math.random().toString(36).slice(2, 7); bets.push({ id: String(b.id), date: String(b.date || D.date).slice(0, 10), game: String(b.game || ""), pick: b.pick, decimal: Number(b.decimal), stake: Number(b.stake), result: b.result, note: String(b.note || "") }); });
          save(bets); rerender();
        } catch (e) { err.textContent = "That file is not a valid export."; err.style.display = ""; }
      };
      fr.readAsText(imp.files[0]);
    });
    box.appendChild(el("div", { class: "banner info", text: "Your bets are stored only in this browser (localStorage). They are never sent anywhere or written to the repository. Use Export to back them up or move them to another device." }));
    box.appendChild(form); box.appendChild(el("h2", { text: "My results" })); box.appendChild(el("div", { class: "card" }, [metrics(s), chart(C.cumulative(bets), "My cumulative profit")]));
    box.appendChild(el("h2", { text: "My bets" }));
    box.appendChild(el("div", { class: "card" }, [table(["Date", "Game", "Pick", "Price", "Stake", "Result", "Profit", ""], rows), el("div", { style: "display:flex;gap:10px;margin-top:10px;align-items:center;flex-wrap:wrap" }, [exp, el("label", { text: "Import JSON:" }), imp])]));
    return box;
  }

  /* ---------- fake (paper) bets ---------- */
  function fakeBets() {
    var box = el("div", {}), names = (D.paper_strategies || []).map(function (s) { return s.name; }), cur = names.indexOf("every_game") >= 0 ? "every_game" : names[0];
    var sel = el("select", {}, (D.paper_strategies || []).map(function (s) { var o = el("option", { value: s.name, text: s.name }); if (s.name === cur) o.selected = true; return o; }));
    var body = el("div", {});
    function draw() {
      body.textContent = "";
      var st = (D.paper_strategies || []).filter(function (s) { return s.name === sel.value; })[0];
      if (!st) { body.appendChild(el("div", { class: "banner info", text: "No paper bets have been logged yet. They appear after the next daily run." })); return; }
      var all = D.paper_bets.filter(function (b) { return b.strategy === sel.value; }), sm = C.summarize(all);
      body.appendChild(el("div", { class: "banner info", text: st.description }));
      body.appendChild(el("div", { class: "card" }, [metrics(sm), chart(C.cumulative(all), "Cumulative fake profit")]));
      var pend = all.filter(function (b) { return b.result === "pending"; });
      body.appendChild(el("h2", { text: "Open fake bets (" + pend.length + ", " + money(sm.open_stake) + " pretend stake)" }));
      body.appendChild(el("div", { class: "card" }, [table(["Date", "Game", "Pretend pick", "Price", "Pretend stake"], pend.map(function (b) { return tr([b.date, b.game, b.pick, b.decimal.toFixed(2), money(b.stake)]); }))]));
      var done = all.filter(function (b) { return b.result !== "pending"; }).sort(function (a, b) { return a.date < b.date ? 1 : a.date > b.date ? -1 : 0; }).slice(0, 200);
      body.appendChild(el("h2", { text: "Settled fake bets (latest 200)" }));
      body.appendChild(el("div", { class: "card" }, [table(["Date", "Game", "Pretend pick", "Price", "Stake", "Result", "Profit"], done.map(function (b) { return tr([b.date, b.game, b.pick, b.decimal.toFixed(2), money(b.stake), b.result, signed(C.settle(b))]); }))]));
    }
    sel.addEventListener("change", draw);
    var rows = (D.paper_strategies || []).map(function (s) { var x = C.summarize(D.paper_bets.filter(function (b) { return b.strategy === s.name; })); return tr([s.name, x.n, x.pending, x.settled, signed(x.profit), pct(x.roi), x.hit === null ? "-" : (x.hit * 100).toFixed(0) + "%"]); });
    box.appendChild(el("div", { class: "banner info", text: "Fake money for measurement only. Every strategy stakes 1% of a $" + D.bankroll.toFixed(0) + " pretend bankroll per bet (the live-policy copies use their own sizing). The no-skill control is market_favorite; every_game bets the model's side of every game, even at a negative edge." }));
    box.appendChild(el("div", { class: "card" }, [table(["Strategy", "Bets", "Open", "Settled", "Profit", "ROI", "Hit rate"], rows)]));
    box.appendChild(el("h2", { text: "Strategy detail" })); box.appendChild(el("div", {}, [el("label", { text: "Strategy" }), sel])); box.appendChild(body); draw();
    return box;
  }

  function main() {
    var app = document.getElementById("app"), view = el("div", {}), tab = "mine";
    function show() { view.textContent = ""; view.appendChild(tab === "mine" ? myBets() : fakeBets()); b1.className = tab === "mine" ? "on" : ""; b2.className = tab === "fake" ? "on" : ""; }
    var b1 = el("button", { text: "My bets" }), b2 = el("button", { text: "Fake bets (paper trading)" });
    b1.addEventListener("click", function () { tab = "mine"; show(); }); b2.addEventListener("click", function () { tab = "fake"; show(); });
    var base = location.pathname.indexOf("/archive/") >= 0 ? "../" : "";
    app.appendChild(el("h1", { text: "Bets tracker" }));
    app.appendChild(el("div", { class: "sub" }, [el("a", { href: base + "index.html", text: "Today's picks" }), " · ", el("a", { href: base + "history.html", text: "Past picks & results" }), " · built " + new Date(D.generated_at).toLocaleString()]));
    app.appendChild(el("div", { class: "banner disc", text: D.disclaimer }));
    app.appendChild(el("div", { class: "tabs" }, [b1, b2])); app.appendChild(view); show();
  }
  main();
})();
