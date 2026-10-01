/* Shared page chrome: sticky top bar with page navigation and a light / dark / system theme switch (remembered in this browser). */
(function (root) {
  "use strict";
  var KEY = "nhlTheme";
  function stored() { try { return localStorage.getItem(KEY) || "system"; } catch (e) { return "system"; } }
  function apply(t) {
    var r = document.documentElement;
    if (t === "light" || t === "dark") r.setAttribute("data-theme", t); else r.removeAttribute("data-theme");
  }
  apply(stored());
  function make(tag, attrs, kids) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { if (k === "text") n.textContent = attrs[k]; else if (k === "class") n.className = attrs[k]; else n.setAttribute(k, attrs[k]); });
    (kids || []).forEach(function (c) { if (c) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function topbar(active) {
    var base = location.pathname.indexOf("/archive/") >= 0 ? "../" : "";
    var links = [["picks", "Picks", "index.html"], ["bets", "Bets", "bets.html"], ["history", "History", "history.html"]].map(function (l) {
      var a = make("a", { href: base + l[2], text: l[1] });
      if (l[0] === active) a.setAttribute("aria-current", "page");
      return a;
    });
    var btn = make("button", { class: "theme-btn", type: "button", "aria-label": "Switch colour theme" });
    function label() { var t = stored(); btn.textContent = t === "system" ? "Theme: auto" : t === "dark" ? "Theme: dark" : "Theme: light"; }
    btn.addEventListener("click", function () {
      var order = ["system", "light", "dark"], next = order[(order.indexOf(stored()) + 1) % 3];
      try { localStorage.setItem(KEY, next); } catch (e) { /* storage unavailable: still switch for this visit */ }
      apply(next); label();
      var cur = document.documentElement.getAttribute("data-theme"); if (next === "system" && cur) apply("system");
    });
    label();
    return make("header", { class: "topbar" }, [make("div", { class: "topbar-in" }, [
      make("a", { class: "brand", href: base + "index.html" }, [make("span", { class: "brand-mark", "aria-hidden": "true" }), "NHL Model"]),
      make("nav", { class: "nav", "aria-label": "Pages" }, links), btn])]);
  }
  root.NHLChrome = { topbar: topbar, make: make };
})(window);
