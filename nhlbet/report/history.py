"""Pick history: every logged recommendation with its result, by day (markdown for GitHub, HTML for the site)."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from nhlbet.privacy import is_public_safe
from nhlbet.report.betlog import final_recommendations, performance, resolved
from nhlbet.report.markdown import DISCLAIMER
from nhlbet.data.store import Store


def _day_table(store: Store) -> pd.DataFrame:
    """One row per game day: games scored, bets, staked, profit (settled only), ROI, pending bets."""
    r = final_recommendations(store)
    if r.empty:
        return pd.DataFrame(columns=["date", "games", "bets", "staked", "profit", "roi", "pending"])
    d = resolved(store)
    settled = set(d.game_id) if len(d) else set()
    r["is_bet"] = (r.action == "BET") & (r.stake > 0)
    rows = []
    for date, part in r.groupby("game_date"):
        bets = part[part.is_bet]
        done = d[d.game_id.isin(bets.game_id) & d.is_bet] if len(d) else d
        staked, profit = (float(done.stake.sum()), float(done.profit.sum())) if len(done) else (0.0, 0.0)
        rows.append({"date": date, "games": int(len(part)), "bets": int(len(bets)), "staked": staked, "profit": profit,
                     "roi": (profit / staked) if staked > 0 else float("nan"), "pending": int((~bets.game_id.isin(settled)).sum())})
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def _bet_table(store: Store, limit: int = 200) -> pd.DataFrame:
    r = final_recommendations(store)
    if r.empty:
        return r
    r = r[(r.action == "BET") & (r.stake > 0)].copy()
    d = resolved(store)
    res = d.set_index("game_id")[["bet_won", "profit", "clv_ev"]] if len(d) else pd.DataFrame(columns=["bet_won", "profit", "clv_ev"])
    r = r.merge(res, left_on="game_id", right_index=True, how="left")
    return r.sort_values(["game_date", "run_at"], ascending=False).head(limit).reset_index(drop=True)


def _report_link(report_dir: Path, date: str, prefix: str) -> str:
    for kind in ("late", "morning"):
        if (report_dir / "daily" / f"{date}-{kind}.md").exists():
            return f"[report]({prefix}daily/{date}-{kind}.md)"
    return ""


def _sp(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:+.1%}"


def _result(row) -> str:
    if pd.isna(row.bet_won):
        return "pending"
    return "won" if row.bet_won == 1 else "lost"


def render_history_md(store: Store, report_dir: str | Path = "reports", site_archive: bool = True, public_safe: bool | None = None,
                      start_bankroll: float = 1000.0) -> str:
    ps = is_public_safe() if public_safe is None else public_safe
    rd = Path(report_dir)
    days, bets = _day_table(store), _bet_table(store)
    perf = performance(store, start_bankroll)
    L = ["# Pick history", "", DISCLAIMER, "",
         "Every recommendation is logged **before** its game and settled afterwards; passes (no bet) are the majority and are not listed here. "
         "[Back to the README](../README.md) · [Today's report](latest.md) · [Bet log CSV](../data/logs/bet_log.csv)", ""]
    b = perf.get("bets") or {}
    if perf.get("resolved_games"):
        clv = perf.get("clv") or {}
        L += ["## Overall", "",
              f"- Games with a logged recommendation and a result: **{perf['resolved_games']}** (bet on {b.get('n', 0)}; passed on {perf.get('no_bet_rate', 0):.0%})"]
        if b.get("n"):
            L += [f"- Bets: staked ${b['staked']:,.2f} · profit ${b['profit']:+,.2f} · **ROI {b['roi']:+.1%}** · win rate {b['win_rate']:.1%}"]
        if clv.get("n", 0) >= 5:
            L += [f"- **Closing-line value**: {clv['mean']:+.2%} per $1 (95% CI {clv['ci'][0]:+.2%} to {clv['ci'][1]:+.2%}), beat the close on {clv['beat_close']:.0%} of {clv['n']} bets"]
        if b.get("n", 0) < 500:
            L += ["- ⚠️ Far too few bets to separate skill from luck (about 6,900 would be needed to detect a true 3% ROI). Closing-line value converges much faster."]
        L += [""]
    else:
        L += ["No settled results yet. This page fills in automatically as games finish.", ""]
    L += ["## By day", "", "| Date | Games | Bets | Staked | Profit | ROI | Pending | Links |", "|---|---|---|---|---|---|---|---|"]
    if days.empty:
        L += ["| - | - | - | - | - | - | - | - |"]
    for r in days.itertuples():
        links = _report_link(rd, r.date, "")
        if site_archive and (rd.parent / "site" / "archive" / f"{r.date}.html").exists():
            links += (" · " if links else "") + f"[page](../site/archive/{r.date}.html)"
        roi = "-" if r.roi != r.roi else f"{r.roi:+.1%}"
        L += [f"| {r.date} | {r.games} | {r.bets} | {'-' if not r.staked else f'${r.staked:,.2f}'} | {'-' if not r.staked else f'${r.profit:+,.2f}'} | {roi} | {r.pending or '-'} | {links} |"]
    L += ["", "## Recommended bets", ""]
    if bets.empty:
        L += ["No recommended bets yet."]
    else:
        L += ["| Date | Game | Pick | Price | Stake | Edge | Result | Profit | CLV |", "|---|---|---|---|---|---|---|---|---|"]
        for r in bets.itertuples():
            price = f"{r.decimal:.2f}" + ("" if ps or not r.book else f" ({r.book})")
            profit = "-" if pd.isna(r.profit) or _result(r) == "pending" else f"${r.profit:+,.2f}"
            L += [f"| {r.game_date} | {r.away} @ {r.home} | {r.team} | {price} | ${r.stake:,.2f} | {_sp(r.edge)} | {_result(r)} | {profit} | {_sp(r.clv_ev)} |"]
    L += ["", "---", DISCLAIMER, ""]
    return "\n".join(L)


def render_history_html(store: Store, report_dir: str | Path = "reports", public_safe: bool | None = None) -> str:
    """Standalone history page for the site: same look as the picks and bets pages, no script (links to the per-day archive pages)."""
    from nhlbet.site.build import HERE

    ps = is_public_safe() if public_safe is None else public_safe
    days, bets = _day_table(store), _bet_table(store, 100)
    e = html.escape
    money = lambda x: f"${x:+,.2f}"  # noqa: E731
    cls = lambda x: "good" if x > 0 else "bad" if x < 0 else ""  # noqa: E731
    rows = "".join(
        f"<tr><td><a href='archive/{e(r.date)}.html'>{e(r.date)}</a></td><td class='num' data-label='Games'>{r.games}</td><td class='num' data-label='Bets'>{r.bets}</td>"
        f"<td class='num' data-label='Staked'>{'-' if not r.staked else f'${r.staked:,.2f}'}</td><td class='num {cls(r.profit) if r.staked else ''}' data-label='Profit'>{'-' if not r.staked else money(r.profit)}</td>"
        f"<td class='num {cls(r.roi) if r.roi == r.roi else ''}' data-label='ROI'>{'-' if r.roi != r.roi else f'{r.roi:+.1%}'}</td><td class='num' data-label='Pending'>{r.pending or '-'}</td></tr>" for r in days.itertuples()
    ) or "<tr><td colspan=7>No days yet.</td></tr>"
    def pill(res: str) -> str:
        k = {"won": "ok", "lost": "alert", "pending": "mute"}.get(res, "mute")
        return f"<span class='pill {k}'>{e(res)}</span>"
    brows = "".join(
        f"<tr><td>{e(r.game_date)}</td><td data-label='Game'>{e(r.away)} @ {e(r.home)}</td><td data-label='Pick'><b>{e(r.team)}</b></td>"
        f"<td class='num' data-label='Price'>{r.decimal:.2f}{'' if ps or not r.book else ' (' + e(str(r.book)) + ')'}</td><td class='num' data-label='Stake'>${r.stake:,.2f}</td><td data-label='Result'>{pill(_result(r))}</td>"
        f"<td class='num {'' if pd.isna(r.profit) or _result(r) == 'pending' else cls(r.profit)}' data-label='Profit'>{'-' if pd.isna(r.profit) or _result(r) == 'pending' else money(r.profit)}</td></tr>" for r in bets.itertuples()
    ) or "<tr><td colspan=7>No recommended bets yet.</td></tr>"
    disc = e(DISCLAIMER.replace("> ", "").replace("**", ""))
    css = (HERE / "theme.css").read_text()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NHL Model: History</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700;800&display=swap">
<style>{css}</style></head>
<body><header class="topbar"><div class="topbar-in"><a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"></span>NHL Model</a>
<nav class="nav" aria-label="Pages"><a href="index.html">Picks</a><a href="bets.html">Bets</a><a href="history.html" aria-current="page">History</a></nav></div></header>
<main class="wrap"><div class="pagehead"><div class="stack"><h1>History</h1><div class="when">Every day's picks and how they settled. Open a day to see that day's full page.</div></div></div>
<div class="note">Research and education only. Not betting advice. Stake only what you can afford to lose.</div>
<nav class="chips" aria-label="Sections"><a class="chip" href="#by-day">By day</a><a class="chip" href="#bets">Recommended bets</a></nav>
<section class="section" id="by-day"><h2>By day <span class="count">{len(days)}</span></h2><div class="card flat scroll"><table class="rtable"><thead><tr><th>Date</th><th class="num">Games</th><th class="num">Bets</th><th class="num">Staked</th><th class="num">Profit</th><th class="num">ROI</th><th class="num">Pending</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section class="section" id="bets"><h2>Recommended bets <span class="count">latest 100</span></h2><div class="card flat scroll"><table class="rtable"><thead><tr><th>Date</th><th>Game</th><th>Pick</th><th class="num">Price</th><th class="num">Stake</th><th>Result</th><th class="num">Profit</th></tr></thead><tbody>{brows}</tbody></table></div></section>
<footer class="footer"><div class="banner disc">{disc}</div></footer></main></body></html>"""
