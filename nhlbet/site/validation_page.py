"""``site/validation.html``: the evidence page. Static HTML, no script, so it works on any phone and with JavaScript off.

It answers one question for an outsider: do the picks have a real, durable edge? Live results (from the append-only logs) are kept apart from the backtest, every
number carries its sample size and an interval, and the verdicts are computed by rule from the data (nothing here is hand-written to look good).
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

from nhlbet.report.markdown import DISCLAIMER

e = html.escape


def _pct(x, d=1) -> str:
    return "-" if x is None or x != x else f"{x * 100:+.{d}f}%"


def _p(x, d=1) -> str:
    return "-" if x is None or x != x else f"{x * 100:.{d}f}%"


def _f(x, d=4) -> str:
    return "-" if x is None or x != x else f"{x:.{d}f}"


def _ci(ci, as_pct=True) -> str:
    if not ci or ci[0] != ci[0]:
        return "-"
    return f"{_pct(ci[0])} to {_pct(ci[1])}" if as_pct else f"{ci[0]:.4f} to {ci[1]:.4f}"


def verdicts(res: dict) -> list[tuple[str, str]]:
    """(label, plain statement) computed from the numbers; the same rules apply whatever the data say."""
    out = []
    gl = res.get("game_level", {})
    d = gl.get("model_minus_close_log_loss")
    if d:
        lo, hi = d["ci"]
        if lo > 0:
            s = "The model's probabilities are WORSE than the market's closing probabilities (the interval is above zero)."
        elif hi < 0:
            s = "The model's probabilities are BETTER than the market's closing probabilities (the interval is below zero)."
        else:
            s = "No demonstrable difference from the market's closing probabilities yet (the interval includes zero)" + (", and the point estimate favours the market." if d["diff"] > 0 else ".")
        out.append((f"Model vs market, {gl['n']} games", f"log loss {d['diff']:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}). {s}"))
    pk = res.get("picks", {}).get("every_game")
    if pk and pk.get("n"):
        rb = res.get("random_baseline", {}).get("every_game", {})
        clv = pk.get("clv", {})
        out.append(("Moneyline picks, one per game", f"{pk['n']} picks, ROI {_pct(pk['roi'])} (95% interval {_ci(pk['roi_ci'])}), win rate {_p(pk['win_rate'])} against a break-even {_p(pk['break_even_win_rate'])}. "
                    f"Random sides at the same margin match or beat this {_p(rb.get('p_random_at_least_as_good'), 0)} of the time. "
                    f"Line movement toward the pick after it was made: {clv.get('mean_prob_points', float('nan')):+.2f} probability points on average (zero means no closing-line value)."))
        rq = res.get("required_bets", {}).get("roi", {})
        if rq:
            out.append(("How many bets before ROI can be trusted", f"At the observed spread of results, detecting a true 3% ROI takes about {rq.get('3%', 0):,} bets and a 5% ROI about {rq.get('5%', 0):,}; a 1% edge takes {rq.get('1%', 0):,}. "
                        "Closing-line value converges much faster, which is why it is the primary measure."))
    bm = res.get("by_market", {})
    for m, v in sorted(bm.items()):
        out.append((m.replace("_", " "), f"{v['picks']} picks, ROI {_pct(v['roi'])} ({_ci(v['roi_ci'])}): {v['verdict']}."))
    return out


def _market_table(rows: dict, title: str) -> str:
    if not rows:
        return ""
    body = "".join(
        f"<tr><td data-label='Market'><b>{e(m.replace('_', ' '))}</b></td><td class='num' data-label='Picks'>{v['picks']}</td><td class='num' data-label='Win rate'>{_p(v['win_rate'])}</td>"
        f"<td class='num {'good' if v['roi'] > 0 else 'bad'}' data-label='ROI'>{_pct(v['roi'])}</td><td class='num' data-label='95% interval'>{_ci(v['roi_ci'])}</td>"
        f"<td class='num' data-label='P(ROI > 0 by luck)'>{_f(v['p_roi_not_above_0'], 2)}</td><td class='num' data-label='Picks for a 5% ROI'>{v['picks_needed_for_5pct_roi']:,}</td><td data-label='Verdict'>{e(v['verdict'])}</td></tr>"
        for m, v in sorted(rows.items()))
    return (f"<h3>{e(title)}</h3><div class='card flat scroll'><table class='rtable'><thead><tr><th>Market</th><th class='num'>Picks</th><th class='num'>Win rate</th><th class='num'>ROI</th>"
            f"<th class='num'>95% interval</th><th class='num'>P(this good by luck)</th><th class='num'>Picks needed to confirm 5%</th><th>Verdict</th></tr></thead><tbody>{body}</tbody></table></div>")


def _split(rows: list[dict], key: str) -> str:
    body = "".join(f"<tr><td data-label='{e(key)}'>{e(str(r[key]))}</td><td class='num' data-label='Picks'>{int(r['n'])}</td><td class='num' data-label='Win rate'>{_p(r['win_rate'])}</td>"
                   f"<td class='num' data-label='ROI'>{_pct(r['roi'])}</td><td class='num' data-label='Avg edge'>{r['avg_edge_pts']:+.1f} pts</td></tr>" for r in rows)
    return f"<div class='card flat scroll'><table class='rtable'><thead><tr><th>{e(key)}</th><th class='num'>Picks</th><th class='num'>Win rate</th><th class='num'>ROI</th><th class='num'>Model edge claimed</th></tr></thead><tbody>{body}</tbody></table></div>"


def render_validation_html(res: dict, css: str, generated_at: str | None = None) -> str:
    gen = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    vs = "".join(f"<li><b>{e(a)}.</b> {e(b)}</li>" for a, b in verdicts(res))
    gl = res.get("game_level", {})
    models = gl.get("models", {})
    mrows = "".join(f"<tr><td data-label='Forecast'>{e(k.replace('_', ' '))}</td><td class='num' data-label='Log loss'>{_f(v['log_loss'])}</td><td class='num' data-label='Brier'>{_f(v['brier'])}</td><td class='num' data-label='Games'>{v['n']}</td></tr>"
                    for k, v in models.items())
    rel = "".join(f"<tr><td data-label='Model bin'>{e(r['bin'])}</td><td class='num' data-label='Games'>{r['n']}</td><td class='num' data-label='Model'>{_p(r['model'])}</td>"
                  f"<td class='num' data-label='Market close'>{_p(r['market_close'])}</td><td class='num' data-label='Actual home wins'>{_p(r['actual'])}</td></tr>" for r in gl.get("reliability", []))
    picks = res.get("picks", {})
    prow = "".join(
        f"<tr><td data-label='Strategy'><b>{e(k)}</b></td><td class='num' data-label='Picks'>{v.get('n', 0)}</td><td class='num' data-label='Win rate'>{_p(v.get('win_rate'))}</td><td class='num' data-label='Break-even'>{_p(v.get('break_even_win_rate'))}</td>"
        f"<td class='num' data-label='ROI'>{_pct(v.get('roi'))}</td><td class='num' data-label='95% interval'>{_ci(v.get('roi_ci'))}</td><td class='num' data-label='Units ($10 each)'>{v.get('units', 0):+.1f}</td>"
        f"<td class='num' data-label='Max drawdown (units)'>{v.get('max_drawdown_units', 0):.1f}</td><td class='num' data-label='Longest losing run'>{v.get('longest_losing_streak', 0)}</td>"
        f"<td class='num' data-label='Edge claimed vs no-vig'>{(v.get('mean_edge_vs_no_vig') or 0) * 100:+.1f} pts</td>"
        f"<td class='num' data-label='Moved our way (prob. pts)'>{v.get('clv', {}).get('mean_prob_points', float('nan')):+.2f}</td></tr>" for k, v in picks.items() if v.get("n"))
    splits = res.get("splits", {}).get("every_game", {})
    split_html = "".join(f"<h4>By {e(lbl)}</h4>{_split(splits[k], k)}" for k, lbl in (("month", "month"), ("side", "home / away"), ("role", "favourite / underdog"), ("odds_bucket", "odds bucket")) if splits.get(k))
    bt = res.get("backtest_by_season", {}).get("stack", [])
    btrows = "".join(f"<tr><td data-label='Season'>{e(r['period'])}</td><td class='num' data-label='Games'>{r['n']}</td><td class='num' data-label='Model'>{_f(r['stack'])}</td><td class='num' data-label='Elo'>{_f(r['elo'])}</td>"
                     f"<td class='num' data-label='Home rate'>{_f(r['home_rate'])}</td><td class='num' data-label='Model - Elo'>{r['stack-elo']:+.4f}</td><td class='num' data-label='95% interval'>{r['ci_low']:+.4f} to {r['ci_high']:+.4f}</td></tr>" for r in bt)
    li = res.get("log_integrity", {})
    irows = "".join(f"<tr><td data-label='Log'>{e(t)}</td><td class='num' data-label='Rows'>{v['rows']:,}</td><td class='num' data-label='Duplicates'>{v['duplicate_keys']}</td>"
                    f"<td class='num' data-label='Logged at or after puck drop'>{v.get('logged_at_or_after_puck_drop', '-')}</td></tr>" for t, v in li.items() if isinstance(v, dict))
    rec = li.get("recommendations", {})
    odds_note = (f"Odds were captured a median of {rec['odds_age_minutes_median']:.1f} minutes (max {rec['odds_age_minutes_max']:.1f}) before each run; {rec['odds_captured_after_run']} captured after the run; "
                 f"{rec['rows_without_model_version']} picks without a model version ({rec['model_versions']} versions in use).") if rec else ""
    method = [
        "The moneyline model is trained only on games before each prediction; features use only games with a strictly earlier date; tuning and feature selection stopped on 2024-06-30, before the evaluation window began (checked by a test).",
        "A pick is the model's side in a game at the best price available in the last odds capture before the game started. Picks are logged with a timestamp, the model version and the odds capture time; the logs are append-only files in the repository history.",
        "Moneylines settle on the final result including overtime and shootout; totals count overtime goals but not the shootout goal; the puck line uses the official margin; whole-number lines return the stake on a push; player props settle on the player's official stats and void if he does not play.",
        "Every paper bet is a flat $10, so profit and ROI read directly. Fake money is used so nothing here is ever a recommendation to stake.",
        "Intervals resample whole days (bets on one slate are not independent). The 'by luck' columns are one-sided tests against zero ROI; the random-pick baseline picks a random side in the same games at the same margin.",
    ]
    limits = [
        "There are no historical sportsbook prices in this project, so the long backtest scores probabilities against base rates, not against the market. Evidence about the market only exists from the day live odds capture began.",
        "Only one pre-game price capture exists per game for totals, puck lines and player props, so closing-line value is measurable for moneylines only.",
        "Samples are small: a few weeks of games. Intervals that include zero mean the data cannot yet tell skill from luck, not that skill is absent.",
        "Player props are priced before the starting lineup is confirmed; a bet on a player who does not play is void. Anytime-goalscorer market probabilities are estimated from one-sided prices with an assumed margin.",
        "The best price across books is used, which flatters results slightly compared with a bettor who has one account.",
    ]
    ul = lambda xs: "<ul>" + "".join(f"<li>{e(x)}</li>" for x in xs) + "</ul>"  # noqa: E731
    disc = e(DISCLAIMER.replace("> ", "").replace("**", ""))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NHL Model: Evidence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700;800&display=swap">
<style>{css}</style></head>
<body><header class="topbar"><div class="topbar-in"><a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"></span>NHL Model</a>
<nav class="nav" aria-label="Pages"><a href="index.html">Picks</a><a href="bets.html">Bets</a><a href="history.html">History</a><a href="validation.html" aria-current="page">Evidence</a></nav></div></header>
<main class="wrap"><div class="pagehead"><div class="stack"><h1>Is there an edge?</h1><div class="when">Updated {e(gen)} from games through {e(str(res.get('data_through', '-')))}. Live results only; the backtest is shown separately below.</div></div></div>
<div class="note">Research and education only. Not betting advice. Past performance does not guarantee future results.</div>
<section class="section"><h2>Plain verdicts</h2><div class="card"><ul>{vs}</ul></div></section>
<section class="section" id="live"><h2>Live record by market <span class="count">paper bets, flat $10</span></h2>
{_market_table(res.get('by_market', {}), "The model's picks")}{_market_table(res.get('controls_by_market', {}), "No-skill controls (what the bookmaker margin alone costs)")}</section>
<section class="section" id="moneyline"><h2>Moneyline picks in detail</h2><div class="card flat scroll"><table class="rtable"><thead><tr><th>Strategy</th><th class="num">Picks</th><th class="num">Win rate</th><th class="num">Break-even</th><th class="num">ROI</th><th class="num">95% interval</th><th class="num">Units</th><th class="num">Max drawdown</th><th class="num">Longest losing run</th><th class="num">Edge claimed</th><th class="num">Line moved our way</th></tr></thead><tbody>{prow}</tbody></table></div>
<div class="note">every_game = the model's side of every game; flat_model_side = only games where the model sees a positive edge; market_favorite = no-skill control. "Line moved our way" is the closing-line value in probability points; a real edge shows up here first.</div>{split_html}</section>
<section class="section" id="scoring"><h2>Model vs market probabilities <span class="count">{gl.get('n', 0)} games</span></h2><div class="card flat scroll"><table class="rtable"><thead><tr><th>Forecast</th><th class="num">Log loss</th><th class="num">Brier</th><th class="num">Games</th></tr></thead><tbody>{mrows}</tbody></table></div>
<div class="note">Lower is better. A model that beats the market must beat the market's own no-vig probability, not a coin flip (0.6931).</div>
<h3>Reliability</h3><div class="card flat scroll"><table class="rtable"><thead><tr><th>Model probability bin</th><th class="num">Games</th><th class="num">Model</th><th class="num">Market close</th><th class="num">Actual</th></tr></thead><tbody>{rel}</tbody></table></div></section>
<section class="section" id="backtest"><h2>Backtest (not live) <span class="count">no market prices exist for it</span></h2><div class="card flat scroll"><table class="rtable"><thead><tr><th>Season</th><th class="num">Games</th><th class="num">Model</th><th class="num">Elo</th><th class="num">Home rate</th><th class="num">Model minus Elo</th><th class="num">95% interval</th></tr></thead><tbody>{btrows}</tbody></table></div>
<div class="note">Walk-forward log loss by season. Skill versus a trivial baseline is not skill versus the betting market; reports/ABLATION.md, reports/model_comparison.md, GOALS.md and PROPS.md in the repository hold the full backtests.</div></section>
<section class="section" id="integrity"><h2>Are the logs honest?</h2><div class="card flat scroll"><table class="rtable"><thead><tr><th>Log</th><th class="num">Rows</th><th class="num">Duplicate rows</th><th class="num">Logged at or after puck drop</th></tr></thead><tbody>{irows}</tbody></table></div><div class="note">{e(odds_note)}</div></section>
<section class="section" id="method"><h2>Method</h2><div class="card">{ul(method)}</div></section>
<section class="section" id="limits"><h2>Known limitations</h2><div class="card">{ul(limits)}</div></section>
<footer class="footer"><div class="banner disc">{disc}</div></footer></main></body></html>"""
