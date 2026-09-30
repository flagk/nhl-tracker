"""Render the daily report (markdown)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nhlbet.report.slate import SlateGame
from nhlbet.risk.policy import RiskConfig

DISCLAIMER = ("> **Disclaimer.** This is a research and educational project. No model guarantees profit, and past or "
              "back-tested results do not predict future results. Sports betting carries a real risk of loss; only stake money you "
              "can afford to lose, and check that betting is legal where you live. Nothing here is financial advice.")


def _pct(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def _signed(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def _matchup(s: SlateGame) -> str:
    t = "" if s.start_utc is None or pd.isna(s.start_utc) else f" ({pd.Timestamp(s.start_utc).tz_convert('America/New_York'):%H:%M} ET)"
    return f"{s.away} @ {s.home}{t}"


def _goalies(s: SlateGame) -> str:
    if s.goalie_status == "unknown":
        return "unknown"
    return f"{s.away_goalie} / {s.home_goalie} ({s.goalie_status})"


def _best_line(s: SlateGame) -> tuple[str, str, str, str]:
    """(price text, market no-vig home %, edge, ev) for the recommended side, else the side with the higher edge."""
    if not s.quotes:
        return "no odds", "-", "-", "-"
    q = s.quotes[s.rec.side] if s.rec.side else max(s.quotes.values(), key=lambda q: q.edge)
    mk = s.quotes["home"].market_prob
    # EV here uses the raw model probability; only show it for actual bets (for a pass it would read as an unkept promise)
    return f"{q.team} {q.best_decimal:.2f} @ {q.best_book}", _pct(mk), _signed(q.edge), (_signed(q.ev) if s.rec.action == "BET" else "-")


def render_report(date: str, run_type: str, slate: list[SlateGame], cfg: RiskConfig, model_version: str, model_meta: dict,
                  perf: dict | None = None, odds_meta: dict | None = None, now: datetime | None = None, shadow=None) -> str:
    now = now or datetime.now(timezone.utc)
    bets = [s for s in slate if s.rec.action == "BET"]
    staked = sum(s.rec.stake for s in bets)
    L = [f"# NHL model report: {date} ({run_type} run)", "", DISCLAIMER, ""]
    drift = model_meta.get("drift", {})
    status = drift.get("status", "UNKNOWN")
    L += [f"*Generated {now:%Y-%m-%d %H:%M} UTC · model `{model_version}` · probability source `{model_meta.get('p_source', '?')}` · "
          f"bankroll ${cfg.bankroll:,.2f} · quarter-Kelly x{cfg.kelly_fraction / 0.25:.2g}, per-bet cap {cfg.max_bet_pct:.0%}, daily cap {cfg.max_daily_exposure_pct:.0%}*", ""]
    if status in ("WARN", "ALERT"):
        reasons = list(drift.get("performance", {}).get("reasons", []))
        flagged = drift.get("features", {}).get("flagged", {})
        if flagged:
            reasons.append("feature drift: " + ", ".join(flagged))
        tail = " Recommendations are suspended." if status == "ALERT" else " Treat picks with extra scepticism."
        L += [f"> **Model health: {status}.** {'; '.join(reasons) or 'see reports/drift.json'}.{tail}", ""]
    if odds_meta:
        stale = " (STALE)" if odds_meta.get("stale") else ""
        credits = "" if odds_meta.get("remaining") is None else f" · API credits left: {odds_meta['remaining']}"
        L += [f"*Odds snapshot: {odds_meta.get('captured_at', 'none')}{stale}{credits}*", ""]
    L += ["## Summary", ""]
    if not slate:
        L += ["No NHL games are scheduled (or none are unplayed) for this date.", ""]
    else:
        L += [f"- {len(slate)} games · **{len(bets)} recommended bet(s)** · total stake ${staked:,.2f} ({staked / cfg.bankroll:.2%} of bankroll)"]
        if not bets:
            L += ["- **No bets today.** Passing is the normal, expected outcome: the model only bets when it sees a sizeable, "
                  "not-too-good-to-be-true edge over the market's no-vig price."]
        L += [""]
        L += ["## Games", "", "| Game | Goalies (away / home) | Model home win | Market no-vig home | Best price | Edge | EV per $1 | Stake | Decision |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in slate:
            price, mk, edge, ev = _best_line(s)
            dec = f"**BET {s.rec.team}**" if s.rec.action == "BET" else "no bet"
            L += [f"| {_matchup(s)} | {_goalies(s)} | {_pct(s.p_home)} | {mk} | {price} | {edge} | {ev} | "
                  f"{'$%.2f' % s.rec.stake if s.rec.action == 'BET' else '-'} | {dec} |"]
        L += ["", "## Why", ""]
        for s in slate:
            L += [f"**{_matchup(s)}**: {s.rec.explain()}"]
            if s.notes:
                L += ["  - Context: " + " ".join(s.notes)]
            if s.odds_stale:
                L += ["  - Odds for this game are stale; no bet is placed on stale prices."]
            L += [""]
    L += ["## Track record (all logged recommendations that have resolved)", ""]
    L += _perf_section(perf)
    L += _shadow_section(shadow)
    L += ["", "---", DISCLAIMER, ""]
    return "\n".join(L)


def _perf_section(perf: dict | None) -> list[str]:
    if not perf or not perf.get("resolved_games"):
        return ["No resolved recommendations yet. Odds snapshots and recommendations are logged from the first run so this table fills in "
                "automatically; **judge the model by closing-line value and log loss vs the market, not by early profit.**"]
    b, m = perf["bets"], perf["model"]
    out = [f"- Resolved games: **{perf['resolved_games']}** (bet on {b['n']}, passed on {perf['no_bet_rate']:.0%}); "
           f"model log loss {m['log_loss']:.4f}, Brier {m['brier']:.4f} (coin flip: 0.6931 / 0.2500)"]
    if "vs_market" in perf:
        v = perf["vs_market"]
        out += [f"- **Model vs market close** ({v['n']} games with odds): model log loss {v['model_log_loss']:.4f} vs market {v['market_close_log_loss']:.4f} "
                f"({'model better' if v['model_log_loss'] < v['market_close_log_loss'] else 'market better'})"]
    if b["n"]:
        c = perf["clv"]
        out += [f"- Bets: {b['n']} · staked ${b['staked']:,.2f} · profit ${b['profit']:+,.2f} · **ROI {b['roi']:+.1%}** · win rate {b['win_rate']:.1%} · avg price {b['avg_decimal']:.2f}",
                f"- Bankroll ${perf['bankroll']:,.2f} · max drawdown {perf['drawdown']['max_drawdown']:.1%} · current drawdown {perf['drawdown']['current_drawdown']:.1%}"]
        if c.get("n", 0) >= 5:
            out += [f"- **CLV**: mean {c['mean']:+.2%} per $1 (95% CI {c['ci'][0]:+.2%} to {c['ci'][1]:+.2%}), beat the close on {c['beat_close']:.0%} of {c['n']} bets"]
        if b["n"] < 500:
            out += [f"- ⚠️ {b['n']} bets is far too few to tell skill from luck (detecting a true 3% ROI takes ~6,900 bets). Watch CLV."]
    if "calibration" in perf:
        out += ["", "| Predicted | Observed | Games |", "|---|---|---|"] + [f"| {r['mean_p']:.1%} | {r['obs']:.1%} | {int(r['n'])} |" for r in perf["calibration"]]
    return out


def _shadow_section(shadow) -> list[str]:
    from nhlbet.risk.shadow import STRATEGIES
    L = ["", "## Paper trading (fake money, for measurement)", "",
         "Every slate is also run through several alternative strategies with pretend stakes. They never affect real recommendations; "
         "they exist to learn what works faster than the selective live policy can. **`market_favorite` is a no-skill control**: a strategy "
         "only means something if it beats it by more than the noise."]
    if shadow is None or len(shadow) == 0:
        return L + ["", "No settled paper bets yet."]
    desc = {s.name: s.description for s in STRATEGIES}
    L += ["", "| Strategy | Bets | ROI | 95% CI | Win rate | Avg CLV/$1 | What it tests |", "|---|---|---|---|---|---|---|"]
    for name, r in shadow.iterrows():
        roi = "-" if r.bets == 0 else f"{r.roi:+.1%}"
        ci = "-" if r.roi_lo != r.roi_lo else f"{r.roi_lo:+.0%} to {r.roi_hi:+.0%}"
        wr = "-" if r.bets == 0 else f"{r.win_rate:.0%}"
        clv = "-" if r.n_clv == 0 else f"{r.avg_clv:+.2%}"
        L += [f"| `{name}` | {int(r.bets)} | {roi} | {ci} | {wr} | {clv} | {desc.get(name, '')} |"]
    small = int(shadow.bets.max()) < 100
    if small:
        L += ["", "*Samples are still small: ROI over fewer than ~100 bets is mostly luck. Compare strategies on CLV and against the control, and wait for volume.*"]
    return L
