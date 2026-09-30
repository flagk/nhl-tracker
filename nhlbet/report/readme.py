"""Keep an auto-updated 'today's picks' block at the top of the README (between markers) with clickable links."""
from __future__ import annotations

import os
from pathlib import Path

START, END = "<!-- PICKS:START -->", "<!-- PICKS:END -->"


def pages_url(env=None) -> str | None:
    env = os.environ if env is None else env
    repo = env.get("GITHUB_REPOSITORY")
    if not repo or "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{name}/"


def run_url(env=None) -> str | None:
    env = os.environ if env is None else env
    repo, rid = env.get("GITHUB_REPOSITORY"), env.get("GITHUB_RUN_ID")
    return f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{rid}" if repo and rid else None


def render_block(date: str, run_type: str, slate, model_status: str, generated: str, bets_today: int, env=None, bet_log_exists: bool | None = None) -> str:
    """``slate``: list of SlateGame-like objects (home, away, p_home, quotes, rec)."""
    pages = pages_url(env)
    if bet_log_exists is None:
        bet_log_exists = Path("data/logs/bet_log.csv").exists()
    L = [START, "", f"### Today's picks: {date}", "",
         f"**{len(slate)} game(s) · {bets_today} recommended bet(s)** · {run_type} run · model health **{model_status}** · updated {generated}", "",
         "| | |", "|---|---|",
         "| 📄 **[Today's full report](reports/latest.md)** | every game: model vs market, edge, stake and the reason for each decision |",
         f"| 🌐 **[Interactive picks page]({pages or 'site/index.html'})** | choose your unit size and staking, top picks, parlay calculator |",
         "| 🗂️ **[Pick history](reports/HISTORY.md)** | every day's picks and how they settled ([web version](" + (f"{pages}history.html" if pages else "site/history.html") + ")) |",
         f"| 🧾 **[My bets & fake bets]({(pages + 'bets.html') if pages else 'site/bets.html'})** | log your own bets (kept only in your browser) and see the pretend-money bets on every game |",
         ("| 📈 [Bet log (CSV)](data/logs/bet_log.csv) | all settled recommended bets |" if bet_log_exists
          else "| 📈 Bet log (CSV) | appears here after the first recommended bet settles |"), ""]
    if slate:
        L += ["| Game | Model: home win | Market (no-vig) | Decision |", "|---|---|---|---|"]
        for s in slate:
            mk = f"{s.quotes['home'].market_prob:.1%}" if s.quotes else "no odds"
            dec = f"**BET {s.rec.team}** ${s.rec.stake:,.2f}" if s.rec.action == "BET" else "no bet"
            L += [f"| {s.away} @ {s.home} | {s.p_home:.1%} | {mk} | {dec} |"]
        if not bets_today:
            L += ["", "*No bets today is normal: the model only bets when it sees a sizeable edge over the market.*"]
    else:
        L += ["*No NHL games to show for this date.*"]
    L += ["", "*Research and educational project; no guarantee of profit. The numbers above are not betting advice.*", "", END]
    return "\n".join(L)


def update_readme(readme: str | Path, block: str) -> bool:
    """Replace the marked block. Returns False (and changes nothing) if the README has no markers."""
    p = Path(readme)
    text = p.read_text()
    if START not in text or END not in text:
        return False
    a, b = text.index(START), text.index(END) + len(END)
    new = text[:a] + block + text[b:]
    if new != text:
        p.write_text(new)
    return True
