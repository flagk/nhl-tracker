"""Does the review agent help? Grade kept, flagged and vetoed picks exactly alike and compare them, on closing-line value first.

Every pick the agent saw is in the append-only logs (data/logs/review/final_picks.jsonl and vetoed_picks.jsonl), vetoed ones included, so they can be graded
after the games like the others: flat $10 paper stake at the logged price, settled by the same rules as every other fake bet (``betlog._leg_status``). The
question is whether picks the agent vetoed did credibly WORSE than picks it kept. Until each group has MIN_N graded picks the answer is "insufficient data";
if the sample is meaningful and vetoed picks did not do credibly worse, the report says the agent adds no demonstrated value. Odds-rule vetoes made in code are a
separate group: they are not the agent's doing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nhlbet.report.validation import cluster_bootstrap, required_n

MIN_N = 100                 # graded picks needed in EACH of the groups compared before any verdict
GROUPS = ["kept", "flagged", "vetoed_by_agent", "vetoed_by_code", "flagged_by_code", "unreviewed"]
FLAT = 10.0


def group_of(rec: dict) -> str:
    r = rec["review"]
    if r.get("status") == "UNREVIEWED" and r.get("decision") is None:
        return "unreviewed"
    by = r.get("decided_by")
    d = r.get("decision")
    if d == "veto":
        return "vetoed_by_code" if by == "code" else "vetoed_by_agent"
    if d == "flag":
        return "flagged_by_code" if by == "code" else "flagged"
    return "kept"


def load_log(root: str | Path = ".") -> pd.DataFrame:
    """One row per pick: the latest record in the logs (a later successful review of the same wave supersedes its UNREVIEWED line)."""
    rows = []
    for name in ("final_picks.jsonl", "vetoed_picks.jsonl"):
        p = Path(root) / "data" / "logs" / "review" / name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines()):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            rows.append({**{k: rec.get(k) for k in ("pick_id", "date", "game_id", "market", "line", "side", "label", "odds", "p_model", "p_market", "edge", "stake", "player_id")},
                         "group": group_of(rec), "logged_at": rec.get("logged_at"), "decision": rec["review"].get("decision"), "decided_by": rec["review"].get("decided_by"),
                         "reason": rec["review"].get("reason"), "run_id": rec["review"].get("run_id"), "reviewed_at": rec["review"].get("reviewed_at")})
    if not rows:
        return pd.DataFrame(columns=["pick_id"])
    d = pd.DataFrame(rows).sort_values("logged_at", kind="stable")
    return d.groupby("pick_id", as_index=False).tail(1).reset_index(drop=True)


def grade(d: pd.DataFrame, games: pd.DataFrame, sk: pd.DataFrame, cons: pd.DataFrame | None = None, start_utc: dict | None = None, quotes=None, method: str | None = None) -> pd.DataFrame:
    """Add ``result`` (won / lost / push / pending), ``profit`` at the flat stake, and the margin-free closing-line move in probability points (``clv_move_pts``).

    With ``quotes`` (an ``evaluation.Quotes`` of the raw per-book prices) the move is measured for moneylines, totals and puck lines from the capture at the review time to the capture
    nearest the start (within the policy's closing gap); without it, only moneylines can be measured, from the stored consensus."""
    from nhlbet.odds.clv import closing_consensus
    from nhlbet.report.betlog import _leg_status
    if d.empty:
        return d.assign(result=[], profit=[], clv_move_pts=[])
    res, clv = [], []
    from nhlbet.policy import policy_get
    method = method or policy_get("evaluation", "devig", "primary", default="proportional")
    gap = float(policy_get("evaluation", "max_close_gap_min", default=60.0))
    teams = {}
    if quotes is not None and games is not None and len(games):
        teams = games.set_index("game_id")[["home", "away"]].to_dict("index")
    for r in d.itertuples():
        leg = {"game_id": int(r.game_id), "market": r.market, "side": r.side, "point": None if pd.isna(r.line) else float(r.line), "player_id": int(r.player_id or 0)}
        res.append(_leg_status(leg, games, sk))
        move = np.nan
        mk = {"h2h": "h2h", "totals": "totals", "spreads": "spreads"}.get(r.market)
        if quotes is not None and mk and int(r.game_id) in teams and start_utc and r.game_id in start_utc and pd.notna(getattr(r, "reviewed_at", None)):
            from nhlbet.report.evaluation import leg_pairs, leg_prob, prob_at
            t = teams[int(r.game_id)]
            start = pd.Timestamp(start_utc[r.game_id])
            p0, _, _, t0 = prob_at(quotes, r.game_id, t["home"], t["away"], mk, r.side, None if pd.isna(r.line) else float(r.line), min(pd.Timestamp(r.reviewed_at), start), method)
            cl = quotes.capture(r.game_id, mk, start, after=t0)
            if cl is not None and (start - cl.cap.iloc[0]).total_seconds() / 60 <= gap and not np.isnan(p0):
                p1 = leg_prob(leg_pairs(cl, t["home"], t["away"], mk, r.side, None if pd.isna(r.line) else float(r.line)), method)[0]
                move = 100.0 * (p1 - p0)
        elif quotes is None and r.market == "h2h" and cons is not None and len(cons) and start_utc and r.game_id in start_utc:
            ch = closing_consensus(cons, int(r.game_id), start_utc[r.game_id])
            if ch is not None:
                p_close = ch if r.side == "home" else 1 - ch
                move = (p_close - r.p_market) * 100
        clv.append(move)
    out = d.copy()
    out["result"] = res
    stake = out.stake.fillna(FLAT).astype(float)
    out["profit"] = np.where(out.result == "won", stake * (out.odds - 1), np.where(out.result == "lost", -stake, 0.0))
    out["clv_move_pts"] = clv
    return out


def _summ(g: pd.DataFrame) -> dict:
    gr = g[g.result.isin(["won", "lost", "push"])]
    bets = gr[gr.result != "push"]                                  # a push refunds the stake: it is neither a win nor a loss and carries no risk
    n = len(bets)
    out = {"picks_logged": int(len(g)), "graded": int(n), "pushes": int((gr.result == "push").sum()), "pending": int((g.result == "pending").sum())}
    if n == 0:
        return out
    stake = bets.stake.fillna(FLAT).astype(float).to_numpy()
    r = (bets.profit.to_numpy(float) / stake)
    lo, hi = cluster_bootstrap(r, bets.date.to_numpy(), np.mean)
    wins = int((bets.result == "won").sum())
    from scipy.stats import beta
    wlo = float(beta.ppf(0.025, wins, n - wins + 1)) if wins else 0.0
    whi = float(beta.ppf(0.975, wins + 1, n - wins)) if wins < n else 1.0
    out.update({"win_rate": wins / n, "win_rate_ci": [wlo, whi], "roi": float(r.mean()), "roi_ci": [lo, hi], "units": float(bets.profit.sum() / FLAT),
                "mean_edge_claimed": float(bets.edge.mean()), "break_even_win_rate": float(np.mean(1 / bets.odds)), "sd_return": float(r.std(ddof=1)) if n > 1 else float("nan")})
    c = bets.clv_move_pts.dropna().to_numpy(float)
    if len(c):
        clo, chi = cluster_bootstrap(c, bets.loc[bets.clv_move_pts.notna(), "date"].to_numpy(), np.mean)
        out["clv_move_pts"] = {"n": int(len(c)), "mean": float(c.mean()), "ci": [clo, chi], "share_positive": float((c > 0).mean())}
    return out


def compare(graded: pd.DataFrame, a: str = "kept", b: str = "vetoed_by_agent", B: int = 4000, seed: int = 0) -> dict:
    """ROI(a) - ROI(b) with a day-clustered bootstrap, and the plain verdict. Positive = what the agent kept did better than what it vetoed."""
    A = graded[(graded.group == a) & graded.result.isin(["won", "lost"])]
    Bv = graded[(graded.group == b) & graded.result.isin(["won", "lost"])]
    out = {"a": a, "b": b, "n_a": int(len(A)), "n_b": int(len(Bv)), "min_n": MIN_N}
    ra = (A.profit / A.stake.fillna(FLAT)).to_numpy(float)
    rb = (Bv.profit / Bv.stake.fillna(FLAT)).to_numpy(float)
    if len(A) and len(Bv):
        da, db = A.date.to_numpy(), Bv.date.to_numpy()
        rng = np.random.default_rng(seed)
        days = np.unique(np.concatenate([da, db]))
        ia = {x: np.flatnonzero(da == x) for x in days}
        ib = {x: np.flatnonzero(db == x) for x in days}
        diffs = []
        for _ in range(B):
            pick = days[rng.integers(0, len(days), len(days))]
            xa = np.concatenate([ra[ia[x]] for x in pick]) if any(len(ia[x]) for x in pick) else np.array([])
            xb = np.concatenate([rb[ib[x]] for x in pick]) if any(len(ib[x]) for x in pick) else np.array([])
            if len(xa) and len(xb):
                diffs.append(xa.mean() - xb.mean())
        out["roi_a"], out["roi_b"] = float(ra.mean()), float(rb.mean())
        out["diff"] = float(ra.mean() - rb.mean())
        out["diff_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else [float("nan")] * 2
        sd = float(np.sqrt((ra.var(ddof=1) if len(ra) > 1 else 0) + (rb.var(ddof=1) if len(rb) > 1 else 0)))
        out["picks_needed_per_group_for_5pct_gap"] = required_n(0.05, sd / np.sqrt(2)) if sd > 0 else None
    if len(A) < MIN_N or len(Bv) < MIN_N:
        out["verdict"] = f"insufficient data: {len(A)} {a} and {len(Bv)} {b} graded picks so far; at least {MIN_N} in each are needed before this means anything"
    elif out["diff_ci"][0] > 0:
        out["verdict"] = "picks the agent vetoed did credibly worse than picks it kept (interval above 0). This is observational: it suggests the vetoes carry information, it does not prove the agent causes the difference."
    else:
        out["verdict"] = "the agent adds no demonstrated value: picks it vetoed did not do credibly worse than picks it kept"
    return out


def compare_clv(graded: pd.DataFrame, a: str = "kept", b: str = "vetoed_by_agent", B: int = 4000, n_perm: int = 20000, seed: int = 0) -> dict:
    """Closing-line value of group ``a`` minus group ``b`` in probability points (positive = what the agent kept moved toward the pick more than what it vetoed).

    CLV needs no results, so it speaks sooner than outcomes. Reported: the difference with a date-clustered bootstrap interval, and a one-sided permutation p-value that shuffles
    the kept / vetoed labels WITHIN each game date (so day effects cannot create a difference; Monte Carlo with ``n_perm`` shuffles). The verdict stays "insufficient data" until
    the policy minimum of CLV observations exists in both groups; it never changes the keep rule."""
    from nhlbet.policy import policy_get
    min_a, min_b = int(policy_get("review", "min_kept_for_verdict", default=100)), int(policy_get("review", "min_vetoed_for_verdict", default=30))
    A = graded[(graded.group == a) & graded.clv_move_pts.notna()]
    Bv = graded[(graded.group == b) & graded.clv_move_pts.notna()]
    out = {"a": a, "b": b, "n_a": int(len(A)), "n_b": int(len(Bv)), "min_a": min_a, "min_b": min_b}
    if len(A) and len(Bv):
        xa, xb = A.clv_move_pts.to_numpy(float), Bv.clv_move_pts.to_numpy(float)
        da, db = A.date.to_numpy(), Bv.date.to_numpy()
        out["mean_a"], out["mean_b"], out["diff"] = float(xa.mean()), float(xb.mean()), float(xa.mean() - xb.mean())
        rng = np.random.default_rng(seed)
        days = np.unique(np.concatenate([da, db]))
        ia = {x: np.flatnonzero(da == x) for x in days}
        ib = {x: np.flatnonzero(db == x) for x in days}
        diffs = []
        for _ in range(B):
            pick = days[rng.integers(0, len(days), len(days))]
            ua = np.concatenate([xa[ia[x]] for x in pick]); ub = np.concatenate([xb[ib[x]] for x in pick])
            if len(ua) and len(ub):
                diffs.append(ua.mean() - ub.mean())
        out["diff_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else [float("nan")] * 2
        both = [x for x in days if len(ia[x]) and len(ib[x])]                   # permute labels within the dates that have both groups
        obs = float(xa.mean() - xb.mean())
        cnt = 0
        for _ in range(n_perm):
            ma, mb = [], []
            for x in both:
                v = np.concatenate([xa[ia[x]], xb[ib[x]]]); rng.shuffle(v)
                ma.append(v[:len(ia[x])]); mb.append(v[len(ia[x]):])
            rest_a = np.concatenate([xa[ia[x]] for x in days if x not in both] or [np.array([])]); rest_b = np.concatenate([xb[ib[x]] for x in days if x not in both] or [np.array([])])
            pa, pb = np.concatenate(ma + [rest_a]), np.concatenate(mb + [rest_b])
            cnt += (pa.mean() - pb.mean()) >= obs
            if cnt > 50 and _ > 2000:
                n_perm = _ + 1
                break
        out["p_permutation_one_sided"] = float((cnt + 1) / (n_perm + 1))
    if len(A) < min_a or len(Bv) < min_b:
        out["verdict"] = (f"insufficient data: {len(A)} {a} and {len(Bv)} {b} picks with a closing-line value; at least {min_a} and {min_b} are needed before this is read")
    elif out["diff_ci"][0] > 0 and out["p_permutation_one_sided"] < float(policy_get("evaluation", "alpha", default=0.05)):
        out["verdict"] = "picks the agent vetoed moved toward the closing line credibly LESS than the ones it kept: the vetoes carry information (observational, not proof of cause)"
    else:
        out["verdict"] = "the agent adds no demonstrated value on closing-line value: vetoed picks did not do credibly worse than kept picks"
    return out


def flag_rate_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per review date: picks reviewed, how many were kept / flagged / vetoed (by the agent) / vetoed or flagged by code / unreviewed, and the flag rate. Reported, never used to tune anything."""
    if d.empty or "date" not in d:
        return pd.DataFrame()
    t = d.groupby(["date", "group"]).size().unstack(fill_value=0)
    for g in GROUPS:
        if g not in t:
            t[g] = 0
    t["picks"] = t[GROUPS].sum(axis=1)
    t["flag_rate"] = t["flagged"] / t["picks"]
    t["veto_rate"] = t["vetoed_by_agent"] / t["picks"]
    return t[["picks"] + GROUPS + ["flag_rate", "veto_rate"]].reset_index()


def report(graded: pd.DataFrame) -> dict:
    groups = {g: _summ(graded[graded.group == g]) for g in GROUPS if (graded.group == g).any()} if len(graded) else {}
    empty = pd.DataFrame(columns=["group", "result", "profit", "stake", "date", "clv_move_pts"])
    fr = flag_rate_table(graded) if len(graded) else pd.DataFrame()
    return {"groups": groups, "clv_kept_vs_vetoed": compare_clv(graded if len(graded) else empty, "kept", "vetoed_by_agent"),
            "clv_kept_vs_flagged": compare_clv(graded if len(graded) else empty, "kept", "flagged"),
            "kept_vs_vetoed": compare(graded, "kept", "vetoed_by_agent") if len(graded) else compare(empty, "kept", "vetoed_by_agent"),
            "flagged_vs_kept": compare(graded, "kept", "flagged") if len(graded) else None, "n_logged": int(len(graded)),
            "flag_rate": {"days": int(len(fr)), "mean": float(fr.flag_rate.mean()) if len(fr) else None, "median": float(fr.flag_rate.median()) if len(fr) else None,
                          "max": float(fr.flag_rate.max()) if len(fr) else None, "by_day": fr.to_dict("records") if len(fr) else []}}


def render_md(rep: dict) -> str:
    def f(x, p=1):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{p}f}%"

    def ci(v):
        return "-" if not v else f"{f(v[0])} to {f(v[1])}"
    lines = ["# Pre-game review: does the agent help?", "",
             "> Research and education only. Every pick the review agent saw is graded here after its game, whatever the agent decided: flat $10 paper stake at the logged price, "
             "the same settlement rules as all other fake bets. Closing-line value (CLV) is the primary comparison because it needs no results: "
             "the margin-free move of the no-vig line from the review time to the close toward the pick, in probability points, for moneylines, totals and puck lines where a closing price was captured. "
             f"Outcome comparisons are secondary and need {MIN_N} graded picks per group.", ""]
    if not rep["groups"]:
        return "\n".join(lines + ["No reviewed picks have been graded yet. **Insufficient data.**", ""])
    lines += ["| group | logged | graded | pushes | pending | win rate (95% CI) | ROI (95% CI) | units | claimed edge | closing-line move, pts (n) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in rep["groups"].items():
        c = s.get("clv_move_pts")
        lines.append(f"| {g} | {s['picks_logged']} | {s['graded']} | {s['pushes']} | {s['pending']} | " + (f"{f(s['win_rate'])} ({ci(s['win_rate_ci'])}) | {f(s['roi'])} ({ci(s['roi_ci'])}) | {s['units']:+.1f} | {f(s['mean_edge_claimed'])} | "
                 + (f"{c['mean']:+.2f} ({c['n']})" if c else "-") if s["graded"] else "- | - | - | - | -") + " |")
    lines += ["", "## Primary: closing-line value, kept vs vetoed (and vs flagged)", ""]
    for key, title in (("clv_kept_vs_vetoed", "kept minus vetoed (agent)"), ("clv_kept_vs_flagged", "kept minus flagged")):
        c = rep.get(key)
        if c:
            lines += [f"**{title}:** {c['n_a']} {c['a']} vs {c['n_b']} {c['b']} picks with a closing-line value." + (f" Mean {c['mean_a']:+.3f} vs {c['mean_b']:+.3f} points; difference {c['diff']:+.3f} (95% CI {c['diff_ci'][0]:+.3f} to {c['diff_ci'][1]:+.3f}); "
                      f"permutation p={c['p_permutation_one_sided']:.3f} (one-sided, labels shuffled within each date)." if "diff" in c else ""), "", f"> {c['verdict']}", ""]
    fr = rep.get("flag_rate") or {}
    if fr.get("days"):
        lines += [f"**Flag rate per review day** ({fr['days']} days): mean {fr['mean'] * 100:.0f}%, median {fr['median'] * 100:.0f}%, maximum {fr['max'] * 100:.0f}% of reviewed picks. "
                  "Reported so the owner can decide; the keep rule is never changed because of it.", ""]
    lines += ["", "## Secondary: results, kept vs vetoed by the agent", ""]
    for key, title in (("kept_vs_vetoed", "kept minus vetoed (agent)"), ("flagged_vs_kept", "kept minus flagged")):
        c = rep.get(key)
        if not c:
            continue
        lines += [f"**{title}:** {c['n_a']} {c['a']} vs {c['n_b']} {c['b']} graded picks." + (f" ROI {f(c.get('roi_a'))} vs {f(c.get('roi_b'))}; difference {f(c['diff'])} (95% CI {ci(c['diff_ci'])})."
                  if "diff" in c else ""), "", f"> {c['verdict']}", ""]
        if c.get("picks_needed_per_group_for_5pct_gap"):
            lines += [f"Picks needed per group to detect a 5-point ROI gap at the observed variance: about {c['picks_needed_per_group_for_5pct_gap']:,}.", ""]
    lines += ["Groups: `kept` / `flagged` / `vetoed_by_agent` are the agent's decisions; `vetoed_by_code` and `flagged_by_code` come from the deterministic odds rule (edge below the minimum after "
              "refreshing prices, or the line gone) and say nothing about the agent; `unreviewed` picks went out without a review because the agent step failed.", ""]
    return "\n".join(lines)


def write_review_report(store, root: str | Path = ".", report_dir: str | Path = "reports") -> dict | None:
    """Grade the logged reviewed picks from the database and write reports/REVIEW.md and REVIEW.json. Returns None when there is no review log yet."""
    d = load_log(root)
    if d.empty or "game_id" not in d:
        return None
    from nhlbet.odds.consensus import consensus_snapshots
    games = store.df("SELECT game_id, home, away, home_score, away_score, last_period, home_win, start_utc FROM games")
    sk = store.df("SELECT game_id, player_id, sog, points, goals, assists FROM skater_game WHERE sog IS NOT NULL")
    starts = dict(zip(games.game_id, games.start_utc))
    from nhlbet.report.evaluation import Quotes
    q = Quotes(store.df("SELECT captured_at, game_id, home, away, book, market, outcome, point, price FROM odds_snapshots WHERE market IN ('h2h','totals','spreads')"))
    g = grade(d, games, sk, consensus_snapshots(store), starts, quotes=q)
    rep = report(g)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir, "REVIEW.json").write_text(json.dumps(rep, indent=1, allow_nan=False, default=lambda x: None))
    Path(report_dir, "REVIEW.md").write_text(render_md(rep) + "\n")
    return rep
