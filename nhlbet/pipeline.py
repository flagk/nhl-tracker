"""The daily job: restore logs -> refresh data -> retrain if needed -> fetch odds -> settle -> slate -> report -> export logs.

Every stage degrades gracefully and says so in the report: a failed data refresh uses the cached database, a failed odds
fetch falls back to a flagged-stale snapshot or 'no odds -> no bet', a missing API key disables odds without crashing.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from nhlbet.data.goalies import load_confirmations
from nhlbet.data.ingest import current_season, import_legacy_csv, ingest_day, ingest_details, ingest_schedule, reparse_skaters
from nhlbet.data.client import NHLClient
from nhlbet.data.store import Store
from nhlbet.hygiene import data_freshness, purge_inplay, validate_inputs
from nhlbet.models.bundle import ModelBundle
from nhlbet.odds.client import OddsAPIError, OddsClient, OddsConfigError
from nhlbet.odds.snapshots import record_fetch
from nhlbet.registry import ModelRegistry
from nhlbet.report.export import export_dataset
from nhlbet.report.betlog import export_logs, performance, plot_performance, restore_logs, shadow_performance
from nhlbet.report.history import render_history_html, render_history_md
from nhlbet.report.markdown import render_report
from nhlbet.report.readme import render_block, update_readme
from nhlbet.report.slate import build_slate
from nhlbet.site.bets import build_bets_page
from nhlbet.site.build import build_payload, build_site
from nhlbet.risk.policy import RiskConfig
from nhlbet.train import retrain

log = logging.getLogger("nhlbet.pipeline")
LEGACY_CSV = Path("data/history/nhl_history.csv")


def game_day(now: datetime | None = None) -> str:
    """The NHL 'game day' is the US/Eastern calendar date."""
    return (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def refresh_data(store: Store, days_back: int = 7, days_ahead: int = 2) -> str | None:
    """Incrementally update schedule + finals. Returns an error string on failure (never raises)."""
    try:
        from datetime import timedelta, date as _d
        c = NHLClient()
        ingest_schedule(c, store, current_season())
        for k in range(-days_back, days_ahead + 1):
            ingest_day(c, store, (_d.today() + timedelta(days=k)).isoformat())
        ok, bad = ingest_details(c, store)
        log.info("data refresh: %d new games ingested, %d failed", ok, bad)
        reparse_skaters(c, store, limit=2500)          # one-off catch-up for shots on goal; a no-op once every game has them
        return None if (ok or not bad) else f"{bad} game detail fetches failed"
    except Exception as e:  # noqa: BLE001 - stage must not take the job down
        log.error("data refresh failed: %s", e)
        return str(e)


DEFAULT_MARKETS = ("h2h", "spreads", "totals")      # 3 credits per fetch
MORNING_MARKETS = ("h2h",)                          # the morning run only needs moneylines: spreads, totals and props are priced closer to game time (late run)
from nhlbet.policy import policy_get as _odds_pol                  # credit limits live in config/policy.yaml (``odds:``)

PROPS_MAX_GAMES = int(_odds_pol("odds", "props_max_games", default=3))                           # props cost 1 credit per market per game
EXTRA_PROPS_MIN_CREDITS = int(_odds_pol("odds", "extra_props_min_credits", default=250))         # assists / anytime-goalscorer prices only while at least this many credits remain
PROPS_MIN_CREDITS = int(_odds_pol("odds", "props_min_credits", default=100))                     # below this no player prices are fetched at all: game moneylines come first
CREDIT_RESERVE = int(_odds_pol("odds", "reserve_credits", default=25))                           # never plan to spend the last credits: they keep the moneylines alive
CORE_AFTER_LATE = int(_odds_pol("odds", "core_after_late", default=4))                           # credits the closing-line snapshots still need after the late run
PROP_REFETCH_MIN = int(_odds_pol("odds", "prop_refetch_min", default=60))                        # a game's player market fetched less than this long ago is not fetched again


def credits_spent_today(store: Store, now: datetime) -> int:
    """Odds credits used since 00:00 UTC today, from the quota counter the API reports with every live call."""
    day = now.strftime("%Y-%m-%d")
    cur = store.df("SELECT MAX(used) AS u FROM odds_fetch_log WHERE source = 'live'")
    prev = store.df("SELECT MAX(used) AS u FROM odds_fetch_log WHERE source = 'live' AND captured_at < ?", [day])
    if cur.empty or cur.u.iloc[0] != cur.u.iloc[0] or cur.u.iloc[0] is None:
        return 0
    base = prev.u.iloc[0] if len(prev) and prev.u.iloc[0] == prev.u.iloc[0] and prev.u.iloc[0] is not None else None
    if base is None:                                    # no history before today: count from the first call today
        first = store.df("SELECT MIN(used) AS u FROM odds_fetch_log WHERE source = 'live' AND captured_at >= ?", [day])
        base = (first.u.iloc[0] - 1) if len(first) and first.u.iloc[0] == first.u.iloc[0] else cur.u.iloc[0]
    return int(max(0, cur.u.iloc[0] - base))


def props_budget(store: Store, remaining: int | None, now: datetime) -> int | None:
    """How many credits player prices may spend today so the whole month lasts.

    The plan is assumed to reset on the 1st: spread what is left (less a reserve) evenly over the days left, subtract what today has already cost and
    what the closing snapshots still need; whatever remains is the player-price budget. ``None`` = unknown quota (no limit applied).
    """
    if remaining is None:
        return None
    nxt = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
    days_left = max(1, (nxt.date() - now.date()).days)
    allowance = max(0, remaining - CREDIT_RESERVE) / days_left
    return max(0, int(allowance - credits_spent_today(store, now) - CORE_AFTER_LATE))


def _recently_fetched(store: Store, event_id: str, market: str, now: datetime, minutes: float = PROP_REFETCH_MIN) -> bool:
    r = store.df("SELECT MAX(captured_at) AS c FROM odds_snapshots WHERE event_id = ? AND market = ?", [event_id, market])
    if r.empty or r.c.iloc[0] is None:
        return False
    return (now - datetime.fromisoformat(str(r.c.iloc[0]).replace("Z", "+00:00"))) < timedelta(minutes=minutes)


def fetch_player_props(store: Store, client: OddsClient, events: list[dict], max_games: int, now: datetime | None = None, regions: str = "us",
                       remaining: int | None = None, budget: int | None = None) -> dict:
    """Player-prop prices (shots, points, assists, anytime goalscorer) for the next few games to start (one credit per market per game).

    One request per market, so a market the plan lacks fails alone. Only live captures are stored: a stale cached answer would put old lines on the
    site. Never raises: a plan without prop access just reports why.
    """
    from nhlbet.odds.props import STATS
    now = now or datetime.now(timezone.utc)
    env = os.environ.get("ODDS_PROP_MARKETS")
    markets = [m.strip() for m in env.split(",") if m.strip()] if env else [v["odds"] for k, v in STATS.items() if not v.get("extra") or remaining is None or remaining >= EXTRA_PROPS_MIN_CREDITS]
    if remaining is not None and remaining < PROPS_MIN_CREDITS:
        return {"games": 0, "error": f"only {remaining} odds credits left: player prices skipped to protect moneylines"}
    upcoming = sorted((e for e in events if e.get("commence_time") and datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > now
                       and datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) < now + timedelta(hours=6)), key=lambda e: e["commence_time"])[:max_games]
    got, errors, ok_markets, spent, skipped = 0, [], set(), 0, 0
    for e in upcoming:
        game_ok = False
        for m in markets:
            if _recently_fetched(store, e["id"], m, now):
                continue                                         # already priced within the last hour: the same credits would buy the same lines
            if budget is not None and spent >= budget:
                skipped += 1
                continue                                         # today's player-price budget is used up
            spent += 1
            try:
                f = client.fetch_event_odds(e["id"], (m,), regions)
            except (OddsAPIError, OddsConfigError) as err:
                log.warning("player props (%s) unavailable: %s", m, err)
                errors.append(f"{m}: {err}")
                continue
            if f.stale:
                errors.append(f"{m}: only an old cached answer was available, not used")
                continue
            record_fetch(store, f, m)
            ok_markets.add(m)
            game_ok = True
        got += int(game_ok)
    out = {"games": got, "markets": sorted(ok_markets), "credits": spent}
    if skipped:
        out["skipped_for_budget"] = skipped
        out["note"] = "Some player prices were skipped to make the monthly odds credits last."
    if errors and not got:
        out["error"] = errors[0]
    elif errors:
        out["warnings"] = errors[:4]
    return out


def fetch_and_store_odds(store: Store, markets: tuple[str, ...] | None = None, regions: str = "us", run_type: str = "late") -> dict:
    markets = markets or tuple(m.strip() for m in os.environ.get("ODDS_MARKETS", ",".join(MORNING_MARKETS if run_type == "morning" else DEFAULT_MARKETS)).split(",") if m.strip())
    if not os.environ.get("ODDS_API_KEY"):
        log.warning("ODDS_API_KEY not set: odds disabled (games will be reported as 'no odds -> no bet')")
        return {"enabled": False}
    try:
        client = OddsClient()
        f = client.fetch_odds(markets, regions)
        record_fetch(store, f, ",".join(markets))
        meta = {"enabled": True, "captured_at": f.captured_at, "stale": f.stale, "remaining": f.remaining, "source": f.source}
        max_props = int(os.environ.get("ODDS_PROPS_MAX_GAMES", PROPS_MAX_GAMES if run_type != "morning" else 0))
        if max_props > 0 and not f.stale:
            meta["props"] = fetch_player_props(store, client, f.events, max_props, regions=regions, remaining=f.remaining,
                                                 budget=props_budget(store, f.remaining, datetime.now(timezone.utc)))
        return meta
    except (OddsAPIError, OddsConfigError) as e:
        log.error("odds fetch failed: %s", e)
        return {"enabled": True, "error": str(e)}


def _betting_status(store: Store, perf: dict, ops: dict, date: str, now: datetime) -> dict:
    """The pre-registered betting gate (config/policy.yaml ``betting_gate``), recomputed from the live log; a change of status is recorded in the state and alerted once."""
    import numpy as np
    from nhlbet.betting_gate import evaluate_gate
    from nhlbet.ops import notify as ops_notify
    from nhlbet.report.betlog import resolved
    d = resolved(store)
    bets = d[d.is_bet & d.clv_move_pts.notna()] if len(d) else d
    curve = perf.get("curve")
    equity = np.r_[float(curve.bankroll.iloc[0] - curve.profit.iloc[0]), curve.bankroll.to_numpy()] if curve is not None and len(curve) else []
    g = evaluate_gate(bets.clv_move_pts if len(bets) else [], bets.game_date if len(bets) else [], equity, ops.get("status", "OK"))
    prev = (ops.get("betting") or {}).get("status")
    ops = {**ops, "betting": {"status": g["status"], "since": (ops.get("betting") or {}).get("since") if prev == g["status"] else now.isoformat(timespec="seconds")}}
    if prev is not None and prev != g["status"]:
        ops = ops_notify.dispatch([{"type": "warning", "check": "betting_status", "key": f"betting_status:{g['status']}:{date}",
                                    "message": f"Betting status changed from {prev} to {g['status']}: {g['why']}. This is research output, not advice."}], ops)
    return {**g, "since": ops["betting"]["since"], "_state": ops}


def run_daily(date: str | None = None, run_type: str = "morning", db: str = "data/nhl.db", bankroll: float = 1000.0, fixed_bankroll: bool = False,
              refresh: bool = True, odds: bool = True, do_retrain: bool | None = None, log_root: str = "data/logs", report_dir: str = "reports",
              model_dir: str = "data/models", site_dir: str = "site", readme_path: str = "README.md", export_dir: str = "data/export", now: datetime | None = None) -> dict:
    date = date or game_day()
    store = Store(db)
    restored = restore_logs(store, log_root)
    if restored:
        log.info("restored logs: %s", restored)
    purge_inplay(store)
    if not store.df("SELECT 1 FROM games LIMIT 1").shape[0] and LEGACY_CSV.exists():
        log.warning("empty database: bootstrapping from %s (28 teams, scores only). Run the backfill workflow for full data.", LEGACY_CSV)
        import_legacy_csv(store, LEGACY_CSV)
    notes = []
    if refresh:
        err = refresh_data(store)
        if err:
            notes.append(f"data refresh failed ({err}); using cached data")
    load_confirmations(store)
    data_ok, data_msg = data_freshness(store, date, err if refresh else None)
    problems = validate_inputs(store)
    if problems:
        data_ok, data_msg = False, "input data failed validation (" + "; ".join(problems) + "): picks withheld"
    if not data_ok:
        notes.append(data_msg)                     # shown on the page and in the report; the slate below is built with the model marked ALERT, so nothing is picked

    reg = ModelRegistry(Path(model_dir) / "registry.json")
    if do_retrain is None:
        do_retrain = run_type == "morning" or reg.latest() is None
    if do_retrain:
        r = retrain(db, model_dir=model_dir, report_dir=report_dir)
        log.info("retrain: %s", {k: v for k, v in r.items() if k != "drift"})
    entry = reg.latest()
    if entry is None:
        raise RuntimeError("no trained model available; run `python scripts/train.py` first")
    bundle = ModelBundle.load(entry["artifact"])
    odds_meta = fetch_and_store_odds(store, run_type=run_type) if odds else {"enabled": False}
    if odds_meta.get("error"):
        notes.append(f"odds fetch failed ({odds_meta['error']})")

    perf = performance(store, bankroll)
    shadow = shadow_performance(store)
    current = bankroll if fixed_bankroll else max(perf.get("bankroll", bankroll), 0.0)
    cfg = RiskConfig(bankroll=current)
    status = entry.get("drift", {}).get("status", "OK")
    if not data_ok:
        status = "ALERT"
    from nhlbet.ops import chain as ops_chain, gate as ops_gate, notify as ops_notify, state as ops_state
    repo_root = Path(log_root).resolve().parent.parent
    state_path = repo_root / "state" / "system_state.json"
    ops = ops_state.load_state(state_path)
    scheduled = int(store.df("SELECT COUNT(*) AS n FROM games WHERE game_date = ? AND game_type IN (2, 3)", [date]).n.iloc[0])
    checks, pol_sha = ops_gate.pre_checks(store, date, err if refresh else None, odds_meta, odds_meta.get("remaining"), scheduled, perf, repo_root, ops)
    ops, ops_events = ops_gate.apply(ops, checks, pol_sha)
    betting = _betting_status(store, perf, ops, date, now or datetime.now(timezone.utc))
    ops = betting.pop("_state")
    if not ops_state.can_publish(ops):                 # a halt (soft or hard) withholds every pick: the slate is built with the model marked ALERT, exactly like stale data
        status = "ALERT"
        notes.append("picks withheld: system " + ops["status"] + " (" + "; ".join(f"{h['check']}: {h['reason']}" for h in ops_state.open_halts(ops))[:300] + ")")
    slate = build_slate(store, bundle, date, cfg, run_type, "OK" if status == "INSUFFICIENT_DATA" else status, now=now)

    text = render_report(date, run_type, slate, cfg, entry["version"], {**entry, "p_source": entry.get("p_source")}, perf, odds_meta if odds_meta.get("enabled") else None, shadow=shadow)
    if notes:
        text = text.replace("## Summary", "> ⚠️ " + " · ".join(notes) + "\n\n## Summary", 1)
    out_dir = Path(report_dir) / "daily"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{date}-{run_type}.md").write_text(text)
    Path(report_dir, "latest.md").write_text(text)
    plot_performance(store, Path(report_dir) / "performance.png", bankroll)
    from nhlbet.report.ranking import learn_weights
    evidence = []
    try:                                   # the previous run's evidence (validation needs this run's exports, which are written after the page)
        from nhlbet.site.validation_page import verdicts
        vj = Path(report_dir, "VALIDATION.json")
        evidence = [list(v) for v in verdicts(json.loads(vj.read_text()))[:5]] if vj.exists() else []
    except Exception as e:
        log.warning("evidence summary unavailable: %s", e)
    payload = build_payload(slate, cfg, perf, entry, odds_meta if odds_meta.get("enabled") else None, date, run_type, notes=notes, shadow=shadow, rank_weights=learn_weights(store), evidence=evidence)
    from nhlbet.ops import health as ops_health
    rc = ops_gate._safe("value_ranges", "soft", lambda: ops_health.range_check(payload))
    if not rc.ok:                          # impossible numbers in what was just built: halt, rebuild the slate with nothing picked
        ops, more = ops_gate.apply(ops, checks + [rc], pol_sha)
        ops_events += more
        status = "ALERT"
        notes.append("picks withheld: system " + ops["status"] + f" ({rc.detail})")
        slate = build_slate(store, bundle, date, cfg, run_type, "ALERT", now=now)
        payload = build_payload(slate, cfg, perf, entry, odds_meta if odds_meta.get("enabled") else None, date, run_type, notes=notes, shadow=shadow, rank_weights=learn_weights(store), evidence=evidence)
    payload["system"] = ops_gate.system_block(ops)
    payload["betting"] = betting
    try:                                   # candidates for the pre-game review agent, and the review decisions made so far today (UNREVIEWED when there are none)
        from nhlbet.review.apply import STATE_NAME, attach_to_payload, load_state
        from nhlbet.review.contract import build_candidates, enabled
        if not enabled():
            raise RuntimeError("switched off (REVIEW_ENABLED=false)")
        review_dir = Path(log_root).parent / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "picks_candidate.json").write_text(json.dumps(build_candidates(payload), indent=1, allow_nan=False))
        attach_to_payload(payload, load_state(review_dir / STATE_NAME))
    except Exception as e:
        log.warning("review candidates unavailable: %s", e)
    build_site(payload, site_dir)
    build_bets_page(store, payload["games"], cfg.bankroll, payload["generated_at"], date, site_dir)
    # history (markdown for GitHub, html for the site) and the front-page block with clickable links
    Path(report_dir, "HISTORY.md").write_text(render_history_md(store, report_dir))
    Path(site_dir, "history.html").write_text(render_history_html(store, report_dir))
    if Path(readme_path).exists():
        block = render_block(date, run_type, slate, status, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), sum(s.rec.action == "BET" for s in slate))
        if not update_readme(readme_path, block):
            log.warning("README has no picks markers; front-page block not updated")
    written = export_logs(store, log_root)
    try:                                   # the hash-chained ledger commits to every log file just written
        ops_chain.append_new(repo_root)
    except Exception as e:
        log.error("ledger append failed: %s", e)
        ops, more = ops_gate.apply(ops, checks + [ops_gate.failed("log_integrity", "hard", f"ledger append failed: {e}", "Do not edit the logs; restore data/logs from the log-backup branch.")], pol_sha)
        ops_events += more
    ops_state.save_state(ops, state_path)
    if not ops_state.can_publish(ops):     # liveness ("ok") pings come from the always-on heartbeat job; a halt reports /fail so the external service alerts at once
        ops_notify.ping("fail")
    try:                                   # BI datasets (Power BI / Excel); context columns come from the as-of features
        from nhlbet.data.loaders import load_tables
        from nhlbet.features.builder import BuilderConfig, FeatureBuilder
        feats = FeatureBuilder(BuilderConfig(**bundle.builder_cfg)).build(load_tables(store))
        export_dataset(store, export_dir, feats, Path(model_dir) / "registry.json", Path(report_dir) / "walkforward_predictions.csv")
    except Exception as e:                 # never let a dashboard export break the daily report
        log.warning("BI export failed: %s", e)
    try:                                   # the evidence page: live results from the logs just written, kept apart from the backtest
        from nhlbet.report.validation import run_validation
        from nhlbet.site.build import HERE
        from nhlbet.site.validation_page import render_validation_html
        res = run_validation(".", export_dir=export_dir, report_dir=report_dir, logs_dir=log_root, archive_dir=Path(site_dir) / "archive")
        Path(report_dir, "VALIDATION.json").write_text(json.dumps(res, indent=1, default=float))
        Path(site_dir, "validation.html").write_text(render_validation_html(res, (HERE / "theme.css").read_text()))
    except Exception as e:                 # same rule: a failed evidence page must not stop the picks
        log.warning("validation page failed: %s", e)
    try:                                   # does the pre-game review agent help? grades kept / flagged / vetoed picks alike (reports/REVIEW.md)
        from nhlbet.review.evaluate import write_review_report
        write_review_report(store, Path(log_root).parent.parent, report_dir)
    except Exception as e:
        log.warning("review report failed: %s", e)
    log.info("report written for %s (%d games, %d bets); exported %d log files", date, len(slate), sum(s.rec.action == "BET" for s in slate), len(written))
    return {"date": date, "games": len(slate), "bets": sum(s.rec.action == "BET" for s in slate), "model": entry["version"], "status": status, "notes": notes}
