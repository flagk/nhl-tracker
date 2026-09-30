"""The daily job: restore logs -> refresh data -> retrain if needed -> fetch odds -> settle -> slate -> report -> export logs.

Every stage degrades gracefully and says so in the report: a failed data refresh uses the cached database, a failed odds
fetch falls back to a flagged-stale snapshot or 'no odds -> no bet', a missing API key disables odds without crashing.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from nhlbet.data.goalies import load_confirmations
from nhlbet.data.ingest import current_season, import_legacy_csv, ingest_day, ingest_details, ingest_schedule
from nhlbet.data.client import NHLClient
from nhlbet.data.store import Store
from nhlbet.models.bundle import ModelBundle
from nhlbet.odds.client import OddsAPIError, OddsClient, OddsConfigError
from nhlbet.odds.snapshots import record_fetch
from nhlbet.registry import ModelRegistry
from nhlbet.report.betlog import export_logs, performance, plot_performance, restore_logs, shadow_performance
from nhlbet.report.history import render_history_html, render_history_md
from nhlbet.report.markdown import render_report
from nhlbet.report.readme import render_block, update_readme
from nhlbet.report.slate import build_slate
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
        return None if (ok or not bad) else f"{bad} game detail fetches failed"
    except Exception as e:  # noqa: BLE001 - stage must not take the job down
        log.error("data refresh failed: %s", e)
        return str(e)


def fetch_and_store_odds(store: Store, markets=("h2h",), regions: str = "us") -> dict:
    if not os.environ.get("ODDS_API_KEY"):
        log.warning("ODDS_API_KEY not set: odds disabled (games will be reported as 'no odds -> no bet')")
        return {"enabled": False}
    try:
        f = OddsClient().fetch_odds(markets, regions)
        record_fetch(store, f, ",".join(markets))
        return {"enabled": True, "captured_at": f.captured_at, "stale": f.stale, "remaining": f.remaining, "source": f.source}
    except (OddsAPIError, OddsConfigError) as e:
        log.error("odds fetch failed: %s", e)
        return {"enabled": True, "error": str(e)}


def run_daily(date: str | None = None, run_type: str = "morning", db: str = "data/nhl.db", bankroll: float = 1000.0, fixed_bankroll: bool = False,
              refresh: bool = True, odds: bool = True, do_retrain: bool | None = None, log_root: str = "data/logs", report_dir: str = "reports",
              model_dir: str = "data/models", site_dir: str = "site", readme_path: str = "README.md") -> dict:
    date = date or game_day()
    store = Store(db)
    restored = restore_logs(store, log_root)
    if restored:
        log.info("restored logs: %s", restored)
    if not store.df("SELECT 1 FROM games LIMIT 1").shape[0] and LEGACY_CSV.exists():
        log.warning("empty database: bootstrapping from %s (28 teams, scores only). Run the backfill workflow for full data.", LEGACY_CSV)
        import_legacy_csv(store, LEGACY_CSV)
    notes = []
    if refresh:
        err = refresh_data(store)
        if err:
            notes.append(f"data refresh failed ({err}); using cached data")
    load_confirmations(store)

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
    odds_meta = fetch_and_store_odds(store) if odds else {"enabled": False}
    if odds_meta.get("error"):
        notes.append(f"odds fetch failed ({odds_meta['error']})")

    perf = performance(store, bankroll)
    shadow = shadow_performance(store)
    current = bankroll if fixed_bankroll else max(perf.get("bankroll", bankroll), 0.0)
    cfg = RiskConfig(bankroll=current)
    status = entry.get("drift", {}).get("status", "OK")
    slate = build_slate(store, bundle, date, cfg, run_type, "OK" if status == "INSUFFICIENT_DATA" else status)

    text = render_report(date, run_type, slate, cfg, entry["version"], {**entry, "p_source": entry.get("p_source")}, perf, odds_meta if odds_meta.get("enabled") else None, shadow=shadow)
    if notes:
        text = text.replace("## Summary", "> ⚠️ " + " · ".join(notes) + "\n\n## Summary", 1)
    out_dir = Path(report_dir) / "daily"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{date}-{run_type}.md").write_text(text)
    Path(report_dir, "latest.md").write_text(text)
    plot_performance(store, Path(report_dir) / "performance.png", bankroll)
    payload = build_payload(slate, cfg, perf, entry, odds_meta if odds_meta.get("enabled") else None, date, run_type, notes=notes, shadow=shadow)
    build_site(payload, site_dir)
    # history (markdown for GitHub, html for the site) and the front-page block with clickable links
    Path(report_dir, "HISTORY.md").write_text(render_history_md(store, report_dir))
    Path(site_dir, "history.html").write_text(render_history_html(store, report_dir))
    if Path(readme_path).exists():
        block = render_block(date, run_type, slate, status, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), sum(s.rec.action == "BET" for s in slate))
        if not update_readme(readme_path, block):
            log.warning("README has no picks markers; front-page block not updated")
    written = export_logs(store, log_root)
    log.info("report written for %s (%d games, %d bets); exported %d log files", date, len(slate), sum(s.rec.action == "BET" for s in slate), len(written))
    return {"date": date, "games": len(slate), "bets": sum(s.rec.action == "BET" for s in slate), "model": entry["version"], "status": status, "notes": notes}
