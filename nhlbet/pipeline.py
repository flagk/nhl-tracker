"""The daily job: restore logs -> refresh data -> retrain if needed -> fetch odds -> settle -> slate -> report -> export logs.

Every stage degrades gracefully and says so in the report: a failed data refresh uses the cached database, a failed odds
fetch falls back to a flagged-stale snapshot or 'no odds -> no bet', a missing API key disables odds without crashing.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from nhlbet.data.goalies import load_confirmations
from nhlbet.data.ingest import current_season, import_legacy_csv, ingest_day, ingest_details, ingest_schedule, reparse_skaters
from nhlbet.data.client import NHLClient
from nhlbet.data.store import Store
from nhlbet.hygiene import purge_inplay
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
PROPS_MAX_GAMES = 3                                 # props cost 1 credit per market (shots, points, assists, anytime goals) per game: 3 games a day is ~12 credits/day on top of ~12 for game odds


EXTRA_PROPS_MIN_CREDITS = 250       # assists / anytime-goalscorer prices are only fetched while at least this many odds credits remain
PROPS_MIN_CREDITS = 100             # below this no player prices are fetched at all: game moneylines come first


def fetch_player_props(store: Store, client: OddsClient, events: list[dict], max_games: int, now: datetime | None = None, regions: str = "us",
                       remaining: int | None = None) -> dict:
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
    got, errors, ok_markets = 0, [], set()
    for e in upcoming:
        game_ok = False
        for m in markets:
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
    out = {"games": got, "markets": sorted(ok_markets)}
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
            meta["props"] = fetch_player_props(store, client, f.events, max_props, regions=regions, remaining=f.remaining)
        return meta
    except (OddsAPIError, OddsConfigError) as e:
        log.error("odds fetch failed: %s", e)
        return {"enabled": True, "error": str(e)}


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
    slate = build_slate(store, bundle, date, cfg, run_type, "OK" if status == "INSUFFICIENT_DATA" else status, now=now)

    text = render_report(date, run_type, slate, cfg, entry["version"], {**entry, "p_source": entry.get("p_source")}, perf, odds_meta if odds_meta.get("enabled") else None, shadow=shadow)
    if notes:
        text = text.replace("## Summary", "> ⚠️ " + " · ".join(notes) + "\n\n## Summary", 1)
    out_dir = Path(report_dir) / "daily"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{date}-{run_type}.md").write_text(text)
    Path(report_dir, "latest.md").write_text(text)
    plot_performance(store, Path(report_dir) / "performance.png", bankroll)
    from nhlbet.report.ranking import learn_weights
    payload = build_payload(slate, cfg, perf, entry, odds_meta if odds_meta.get("enabled") else None, date, run_type, notes=notes, shadow=shadow, rank_weights=learn_weights(store))
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
    try:                                   # BI datasets (Power BI / Excel); context columns come from the as-of features
        from nhlbet.data.loaders import load_tables
        from nhlbet.features.builder import BuilderConfig, FeatureBuilder
        feats = FeatureBuilder(BuilderConfig(**bundle.builder_cfg)).build(load_tables(store))
        export_dataset(store, export_dir, feats, Path(model_dir) / "registry.json", Path(report_dir) / "walkforward_predictions.csv")
    except Exception as e:                 # never let a dashboard export break the daily report
        log.warning("BI export failed: %s", e)
    log.info("report written for %s (%d games, %d bets); exported %d log files", date, len(slate), sum(s.rec.action == "BET" for s in slate), len(written))
    return {"date": date, "games": len(slate), "bets": sum(s.rec.action == "BET" for s in slate), "model": entry["version"], "status": status, "notes": notes}
