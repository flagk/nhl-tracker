# Phase 2: raw data and point-in-time rebuild

## What was built

| Piece | Where | Status |
|---|---|---|
| Immutable, content-addressed, versioned snapshot store with fetch time (`knowledge_time`) and a hash manifest | `nhlbet/pit/snapshots.py` | built, tested (`tests/test_pit.py`): rewrites impossible, refetches become new versions, corrupt or missing objects detected, bytes reproducible |
| Fetch of play-by-play, boxscore and shift charts with shape checks; endpoint probe | `nhlbet/pit/fetch.py`, `scripts/pit_ingest.py` | built, tested against recorded fixtures. **Endpoints verified on 2026-10-26** by the probe run in Actions (the NHL hosts return 403 from the authoring sandbox): the schedule, play-by-play and boxscore endpoints on `api-web.nhle.com` and the shift-chart endpoint on `api.nhle.com/stats/rest/en/shiftcharts` all answered HTTP 200 for a final game (2026020183) with the keys the parsers need (`audit/data/endpoint_probe.json`). Shift charts are a community-documented endpoint, not an official one, so they may change without notice. Ingestion into the snapshot store has **not** been run (it needs the private DVC remote) |
| As-of observation table: `event_time` and `knowledge_time` stored separately, as-of resolver, revision report | `nhlbet/pit/asof.py` | built, tested |
| Point-in-time mode in the feature builder (`FeatureBuilder.build(..., adv_asof=resolver)`) | `nhlbet/features/builder.py` | built. Test proves (1) output identical to the plain builder to 1e-12 when nothing was revised, (2) a late correction changes only decisions made after it was fetched |
| Goalie confirmation events with first-seen time (the table keeps only the latest per team and date) | `nhlbet/pit/goalie_log.py`, wired into `run_daily` | built, tested; inside the hash-chained ledger |
| Odds archive (raw, timestamped, all books) | `scripts/odds_archive.py`, `audit.odds_archive` in the policy | built, tested, **off by default** (cost, below) |
| DVC | `.dvc/`, `pit-ingest.yml`, `odds-archive.yml` | configured; `dvc` could not be installed in the sandbox (the `antlr4-python3-runtime` wheel fails to build), so **DVC itself has not been executed**. The workflows install `dvc[s3]` on the runner and fail closed without a private remote |

## Scope of the as-of mode

Point-in-time resolution applies to the play-by-play-derived team statistics (shot attempts, xG, high danger, special teams: the `team_game` columns), which are the values the NHL corrects after the fact. Final scores, goalie boxscores and the schedule are still read from the database as last written. Scores are almost never revised, but goalie and skater boxscore rows are, and have the same gap. It is listed in Phase 1 as open item 10 and is only partly closed.

## Why this cannot be backfilled

Knowledge time only exists from the first fetch onward. Past seasons can be re-fetched, but a re-fetch today returns the settled values, stamped today: historical backtests can therefore never be made point-in-time; only the forward period can. The revision report (`nhlbet.pit.asof.revision_report`) will quantify how much the provider changes values once two fetches of the same game exist.

## The odds archive: why "every few minutes" is not switched on

The Odds API charges one credit per market per region per call. The plan has 500 credits a month and the live pipeline already uses most of them.

| Cadence | Calls per 8-hour evening | Credits per month |
|---|---|---|
| every 5 minutes | 96 | about 2,900 |
| every 10 minutes | 48 | about 1,440 |
| every 30 minutes | 16 | about 480 (the whole plan) |

So the request cannot be met on this plan. The archiver exists, respects a credit floor and a minimum interval from the policy, and is disabled. **Owner decision needed:** a larger plan, or a lower cadence that leaves the live pipeline its credits. Raw per-book quotes must also go to a private store (the repository is public), so the archive needs the secrets `DVC_REMOTE_URL`, `DVC_AWS_ACCESS_KEY_ID`, `DVC_AWS_SECRET_ACCESS_KEY` (or another DVC remote type).

## Goalie confirmations

No automated confirmation source exists (the NHL API does not publish starters before puck drop, per `nhlbet/data/goalies.py`); confirmations are a manual file. What is now preserved is the first-seen time of every distinct confirmation, including ones later overwritten. A confirmation feed is not something this audit can create.

## Not done in this phase

- Shift data are fetched and stored but not yet turned into features (the production model does not use them).
- Re-ingesting the three historical seasons into snapshots: needs the NHL API and a DVC remote; roughly 1,300 games per season, three payloads each.
