# Audit

Goal: decide whether a real, durable betting edge exists, how confident we can be, and how decay would be detected. Negative findings are reported as found.

| Phase | Status | Document |
|---|---|---|
| 1 Reconnaissance | done | [PHASE1_RECON.md](PHASE1_RECON.md) |
| 2 Raw data and point-in-time rebuild | built; not run against the live NHL API (blocked from the sandbox) | [PHASE2_PIT.md](PHASE2_PIT.md) |
| 3 Leakage attack | done | [PHASE3_LEAKAGE.md](PHASE3_LEAKAGE.md) |
| 4 to 7 | not started: stopped for review after Phase 3 as instructed | |

Older files in this folder (`leakage_probes.*`, `statistics.*`, `legacy_predictions.csv`, `reproduce_legacy.py`) belong to the first audit (`AUDIT.md` at the repository root).

Scripts are in `scripts/`; results the scripts wrote are in `results/` and `data/`; `jobs.txt` lists what the `audit-realdata` workflow runs (push a change to `ci/audit_trigger.txt` on the `audit` branch to rerun it).
