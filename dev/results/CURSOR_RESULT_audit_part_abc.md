# C-EXPORT-GAP — headless export wiring (report only)

**Date:** 2026-08-13  
**Status:** Report for Milan decision; no implementation in this task.

## Current state

| Path | Export step |
|------|-------------|
| `night_run.py` Step 15 | PDF via `generate_all_method_photometry_reports` |
| UI / manual CLI | `export_lightcurve_reports`, `export_all_method_lightcurve_reports` in `export_reports.py:904` |

AAVSO/VarAstro are **not** on the automated night path.

## What wiring would take

1. After Step 14 photometry (or after trust/comp QA), call `export_all_method_lightcurve_reports(photometry_dir, cfg, ...)` per obs_group — same entry the UI uses.
2. Pass `observer_code`, filter map, BJD column selection, and method key (`aperture` default).
3. Respect `INV-CFG-01` / trust: optionally skip export when trust is RED (policy choice).
4. Stamp export provenance in `pipeline_meta.json` (new keys or extend INV-PROV-01).

## Gates that apply

| Gate | Effect |
|------|--------|
| **G-EXP-02 / T1** | Refuses AAVSO if `time_base != BJD_TDB` — headless must ensure Phase 2A wrote `bjd_tdb_mid` |
| **OSC-03** | Blocks `oneRGGB` rows; R/G/B must use TR/TG/TB — mono rig unaffected |
| **Trust GREEN** | Not enforced today on export; wiring could add optional hard gate |

## Risks of unattended export

- Submitting a RED-trust or unreviewed LC to AAVSO under observer code.
- Wrong filter code for unfiltered rig (D10-1 CR band must match config).
- Incomplete comp set exported if photometry partial-failed but PDF warning-only continued.
- Duplicate submission if re-run night without idempotency check.

## Recommendation

Add an **opt-in** `night_run` flag (`export_aavso: bool = False`) default off; require trust != RED and explicit observer code before write.
