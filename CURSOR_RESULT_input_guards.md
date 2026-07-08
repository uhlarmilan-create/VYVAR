CURSOR RESULT ù 2026-07-08 (INPUT-GUARDS-0708)

What I did
Implemented null-island observer guard at `resolve_site` and PDF config snapshot sourcing
from `pipeline_meta.provenance` (G7-F003c / PROV-FIX synergy). Tests + full gate green; draft_424
PDF rebuild with 0 overflow violations.

## Output / findings

### Item 1 ù TIER1-OBSLOC-ZERO (`166cbf4`)

**Resolution chain (observer lat/lon ? physics):**

| Step | File:line | Role |
|------|-----------|------|
| Choke | `param_resolver.py:resolve_site` | draft `ID_LOCATION` ? header `SITELAT/LONG/ELEV` ? flagged config |
| Wrapper | `time_utils.py:155-182` `resolve_observer_location` | calls `resolve_site` |
| Airmass | `pipeline.py:9051-9116` `_compute_airmass_from_altaz` | AltAz; NaN if unresolved |
| Airmass hdr | `pipeline.py:9119+` `_extract_airmass_from_header` | calls AltAz fallback |
| Phase 2A site | `photometry_core.py:6868-6953` | once per draft; `site_ok` threaded |
| BJD/HJD | `photometry_core.py:6965-7037` `_recompute_bjd_hjd_with_status` | `BJD_TDB` or `JD_FALLBACK` |
| Lunar | `photometry_core.py:5509+` `_phase2a_compute_lunar_context` | skipped if `site_ok=False` |
| Lunar core | `lunar_context.py:117-139` `get_lunar_context` | Moon position from site |
| Export | `export_reports.py:146-207` | per-draft `observer_location` from meta |
| Report | `photometry_report.py:584+` | site from `pipeline_meta.observer_location` |

**Guard:** `NULL_ISLAND_LAT_LON_THRESHOLD_DEG = 0.01` ù both |lat| and |lon| below ? UNRESOLVED;
ERROR names `ID_LOCATION` when from draft row.

**Tests:** `tests/test_obsloc_null_island.py` (7 passed) ù null island unresolved, real site OK,
airmass NaN, BJD `JD_FALLBACK`.

### Item 2 ù G7-F003c (`80aab21`)

**PDF cfg fields sourced from snapshot (via `self._cfg` / `resolve_report_config`):**
`phase01_use_bprp_primary`, `aperture_comp_factor`, `aperture_variable_factor`,
`gs11_dilution_enabled`, `phase01_plate_scale_arcsec_per_px`, `auto_fwhm_enabled`,
`auto_fwhm_k_factor`, `psf_chi2_threshold`, `psf_photometry_enabled`, `psf_adaptive_enabled`,
`observer_name`, `observer_code`, variability edge-safe thresholds (`to_dict()`), methods/citations
via `build_run_citation_context`.

**Fallback:** no `provenance.config_snapshot` ? live `AppConfig` + footer
`config: live (no run snapshot)`.

**draft_424 PDF rebuild:** `VYVAR_report_NoFilter_60_2_inputguards_test.pdf` ù
**overflow_violations = 0**.

**Tests:** `tests/test_g7_f003c_report_cfg_snapshot.py` (3 passed); existing G7-F003b tests pass.

### Gate
`587 passed`, 15 skipped; ruff BLE001/E722 ù **PASS**

## Errors (if any)
None.

## Files changed
| Item | Commit | Files |
|------|--------|-------|
| 1 | `166cbf4` | `param_resolver.py`, `photometry_core.py`, `tests/test_obsloc_null_island.py` |
| 2 | `80aab21` | `photometry_report.py`, `tests/test_g7_f003c_report_cfg_snapshot.py` |
| close | `ab8e5da` | `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `CURSOR_RESULT_input_guards.md` |
