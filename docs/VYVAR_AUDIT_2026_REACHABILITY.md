# VYVAR audit 2026 - module reachability (exact)

**Total `src_py` modules:** 119  
**Method:** AST import closure from 10 entry modules (`app`, `night_run`, `simulate_night_run`, `xval_run`, `psf_runner`, `comp_qa`, `trust_flag`, `inspect_drafts`, `repair_catalog_ids`, `run_crowding_index`, `run_smoothness_report`) plus 11 lazy Streamlit tab modules added explicitly.

**Lazy-import limit:** dynamic `importlib` / inline imports inside Streamlit render callbacks are not parsed. Modules reachable only that way remain `not_statically_reachable` unless listed in the LAZY set below.

## Counts

| Class | Count |
|-------|------:|
| production_reachable | 88 |
| cli_entry | 12 |
| unwired_ui | 4 |
| not_statically_reachable | 15 |

## Lazy-import modules (explicitly added to closure)

- `ui_aperture_photometry`
- `ui_calibration_library`
- `ui_dao_stars`
- `ui_epsf_dashboard`
- `ui_finalization` (banner only; full panel unwired)
- `ui_hrd`
- `ui_params_dashboard`
- `ui_photometry`
- `ui_photometry_quality`
- `ui_settings`
- `ui_variability`

## not_statically_reachable (15)

These exist in `src_py/` but are not in the static closure above. They are **not dead** until proven unreachable on all config/UI paths:

| Module | Notes |
|--------|-------|
| `band_classify.py` | Imported lazily from photometry_core helpers |
| `cal_stage.py` | Imported from pipeline/cal_diag paths |
| `dao_reconcile.py` | Dev/diagnostic |
| `except_fix_counters.py` | Exception telemetry |
| `gaia_johnson.py` | OSC/colour transforms |
| `hrd_colorfield.py` | HRD optional layer |
| `k2_cohort_core.py` | K2 fit internals |
| `mag_constants.py` | Constants |
| `osc_align.py` | OSC equipment only |
| `osc_extract.py` | OSC equipment only |
| `phase0_funnel.py` | Phase 0 funnel |
| `plain_stats.py` | Stats helpers |
| `run_preflight_log.py` | Preflight CLI |
| `sigma_budget.py` | Error budget analysis |
| `vyvar_runtime.py` | Runtime hooks |

## unwired_ui (4)

| Module | Notes |
|--------|-------|
| `ui_finalization.py` | `render_finalization()` not in `app.py` |
| `ui_photometry_results.py` | Legacy tab |
| `ui_select_stars.py` | Legacy |
| `ui_suspected_lightcurves.py` | Legacy |

## cli_entry (12)

`comp_qa`, `dao_reconcile`, `inspect_drafts`, `psf_runner`, `repair_catalog_ids`, `run_crowding_index`, `run_preflight_log`, `run_smoothness_report`, `simulate_night_run`, `trust_flag`, `validate_lc_crossval`, `xval_run` (+ `xval_harness_core` via xval_run)
