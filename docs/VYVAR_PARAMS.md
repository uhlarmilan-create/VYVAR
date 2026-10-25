# VYVAR -- Config <-> UI parameter registry

<!-- GENERATED FILE -- DO NOT EDIT BY HAND. -->
Regenerate with `python tools/gen_params_md.py`. Hand edits will be overwritten.
Source: `validation/params_registry.json` (editorial metadata) + `dataclasses.fields(AppConfig)` (defaults and types, from code).
Human-readable guide: `VYVAR_CONFIG_GUIDE_EN.md` / `VYVAR_CONFIG_GUIDE_CZ.md` (per-parameter plain-language explanations, hand-authored).
In-depth handbook: `VYVAR_PARAMETER_HANDBOOK_CZ.pdf` (Czech; per-parameter reasoning, ranges, math and literature; regenerate with `python dev/tools/docs_pdf/build_parameter_handbook.py`).

_Generated 2026-10-25T08:49:52Z at git HEAD 6259312._

## Summary

- Entries: 57
- Tier: basic 13, advanced 24, expert 20
- Kind: static 56, derived 0, resolved 1
- Widget: auto 54, custom 0, hidden 3
- Owner: db_static 5, config_runtime 45, fits_dynamic 0, internal 7
- Scope: universal 41, rig 7, site 9, session 0
- Scope key: none 41, rig 4, rig_band 2, rig_sampling 1, site 9, frame 0
- Rig triage group: a 5, c 2

Columns: key, default, range, tier, kind, owner, scope, scope_key, scope_group, widget, label. `kind=resolved` means the runtime value can be auto-derived/overridden by the pipeline (the configured value is the base/fallback). `owner` is the storage-and-ownership axis: `db_static` (DB reference tables), `config_runtime` (user-tuned config.json), `fits_dynamic` (resolved from FITS/WCS at run time), `internal` (plumbing). `widget=custom` keys keep their hand-built UI; `widget=hidden` keys are plumbing not surfaced in the generated dashboard.

## observer

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `aavso_filter_map` | {} | - | expert | static | config_runtime | site | site | n/a | auto | AAVSO filter names |
| `observer_alt_m` | nan | - | basic | static | db_static | site | site | n/a | auto | Observer Alt M |
| `observer_code` |  | - | basic | static | config_runtime | site | site | n/a | auto | AAVSO observer code |
| `observer_lat` | nan | - | basic | static | db_static | site | site | n/a | auto | Observer Lat |
| `observer_location_id` | 0 | - | expert | static | db_static | site | site | n/a | hidden | Observing site (run time) |
| `observer_location_name` |  | - | basic | static | db_static | site | site | n/a | auto | Observer Location Name |
| `observer_lon` | nan | - | basic | static | db_static | site | site | n/a | auto | Observer Lon |
| `observer_name` | Unknown Observer | - | basic | static | config_runtime | site | site | n/a | auto | Observer name |

## paths

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `archive_root` | (resolved at runtime) | - | expert | static | internal | universal | none | n/a | auto | Archive |
| `astrometry_index_dir` |  | - | advanced | static | config_runtime | universal | none | n/a | auto | Blind plate-solve index |
| `calibration_library_root` | (resolved at runtime) | - | expert | static | internal | universal | none | n/a | auto | Calibration library |
| `database_path` | (resolved at runtime) | - | expert | static | internal | universal | none | n/a | auto | App database |
| `gaia_db_path` |  | - | expert | static | internal | universal | none | n/a | auto | Gaia DR3 catalog |
| `objects_db_path` | catalogs/vyvar_objects.db | - | expert | static | internal | universal | none | n/a | auto | Objects catalog (VSX + exoplanets) |
| `project_root` | (git toplevel) | - | expert | static | internal | universal | none | n/a | hidden | Project Root |

## calibration

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | 0.01 .. 20 | expert | static | config_runtime | rig | rig | a | auto | Dark temperature tolerance |
| `masterdark_validity_days` | 90 | 1 .. 3650 | basic | static | config_runtime | rig | rig | c | auto | Master dark valid for |
| `masterflat_validity_days` | 200 | 1 .. 3650 | basic | static | config_runtime | rig | rig | c | auto | Master flat valid for |

## qc

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `auto_fwhm_enabled` | True | - | advanced | static | config_runtime | universal | none | n/a | auto | Night FWHM limit from the data |
| `auto_fwhm_k_factor` | 3.0 | 1 .. 4 | advanced | static | config_runtime | universal | none | n/a | auto | Night FWHM limit k |
| `osc_channel_binning` | 2 | 1 .. 4 | advanced | static | config_runtime | rig | rig | a | auto | OSC channel binning |
| `preprocess_sky_surface_order` | 2 | 0 .. 2 | expert | static | config_runtime | universal | none | n/a | auto | Sky background surface order |
| `qc_after_calibrate_enabled` | True | - | basic | static | config_runtime | universal | none | n/a | auto | Quality check after calibration |

## alignment

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `alignment_max_stars` | 160 | 10 .. 5000 | advanced | static | config_runtime | universal | none | n/a | auto | Stars used for alignment |

## detection

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `detection_threshold_sigma` | 4.5 | 2 .. 20 | advanced | static | config_runtime | rig | rig_sampling | a | auto | Star detection threshold |
| `masterstar_best_of_n` | 10 | 1 .. 25 | expert | static | config_runtime | universal | none | n/a | auto | MASTERSTAR candidates |
| `vsx_out_of_scope_types` | [] | - | basic | static | config_runtime | universal | none | n/a | auto | VSX types not measured |

## photometry

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `annulus_inner_fwhm` | 2.7 | 1 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | Sky annulus inner radius |
| `annulus_outer_fwhm` | 5.2 | 1.5 .. 12 | advanced | static | config_runtime | universal | none | n/a | auto | Sky annulus outer radius |
| `aperture_fwhm_factor` | 1.35 | 0.25 .. 6 | advanced | static | config_runtime | universal | none | n/a | auto | Aperture radius |
| `epsf_auto_run` | False | - | basic | static | config_runtime | universal | none | n/a | auto | Run ePSF in every night run (beta) |
| `epsf_export_auto` | True | - | basic | static | config_runtime | universal | none | n/a | auto | Write ePSF light-curve files |
| `epsf_targets` | ALL | - | basic | static | config_runtime | universal | none | n/a | auto | ePSF targets |
| `gs11_dilution_enabled` | False | - | expert | static | config_runtime | universal | none | n/a | auto | Neighbour light estimate |
| `per_frame_sat_min_clean_frac` | 0.5 | 0.1 .. 1 | advanced | static | config_runtime | universal | none | n/a | auto | Unsaturated frames needed |
| `save_lightcurve_png` | False | - | advanced | static | config_runtime | universal | none | n/a | auto | Save light-curve PNG files |

## comp_selection

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `comp_catalog_offset_max_mag` | 0.2 | 0 .. 2 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison catalogue offset limit |
| `comp_color_tiers` | [{'bprp': 0.15}, {'bprp': 0.3}, {'bprp': 0.55}] | - | advanced | static | config_runtime | universal | none | n/a | auto | Colour classes |
| `comp_isolation_fwhm` | 3.0 | 1 .. 12 | expert | static | config_runtime | universal | none | n/a | auto | Comparison isolation |
| `comp_max_delta_bprp` | 0.79 | 0 .. 5 | advanced | static | config_runtime | universal | none | n/a | auto | Largest colour difference |
| `comp_max_slope_mmag_hr` | 5.0 | 0 .. 500 | expert | static | config_runtime | universal | none | n/a | auto | Comparison drift limit |
| `phase01_comparison_max_comp_rms` | 0.1 | 0.01 .. 0.5 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison scatter limit |
| `phase01_comparison_min_frames_frac` | 0.2 | 0.05 .. 0.95 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison frame coverage |
| `phase01_comparison_n_comp_max` | 8 | 3 .. 20 | advanced | static | config_runtime | universal | none | n/a | auto | Most comparison stars |
| `phase01_comparison_n_comp_min` | 3 | 2 .. 12 | advanced | static | config_runtime | universal | none | n/a | auto | Fewest comparison stars |
| `snr_min` | 10.0 | 3 .. 1000 | advanced | static | config_runtime | universal | none | n/a | auto | Minimum SNR |

## trust

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `comp_trust_min_comps` | 5 | 3 .. 20 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison stars for green |
| `lc_quality_min_frames` | 20 | 3 .. 500 | advanced | static | config_runtime | universal | none | n/a | auto | Light curve minimum frames |

## extinction

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `apply_color_term` | auto | - | expert | static | config_runtime | rig | rig_band | a | auto | Colour term |
| `k2_defaults_bprp` | {} | - | expert | static | config_runtime | rig | rig_band | a | auto | Own k2 per band |
| `k2_mode` | literature | - | advanced | static | config_runtime | site | site | n/a | auto | Second-order extinction |

## reports

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `hrd_nss_category_enabled` | False | - | expert | static | config_runtime | universal | none | n/a | auto | HRD binaries category |
| `hrd_online_enrich_enabled` | True | - | expert | static | config_runtime | universal | none | n/a | auto | HRD online details |

## export

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `tess_enabled` | False | - | advanced | static | config_runtime | universal | none | n/a | auto | TESS check of new candidates |
| `transient_search_enabled` | False | - | advanced | static | config_runtime | universal | none | n/a | auto | Search for transients and minor bodies |

## system

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `per_frame_mp_reserve_ram_gb` | 1.5 | 0 .. 64 | expert | static | config_runtime | universal | none | n/a | auto | RAM kept free per worker |
| `qc_preprocess_workers` | 1 | - | expert | resolved | internal | universal | none | n/a | hidden | QC Preprocess Workers |

