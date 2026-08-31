# CURSOR RESULT - CONSOLIDATE-01A v2 src_py reachability sweep

Date: 2026-08-31. Branch: consolidate-01. English. ASCII.
Base: origin/consolidate-01 @ d320697. Live draft 516 was not written.
No refit. No numeric change. No config prerez. No pipeline/photometry_core split.

## Premise (Rule 0.1)

What is compared: src_py modules and functions reachable from roots R1-R5
versus vulture unused-function candidates (conf >= 60) plus whole-repo grep.
How they differ: import-count zero is not unreachability; R2/R3/R4/R5 keep
entry points and test-imported modules. Deletions are only unreachable
modules and functions that both vulture and grep called dead.

## What I did

Reachability closure of 139 src_py modules from R1-R5. Deleted three
unreachable CLI modules and dead functions in 26 other modules (one
commit per module). Inventory of 35 duplicated top-level helper names
and CircularAnnulus construction sites (no dedup this stage).

## 1. Roots (file:line)

### R1 Streamlit app

`src_py/app.py` (entry `main` at app.py:2310; sidebar radio :2339).
Direct imports: app.py:18-52 (config, database, infolog, importer,
optics_selection, pipeline, ui_calibration, ui_database_explorer,
ui_masterstar_qa, ui_quality_dashboard, ui_components,
ui_aperture_photometry, ui_variability).
Lazy: ui_finalization :1946, ui_epsf_dashboard :2431,
ui_calibration_library :2448, ui_settings :2459.
ui_settings.py:11-13 loads ui_dao_stars, ui_params_dashboard, ui_photometry.
ui_settings.py:1039 loads ui_photometry_quality.
ui_aperture_photometry.py:1791 loads ui_hrd.

### R2 CLI (def main + argparse) -- keep as roots

| Module | Evidence |
| --- | --- |
| night_run | parse_night_run_cli :223, main :1906 |
| simulate_night_run | argparse :13, main :43 |
| psf_runner | argparse :3, main :1499 (CLI wrapper over epsf_stage) |
| psf_internal_lc | main + argparse |
| dao_gaia_stage_01 / iter2 / iter3 / iter4 | main + argparse |
| xval_run | argparse + main :88; also R5 |
| repair_catalog_ids | main + argparse |
| comp_qa | main + argparse |
| trust_flag | argparse :15, main :15 |

### R2 rest -- main, no argparse (classified, not R2 roots)

| Module | Class | Disposition |
| --- | --- | --- |
| app | Streamlit | R1 |
| inspect_drafts | one-off CLI, hardcoded usage in docstring | deleted (3a) |
| run_crowding_index | hardcoded DRAFTS list, "no pipeline wire-up" | deleted (3a) |
| run_smoothness_report | hardcoded draft_000249 | deleted (3a) |
| validate_lc_crossval | offline LC script, no argparse | kept (dev/scripts importers) |

### R3 Tests

Any src_py module imported by `dev/tests/**/*.py` is a root.
Keeps (among 3a candidates): wide_slope_noise_core, psf_neighbor_sub,
method_lc_output, phase0_funnel, aperture_scatter_select, snr_cog_gates.

### R4 Gate script

`dev/scripts/session_baseline_check.py`:
config :231, database :321, draft_provenance :322, photometry_core :479/:768,
epsf_stage :528, epsf_zp_ok :575, catalog_provenance :603,
phase0_funnel :635/:1068, pipeline :636/:768/:1260, night_run :1259.

### R5 Docs (STATE / ROADMAP / PROCESS named commands)

xval_run named in all three (PROCESS.md:330/:354, STATE.md:1274/:1469).
Others named as architecture (photometry_core, pipeline, method_lc_output,
psf_neighbor_sub, ...) -- listed in reachability_meta.json r5_doc_hits.

## 2. Reachability

JSON: `dev/results/context/session_20260831_c01a/reachability.json`
(module -> roots, importers, src_py_importers, reachable).

Architect baseline at d320697: 139 files, 137536 lines. Reproduced.
Zero src_py importers: 15 including `app` (14 if app excluded as the R1 root;
architect 14 matches that convention). `epsf_zp_ok` is the 01B extra with
zero src_py importers (imported by R4).

Closure: 135 reachable / 4 unreachable from strict R1-R5.

## 3a Whole-module verify

| Candidate | Lines | Disposition | Evidence |
| --- | --- | --- | --- |
| wide_slope_noise_core.py | 1271 | KEEP R3 | dev/tests/test_wide_slope_noise_core.py:10 |
| validate_lc_crossval.py | 584 | KEEP | not R1-R5 strict; `dev/scripts/test_aperture_sweep.py:31` and `build_lc_from_fits_aperture.py:32` import it. Scope line: dev/ is a reachability root. Not named in STATE/ROADMAP/PROCESS. |
| xval_run.py | 255 | KEEP R2+R5 | xval_run.py:88 main+argparse; PROCESS.md:330/:354 |
| psf_neighbor_sub.py | 424 | KEEP R3+R5 | test_psf_neighbor_sub.py:7; ROADMAP names the module. Production src_py has zero imports (gated OFF). |
| method_lc_output.py | 373 | KEEP R3+R5 | test_get_lc_psf_strict.py:10; ROADMAP names it. Production src_py has zero imports. |
| trust_flag.py | 65 | KEEP R2 | CLI over trust_flag_core; argparse :15 |
| phase0_funnel.py | 72 | KEEP R4 | session_baseline_check.py:635 |
| run_crowding_index.py | 86 | DELETE | no importers in src_py/dev/tests; main-only CLI |
| run_smoothness_report.py | 61 | DELETE | no importers; hardcoded draft_000249 |
| inspect_drafts.py | 160 | DELETE | no importers; main without argparse |

### aperture_scatter_select / snr_cog_gates -- KEEP, no STOP

Not UI-selectable. `aperture_policy_mode` is only `f_fixed_night` /
`f_per_frame` (config.py:737). No scatter mode in the UI.

They are still reachable on the production aperture path:
pipeline.py:16021 `precompute_and_save_snr_aperture_table_for_draft` (always
when `_run_aperture`); photometry_core.py:1837 and :2956 import them.
The per-star snr_table radius branch is skipped when APERTURE-01 is set
(photometry_core.py:14345 `snr_aperture_table is not None and _ap01_mode is None`).
Deleting the modules would require deleting that call site and would drop
`aperture_snr_table.json` from the draft product. Not "reachable only through
a closed config mode". KEEP.

No exclusive config keys on the three deleted modules. 3d: nothing to remove.

## 3b Dead functions

vulture unused-function reachable: 183 (architect 184 unused-function
candidates on the whole tree; 183 in reachable modules).
Grep classified: 119 kept-used, 0 kept-dynamic, 64 dead.

Deleted 57 functions in 26 modules (one commit each). Names are in the
commit subjects.

Retained (dead by vulture+grep, not deleted):
- `psf_photometry`: `_epsf_allowed_catalog_ids`, `_epsf_positions_from_csvs`, `_psf_fit_region_mask` (107 lines)
- `psf_runner`: `step_1_build_epsf`, `step_3_run_psf_on_frames`, `step_4_build_summary`, `step_5_calibrate_lightcurve` (745 lines)

Reason: those two modules are on the G-EPSF import graph. Leaving them
untouched skips `--full-epsf` (hours saved, honesty kept). Inventory only.

F401: ruff --fix only on modules where the deleted functions owned the
import (cal_stage, dao_gaia_stage_validation, except_fix_counters,
osc_align, sat_diag, ui_variability). Pre-existing F401 in pipeline.py /
photometry_core.py left in place.

## 3c Inventory (no code change)

### 35 duplicated top-level helper names

Line numbers frozen at d320697 (before 3b splices). Matches architect count
when class methods are excluded. Full table: `dup_helpers_toplevel.json`.
Byte-equal (identical body sha): `_detail_help`,
`_is_corner`, `_peak_at`, `_saturation_limit`, `asinh_rgb`. All others differ.

| name | files:lines | byte-equal? |
| --- | --- | --- |
| _as_fits_float32_image | pipeline.py:2587; vyvar_alignment_frame.py:40 | no |
| _clamp | dao_gaia_calibration.py:232; param_resolver.py:153 | no |
| _coerce_bool | epsf_science_set.py:19; psf_internal_lc.py:190 | no |
| _compute_masterstar_score | night_run.py:457; ui_quality_dashboard.py:29 | no |
| _dao_full_to_binned_xy | masterstar_gaia_accounting.py:356; pipeline.py:7629 | no |
| _dao_pass2_annulus_stats | masterstar_gaia_accounting.py:80; pipeline.py:7660 | no |
| _dao_xy_binned_to_full | masterstar_gaia_accounting.py:348; pipeline.py:7619 | no |
| _detail_help | ui_dao_stars.py:16; ui_photometry.py:10; ui_settings.py:21 | YES |
| _eligible_mask | dao_gaia_stage_01_iter2.py:274; dao_gaia_stage_01_iter4.py:128 | no |
| _eval_poly | astrometry_optimizer.py:389; gaia_johnson.py:93 | no |
| _fit_shape_for_cutout | psf_photometry.py:431; psf_runner.py:233 | no |
| _flux_to_mag | comp_rms_loo.py:43; photometry_core.py:755; psf_neighbor_sub.py:23 | no |
| _fmt_opt_num | export_reports.py:387; ui_aperture_photometry.py:437 | no |
| _gaia_on_chip | dao_gaia_stage_01.py:160; dao_gaia_stage_01_iter2.py:101 | no |
| _header_has_vy_skysf | cal_stage.py:139; pipeline.py:18531 | no |
| _is_corner | dao_gaia_stage_01.py:300; dao_gaia_stage_01_iter2.py:139 | YES |
| _load_radec_map | crossmatch_runner.py:99; tess_runner.py:83 | no |
| _mad_sigma | comp_selection_per_target.py:1205; photometry_core.py:510; variability_detector.py:42 | no |
| _masterstar_candidate_path_for_job | night_run.py:493; ui_quality_dashboard.py:81 | no |
| _norm_cid | masterstar_gaia_accounting.py:561; photometry_report.py:49; psf_internal_lc.py:178; validate_lc_crossval.py:111 | no |
| _norm_id | astrometry_optimizer.py:363; epsf_science_set.py:26 | no |
| _normalize_ids | dao_reconcile.py:221; hrd_enrich.py:121 | no |
| _peak_at | dao_gaia_stage_01.py:317; dao_gaia_stage_01_iter2.py:156 | YES |
| _safe_float | catalog_crossmatch.py:26; hrd_enrich.py:36 | no |
| _saturation_limit | dao_gaia_stage_01.py:305; dao_gaia_stage_01_iter2.py:144 | YES |
| _sep_arcsec | catalog_crossmatch.py:42; repair_catalog_ids.py:38 | no |
| _warn_once | param_resolver.py:40; time_utils.py:26 | no |
| asinh_rgb | dao_gaia_stage_01.py:492; dao_gaia_stage_01_iter2.py:467 | YES |
| assign_states | dao_gaia_stage_01.py:326; dao_gaia_stage_01_iter2.py:207 | no |
| g3_spurious | dao_gaia_stage_01.py:424; dao_gaia_stage_01_iter2.py:246 | no |
| load_pipeline_meta | citations.py:357; invariants_runtime.py:611 | no |
| log | comp_qa.py:13; xval_run.py:42 | no |
| mad_sigma | comp_frame_normalize.py:140; comp_qa_core.py:33; comp_rms_loo.py:34 | no |
| main | 17 CLI/app entries (see json; includes deleted inspect_drafts / run_* at inventory time) | no |
| plate_scale_arcsec_per_px_from_wcs | dao_gaia_calibration.py:236; unit_resolver.py:34 | no |

One-line diff for non-equal rows: function bodies differ (distinct sha256 of
source span); not a comment-only delta. Input for the later single-helper stage.

### Aperture / annulus construction sites

Production geometry (APERTURE-01): `aperture_policy.resolve_aperture_geometry`
(aperture_policy.py:75-92):
- r_ap = max(0.5, f * FWHM)
- r_in = max(r_ap + 0.5, annulus_inner_fwhm * FWHM)
- r_out = max(r_in + 0.5, annulus_outer_fwhm * FWHM)
- f default 1.35; annulus_inner/outer_fwhm 2.7 / 5.2
- FWHM source: qc_metrics.fwhm_px / header VY_FWHM (FWHM-AUTH-01). Mode
  f_fixed_night uses night median; f_per_frame uses per-frame QC.
- Sky statistic (production photometry_core): plain median of annulus
  mask pixels, no rejection (SKY-CLIP-01). `_sky_pp_from_annulus_image`
  photometry_core.py:13880; used from `_annulus_sky_subtracted_flux` :3706
  and `enhance_catalog_dataframe_aperture_bpm` :14424 / `_aperture_flux_sky_batch` :13933.

Other CircularAnnulus sites (not the APERTURE-01 product path):

| Site | Formula | FWHM / r source | Sky statistic |
| --- | --- | --- | --- |
| photometry_core.py:2854 `_estimate_annulus_sky_pp` | caller r_in/r_out | caller | mean = aperture_sum / area (exact) |
| photometry_core.py:3692 `_annulus_sky_subtracted_flux` | caller r_ap/r_in/r_out | caller | median mask (SKY-CLIP-01) |
| photometry_core.py:13933 / :13989 `_aperture_flux_sky_batch` | caller | caller | median mask |
| photometry_core.py:14424 enhance_catalog APERTURE-01 | resolve_aperture_geometry | QC VY_FWHM | median mask |
| aperture_scatter_select.py:335 | ladder r_in/r_out | scatter ladder FWHM | median of mask values, n>=8 |
| psf_photometry.py:2209 `_annulus_median_per_px` | caller r_in/r_out | PSF path | median of mask values, n>=8 |
| masterstar_gaia_accounting.py:478 forced seed | r_in=r_px+4, r_out=r_px+8 | 1.5*FWHM or param | 3-sigma mean/med/std (plain_mean_med_std) |
| xval_run.py:181 offline harness | r_in, r_out CLI | fwhm/2.3548 sigma | ApertureStats.median, sigma_clip=None |

Full windows: `annulus_sites.json`.

## 3d Config

No config key was owned exclusively by a deleted module. No registry edit.

## G6 line counts

| | files | lines |
| --- | --- | --- |
| before (d320697) | 139 | 137536 |
| after | 136 | 136050 |
| delta | -3 | -1486 (-1.08%) |

Finding: sweep removed < 3%. Not a failure. Bulk of remaining unused
candidates sit in psf_runner step_* (745 lines, retained for G-EPSF skip)
and in kept-used functions that tests or production still call.

Deleted list:
- modules: inspect_drafts.py, run_crowding_index.py, run_smoothness_report.py
- functions: see commit subjects 3f4ab23..5d9cdd0 (26 modules)

## G-EPSF

Skipped. No deleted line in epsf_stage, psf_photometry, psf_internal_lc,
epsf_psf_merge, epsf_zp_ok, or psf_runner. (epsf_frame_accounting lost two
unused helpers; that module is not on the listed import graph.)

## Gates

| Gate | Status | Detail |
| --- | --- | --- |
| G1 before | PASS d320697; pytest 1628 passed, 32 skipped; clean-tree PASS | |
| G1 after | PASS 5d9cdd0; pytest 1628 passed, 32 skipped; clean-tree PASS | |
| G2 --full | PASS era04_aperture d55fcc9d n=53 / ext cc8b532e n=157. Pipeline 1423s. No full-epsf checks (ePSF OFF). Provenance DRIFT is git_hash (not FAIL). | |
| G4 live 516 | PASS csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d | |
| G6 | -1.08% lines (137536 -> 136050) | finding, not failure |
| G-EPSF | skipped | |

## STOPs and refutations

None. Spec vs code notes (not STOPs):
1. Architect "14 zero-importer modules" vs 15 including `app`. Convention.
2. validate_lc_crossval is unreachable from strict R1-R5 but imported by
   two dev/scripts. Kept because "dev/ is a reachability root".
3. aperture_scatter_select / snr_cog_gates still called from pipeline
   despite APERTURE-01 owning radii. Kept.

## Files / commits

Product tip after 3b: `5d9cdd0`.
3a `5f99d14`. 3b one commit per module through `5d9cdd0`.
Session data under `dev/results/context/session_20260831_c01a/`.
Push: `git push origin consolidate-01:consolidate-01`.
