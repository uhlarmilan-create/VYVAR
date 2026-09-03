# CURSOR RESULT - CONSOLIDATE-01 E-FINAL: glue dismantling, constants
# leaf, permanent facades

Date: 2026-09-03. Branch: consolidate-01. English. ASCII.
Architect: Claude. Implementer: Cursor.
Base: origin/consolidate-01 @ 5347b7f (E-DEAD report tip).
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Standing authority: refute before executing. ePSF-graph modules
untouched (no G-EPSF).

## What I did

Milan 2026-09-03 decisions written to the ledger:
D-FACADE-PERMANENT-01, D-CONSTANTS-LEAF-01, D-RUNFULL-HOME-01.

Phase A: new leaf `pipeline_constants.py` (imports nothing from
VYVAR). Moved pipeline-physical constants; SAT_LIMIT twin in
`pipeline_catalog.py` deleted; `test_sat_limit_twin_guard.py`
retired. `_PIXEL_MATCH_DEBUG_LOGGED` assigned-never-loaded, deleted.
Facade re-exports every moved constant.

Phase B: follow/inject pairs with test retargets; LAST_EXCLUDED_TARGETS
home is photometry_comp (PEP 562 on the facade); photometry_shared
placeholders converted to module-level imports; stale call-time
facade imports retargeted to real homes. ePSF-graph call-time facade
imports left as permanent state.

Phase C: eight per-wave facade getattr tests consolidated into
`test_facade_inventory.py` (bcead65 scalar-rebind rule) plus an AST
guard of sanctioned physical defs. `photometry.py` is a pure
`from photometry_core import *` alias shim (no logic; left).

## Architect-error ledger addition (verbatim)

errors 12 and 13 (E-DEAD hypotheses refuted:
band_classify comment misread as caller; _robust_scatter_mag test
misread as _robust_scatter_mad evidence; root cause both times =
substring grep asserted without reading the hit lines - same class
as errors 9-11).

## STOP points

None hit. Align body stayed call-time (cycle: astrometry_align
imports pipeline_astrometry). merge_photometry_pipeline_meta
patches on photometry_core stay because epsf_stage / epsf_psf_merge
are untouchable. pl.LOGGER is physical on the facade; log_event is
infolog (patches stay). run_full stays (D-RUNFULL-HOME-01).
photometry_core skip-if-incomplete phase2a import remains (core <->
phase2a first-import cycle; not the None+inject mechanism).

vyvar_platesolver.py still has its own
`_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO = 1.15` numeric twin (out of
scope; only SAT_LIMIT twin was dismantled).

## Commits

Phase A:
| SHA | concern |
| --- | --- |
| 8136282 | pipeline_constants leaf + facade re-export; delete _PIXEL_MATCH_DEBUG_LOGGED |
| f2faeb1 | pipeline_catalog SAT_LIMIT from leaf; retire test_sat_limit_twin_guard |
| ac4c9d2 | masterstar_build constants from leaf |
| d130f4d | pipeline_calibrate SAT_LIMIT_CONTAINER_CLIP_ADU from leaf |
| 83e4f60 | pipeline_astrometry _EXO_HOST_ANNOTATION_COLUMNS from leaf |
| d8d1ccc | photometry_lightcurve SAT_LIMIT_CONTAINER_CLIP_ADU from leaf |

Phase B pairs:
| SHA | facade name | patching tests | follow/inject site |
| --- | --- | --- | --- |
| fa1ccac + 20f36d7 | _fit_subtract_preprocess_sky_surface | test_osc1_extraction.py:200; E5 getattr | pipeline.py follow; cal_diag call-time (cycle) |
| 3db4e01 | _plate_solve_input_bundle | none (dead glue); E6a follow list emptied | pipeline.py follow |
| 9b0a414 | extract_fits_metadata (gate_helpers) | test_except_fix2 exc0389 | pipeline.py follow into gate_helpers |
| 483780a | _resolve_git_provenance | test_f431_labbe_provenance (photometry_provenance) | photometry_core lambda inject |
| 2787569 | _enrich_active_targets_bp_rp / _ensure_active_target_display_names | test_f428_fixbatch.py | photometry_core self-injects |
| 449efb0 | LAST_EXCLUDED_TARGETS / select_active_targets wrap | test_f428_fixbatch.py:277 | functools wrapper + sync |
| 33328e8 | _fill_masterstars_gaia_matched_bp_rp_from_local_db | test_invariants_p2.py | masterstar_build call-time |
| a8494e9 | _astrometry_align_impl_body | test_astrometry_fault_isolation.py:59/82/96 | pipeline_astrometry call-time (kept; pointed at astrometry_align) |
| 29cdc40 | extract_fits_metadata (bundle) | test_except_fix2 exc0312 | pipeline_astrometry call-time |
| b441a62 | catalog_cone_radius_deg_from_optics | test_except_fix2 exc0342 | pipeline_catalog call-time |
| eb95f37 | _all_pix2world_icrs_deg (+ maybe_rescale) | test_except_fix2 exc0317 | pipeline_astrometry call-time |
| 5a1818b | pipeline.AppConfig | test_field_run_findings / test_border_ram_handoff | write_photometry_plan uses config.AppConfig() |
| 8176f92 | calibrate MP spawn imports | test_calibrate_mp_spawn.py | test-only retarget |
| 6c70198 | read_flux_from_csv inject | test_phase2a_saturated_skip.py | photometry_phase2a lambda into phase2a_target |
| c7e16e0 | photometry_shared placeholders (12) | facade getattr (later inventory) | None+inject dismantled |
| c243973 | find_qc_metrics_csv / inv_sat_limit / stamp_vsx / bad_columns | none (unpatched) | stale call-time |

Phase C / docs:
| SHA | concern |
| --- | --- |
| 587d1a4 | test_facade_inventory.py + AST guard; delete eight per-wave facade tests |
| (report) | decisions + ROADMAP + this report + ledger stamp |

Product SHA for gates: `587d1a4`.

## Pair table (facade name x tests x follow x commit)

See Phase B table above. Additional classification:

- photometry_core.merge_photometry_pipeline_meta: STAY on facade
  (epsf_stage.py / epsf_psf_merge.py / test_epsf_stage.py; ePSF graph
  untouchable). test_epsf_chain_01b night_run path also patches the
  facade; run_full patch stays (physical home).
- pl.log_event: infolog re-export; test_invariants_p2 patches and
  calls pl.log_event. Stay.
- pl.LOGGER: physical on pipeline.py. Stay.

## Placeholder conversion table

| name | home | mechanism |
| --- | --- | --- |
| _sky_pp_for_photometric_error | photometry_phase2a | module-level import at photometry_shared EOF |
| _coerce_bool_cell | photometry_lightcurve | module-level import at photometry_shared EOF |
| _assert_inv_err_sigma_acct_01 | photometry_gate_helpers | module-level import at photometry_shared EOF |
| _clamp_err_empty_apertures_n | photometry_gate_helpers | same |
| _clamp_err_empty_apertures_min | photometry_gate_helpers | same |
| _labbe_content_seed_from_header | photometry_gate_helpers | same |
| _sigma_bkg_r_key | photometry_gate_helpers | same |
| _sky_pp_from_annulus_image | photometry_gate_helpers | same |
| bkg_scale_ratio_empirical_over_howell | photometry_gate_helpers | same |
| compute_setup_bkg_scale_r | photometry_gate_helpers | same |
| measure_empty_aperture_sigma_bkg | photometry_gate_helpers | same |
| scaled_sigma_bkg_ap_from_howell | photometry_gate_helpers | same |

gate_helpers `_normalize_gaia_id` now imports from photometry_shared
directly (cycle rationale dismantled). compute_per_frame_cog_correction
also imported from photometry_shared (was facade; blocked the
partial-core import).

No placeholder kept as lazy import.

## Constants consumer table

| consumer | names | commit |
| --- | --- | --- |
| pipeline.py | all moved constants re-exported | 8136282 |
| pipeline_catalog.py | SAT_LIMIT_NO_KNEE_FRAC (twin deleted) | f2faeb1 |
| masterstar_build.py | SAT_LIMIT_* + _MASTERSTAR_* | ac4c9d2 |
| pipeline_calibrate.py | SAT_LIMIT_CONTAINER_CLIP_ADU | d130f4d |
| pipeline_astrometry.py | _EXO_HOST_ANNOTATION_COLUMNS | 83e4f60 |
| photometry_lightcurve.py | SAT_LIMIT_CONTAINER_CLIP_ADU | d8d1ccc |

## Retired tests

- `dev/tests/test_sat_limit_twin_guard.py` (single-source restored)
- `dev/tests/test_consolidate_e1_facade.py` through
  `test_consolidate_edead_facade.py` (eight files; history in git)

## Final facade inventory

`pipeline.py`: **1047** lines.
Physical defs: `_analyze_calibrated_qc_one`, `analyze_calibrated_qc`,
`AstroPipeline`. LOGGER is physical. Constants are leaf re-exports.

`photometry_core.py`: **1233** lines.
Physical defs: `compute_auto_fwhm_limit`, `run_full_photometry_pipeline`,
plus PEP 562 `__getattr__` for LAST_EXCLUDED_TARGETS.

`photometry.py`: star-import alias of photometry_core. No logic.

AST guard: `test_facade_inventory.py::test_facade_physical_def_ast_guard`.

## Gates

| gate | result | log |
| --- | --- | --- |
| G1 C1 8136282 | PASS 1663 / 32 skip | g1_c1.txt |
| G1 C2 f2faeb1 | PASS 1662 / 32 skip (twin test retired) | g1_c2.txt |
| G1 after each Phase A/B commit | PASS 1662 / 32 skip | g1_c3.txt .. g1_c23.txt |
| G1 C15 a8494e9 --fast | contaminated (dirty tree mid-exc0312); --clean PASS | g1_c15.txt |
| G1 C24 587d1a4 | PASS 1621 / 32 skip (eight per-wave facade tests -> 7 inventory tests) | g1_c24.txt |
| G2 --full at 587d1a4 | PASS era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157 (1348s) | g2_full.txt |
| G4 live 516 before G2 | PASS csv `bfa24039` / fits `13e77cf8` / epsf `172f9540` | g4_before.txt |
| G4 live 516 after G2 | PASS same prefixes (unchanged) | g4_after.txt |
| G-EPSF | not run | ePSF-graph files untouched |

Ledger stamp: `VL-COUNTERS-ZERO` and `VL-ANCHOR-WCSINV` -> `587d1a4`.

## CONSOLIDATE-01 closure readiness

E-program product work is on the branch. After G2 `--full` (era04_aperture
d55fcc9d n=53 / ext cc8b532e n=157) and G4 live 516 (csv bfa24039 /
fits 13e77cf8 / epsf 172f9540) PASS at product SHA 587d1a4, the
branch is ready for Milan's fast-forward of main (PUSH_AUTH).
Next product work per ROADMAP: C-EXPORT-GAP, then FRAME-QC-PARITY
remainder, then SKY-SURFACE-BLAST-RADIUS.

MERGE_SHA is the report-commit SHA (printed in the close result).

Milan fast-forward of main (PUSH_AUTH):

```
git fetch origin
git merge --ff-only origin/consolidate-01
```
