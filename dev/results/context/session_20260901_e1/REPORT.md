# CURSOR RESULT - CONSOLIDATE-01E1 extraction wave 1

Date: 2026-09-01 (gates finished 2026-09-02 UTC). Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ 3f20fc1.
Work started from current tip `4af3c90` (E0 map + SEL-GHOST-01 MERGED ROADMAP close).
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Compared with: E0 `module_map_proposal.md` buckets vs this wave's actual moves.
They differ in three architect-mandated ways (section 1): no `*_dead.py`;
giant-module names reserved for E4/E6; AstroPipeline stays in pipeline.py.
Item 7 also differs from the E0 split of two ePSF-hook files: one `epsf_hooks.py`,
and the two annulus helpers go to `photometry_gate_helpers.py` (evidence below).

## What I did

Pure moves of seven small isolated modules (one commit each), facades re-export,
external `src_py` / `dev` callers untouched except FLOW facts (G1 guard) and
one facade follow-proxy for a risk-register patch string. No body edits of
moved defs, no renames, no signature changes.

## Moves per module

| # | module | commit | n defs | facade | notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `fits_meta.py` | `6f6d3c4` | 20 | pipeline.py (top import) | E0 `pipeline_import` bucket. Self-contained (no remaining pipeline callees). |
| 2 | `pipeline_ui_helpers.py` | `75d95d7` | 5 | pipeline.py (late) | stay-callees imported from pipeline at module load (after facade is complete). |
| 3 | `photometry_ui_helpers.py` | `fd2fe46` | 2 | photometry_core.py (late) | `TIME_BASE_*` imported from photometry_core. |
| 4 | `pipeline_gate_helpers.py` | `da0cd06` | 1 | pipeline.py (late) | `validate_comparison_ensemble_flatness`. |
| 5 | `photometry_gate_helpers.py` | `2c3cea0` | 26 | photometry_core.py (late) | E0 25 + `_annulus_sky_subtracted_flux` (item 7). |
| 6 | `photometry_exports.py` | `0dd8d51` | 5 | photometry_core.py (late) | sibling import of `_resolve_star_flux_method`, `comp_quality_quality_strings`. |
| 7 | `epsf_hooks.py` | `08ce4ed` | 3 | both facades (late) | remaining ePSF-hook defs after item-7 verdict. |

Moved names (commit messages list the same sets):

- **fits_meta.py:** `_safe_filter_token`, `observation_group_key_from_metadata`, `_summarize_lights_binning_from_headers`, `log_lights_binning_from_headers_preflight`, `generate_observation_hash`, `_fits_pixel_raw_to_micrometres`, `_focal_mm_plausible`, `_merge_equipment_pixel_into_metadata`, `_recompute_effective_pixel_from_physical`, `_header_pick_first`, `_enrich_calibration_metadata_from_header`, `_apply_draft_combined_to_pipeline_meta`, `_fits_meta_ra_deg`, `_fits_meta_dec_deg`, `_parse_fits_binning_int`, `_log_effective_pixel_pitch`, `fits_metadata_from_primary_header`, `_valid_bayerpat_from_header`, `extract_fits_metadata`, `scan_usb_folder`.
- **pipeline_ui_helpers.py:** `_resolve_light_fits_for_quality_inspection`, `run_quality_analysis`, `list_best_processed_light_paths_for_masterstar`, `resolve_masterstars_metadata_csv`, `preprocess_sky_summary_from_df`.
- **photometry_ui_helpers.py:** `resolve_lc_time_base`, `lc_time_axis_short_label`.
- **pipeline_gate_helpers.py:** `validate_comparison_ensemble_flatness`.
- **photometry_gate_helpers.py:** `_sigma_bkg_r_key`, `_assert_inv_err_sigma_acct_01`, `comp_quality_quality_strings`, `_clamp_err_empty_apertures_n`, `_normalize_err_background_mode`, `_labbe_content_seed_from_header`, `measure_empty_aperture_sigma_bkg`, `estimate_star_free_per_pixel_variance_adu2`, `_howell_bkg_variance_adu2`, `_clamp_bkg_scale_r`, `bkg_scale_ratio_empirical_over_howell`, `compute_setup_bkg_scale_r`, `scaled_sigma_bkg_ap_from_howell`, `measure_growth_curve_ee`, `_phase2a_star_mag_lookup`, `discover_aligned_science_fits`, `_median_bkg_var_from_aligned_frames`, `_estimate_annulus_sky_pp`, `_annulus_sky_subtracted_flux`, `_resolve_star_flux_method`, `_frame_quality_gate_select`, `_recompute_bjd_hjd_per_target`, `photometer_check_star_production_path`, `_compute_fov_max_dist`, `_sky_pp_from_annulus_image`, `_aperture_flux_sky_per_star`.
- **photometry_exports.py:** `lc_has_finite_airmass`, `apply_comp_w_rel_for_display`, `ensemble_member_ids`, `_get_lc_psf_strict`, `_get_lc_adaptive_per_star`.
- **epsf_hooks.py:** `_add_catalog_ids_from_csv`, `_epsf_lc_catalog_ids` (from pipeline.py), `load_epsf_metrics_for_draft` (from photometry_core.py).

C-A: no `*_dead.py`. Dead-bucket defs stayed in the facades.
C-B / C-C: not this wave (giant names reserved; AstroPipeline stays in pipeline.py).

Facade line counts after the wave: pipeline.py 19095 (was 20124 at E0); photometry_core.py 16455 (was 17525).

## Facade re-export

- `pipeline.py`: `from fits_meta import (...)` immediately after the existing top import block (`import itertools`). Late imports at EOF for `pipeline_ui_helpers`, `pipeline_gate_helpers`, `epsf_hooks` (pipeline names only).
- `photometry_core.py`: late imports at EOF (after `__all__`) for `photometry_ui_helpers`, `photometry_gate_helpers`, `photometry_exports`, `epsf_hooks.load_epsf_metrics_for_draft`. `__all__` still contains `load_epsf_metrics_for_draft`; `photometry.py` star-import unchanged.
- Call-time follow (`a3ddce7`): after importing `pipeline_gate_helpers`, pipeline binds `_pipeline_gate_helpers.extract_fits_metadata = lambda *a, **k: extract_fits_metadata(*a, **k)` so `monkeypatch.setattr("pipeline.extract_fits_metadata", ...)` still reaches `validate_comparison_ensemble_flatness`. Not a body edit of the moved def.

## Item-7 verdict

**`_annulus_sky_subtracted_flux` and `_sky_pp_from_annulus_image` are shared production photometry, not ePSF-only.** They landed in `photometry_gate_helpers.py` (bucket 5), not `epsf_hooks.py`.

Evidence:

1. The moved docstring still says "shared DAO/PSF path" (`photometry_gate_helpers.py` `_annulus_sky_subtracted_flux`).
2. Aperture path: `measure_empty_aperture_sigma_bkg` (same gate-helpers module) calls `_annulus_sky_subtracted_flux`. `_aperture_flux_sky_per_star` (gate-helpers) and remaining `_aperture_flux_sky_batch` (photometry_core, dead-bucket / stays) call `_sky_pp_from_annulus_image`.
3. ePSF path (unchanged import site, facade re-export): `psf_photometry.py` does `from photometry_core import _annulus_sky_subtracted_flux`; `psf_neighbor_sub.py` does the same.
4. E0 already listed `photometry_gate_helpers` as a caller of both; the map's ePSF-hook tag was a misplaced stage tag.

`epsf_hooks.py` therefore holds the other three ePSF-hook defs only. Those three are **not** imported by `psf_photometry` / `psf_internal_lc`. G-EPSF still ran because R-GATE keys off any moved def on the ePSF import graph: `_annulus_sky_subtracted_flux` is imported by `psf_photometry`.

## Risk-register hits and tests

No MP spawn workers appeared in wave 1.

| name | register | test |
| --- | --- | --- |
| `extract_fits_metadata` | string/getattr patch `pipeline.extract_fits_metadata` | `test_e1_extract_fits_metadata_patch_string_path`; `test_exc0389_stress_sidecar_skip_counted` (follow-proxy) |
| `_annulus_sky_subtracted_flux` | patch-string + private test imports | `test_e1_annulus_patch_string_path`; getattr loop |
| `load_epsf_metrics_for_draft` | `in__all__` | star-import + `__module__ == "epsf_hooks"` |
| `_epsf_lc_catalog_ids` | private test imports | `test_e1_epsf_hooks_private_and_all` |
| `_get_lc_psf_strict` | private test imports | getattr loop |
| `_assert_inv_err_sigma_acct_01`, `_clamp_bkg_scale_r`, `_clamp_err_empty_apertures_n`, `_frame_quality_gate_select`, `_howell_bkg_variance_adu2`, `_labbe_content_seed_from_header`, `_normalize_err_background_mode`, `_recompute_bjd_hjd_per_target`, `_resolve_star_flux_method`, `_sigma_bkg_r_key` | private test imports | getattr loop in `dev/tests/test_consolidate_e1_facade.py` |

## Follow-up commits (not new modules)

- `dfe7145` FLOW sync: `DOC_FUNCTIONS` now points `measure_empty_aperture_sigma_bkg` at `photometry_gate_helpers.py`; builder footnote + `docs/VYVAR_FLOW_CZ.pdf` regenerated. First G1 failed `test_flow_doc_functions_exist`.
- `a3ddce7` patch-path follow for `pipeline.extract_fits_metadata` (second G1 failed `test_exc0389_stress_sidecar_skip_counted`).

## Gates

Product SHA for gates: `a3ddce7`.

| gate | status | detail |
| --- | --- | --- |
| G1 `--fast --clean` | PASS at `a3ddce7` | 1618 passed, 32 skipped (1612 at E0 + 6 facade tests). clean-tree PASS. db-quick-check WARN waived. Log: `g1.txt` |
| G2 `--full` aperture | PASS at `a3ddce7` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1496s. Log: `g2_full.txt` |
| G-EPSF `--full-epsf` | PASS at `a3ddce7` | epsf01 `c743b8ba` n=53. ePSF stage 15987s, n_stars=63, 53 PSF LCs. G3 residual BO dem=12.505 (ref 12.505), n_full=134. Aperture hashes unchanged. Pipeline 1860s. Log: `g_epsf.txt` |
| G4 live 516 | PASS before G2, after G2, after G-EPSF | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. |

G4 path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/`
(`masterstars_full_match.csv`, `MASTERSTAR.fits`, `masterstar_epsf.fits`).

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` stamped to `a3ddce7` by `--full-epsf`.

## Files changed

Seven module commits + FLOW + follow-proxy + this report/logs/ledger:

- `src_py/fits_meta.py` (new)
- `src_py/pipeline_ui_helpers.py` (new)
- `src_py/photometry_ui_helpers.py` (new)
- `src_py/pipeline_gate_helpers.py` (new)
- `src_py/photometry_gate_helpers.py` (new)
- `src_py/photometry_exports.py` (new)
- `src_py/epsf_hooks.py` (new)
- `src_py/pipeline.py`, `src_py/photometry_core.py` (cuts + re-exports)
- `dev/tests/test_consolidate_e1_facade.py`
- `dev/tools/docs_pdf/flow_doc_facts.py`, `build_flow_doc.py`, `docs/VYVAR_FLOW_CZ.pdf`
- `dev/validation/VYVAR_VALIDATION_LEDGER.json`
- `dev/results/context/session_20260901_e1/` (this REPORT + gate logs)

## Errors

1. First G1 FAIL: FLOW `def measure_empty_aperture_sigma_bkg` no longer in `photometry_core.py`. Fixed `dfe7145`.
2. Second G1 FAIL: `test_exc0389` patches `pipeline.extract_fits_metadata` after the caller moved. Fixed `a3ddce7`.
3. First G2 attempt: Tee-Object could not write `g2_full.txt` (file lock); rerun to `tmp/e1_g2_full.txt` then copied into the session dir.

None remaining.
