# CURSOR RESULT - CONSOLIDATE-01C helper dedup (no numeric change)

Date: 2026-08-31. Branch: consolidate-01. English. ASCII.
Base: origin/consolidate-01 @ 8cfff34. Live draft 516 was not written.
Push: origin/consolidate-01 only (after G1-after / G2 / G-EPSF / G4).

Input: `dev/results/context/session_20260831_c01a/dup_helpers_toplevel.json`
(frozen at d320697; lines re-located at 8cfff34).

## Premise (Rule 0.1)

Compared: 35 duplicated top-level helper *names* (01A inventory) vs bodies
at tip 8cfff34. A name with two bodies is the defect. Merge only when the
survivor is call-compatible and bit-identical in effect at every live call
site (R2). Otherwise rename so each behaviour has one honest name.
No photometry numeric policy change (R1); G2 byte-identity is the proof.

## What I did

C-1: five byte-equal names to one home (UI `_detail_help` -> `ui_help.py`;
dao_gaia family `_is_corner` / `_peak_at` / `_saturation_limit` / `asinh_rgb`
-> family-local `dao_gaia_common.py`, not a global home, R4).

C-2: merge when bit-identical; otherwise rename in place. `ids_norm.py` was
not created (id family does not share semantics). `stats_core.py` holds
`_flux_to_mag` and `_coerce_bool` survivors.

C-3: left CLI `main` / `log` and dao_gaia generation helpers (R4).
`_compute_masterstar_score` + `_masterstar_candidate_path_for_job` were
the same job logic: UI copies deleted (C-2). `_fit_shape_for_cutout` was
not the same algorithm: renamed the psf_runner copy.

R3: one duplicated name per commit (30 commits on consolidate-01).

## Per-name verdict

| name | survivor | deleted / renamed copies | category | evidence |
| --- | --- | --- | --- | --- |
| `_detail_help` | `ui_help.py` | ui_dao_stars, ui_photometry, ui_settings | C-1 merge | byte-equal expander body |
| `_is_corner` | `dao_gaia_common.py` | stage_01, iter2 | C-1 merge | byte-equal; CORNER_MARGIN_PX=120; family-local R4 |
| `_peak_at` | `dao_gaia_common.py` | stage_01, iter2 | C-1 merge | byte-equal r=3 patch max |
| `_saturation_limit` | `dao_gaia_common.py` | stage_01, iter2 | C-1 merge | SATURATE/VY_SATURATE/HISTCUTLO else 60000 |
| `asinh_rgb` | `dao_gaia_common.py` | stage_01, iter2 | C-1 merge | percentile 99.5 asinh stretch |
| `_flux_to_mag` | `stats_core.py` | photometry_core, comp_rms_loo | C-2 merge | nan if not finite or flux<=0; -2.5 log10 |
| `_flux_to_mag` (psf) | `_flux_to_mag_with_zp` in psf_neighbor_sub | (renamed, not merged) | C-2 rename | zp, returns +inf not nan |
| `_mad_sigma` | `_mad_sigma_or_std_floor` (photometry_core); `_mad_sigma_or_nan` (variability_detector) | unused copy deleted in comp_selection_per_target | C-2 rename | std+1e-9 floor vs nan on n<3; dead third copy |
| `mad_sigma` | `mad_sigma_or_std` / `mad_sigma_normalized_or_nan` / `mad_sigma_scale_or_zero` | three live copies renamed | C-2 rename | std/inf vs nan n<2 vs 1.4826*mad (0 when MAD=0) |
| `_norm_cid` | four honest names (see below) | four bodies | C-2 rename | not the same: strip .0 vs Gaia vs Decimal vs regex \d+.0+ |
| `_norm_id` | normalize_gaia_source_id (astrometry); `_norm_id_gaia_or_raw` (epsf); `_strip_id_series` (night_run nested) | alias deleted / renamed | C-2 rename | night_run was Series.str.strip, not Gaia |
| `_normalize_ids` | `_normalize_id_series` / `_unique_normalized_gaia_ids` | dao_reconcile vs hrd_enrich | C-2 rename | DataFrame column vs unique list |
| `_clamp` | `_clamp_lo_hi` / `_clamp_param_sanity` | dao_gaia_calibration vs param_resolver | C-2 rename | (value,lo,hi) vs SANITY[param] |
| `_safe_float` | `_safe_float_blank_tokens` / `_safe_float_masked` | catalog vs hrd | C-2 rename | `--` tokens vs .mask |
| `_coerce_bool` | `stats_core.py` | epsf_science_set, psf_internal_lc | C-2 merge | None/nan -> False at all live map() sites |
| `_warn_once` | `_warn_once_logger` / `_warn_once_infolog` | param_resolver vs time_utils | C-2 rename | logger.warning vs log_event |
| `_eval_poly` | `_eval_poly2d` / `_eval_poly1d` | astrometry vs gaia_johnson | C-2 rename | packed 2-D vs 1-D coeffs |
| `_load_radec_map` | `crossmatch_runner.py` | tess_runner wrapper deleted | C-2 merge | tess already delegated at 8cfff34 |
| `_sep_arcsec` | `_sep_arcsec_skycoord` / `_sep_arcsec_small_angle` | catalog vs repair | C-2 rename | SkyCoord vs small-angle formula |
| `_fmt_opt_num` | `_fmt_opt_num_na` / `_fmt_opt_num_empty` | export vs UI | C-2 rename | na= vs empty=- |
| `_header_has_vy_skysf` | `cal_stage.py` (pipeline re-export) | pipeline body deleted | C-2 merge | missing/unrecognized -> False both ways |
| `_as_fits_float32_image` | `vyvar_alignment_frame.py` | pipeline body deleted | C-2 merge | ascontiguousarray float32 |
| `plate_scale_arcsec_per_px_from_wcs` | unit_resolver keeps name; dao_gaia -> `_nan` | dao_gaia_calibration renamed | C-2 rename | no-abs/nan vs abs/None |
| `_dao_xy_binned_to_full` | `masterstar_gaia_accounting.py` (pipeline re-export) | pipeline body deleted | C-2 merge | x*f+(f-1)*0.5; f<=1 identity |
| `_dao_full_to_binned_xy` | same | pipeline body deleted | C-2 merge | inverse; bfac<=1 identity |
| `_dao_pass2_annulus_stats` | same | pipeline body deleted | C-2 merge | r=8-12, n>=10, plain_mean_med_std sigma=3 maxiters=2 |
| `load_pipeline_meta` | `invariants_runtime.py` (citations re-export) | citations body deleted | C-2 merge | None->{}; non-dict JSON->{} (citations used to return raw list) |
| `_compute_masterstar_score` | `night_run.py` | ui_quality_dashboard deleted | C-2 merge | weights 0.45/0.30/0.15/0.10; FWHM, ELONGATION_MEAN |
| `_compute_masterstar_score` (report) | `_compute_masterstar_score_fwhm_px` | photometry_report method renamed | C-2 rename | FWHM_PX/ELONGATION cols; unused |
| `_masterstar_candidate_path_for_job` | `night_run.py` | UI copy deleted | C-2 merge | same path logic; archive Optional; log_event -> LOGGER.warning |
| `_fit_shape_for_cutout` | psf_photometry keeps name | psf_runner -> `_legacy_even_down` | C-2 rename | even-down vs FWHM/even-up; cutout=20 -> 15 vs 17 |

### `_norm_cid` honest names

| module | new name | behaviour |
| --- | --- | --- |
| masterstar_gaia_accounting | `_norm_cid_strip_dotzero` | strip; trailing `.0` on all-digit |
| psf_internal_lc | `_norm_cid_gaia` | `normalize_gaia_source_id`, except -> raw |
| photometry_report | `_norm_cid_decimal` | `int(Decimal(s))` |
| validate_lc_crossval | `_norm_cid_int_dotzero` | regex `\d+\.0+` |

`ids_norm.py` not created.

## Coordinate transforms -- per-argument-range identity

Survivor: `masterstar_gaia_accounting.py` (production already imports it;
pipeline importing accounting is the existing direction; reverse would cycle).

Bodies at 8cfff34 were text-identical except pipeline's inner `import numpy`
and a docstring. Effect:

- `f<=1` / `bfac<=1`: return float64 copies / identity floats.
- `f>=2`: `x_full = x_binned * f + (f-1)*0.5`; inverse `(x_full - off)/f`.
- Roundtrip for integer f in {1,2,3,4} and sample x in {0, 0.5, 100, 2047.5}
  is exact in IEEE float64 (affine with dyadic offset).
- Annulus: same rmax=13, rr in [8,12], n<10 -> (nan,nan), same
  `plain_mean_med_std(..., sigma=3.0, maxiters=2)`.

Wrong merge would move stars; this merge does not change the formula.

## C-3 left (intentional)

| name | why left |
| --- | --- |
| `main` | CLI convention; many R2 CLI roots. 01A already deleted inspect_drafts / run_crowding_index / run_smoothness_report mains. |
| `log` | `comp_qa.py:13` and `xval_run.py:42` -- two tiny print flush loggers, not shared. |
| `_eligible_mask` / `_gaia_on_chip` / `assign_states` / `g3_spurious` | dao_gaia generation family (stage_01 vs iter2/3/4). R4: do not merge across generations. |
| `_fit_shape_for_cutout` | spec listed C-3; bodies differed -- treated as C-2 rename (psf_runner still live via `_load_psf_photometry_bundle`). |

## R4 -- production imports of dao_gaia generations

Grep at tip (`pipeline.py`, `night_run.py`, `app.py`): **no**
`dao_gaia_stage_01*` imports.

Family-internal: iter2 imports stage_01; iter3 imports iter2+stage_01;
iter4 imports iter2+iter3+stage_01. `dao_gaia_stage_validation.py`
imports **iter4** as the CLI-family production plus `FRAMES` from stage_01.

Production MASTERSTAR path uses `masterstar_gaia_accounting`, not the
stage CLI modules.

### Question for Milan

Can dao_gaia_stage_01 / `_iter2` / `_iter3` modules older than the
CLI-family production generation (iter4, as used by
`dao_gaia_stage_validation.py`) be retired as modules? No deletion in 01C.

## STOPs / spec vs code

1. `tess_runner._load_radec_map` was already a one-line wrapper around
   `crossmatch_runner` at 8cfff34 (not two bodies). Deleted the wrapper.
2. Inventory `_norm_id` at night_run:792 was a **nested** Series.strip,
   not Gaia normalize. Renamed `_strip_id_series`.
3. `comp_selection_per_target._mad_sigma` had **zero call sites** (dead).
   Deleted.
4. `photometry_core._flux_to_mag` line moved vs d320697 (01B SNR deletion).
5. G-EPSF: spec expected skip (`psf_neighbor_sub._flux_to_mag` does not
   trigger it). **Not skipped**: `_coerce_bool` merge and `_norm_cid` /
   `_norm_id` renames edited `epsf_science_set.py` and `psf_internal_lc.py`.
   G-EPSF is required.

## Gates

| gate | result |
| --- | --- |
| G1 --fast --clean before | PASS at 8cfff34: 1603 passed, 32 skipped. OVERALL PASS. Log: g1_before.txt |
| G1 --fast --clean after | PASS at dcc52b4: 1603 passed, 32 skipped. OVERALL PASS. Log: g1_after.txt |
| G2 --full aperture | PASS era04_aperture d55fcc9d n=53 / ext cc8b532e n=157. Log: g2_full.txt |
| G-EPSF --full-epsf | PASS (`g_epsf.txt`). epsf01 c743b8ba89f4ac54 n=53. G3 residual PASS BO dem=12.505 n_full=134 (ref 12.505); FW same PASS row as 01B (column truncated). ePSF stage 16744 s, 53 PSF LCs. Aperture hashes unchanged. Pipeline 1687 s. |
| G4 live 516 | PASS after G2 and after G-EPSF: csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d |

## Files changed

New: `src_py/ui_help.py`, `src_py/dao_gaia_common.py`, `src_py/stats_core.py`.
No `ids_norm.py`. Live 516 untouched. Historical `dev/results/**` untracked,
not committed.
