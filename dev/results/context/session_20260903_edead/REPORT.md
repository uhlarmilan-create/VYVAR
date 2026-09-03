# CURSOR RESULT - CONSOLIDATE-01 E-DEAD: the two E0 dead buckets
# through the FULL 01A R1-R5 reachability

Date: 2026-09-03. Branch: consolidate-01. English. ASCII.
Architect: Claude. Implementer: Cursor.
Base: origin/consolidate-01 @ b44c82f (E6b report tip).
Work started from local tip `b44c82f`.
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Literature: 01A v2 protocol
(`dev/results/context/session_20260831_c01a/REPORT.md`).
Deletion rule: R1-R5 closure miss AND vulture (conf >= 60) AND
word-boundary grep. One deletion commit per module. Deletions
before moves. INV-GATE-REMOVAL analog: G2/G4 byte-identity proves
only that deleted code was not on the anchored path; deadness
evidence is the reachability protocol.

## What I did

Independent ref classification of the 27 E0 names (section 2
classes). Two architect hypotheses refuted. Deleted 10
unreachable defs (3 photometry_core, 7 pipeline) including
glue/docstring rewords. R-MOVE of 14 alive defs to stage homes
(facade re-export + call-time policy as E5/E6). Left
`analyze_calibrated_qc` + `_analyze_calibrated_qc_one` in
`pipeline.py` (AstroPipeline-adjacent; E-final revisits).
No G-EPSF.

## ePSF import graph (G-EPSF not required)

Sweep of `src_py/*epsf*` / `src_py/*psf*` imports from
`photometry_core`: `merge_photometry_pipeline_meta`,
`_annulus_sky_subtracted_flux`, `_resolve_git_provenance`,
`ensemble_normalize`. From `pipeline`: the four E6a names in
`epsf_psf_merge.py`. None of the 27 E-DEAD names. Architect
claim stands. See `classification.json`.

## Refutations

1. `_is_broadband_photometric_filter`: architect said REAL
   caller `band_classify.py`. Measured: comment only at
   `band_classify.py:250` ("legacy heuristic"). DELETE.
2. `test_comp_pool_noise_s1.py:22`
   `test_robust_scatter_mad_on_gaussian` calls
   `comp_pool_noise._robust_scatter_mag`, not
   `photometry_core._robust_scatter_mad`. Not a TEST-ROOT.
   `_robust_scatter_mad` is still ALIVE via
   `measure_empty_aperture_sigma_bkg`.

vulture does not follow cross-module imports, so it marks many
ALIVE names unused. 01A treated that as keep/move, not a wave
STOP. Intra-module transitive pairs become unused after the
parent is deleted (fixed-point: `_per_frame_noise_error_map` /
`_frame_gain_readnoise_for_error_map` and `_star_saturation_flags`
/ `_saturated_core_plateau`).

R5: none of the 10 delete names appear in STATE / ROADMAP /
PROCESS as a command or architecture element. STATE names the
config key `cog_aperture_correction_enabled`, not
`compute_per_frame_cog_correction`. TXT-DUMP `.txt` files under
`dev/scripts/` are captured audit stdout, not executed
(`vyvar_audit.txt`, `solver_audit.txt` have no runner).

CoG ledger: `compute_per_frame_cog_correction` is AUDITED-CLEAN
in `dev/results/VYVAR_FULL_AUDIT_LEDGER.md` (row 9687). The
CONSOLIDATE-01 CoG/scatter product deletion removed SNR-table
modules, not this gated REAL path.
`enhance_catalog_dataframe_aperture_bpm` calls it when
`cog_params` is set (`cog_aperture_correction_enabled` default
False; INV-CFG-01 keeps the function reachable).

Header `from photometry_core import ...` lines in shared /
gate_helpers are GLUE. Call sites inside those modules are REAL.

## Commits

| # | SHA | concern |
| --- | --- | --- |
| C1 | `89a34c7` | delete 3 unreachable photometry_core helpers |
| C2 | `980c57f` | delete 7 unreachable pipeline.py helpers |
| C3 | `0b37a7a` | R-MOVE 4 helpers to photometry_shared |
| C4 | `ce0eae2` | R-MOVE 7 helpers to photometry_gate_helpers |
| C5 | `89fc0b1` | R-MOVE `_get_lc_star_method` to photometry_exports |
| C6 | `8006c88` | R-MOVE 3 helpers to pipeline_ui_helpers |
| C7 | (this) | report + lists + ledger stamp |

Product SHA for gates: `8006c88`.

## Disposition (27)

Full table: `disposition_table.md`.
Delete-candidate grep on b44c82f: `grep_b44c82f_delete_candidates.txt`.

DELETE (10):
- `_median_bkg_var_adu2_per_px_from_proc_cache` (0 refs)
- `_star_mag_for_aperture_sizing` (0 refs; SNR-table relic;
  `_APERTURE_SIZING_MAG_COLS` kept -- still used)
- `_is_broadband_photometric_filter` (DOCSTRING only; comment reworded)
- `get_auto_fov` (0 refs)
- `_solve_wcs_solve_field_cli` (TXT-DUMP only)
- `_solve_wcs_astrometry_net` (TXT-DUMP only)
- `_per_frame_noise_error_map` + `_frame_gain_readnoise_for_error_map`
  (transitive pair)
- `_saturated_core_plateau` + `_star_saturation_flags`
  (DOCSTRING only; catalog docstrings reworded)

STAY in pipeline.py (2):
- `analyze_calibrated_qc` + `_analyze_calibrated_qc_one`
  (REAL caller AstroPipeline)

R-MOVE (14):
- photometry_shared (4): `compute_per_frame_cog_correction`,
  `_aperture_flux_sky_batch`, `_finite_pixel_bbox_from_array`,
  `_intersection_bbox_from_frame_bboxes`
- photometry_gate_helpers (7): `_build_star_exclusion_mask`,
  `_canonicalize_star_xy`, `_robust_scatter_mad`,
  `_clamp_err_empty_apertures_min`, `_labbe_append_debug_record`,
  `_labbe_debug_dump_enabled`, `_labbe_debug_dump_path`
- photometry_exports (1): `_get_lc_star_method`
- pipeline_ui_helpers (3): `_quality_inspection_dao_metrics`,
  `_estimate_fov_deg_from_fits_path`,
  `_obs_fwhm_basename_map_from_db`

## Strongest non-counting refs (deleted defs)

See `disposition_table.md` and `grep_b44c82f_delete_candidates.txt`.
Short form:

- median bkg / star mag / get_auto_fov: no executable hit
- broadband filter: `band_classify.py:250` comment
- solve-field / astrometry.net: `dev/scripts/*.txt` audit dumps
- noise-map pair: internal-only; parent had zero external refs
- saturation pair: `pipeline_catalog.py` docstrings of vectorized
  replacements; `_saturated_core_plateau_vectorized` is a
  different name (SUBSTRING) and stays

## vulture (conf >= 60)

Post-C6: zero unused-function hits among the 27 names
(`vulture_edead.txt` empty; full dump `vulture_raw.txt`).
Deleted names are gone. Moved names have local callers in the
stage-home module, so vulture no longer flags them.

Pre-delete (b44c82f): vulture flagged the 10 delete names as
unused (and also flagged ALIVE cross-module names). That
disagreement is the 01A keep/move class, not a STOP. Grep
confirmed the 10 had no REAL / TEST-ROOT / R5 hit.

## Gates

| gate | result | log |
| --- | --- | --- |
| G1 after C1 `89a34c7` | PASS 1658 / 32 skip | `g1_c1.txt` |
| G1 after C2 `980c57f` | PASS 1658 / 32 skip | `g1_c2.txt` |
| G1 after C3 `0b37a7a` | PASS 1659 / 32 skip | `g1_c3.txt` |
| G1 after C4 `ce0eae2` | PASS 1660 / 32 skip | `g1_c4.txt` |
| G1 after C5 `89fc0b1` | PASS 1661 / 32 skip | `g1_c5.txt` |
| G1 after C6 `8006c88` | PASS 1663 / 32 skip | `g1_c6.txt` |
| G2 `--full` at `8006c88` | PASS era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157 (1975s) | `g2_full.txt` |
| G4 live 516 before G2 | PASS csv `bfa24039` / fits `13e77cf8` / epsf `172f9540` | `g4_before.txt` |
| G4 live 516 after G2 | PASS same prefixes (unchanged) | `g4_after.txt` |
| G-EPSF | not run | none of the 27 on the ePSF graph |

Ledger stamp: `VL-COUNTERS-ZERO` and `VL-ANCHOR-WCSINV` -> `8006c88`.

No non-facade test broke on deletion. Facade getattr tests added
in `dev/tests/test_consolidate_edead_facade.py`.

## Facade inventory (feeds E-final)

See `facade_inventory.md`.

`pipeline.py` after C6: **1089** lines. Physical defs:
`_analyze_calibrated_qc_one`, `analyze_calibrated_qc`,
`AstroPipeline` (C-C), plus constants (`SAT_LIMIT_*`,
`_MASTERSTAR_*`, `_EXO_HOST_ANNOTATION_COLUMNS`, ...).

`photometry_core.py` after C6: **1273** lines. Physical defs:
`compute_auto_fwhm_limit`, `run_full_photometry_pipeline` (C-D),
`select_active_targets` (E3 wrap).

Remaining E4 glue: header imports from the facade in
photometry_shared / photometry_gate_helpers / photometry_exports;
`_clamp_err_empty_apertures_min` inject onto shared (same pattern
as `_clamp_err_empty_apertures_n`). E-final: glue dismantling,
facade permanence (Milan), twin-dismantle of SAT_LIMIT to a leaf
constants module, test retargets.
