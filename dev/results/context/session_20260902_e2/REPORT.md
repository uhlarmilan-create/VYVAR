# CURSOR RESULT - CONSOLIDATE-01E2 extraction wave 2

Date: 2026-09-02. Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ dd7f21c.
Work started from local tip `dd7f21c` (matches origin/consolidate-01).
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Compared with: E0 `photometry_shared` bucket (32 defs, 1936 lines) vs this wave.
They differ by architect corrections C-D and C-E (section 1). The 32-name
bucket is not one module, and `run_full_photometry_pipeline` does not move.

## What I did

Pure moves of two shared-layer modules (one commit each), facades re-export,
external `src_py` / `dev` callers untouched. No body edits of moved defs, no
renames, no signature changes. One follow-proxy (same shape as E1
`extract_fits_metadata`). One header-only placeholder block so ruff can see
names that the facade injects after a cycle break.

## 1. Architect corrections

**C-D:** `run_full_photometry_pipeline` stayed in `photometry_core.py`
(`__module__ == "photometry_core"`). `run_phase2a` was not in the E0 shared
bucket (phase2a bucket); verified still in `photometry_core.py`.

**C-E:** two modules, not one. Provenance set (8 defs) vs the remaining 23
(32 minus C-D minus provenance). `measure_fwhm_from_masterstar` is in the E0
**phase2a** bucket, not shared; it did not move.

## Moves per module

| # | module | commit | n defs | facade | notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `photometry_provenance.py` | `cdddfda` | 8 | photometry_core.py (late EOF) | stay-callees: `LOGGER`, `_REPO_ROOT_FOR_PROVENANCE`. Header: `_GIT_PROVENANCE_WARNED = False` (moved function uses `global`). |
| 2 | `photometry_shared.py` | `0a82abd` | 23 | photometry_core.py (late, **before** `photometry_gate_helpers`) | C-D leftover stays in facade. |

Moved names:

- **photometry_provenance.py:** `_is_import_relevant_py_path`, `_porcelain_status_by_path`, `classify_git_dirty_paths`, `_resolve_git_provenance`, `_json_safe_snapshot_value`, `_complete_config_snapshot`, `_build_pipeline_provenance_block`, `merge_photometry_pipeline_meta`.
- **photometry_shared.py:** `_safe_polyfit`, `_normalize_gaia_id`, `finalize_hybrid_bkg_fallback_proc_dir`, `stamp_masterstar_snr_columns`, `_target_display_name`, `stamp_vsx_known_variable_on_masterstars`, `build_gs11_summary`, `_get_lc_adaptive`, `_get_plate_scale_from_cfg`, `_resolve_plate_scale_arcsec_per_px`, `_cd_matrix_scale_arcsec_per_px`, `_read_plate_scale_from_fits_path`, `_angular_distance_deg`, `StressTestResult`, `stress_test_relative_rms_from_sidecars`, `vsx_is_known_variable_top3_per_bin`, `common_field_intersection_bbox_px_from_arrays`, `common_field_intersection_bbox_px`, `recommended_aperture_by_color`, `bad_columns_for_light_frame`, `_fwhm_moment_at`, `compute_fwhm_gaussian_for_aperture_catalog`, `enhance_catalog_dataframe_aperture_bpm`.

Follow-up (not a new module):

- `1ec26be` header placeholders in `photometry_shared.py` for nine gate-helper names (ruff F821). Same commit rewrote E1 `g1.txt` from UTF-16 (Tee-Object BOM) to ASCII so the tracked-text encoding gate can pass.

Facade line counts after the wave: `photometry_core.py` 14790 (was 16455 after E1). `pipeline.py` untouched except as an importer.

## Cycle break (shared vs gate-helpers)

`photometry_gate_helpers` (E1) does `from photometry_core import _normalize_gaia_id`.
`_normalize_gaia_id` moved to shared. Shared bodies call nine E1 gate-helper
names. A sibling import either way at load time is a cycle.

Break (no body edit of a moved def):

1. Late-import `photometry_shared` **before** `photometry_gate_helpers` so the
   facade binds `_normalize_gaia_id` first.
2. Shared does **not** import `photometry_gate_helpers`.
3. After gate-helpers re-export, photometry_core injects the nine names onto
   `photometry_shared`.
4. Shared header placeholders (`= None`) exist only so ruff/pyflakes see the
   names; inject overwrites them before any caller can run.

Removed stay-code: dead mid-file alias `_normalize_id_value = _normalize_gaia_id`
(overwritten 28 lines later by the stay `def _normalize_id_value`). Necessary
so load does not NameError after the cut.

## Import-direction table

| from | to | when | notes |
| --- | --- | --- | --- |
| `pipeline.py` | `photometry.py` | top | `enhance_catalog_dataframe_aperture_bpm` and other `__all__` names |
| `pipeline.py` | `photometry_core.py` | top | `_fwhm_moment_at`, `merge_photometry_pipeline_meta`, `stamp_masterstar_snr_columns` |
| `pipeline.py` | `photometry_core.py` | lazy | `finalize_hybrid_bkg_fallback_proc_dir` (~10124), `stamp_vsx_known_variable_on_masterstars` (~13304), `bad_columns_for_light_frame` (~13534) |
| `pipeline.py` | `photometry_phase2a.py` | lazy | `measure_fwhm_from_masterstar` (masterstar build, ~13223) |
| `photometry.py` | `photometry_core.py` | star-import | unchanged |
| `photometry_phase2a.py` | `photometry_core.py` | star-import + `_normalize_gaia_id` | unchanged; `_normalize_gaia_id` now the shared object via facade |
| `photometry_core.py` | `photometry_shared.py` | late, before gate-helpers | re-export 23 names |
| `photometry_core.py` | `photometry_gate_helpers.py` | late, after shared | existing E1 re-export |
| `photometry_core.py` | `photometry_provenance.py` | late EOF | re-export 8 names |
| `photometry_shared.py` | `photometry_core.py` | module load | stay constants + stay helpers only |
| `photometry_provenance.py` | `photometry_core.py` | module load | `LOGGER`, `_REPO_ROOT_FOR_PROVENANCE` |
| `photometry_gate_helpers.py` | `photometry_core.py` | module load | stay names + `_normalize_gaia_id` (facade, after shared) |
| `photometry_shared.py` | `pipeline.py` | never | no module-level pipeline import |
| `photometry_provenance.py` | `pipeline.py` | never | no module-level pipeline import |
| `photometry_core.py` | `pipeline.py` | lazy inside **stay** defs | existing facade pattern; not added by E2 |

`enhance_catalog_dataframe_aperture_bpm` and
`finalize_hybrid_bkg_fallback_proc_dir` sit on the aperture product path and
are called from E0 `pipeline_astrometry_2` wrappers that **stay in pipeline.py**:
`_apply_aperture_catalog_enhancements_from_st` (~181, top-bound enhance) and
`_finalize_hybrid_bkg_fallback_sidecar` (~10110, lazy finalize). Direction
remains pipeline -> photometry.

## `measure_fwhm_from_masterstar` (not moved)

E0 placed it in the phase2a bucket. Physical home still
`photometry_core.py:368`. Callers:

1. `pipeline.py:13223` -- masterstar build; `from photometry_phase2a import measure_fwhm_from_masterstar`.
2. `photometry_core.py:7050` -- `_phase2a_prepare_shared_state` (phase2a state prep); bare name, same module.

`photometry_phase2a.py` remains `from photometry_core import *` plus explicit
`_normalize_gaia_id`.

## Call-time follows

| name | test patches | caller after move | follow? |
| --- | --- | --- | --- |
| `_resolve_git_provenance` | `photometry_core._resolve_git_provenance` (`test_f431_labbe_provenance`) | `_build_pipeline_provenance_block` in `photometry_provenance` (LOAD_GLOBAL in the new module) | **yes** -- lambda on the facade, same as E1 `extract_fits_metadata` |
| `merge_photometry_pipeline_meta` | `photometry_core.merge_photometry_pipeline_meta` (`test_epsf_stage`, `test_epsf_chain_01b`) | `night_run` / `epsf_stage` / `epsf_psf_merge` do `from photometry_core import merge...` at call time | **no** -- lookup is the facade |
| `enhance_catalog_dataframe_aperture_bpm` | `pipeline.enhance_catalog_dataframe_aperture_bpm` (`test_exc0275`) | `_apply_aperture_catalog_enhancements_from_st` stays in `pipeline.py` | **no** -- same-module lookup |

## G-EPSF import list (why it runs)

Moved E2 names on the ePSF graph (callers still import the facade):

- `psf_internal_lc.py:272` -- `from photometry_core import _resolve_git_provenance`
- `epsf_psf_merge.py:308,483` -- `from photometry_core import merge_photometry_pipeline_meta`
- `epsf_stage.py:230` -- `from photometry_core import merge_photometry_pipeline_meta`

`psf_photometry.py` still only lazy-imports `_annulus_sky_subtracted_flux` (E1).
R-GATE: run G-EPSF because moved defs are on that graph.

## Gates

Product SHA for gates: `1ec26be`.

| gate | status | detail |
| --- | --- | --- |
| G1 `--fast --clean` | PASS at `1ec26be` | 1624 passed, 32 skipped (1618 at E1 + E2 facade tests). clean-tree PASS. db-quick-check WARN waived. Log: `g1.txt` |
| G2 `--full` aperture | PASS at `1ec26be` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1360s. Log: `g2_full.txt` |
| G-EPSF `--full-epsf` | PASS at `1ec26be` | epsf01 `c743b8ba` n=53. ePSF stage 14676s, n_stars=63, 53 PSF LCs. G3 residual BO dem=12.505 (ref 12.505), n_full=134. Aperture hashes unchanged. Pipeline 1362s. Log: `g_epsf.txt` |
| G4 live 516 | PASS before G2, after G2, after G-EPSF | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. |

G4 path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/`
(`masterstars_full_match.csv`, `MASTERSTAR.fits`, `masterstar_epsf.fits`).

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` stamped to `1ec26be` by `--full-epsf`.

## STOPs

None. Cycle was breakable with late import + inject (E1 pattern). No moved
def needed a body edit.

## Files changed

Two module commits + one wiring/encoding follow + this report/logs/ledger
(after remaining gates):

- `src_py/photometry_provenance.py` (new)
- `src_py/photometry_shared.py` (new)
- `src_py/photometry_core.py` (cuts + re-exports + inject + follow)
- `dev/tests/test_consolidate_e2_facade.py`
- `dev/results/context/session_20260901_e1/g1.txt` (UTF-16 -> ASCII)
- `dev/results/context/session_20260902_e2/` (this REPORT + gate logs)
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` (stamped by `--full` / `--full-epsf`)

## Errors

1. First G1 FAIL: ruff F821 on injected gate names in `photometry_shared.py`.
   Fixed `1ec26be` (header placeholders, not body edits).
2. Same first G1 FAIL: `test_tracked_text_files_are_ascii` on E1 `g1.txt`
   (UTF-16 BOM from Tee-Object). File was committed after E1 G1, so E1 did
   not see it. Rewrote to ASCII in `1ec26be`.
3. First G1 write to `tmp/e2_g1.txt` was file-locked; rerun to `tmp/e2_g1b.txt`.

None remaining. G2 and G-EPSF passed on the first attempt.
