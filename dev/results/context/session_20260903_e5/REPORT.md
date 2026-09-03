# CURSOR RESULT - CONSOLIDATE-01E5 calibrate wave (pipeline_calibrate.py)

Date: 2026-09-03. Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ 09c62a4 (SYNC-CLOSE stamp).
Work started from local tip `09c62a4` (matches origin/consolidate-01).
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Compared with: E0 `pipeline_calibrate.py` bucket (76 defs / 4121 lines).
This wave moves the 75 defs minus AstroPipeline (C-C). AST on the tip
confirmed zero membership drift (76 found, 0 missing; move spans 3614
lines; AstroPipeline pipeline.py:18399-18905 historically, 507 lines,
stays). After the move `src_py/pipeline_calibrate.py` is **3847** lines
(under the 4000 cap; not trimmed).

## What I did

M1 first on the unmodified tree (smoke test written, not committed).
C1 signature fix + the now-green smoke test. C2 pure move of 75 defs
plus spawn-worker globals and `_CALIB_MASTER_NB_UNSET`. C3 facade
getattr-loop test. External callers untouched. One call-time follow.

## M1 - initializer arity (CONFIRMED)

`_init_calibrate_batch_worker` took one parameter `initargs`.
CPython `ProcessPoolExecutor` calls `initializer(*initargs)`.
Both production sites pass a 3-tuple. Pre-fix smoke on 09c62a4:

- Log: `m1_smoke_pre_fix.txt`
- FAIL: `BrokenProcessPool`
- stderr: `TypeError: _init_calibrate_batch_worker() takes 1 positional argument but 3 were given`

Architect prediction matched. Direction A applied (three parameters,
unpack line deleted, zero call-site changes). Post-fix:

- Log: `m1_smoke_post_fix.txt`
- PASS in 4.70s

## ePSF import graph (G-EPSF not required)

Sweep of `src_py/*epsf*` and `src_py/*psf*` `from pipeline import`:
only `epsf_psf_merge.py:107,337,395` (`_vyvar_df_to_csv`,
`_fill_psf_catalog_columns`, `_epsf_fit_catalog_ids`,
`_export_catalog_psf_st_fields`). Those are astrometry-bucket names,
not E5. Architect claim stands. G-EPSF not run.

## Module-level state moved with the bucket

- `_cal_batch_flat_cache`, `_cal_batch_flat_median`,
  `_cal_batch_md_preload`, `_cal_batch_native_binning`,
  `_cal_batch_cal_diag` (spawn-worker globals; initializer + worker
  + `global` stay in one module)
- `_CALIB_MASTER_NB_UNSET` (identity-sensitive `is` default; facade
  re-export is the same object)

No other module-level constant is referenced only by bucket defs
(AST sweep: `only_bucket_mod` empty). `SAT_LIMIT_*` stays in
pipeline.py (stay-behind `inv_sat_limit_peak_test_adu` and MASTERSTAR
zone path). `LOGGER` stays; see lazy-import note.

## Commits

| # | SHA | concern |
| --- | --- | --- |
| C1 | `881b23c` | initializer arity fix + MP spawn smoke test |
| C2 | `8fd4881` | extract `pipeline_calibrate.py` (pure move + facade + follow) |
| C3 | `d902299` | facade getattr-loop test |

Product SHA for post-move gates: `d902299`.

## Facade / wiring

- `pipeline.py` late-imports the 75 names plus `_CALIB_MASTER_NB_UNSET`
  **before** `pipeline_ui_helpers` so that helper's
  `from pipeline import draft_median_pointing_icrs_deg,
  sync_obs_files_drift_arcmin_for_draft` still resolves.
- `AstroPipeline` stays in `pipeline.py` (`__module__ == "pipeline"`).
- Stay-behind `analyze_calibrated_qc` resolves `_vyvar_parallel_pool`
  through the facade re-export (call-time global lookup).

## Lazy imports (E4 mechanism)

AST Load sweep of the 75 moved defs: **no stay-behind function names**
are called (`stay_top_called` empty). The only stay-behind *module*
name used is `SAT_LIMIT_CONTAINER_CLIP_ADU` inside
`_effective_saturation_limit`. That function lazy-imports it from
`pipeline` (spawn children import `pipeline_calibrate` first; a
module-level `from pipeline import SAT_LIMIT_*` would cycle).

`LOGGER = logging.getLogger("pipeline")` in the new module (same
named logger as `pipeline.LOGGER`; `logging.getLogger` singleton).
A module-level `from pipeline import LOGGER` would be the same cycle.

Already-extracted names come from real modules (`fits_meta`,
`cal_stage`, `cal_diag`, `calibration`, `vyvar_alignment_frame`,
`plain_stats`, `utils`, `vyvar_platesolver`).

## Call-time follows

Sweep of `monkeypatch.setattr` on pipeline names in `dev/tests`:

| name | test | follow? |
| --- | --- | --- |
| `_fit_subtract_preprocess_sky_surface` | `test_osc1_extraction.py:200` | **yes** -- lambda on `pipeline_calibrate` looking up the facade name at call time. Caller `_qc_enrich_one_frame`. |
| `extract_fits_metadata` | already E1 | unchanged |
| others (`AppConfig`, `_astrometry_align_impl_body`, `_fill_psf_catalog_columns`, ...) | not E5 | none |

`SkySurfaceOrderConflictError` identity preserved by re-export
(`test_skysf_double_guard.py` still imports from the facade).

E-final retarget note: `dev/tests/test_calibrate_mp_spawn.py` imports
the two worker names from the `pipeline` facade on purpose (survives
C1 and C2 unchanged).

## AstroPipeline cross-reference (bucket names)

`_cal_diag_export_for_workers`, `_calibrate_batch_process_one`,
`_calibrate_one_light_disk`, `_cfg_calibration_library_native_binning`,
`_db_for_calibration_tasks`, `_init_calibrate_batch_worker`,
`_light_binning_from_path`, `_log_calibration_io_preflight`,
`_match_and_crop_pair`, `_obs_group_key_from_light_path`,
`_pipeline_ui_error`, `_qc_pack_from_config`,
`_saturation_adu_for_cal_diag`,
`_vyvar_calibrate_multiprocessing_enabled`,
`apply_perf10_dao_qc_to_obs_files`, `calibrate_lights_to_calibrated`,
`run_osc_channel_extraction_for_archive`.

Full lists: `moved_names.txt`, `astropipeline_xref.txt`,
`lazy_imports.txt`, `call_time_follows.txt`.

## Smoke-test contract (passthrough)

`_calibrate_batch_process_one` with `md_s=None` and empty masterflat
map still writes `dst` via `_calibrate_one_light_disk` /
`_calibrate_one_light_apply_masters_in_ram`. With no dark and no flat,
`_calibration_flags` returns `"P"`. Asserted: `ok is True`, `error is
None`, `vy_cflag == "P"`, `VYVARCAL` on the written FITS. 16x16
float32 synthetic light. No DB, no network.

## Gates

| gate | status | detail |
| --- | --- | --- |
| M1 pre-fix smoke | FAIL as predicted | BrokenProcessPool + initializer TypeError. `m1_smoke_pre_fix.txt` |
| C1 post-fix smoke | PASS | 1 passed in 4.70s. `m1_smoke_post_fix.txt` |
| G1 after C1 `--fast --clean` | PASS at `881b23c` | 1639 passed, 32 skipped. clean-tree PASS. `g1_c1.txt` |
| G1 after C3 `--fast --clean` | PASS at `d902299` | 1643 passed, 32 skipped (1639 + 4 facade). clean-tree PASS. `g1.txt` |
| G2 `--full` aperture | PASS at `d902299` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1543s. `g2_full.txt` |
| G-EPSF | not run | no E5 name on the ePSF import graph |
| G4 live 516 | PASS before G2 and after G2 | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. `g4_before.txt` / `g4_after.txt` |

G4 path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/`
(`masterstars_full_match.csv`, `MASTERSTAR.fits`, `masterstar_epsf.fits`).

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` / `commit`
stamped to `d902299` by `--full`.

Facade line count after the wave: `pipeline.py` **15377** (was 19096
at E4 close). `pipeline_calibrate.py` **3847**.

## STOPs

None. M1 matched the architect. No def resisted a pure move. Line
count 3847 < 4000. ePSF graph clean.

## Files changed

- `src_py/pipeline.py` (C1 signature; C2 cuts + late re-export + follow)
- `src_py/pipeline_calibrate.py` (new)
- `dev/tests/test_calibrate_mp_spawn.py` (C1; facade import, E-final note)
- `dev/tests/test_consolidate_e5_facade.py` (C3)
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` (`--full` stamp)
- `dev/results/context/session_20260903_e5/` (this REPORT + logs + lists)

## Docs impact

none (extraction wave; no science-path behavior change except the
latent MP initializer which was opt-in and previously broken).
Facade names and `AstroPipeline` stay on `pipeline`.

## Recurrence

new test `test_calibrate_batch_mp_spawn_passthrough_roundtrip` (C1)
plus `test_e5_init_arity_three` (C3).

## Errors

None on the C1/C2/C3 path. G1/G2 OVERALL PASS on the first attempt
at `d902299`.
