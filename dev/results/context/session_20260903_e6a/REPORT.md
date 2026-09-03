# CURSOR RESULT - CONSOLIDATE-01E6a packed astrometry buckets
# (pipeline_preprocess.py + pipeline_astrometry.py + pipeline_catalog.py)

Date: 2026-09-03. Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ 9b48a32 (E5 report tip).
Work started from local tip `9b48a32`.
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Compared with: E0 two packed modules (`pipeline_astrometry.py` 65 defs /
3995 def-lines; `pipeline_astrometry_2.py` 73 defs / 3687 def-lines).
AST on `9b48a32`: zero membership drift vs E0.

Architect amendment V1 (accepted; not refuted): three modules. E5-scale
overhead on bucket 1 as-is would land ~4225 (over the 4000 cap). The
13-def / 509-line preprocess/QC group is stage-wise not astrometry.

| module | defs | def-lines (AST) | assembled | cap |
| --- | --- | --- | --- | --- |
| `pipeline_preprocess.py` | 13 | 509 | **570** | OK |
| `pipeline_astrometry.py` | 52 | 3486 | **3675** | OK |
| `pipeline_catalog.py` | 73 | 3687 | **3927** | OK (tight; not trimmed) |

Four giants stay physical in `pipeline.py` this wave (E6b):
`generate_masterstar_and_catalog`, `detect_stars_and_match_catalog`,
`export_per_frame_catalogs`, `_astrometry_align_impl_body`.
Facade after the wave: `pipeline.py` **7533** (was 15377 at E5 close).

## What I did

Measure on `9b48a32` (defs, only_bucket_mod, patches, ePSF graph,
giants xref, export initializer arity). C1/C2/C3 pure moves + facade
re-exports before `pipeline_ui_helpers`. C3 spawn smoke. C4 getattr
loops. No giant-body edits. No pre-registered body fix (initializer
arity already 1-param / 1-tuple).

## M1 measurement (no STOP; no arity fix)

`_init_export_per_frame_worker` takes one parameter `state`.
Both production sites (`export_per_frame_catalogs`, giant, stays)
pass `initargs=(ws,)`. Measured at pipeline.py:8223 / 9795 / 9835
on `9b48a32`. Direction: leave it.

The RAM path pickles FITS headers by design (`pickle.dumps(_hdr.copy())`).
Location-independent; not "fixed".

Alignment MP is out of scope (workers in `vyvar_alignment_frame`;
giant stays).

## ePSF import graph (G-EPSF required)

Sweep of `src_py/*epsf*` and `src_py/*psf*` `from pipeline import`:
only `epsf_psf_merge.py:107,337,395`:

- `_vyvar_df_to_csv` (M2)
- `_fill_psf_catalog_columns` (M3)
- `_epsf_fit_catalog_ids` (M3)
- `_export_catalog_psf_st_fields` (M2)

No fifth ePSF-graph name. Those callers already use call-time facade
imports; re-export suffices (E-final retarget material). G-EPSF run
once at product SHA `bcead65`.

## Module-level state (only_bucket_mod)

M1: empty.

M2 (moved):
- `_EPSF_SKIP_LOGGED`
- `_VYVAR_TIME_JD_CSV_COLS`

M3 (moved):
- `_BATCH_E_N_EQUIV_LOGGED`
- `_EXPORT_PER_FRAME_WORKER_STATE` (with initializer, both worker
  tasks, `_sat_ctx_from_worker`, `_cfg_from_export_worker_state`)
- `_MASTERSTAR_ZONE_LOG_ONCE`
- `_MOFFAT_CHI2_LIMIT`

Left in `pipeline.py` (not a spawn-global co-location conflict):
- `_EXO_HOST_ANNOTATION_COLUMNS` -- immutable tuple used by M2 and M3.
  Both modules lazy-import. SAT_LIMIT analog.
- `SAT_LIMIT_PEAK_TEST_SOURCE` -- f-string over stay-behind
  `SAT_LIMIT_*`. Moving it plus `from pipeline import SAT_LIMIT_*` at
  catalog load caused a circular import. Reverted.
- `_PIXEL_MATCH_DEBUG_LOGGED` -- assigned, never loaded.

Cone cache is file-based (`_field_catalog_cone_meta_path` /
`_write_field_catalog_cone_meta`). No module global. No KD-tree or
DB-handle module cache.

`LOGGER = logging.getLogger("pipeline")` in each new module.
No module-level `from pipeline import` (the SAT_LIMIT attempt was
reverted).

Catalog has a numeric twin `SAT_LIMIT_NO_KNEE_FRAC = 0.80` for
default-arg evaluation only; not re-exported. Canonical constants
stay on `pipeline`.

Identity-sensitive objects among the 138: none (no Exception
subclass, no sentinel). Facade re-export would have preserved
identity.

## Commits

| # | SHA | concern |
| --- | --- | --- |
| C1 | `e6c43f1` | extract `pipeline_preprocess.py` (13 defs) |
| glue | `74bba99` | ASCII-fold E5 gate logs (UTF-16 BOM from `cmd >`) |
| C2 | `29836fc` | extract `pipeline_astrometry.py` (52 defs) |
| glue | `17905aa` | Sequence import (F821) + E3 spatial-grid `__module__` |
| glue | `d6e694d` | E4 spatial-grid retarget + exoplanet source scan via `inspect.getfile` |
| C3 | `e803655` | extract `pipeline_catalog.py` (73 defs) + spawn smoke |
| C4 | `9a1ab20` | facade getattr-loop tests for the three modules |
| glue | `969a274` | harden SAT_LIMIT_PEAK_TEST_SOURCE facade identity test |
| glue | `bcead65` | facade getattr allows rebound imported scalars |

Product SHA for post-move gates: `bcead65`.

C4 suite-fail root cause: `_BATCH_E_N_EQUIV_LOGGED` is a bool.
Facade `from pipeline_catalog import` copy-binds `False` at import.
A later `global` rebind in the home module diverges identity.
Isolated file 6/6; full suite 1649 passed + 1 fail
(`assert False is True`). Fix: bool/int/float/str/tuple only check
`hasattr`; dict/set still `is`. Isolated file still 6/6 after.

## Facade / wiring

`pipeline.py` late-imports M1 then M2 then M3 **before**
`pipeline_ui_helpers` (same E5 mechanism). Giants call moved names
as plain globals; facade re-export binds them. No giant-body edits.

Movers calling still-physical names (giants, patched facade names,
leave-behind constants) use call-time `from pipeline import`
(E4 mechanism). This preserves
`test_astrometry_fault_isolation.py:59/82/96` patches on
`pipeline._astrometry_align_impl_body`.

## Call-time follows

Sweep of `monkeypatch` / `mock.patch` for all 138 moved names.

Lambda follow (same-module caller + facade patch):
- `_plate_solve_input_bundle` --
  `test_except_fix2_top10.py:118` (also direct-call at :77).
  Home `pipeline_astrometry`. Follow on the facade looks up the
  facade name at call time.

Call-time `from pipeline import` (no lambda; patch bites):
- `extract_fits_metadata` -- M2 `_plate_solve_input_bundle`
- `maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel` -- M2
- `_all_pix2world_icrs_deg` -- M2
  `_try_rescale_masterstar_linear_wcs_to_expected_plate_scale`
  (mover exercised by `test_except_fix2_top10.py:130`)
- `catalog_cone_radius_deg_from_optics` -- M3
- `enhance_catalog_dataframe_aperture_bpm` -- M3
- `_dao_targeted_pass2_unmatched_gaia` -- M3
  `detect_stars_match_master_reference`
  (`test_g1_f003_alignment_pixel_fallback.py:87`)

Re-export suffices:
- `_fill_psf_catalog_columns` -- exercised caller is
  `epsf_psf_merge.merge_psf_into_sidecar` (call-time facade import).
  Internal M3 `_export_per_frame_run_catalog_core` is same-module
  and is NOT the path that test exercises. No follow.

NO follow this wave (caller is a staying giant):
- `_fill_masterstars_gaia_matched_bp_rp_from_local_db` --
  `test_invariants_p2.py:362`. Sole caller
  `generate_masterstar_and_catalog` (stays).
  **E6b NOTE: add the lambda follow when that giant moves.**

Name STAYS (giant):
- `_astrometry_align_impl_body` -- moved callers use call-time
  facade import. Covered.

Delta vs architect's four names: fifth patched E6a name
`_dao_targeted_pass2_unmatched_gaia` (follow mechanism fits: lazy
facade import). Plus two patched utils/photometry names that live
on the facade (`catalog_cone_radius_deg_from_optics`,
`enhance_catalog_dataframe_aperture_bpm`).

No fifth patched facade name whose exercised caller moves and whose
follow does not fit.

## Spawn smoke (C3)

`dev/tests/test_export_mp_spawn.py`:
`multiprocessing.get_context("spawn")` + ProcessPoolExecutor with
`initializer=_init_export_per_frame_worker, initargs=(state,)`
then submit `_probe_worker_state_key` against
`_EXPORT_PER_FRAME_WORKER_STATE` / `_cfg_from_export_worker_state`.
Asserts clean spawn import of `pipeline_catalog`, initializer runs,
state global lives in the right namespace. No DB, no network.
A full export job is not fixtured (G2 `--full` already exercises
`export_per_frame_catalogs` end-to-end). Facade imports on purpose
(E-final retarget note).

## Gates

| gate | status | detail |
| --- | --- | --- |
| G1 after C1 `--fast --clean` | PASS at `74bba99` | 1643 passed, 32 skipped. `g1_c1.txt` |
| G1 after C2 `--fast --clean` | PASS at `d6e694d` | 1644 passed, 32 skipped. `g1_c2.txt` |
| G1 after C3 `--fast --clean` | PASS at `e803655` | 1644 passed, 32 skipped. `g1_c3.txt` |
| G1 at C4 `--fast --clean` | PASS at `bcead65` | 1650 passed, 32 skipped. clean-tree PASS. `g1.txt` |
| G2 `--full` aperture | PASS at `bcead65` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1780s. `g2_full.txt` |
| G-EPSF `--full-epsf` | PASS at `bcead65` | aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157 / epsf01 `c743b8ba` n=53. Pipeline 1356s; ePSF stage 13821s (53 PSF LCs). G3 residual PASS. `g_epsf.txt` |
| G4 live 516 | PASS before G2 and after G-EPSF | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. `g4_before.txt` / `g4_after.txt` |

G4 path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/`
(`masterstars_full_match.csv`, `MASTERSTAR.fits`, `masterstar_epsf.fits`).

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` / `commit`
stamped to `bcead65` by `--full`.

## STOPs

None. No module over 4000 (catalog 3927). No def resisted a pure
move. Shared `_EXO_HOST_ANNOTATION_COLUMNS` left in `pipeline.py`
(architect-analog of SAT_LIMIT; not a spawn-global conflict).
ePSF graph matches the four listed names.

## Files changed

- `src_py/pipeline.py` (cuts + late re-export + one lambda follow)
- `src_py/pipeline_preprocess.py` (new)
- `src_py/pipeline_astrometry.py` (new)
- `src_py/pipeline_catalog.py` (new)
- `dev/tests/test_export_mp_spawn.py` (C3; facade import, E-final note)
- `dev/tests/test_consolidate_e6a_facade.py` (C4)
- `dev/tests/test_consolidate_e3_facade.py` / `test_consolidate_e4_facade.py`
  / `test_exoplanet_local_match.py` (spatial-grid / source-scan retarget)
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` (`--full` stamp)
- `dev/results/context/session_20260903_e5/` (ASCII-fold of UTF-16 logs)
- `dev/results/context/session_20260903_e6a/` (this REPORT + logs + lists)

## Docs impact

none (extraction wave; no science-path behavior change).
Facade names and the four giants stay on `pipeline`.

## Recurrence

new test `test_export_per_frame_mp_spawn_state_roundtrip` (C3)
plus `test_e6a_*` facade getattr / giant-stay / init-arity (C4).

## Errors

C4 first G1 FAIL: `_BATCH_E_N_EQUIV_LOGGED` identity after a prior
test rebound the home-module global. Fixed at `bcead65`. G1/G2/
G-EPSF OVERALL PASS at `bcead65`.

E5 session logs written via `cmd >` were UTF-16-LE (BOM `0xFF`) and
blocked G1 `ascii_policy`. Folded at `74bba99`. Subsequent logs
written via Python `Path.write_bytes`.

## Lists

`moved_names_preprocess.txt`, `moved_names_astrometry.txt`,
`moved_names_catalog.txt`, `giants_xref.txt`, `lazy_imports.txt`,
`call_time_follows.txt`, `only_bucket_mod.txt`, `patch_sweep.json`.
