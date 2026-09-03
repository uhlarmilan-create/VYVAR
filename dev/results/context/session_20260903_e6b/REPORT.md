# CURSOR RESULT - CONSOLIDATE-01E6b the four giants
# (catalog_match.py, frame_export.py, masterstar_build.py,
#  astrometry_align.py) + SAT_LIMIT twin guard

Date: 2026-09-03. Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ ff63734 (E6a report tip).
Work started from local tip `ff63734`.
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Spans measured on `ff63734`: zero drift vs the task table
(1372 / 1101 / 2540 / 1049 def-lines). Assembled modules all under
the 4000 cap. `pipeline.py` after C4: **1476** lines (was 7533 at
E6a close). AstroPipeline stays (C-C).

| module | def | def-lines | assembled |
| --- | --- | --- | --- |
| `catalog_match.py` | `detect_stars_and_match_catalog` | 1372 | **1450** |
| `frame_export.py` | `export_per_frame_catalogs` | 1101 | **1179** |
| `masterstar_build.py` | `generate_masterstar_and_catalog` | 2540 | **2643** |
| `astrometry_align.py` | `_astrometry_align_impl_body` | 1049 | **1135** |

E-BODY not this wave; giant bodies stay whole.

## What I did

Measure on `ff63734` (Load classification, monkeypatch sweep,
identity, ePSF graph confirm). C1-C4 pure moves + call-time
imports. C5 twin guard. C6 facade getattr tests. Source-scan
allowlist follow for MASTER_SOURCES. No giant-body edits beyond
the sanctioned call-time import class.

## ePSF import graph (G-EPSF not required)

Sweep of `src_py/*epsf*` / `src_py/*psf*` `from pipeline import`:
only `epsf_psf_merge.py:107,337,395` (four E6a names). No giant
is imported by any ePSF-graph module. Architect claim stands.
G-EPSF not run.

## Wiring verified

- `_astrometry_align_impl_body`: all callers in
  `pipeline_astrometry.py` already use call-time facade imports.
  Facade re-export + those call-time imports keep
  `test_astrometry_fault_isolation.py:59/82/96` biting. No lambda.
- `detect_stars_and_match_catalog`: call-time imported by
  `pipeline_catalog.py`; facade re-export covers. Also E3-imported
  by `frame_export.py` from `catalog_match` (sibling giant).
- `export_per_frame_catalogs`: facade import by
  `test_pre_cal_proc_csv_naming_e2e.py:70`. Workers / initializer /
  `_EXPORT_PER_FRAME_WORKER_STATE` E3-imported from
  `pipeline_catalog`. E6a spawn smoke unchanged (facade imports on
  purpose).
- `generate_masterstar_and_catalog`: call-time follow for
  `_fill_masterstars_gaia_matched_bp_rp_from_local_db` at the call
  site in `masterstar_build.py` (E6a carry-forward DONE).
- Alignment MP: `vyvar_alignment_frame` fresh-attr lookup moves
  as-is. Header pickling untouched.

## Commits

| # | SHA | concern |
| --- | --- | --- |
| C1 | `066fd4e` | extract `catalog_match.py` + pipeline.py cut + facade imports for all four |
| C2 | `c8c594f` | extract `frame_export.py` |
| C3 | `dce2932` | extract `masterstar_build.py` + call-time follow |
| C4 | `1bc29f8` | extract `astrometry_align.py` |
| glue | `de4f0da` | update E6a giant-stay test for E6b module locations |
| C5 | `76e659d` | SAT_LIMIT_NO_KNEE_FRAC twin guard test |
| C6 | `e7a8f06` | facade getattr tests for the four modules |
| glue | `bf0ddfb` | MASTER_SOURCES source-scan allowlist for masterstar_build.py |

Product SHA for gates: `bf0ddfb`.

## Call-time follows

See `call_time_follows.txt`. Summary:

- `_fill_masterstars_gaia_matched_bp_rp_from_local_db` -- call-time
  facade import at call site in masterstar_build (patched by
  `test_invariants_p2.py:362`).
- masterstar_build function-start call-time import of
  `SAT_LIMIT_*`, `_MASTERSTAR_*` / `_PLATESOLVE_*` constants, and
  `detect_stars_and_match_catalog`.
- astrometry_align function-start call-time import of
  `export_per_frame_catalogs`, `generate_masterstar_and_catalog`.

Monkeypatch sweep: no fifth patched name whose exercised caller is
a giant and whose call-time-import mechanism does not fit.

## SAT_LIMIT twin guard (C5)

`pipeline_catalog.SAT_LIMIT_NO_KNEE_FRAC = 0.80` equals
`pipeline.SAT_LIMIT_NO_KNEE_FRAC`. Guarded by
`dev/tests/test_sat_limit_twin_guard.py`.
`_EXO_HOST_ANNOTATION_COLUMNS` is shared via lazy imports, not
duplicated -- left as-is. No other numeric twin introduced this wave.

**Rule (restated):** any numeric twin created for default-arg or
import-order reasons MUST ship with an equality-guard test in the
same commit.

**E-final inventory note:** permanent fix = canonical constants in a
leaf module (dismantle the twin).

## Identity

No Exception subclass or sentinel defined in the four bodies.
`InvariantViolation` is already call-time imported from
`invariants_runtime` inside masterstar_build. Builtins only.
See `identity_list.txt`.

## Remaining pipeline.py physical names (post-E6b facade inventory)

Defs/classes (13):
`_frame_gain_readnoise_for_error_map`, `_per_frame_noise_error_map`,
`_quality_inspection_dao_metrics`, `_estimate_fov_deg_from_fits_path`,
`_obs_fwhm_basename_map_from_db`, `get_auto_fov`,
`_solve_wcs_solve_field_cli`, `_solve_wcs_astrometry_net`,
`_saturated_core_plateau`, `_star_saturation_flags`,
`_analyze_calibrated_qc_one`, `analyze_calibrated_qc`,
`AstroPipeline` (C-C).

Constants / module state:
`pointing_hint_from_header` (public alias), `LOGGER`,
`_SKY_ADU_FALLBACK`, `_MASTERSTAR_*` / `_PLATESOLVE_*` constants,
`_EXO_HOST_ANNOTATION_COLUMNS`, `SAT_LIMIT_*` (incl. peak-test
provenance string), `_PIXEL_MATCH_DEBUG_LOGGED`.

None of these were assigned to E6 buckets in the E0 map (no
membership drift STOP). They are E-DEAD / E-final candidates or
intentionally stay-behind constants.

## Gates

| gate | status | detail |
| --- | --- | --- |
| G1 after C1 | FAIL then fixed | E6a giant-stay test expected `__module__==pipeline`; updated at `de4f0da`. Later MASTER_SOURCES allowlist at `bf0ddfb`. `g1_c1.txt` |
| G1 `--fast --clean` | PASS at `bf0ddfb` | 1658 passed, 32 skipped. clean-tree PASS. `g1.txt` |
| G2 `--full` aperture | PASS at `bf0ddfb` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1358s. `g2_full.txt` |
| G-EPSF | not run | no giant on the ePSF import graph |
| G4 live 516 | PASS before and after G2 | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. `g4_before.txt` / `g4_after.txt` |

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` / `commit`
stamped to `bf0ddfb` by `--full`.

## STOPs

None. No module over 4000. No body edit beyond call-time imports.
No E3 import cycle. No membership drift vs E0 E6 buckets.

## Files changed

- `src_py/pipeline.py` (giant cuts + facade re-exports)
- `src_py/catalog_match.py` (new)
- `src_py/frame_export.py` (new)
- `src_py/masterstar_build.py` (new)
- `src_py/astrometry_align.py` (new)
- `dev/tests/test_consolidate_e6a_facade.py` (giant `__module__` update)
- `dev/tests/test_sat_limit_twin_guard.py` (C5)
- `dev/tests/test_consolidate_e6b_facade.py` (C6)
- `dev/tests/test_database_master_sources_retire.py` (allowlist)
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` (`--full` stamp)
- `dev/results/context/session_20260903_e6b/` (this REPORT + logs + lists)

## Docs impact

none (extraction wave; no science-path behavior change).

## Recurrence

new tests: twin guard (C5), E6b facade getattr (C6).

## Errors

G1 after C1 failed on the E6a giant-stay assertion (expected; E6b
moved them). Fixed at `de4f0da`. Second G1 fail:
`test_no_master_sources_outside_drop_migration` found
`MASTER_SOURCES` in `masterstar_build.py` (source moved with the
giant). Allowlist follow at `bf0ddfb`. G1/G2 OVERALL PASS at
`bf0ddfb`.

## Lists

`moved_names.txt`, `e3_imports.txt`, `call_time_follows.txt`,
`identity_list.txt`, `measure.json`.

## E-program remainder (for the architect)

After E6b: **E-DEAD** (two dead buckets through full 01A R1-R5
reachability) and **E-final** (glue dismantling + facade permanence
decision, Milan's). Post-E6b facade inventory is above. Twin
dismantle to a leaf constants module is E-final inventory.
