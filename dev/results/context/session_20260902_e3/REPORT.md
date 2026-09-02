# CURSOR RESULT - CONSOLIDATE-01E3 extraction wave 3

Date: 2026-09-02. Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ daa6315.
Work started from local tip `daa6315` (matches origin/consolidate-01).
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Compared with: E0 `photometry_comp` bucket (29 defs, 3017 lines) plus
`run_phase0_and_phase1` (1 def, 1078 lines, E0 name
`photometry_comp__run_phase0_and_phase1.py`). This wave uses the C-B name
`phase01_run.py` for the runner. They differ from E0 only by that C-B name
and by E3 import policy (real-module imports for E1/E2 names).

## What I did

Pure moves of two modules (one commit each), facades re-export, external
`src_py` / `dev` callers untouched except FLOW facts (G1 guard) and one
source-path test (`test_no_fixed_radius_in_select_active_targets_source`).
No body edits of moved defs, no renames, no signature changes.

## Moves per module

| # | module | commit | n defs | facade | notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `photometry_comp.py` | `6e348ea` | 29 | photometry_core.py (late EOF) | E1/E2 names from `photometry_shared` / `photometry_provenance`. Stay: `LOGGER`, `_GAIA_ID_DTYPE`, `_PHASE_USECOLS_PERFRAME`. |
| 2 | `phase01_run.py` | `e985074` | 1 | photometry_core.py (late, after comp wrap) | C-B name. Imports `photometry_comp` (not the reverse). FLOW sync in the same commit. |

Moved names:

- **photometry_comp.py:** `select_comparison_stars_per_target`, `select_active_targets`, `build_global_comp_pool`, `_select_comps_by_rms_then_color`, `_write_suspected_variables`, `_enrich_comp_bp_rp`, `_enrich_active_targets_bp_rp`, `_select_comps_tiered`, `_enrich_target_bp_rp_from_gaia_db`, `_batch_enrich_targets_bp_rp_from_gaia_db`, `_refresh_variable_targets_xy`, `_read_field_density_inputs`, `_resolve_frame_hw_px_from_masterstar`, `ensure_full_variable_targets_if_presel_stub`, `_auto_repair_catalog_ids`, `_warn_zero_compstars_edge`, `_count_gate_passing_comps`, `_attach_predicted_dilution_report`, `_phase0_effective_frame_hw_px`, `_select_comps_by_color_then_rms`, `_dedupe_comp_pool_by_gaia_key`, `_bprp_tier_ladder_for_selection`, `_variable_targets_looks_like_ct_presel_stub`, `_active_target_zone_flag`, `_ensure_active_target_display_names`, `_normalize_id_value`, `_sid_int`, `_bool_col`, `_normalize_id_series`.
- **phase01_run.py:** `run_phase0_and_phase1`.

`select_comparison_stars_spatial_grid` stayed in `pipeline.py` (E0 astrometry bucket).

Facade line counts after the wave: `photometry_core.py` 10751 (was 14790 after E2).

## Sharp edges

**Dead alias:** E2 removed `_normalize_id_value = _normalize_gaia_id`. Grep of the
tree finds that string only in the E2 REPORT. Live `def _normalize_id_value`
moved with this bucket. Remaining callers (`comp_selection_per_target.py`,
`pinned_ensembles.py`, `run_phase0_and_phase1`) import/use the facade name;
none depended on the dead alias ordering.

**LAST_EXCLUDED_TARGETS:** module global written by `select_active_targets` and
read by `run_phase0_and_phase1` plus `test_f428_fixbatch` via
`photometry_core.LAST_EXCLUDED_TARGETS`. The constant stays in
`photometry_core.py`. `photometry_comp` has its own copy for the moved
function's `global`. A `functools.wraps` facade wrapper copies the DataFrame
back to the facade (and to `phase01_run` if loaded). Not a body edit of the
moved def.

**ensure_full_variable_targets_if_presel_stub / `_auto_repair_catalog_ids`:**
pure move; lazy `from pipeline import write_photometry_plan_files` unchanged.
Facade smoke test calls `ensure_full_...` on a tmp non-stub VT (returns False,
file bytes unchanged).

## Import-direction table (E2 + this wave)

E2 rows unchanged. Additions/changes:

| from | to | when | notes |
| --- | --- | --- | --- |
| `photometry_core.py` | `photometry_comp.py` | late EOF, before wrap | re-export 29 names |
| `photometry_core.py` | `phase01_run.py` | late, after LAST_EXCLUDED wrap | re-export `run_phase0_and_phase1` |
| `photometry_comp.py` | `photometry_shared.py` | module load | `_normalize_gaia_id`, `_safe_polyfit`, `_target_display_name` |
| `photometry_comp.py` | `photometry_provenance.py` | module load | `merge_photometry_pipeline_meta` |
| `photometry_comp.py` | `photometry_core.py` | module load | stay: `LOGGER`, `_GAIA_ID_DTYPE`, `_PHASE_USECOLS_PERFRAME` |
| `photometry_comp.py` | `pipeline.py` | lazy inside stay-moved def | existing `write_photometry_plan_files` |
| `photometry_comp.py` | `phase01_run.py` | never | |
| `phase01_run.py` | `photometry_comp.py` | module load | machinery under the runner |
| `phase01_run.py` | `photometry_shared.py` | module load | `_angular_distance_deg`, `_resolve_plate_scale_arcsec_per_px`, `build_gs11_summary` |
| `phase01_run.py` | `photometry_provenance.py` | module load | `merge_photometry_pipeline_meta` |
| `phase01_run.py` | `photometry_core.py` | module load | `LAST_EXCLUDED_TARGETS`, `_GAIA_ID_DTYPE` (still physical in facade) |
| `comp_selection_per_target.py` | `photometry_core.py` | unchanged | facade `_normalize_id_series` / `_normalize_id_value` |
| `pinned_ensembles.py` | `photometry_core.py` | lazy, unchanged | `_bool_col`, `_enrich_comp_bp_rp`, `_normalize_id_series`, `_normalize_id_value` |

## Call-time follows

| name | test patches | caller after move | follow? |
| --- | --- | --- | --- |
| `_enrich_active_targets_bp_rp` | `photometry_core._enrich_active_targets_bp_rp` (`test_f428_fixbatch`) | `select_active_targets` in `photometry_comp` | **yes** -- lambda on the facade |
| `_ensure_active_target_display_names` | `photometry_core._ensure_active_target_display_names` (`test_f428`) | same | **yes** |
| `select_active_targets` | none (LAST_EXCLUDED side channel) | `run_phase0_and_phase1` + `test_f428` read `photometry_core.LAST_EXCLUDED_TARGETS` | wrap (not a patch-string follow) |

## G-EPSF evidence (why it runs)

Moved E3 names are on the ePSF import graph via `pinned_ensembles`, not via
direct `psf_photometry` imports:

- `psf_internal_lc.py:287` -- `from pinned_ensembles import get_pinned_members_for_target`
- `pinned_ensembles.py:521` -- `from photometry_core import _PHASE_USECOLS_PERFRAME, _bool_col, _enrich_comp_bp_rp, _normalize_id_series, _normalize_id_value`

`_bool_col`, `_enrich_comp_bp_rp`, `_normalize_id_series`, `_normalize_id_value`
moved to `photometry_comp.py` and are re-exported on the facade. Callers
unchanged. `psf_photometry.py` still only lazy-imports
`_annulus_sky_subtracted_flux` (E1). R-GATE: run G-EPSF.

## Gates

Product SHA for gates: `e985074`.

| gate | status | detail |
| --- | --- | --- |
| G1 `--fast --clean` | PASS at `e985074` | 1630 passed, 32 skipped (1624 at E2 + E3 facade tests). clean-tree PASS. db-quick-check WARN waived. Log: `g1.txt` |
| G2 `--full` aperture | PASS at `e985074` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1494s. Log: `g2_full.txt` |
| G-EPSF `--full-epsf` | PASS at `e985074` | epsf01 `c743b8ba` n=53. ePSF stage 14814s, n_stars=63, 53 PSF LCs. G3 residual BO dem=12.505 (ref 12.505), n_full=134. Aperture hashes unchanged. Pipeline 1376s. Log: `g_epsf.txt` |
| G4 live 516 | PASS before G2, after G2, after G-EPSF | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. |

G4 path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/`
(`masterstars_full_match.csv`, `MASTERSTAR.fits`, `masterstar_epsf.fits`).

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` stamped to `e985074` by `--full-epsf`.

## STOPs

None. `photometry_comp` <-> `phase01_run` is one-way (runner above machinery).
LAST_EXCLUDED_TARGETS was a side channel, not a cycle; wrapper + facade
constant kept tests and `run_phase0_and_phase1` without body edits.

## Files changed

Two module commits (FLOW in the phase01 commit) + this report/logs/ledger
(after remaining gates):

- `src_py/photometry_comp.py` (new)
- `src_py/phase01_run.py` (new)
- `src_py/photometry_core.py` (cuts + re-exports + wrap + follows)
- `dev/tests/test_consolidate_e3_facade.py`
- `dev/tests/test_phase0_identity_gate.py` (source path)
- `dev/tools/docs_pdf/flow_doc_facts.py`, `build_flow_doc.py`, `docs/VYVAR_FLOW_CZ.pdf`
- `dev/results/context/session_20260902_e3/` (this REPORT + gate logs)
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` (stamped by `--full` / `--full-epsf`)

## Errors

None. G1/G2/G-EPSF passed on the first attempt.
