# CURSOR RESULT - CONSOLIDATE-01E4 extraction wave 4

Date: 2026-09-02. Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Base stated in the task: origin/consolidate-01 @ 93501b9.
Work started from local tip `93501b9` (matches origin/consolidate-01).
Push only `git push origin consolidate-01:consolidate-01`. main not pushed.

Compared with: E0 phase2a buckets (63 defs / 3995 ln, 46 defs / 2584 ln)
plus `_phase2a_process_one_target` (1611 ln) and
`_phase2a_prepare_shared_state` (940 ln). This wave uses C-B names
`phase2a_state.py` / `phase2a_target.py` and the honest name
`photometry_lightcurve.py` for the E0 `photometry_phase2a_2` bucket.
Membership is exactly the E0 lists; nothing was reassigned.

## What I did

Pure moves of four modules (one commit each), facades re-export,
external `src_py` callers untouched except V1 stub fill, E3-style
source-path tests, FLOW facts, and iron-gate path for
`ensemble_normalize`. No body edits of moved defs, no renames, no
signature changes.

## V1 - Module-name collision

`src_py/photometry_phase2a.py` already existed as a compatibility stub
(not a typo, not an alias):

```
"""Backward compatibility - povodny modul; implementacia je v ``photometry_core``."""
from photometry_core import *  # noqa: F401, F403
from photometry_core import _normalize_gaia_id
```

Inventoried at `photometry_phase2a_stub_before.txt`. This wave filled
it. Call paths that imported the stub first:

- `pipeline.py:13223` -- lazy `from photometry_phase2a import measure_fwhm_from_masterstar` (masterstar build). Live, not latent.
- `ui_aperture_photometry.py:283,469` -- `from photometry_phase2a import _normalize_gaia_id`.

After the fill, `_normalize_gaia_id` is still re-exported from
`photometry_shared` so the UI import keeps working.
`measure_fwhm_from_masterstar` now lives in this module.

Because the stub is imported *before* `photometry_core` on those paths,
header stay-imports re-enter the facade while this module is incomplete.
Break (no body edit of a moved def):

1. Facade late-import of `photometry_phase2a` is skipped when the module
   is already in `sys.modules` without `run_phase2a`.
2. EOF inject binds the 63 names onto `photometry_core`.
3. `photometry_shared` / `photometry_provenance` imports sit after the
   63 defs (header import of those siblings re-enters core mid-init).
4. `photometry_gate_helpers` no longer imports `_Phase2AState` /
   `_phase2a_process_one_target` / `_recompute_bjd_hjd_with_status` at
   module top; those three are late-imported inside the two functions
   that use them (`TYPE_CHECKING` keeps the annotation).
5. Giants are not late-imported from the facade EOF. `photometry_phase2a`
   footer imports `phase2a_state` / `phase2a_target` and injects the
   names onto the facade. That keeps state/target above the two
   libraries and avoids loading the giants while the 63 defs are still
   missing.

## V2 - `_Phase2AState` pickling

The class lands in `photometry_phase2a.py`. Grep of moved defs and
callers: no `pickle` of `_Phase2AState`. Apparent `dump`/`loads` hits
in the analyzer are `json.dump` / `json.loads` (summary, dynamic
params, aperture policy). `pipeline.py` pickles FITS headers for
alignment MP workers, not this class. Home can move; no pickle
stability risk.

## V3 - LAST_EXCLUDED-style side channels

No `global` statement in any moved def. `_ADAPTIVE_BLEND_CACHE` is a
module-level dict that stays physical in `photometry_core.py` (not a
def; annotation uses `BlendMapEntry` under `from __future__ import
annotations`). `_load_blend_worklist` (lightcurve) item-assigns;
`_phase2a_prepare_shared_state` pops. Both import the same dict object.
No facade wrap.

## Moves per module

| # | module | commit | n defs | facade | notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `photometry_phase2a.py` | `db78a41` | 63 | photometry_core.py (before gate-helpers) | Filled the V1 stub. Includes `run_phase2a` and `measure_fwhm_from_masterstar`. E1/E2 names from real homes. Stay: constants + (until later commits) giants / `_coerce_bool_cell`. |
| 2 | `photometry_lightcurve.py` | `e3f3630` | 46 | photometry_core.py (before gate-helpers) | E0 `photometry_phase2a_2` membership. Honest name. |
| 3 | `phase2a_state.py` | `c9db8da` | 1 | inject via photometry_phase2a footer | C-B. Imports machinery from `photometry_phase2a` + `photometry_lightcurve`. |
| 4 | `phase2a_target.py` | `924e3d5` | 1 | inject via photometry_phase2a footer | C-B. Same. |

Follow-up (not a new module): `1abba69` retargets source-scan tests
(`test_field_map_no_catalog_only`, `test_recur_shatext_templates`,
`iron_gates_scan.check_comp_membership_ensemble_normalize`).

Facade line counts after the wave: `photometry_core.py` **1739**
(was 10751 after E3). `run_full_photometry_pipeline` stays (C-D).

`select_comparison_stars_spatial_grid` stayed in `pipeline.py`.

## Semantic placements reported (not reassigned)

| module | def | why it looks odd |
| --- | --- | --- |
| `photometry_phase2a.py` | `democratic_detrend_lc` | other detrend family is in lightcurve |
| `photometry_phase2a.py` | `classify_lc_quality`, `build_lc_quality_summary` | LC quality, not stage machinery |
| `photometry_lightcurve.py` | `_coerce_bool_cell` | generic helper; `read_flux_from_csv` (phase2a) is the only 63-to-46 call |
| `photometry_lightcurve.py` | `_build_phase2a_resolved_facts`, `_fits_header_facts` | prep/state-ish, used by the state giant |

## Import-direction table (E3 + this wave)

E3 rows unchanged. Additions/changes:

| from | to | when | notes |
| --- | --- | --- | --- |
| `photometry_core.py` | `photometry_phase2a.py` | late, before gate-helpers, skip-if-incomplete | re-export 63 names; inject if skipped |
| `photometry_core.py` | `photometry_lightcurve.py` | late, before gate-helpers | re-export 46 names |
| `photometry_phase2a.py` | `photometry_shared.py` | after bodies | `_normalize_gaia_id`, `_safe_polyfit`, `_target_display_name`, `build_gs11_summary` |
| `photometry_phase2a.py` | `photometry_provenance.py` | after bodies | `merge_photometry_pipeline_meta` |
| `photometry_phase2a.py` | `photometry_lightcurve.py` | header (after core stay) | `_coerce_bool_cell` only |
| `photometry_phase2a.py` | `phase2a_state.py` / `phase2a_target.py` | footer | giants; then inject onto facade |
| `photometry_phase2a.py` | `photometry_core.py` | header | stay constants + LOGGER |
| `photometry_lightcurve.py` | `photometry_shared.py` | module load | `_normalize_gaia_id` |
| `photometry_lightcurve.py` | `photometry_core.py` | module load | stay constants + `_ADAPTIVE_BLEND_CACHE` |
| `photometry_lightcurve.py` | `photometry_phase2a.py` | never | |
| `phase2a_state.py` / `phase2a_target.py` | `photometry_phase2a.py` + `photometry_lightcurve.py` | module load | machinery under the giants |
| `photometry_shared.py` | `_sky_pp_for_photometric_error` / `_coerce_bool_cell` | placeholder + inject | E2-style; those names left the facade |
| `pipeline.py` | `photometry_phase2a.py` | lazy | `measure_fwhm_from_masterstar` (was stub, now real) |
| `ui_aperture_photometry.py` | `photometry_phase2a.py` | lazy | `_normalize_gaia_id` still re-exported |
| `psf_internal_lc.py` | `photometry_core.py` | lazy, unchanged | `ensemble_normalize` (now lightcurve via facade) |
| `photometry_gate_helpers.py` | `photometry_core.py` | lazy inside 2 fns | `_phase2a_process_one_target`, `_recompute_bjd_hjd_with_status` |

## Call-time follows

| name | test patches | caller after move | follow? |
| --- | --- | --- | --- |
| `read_flux_from_csv` | `photometry_core.read_flux_from_csv` (`test_phase2a_saturated_skip`) | `_phase2a_process_one_target` in `phase2a_target` | **yes** -- lambda on `phase2a_target` looking up the facade name at call time |
| `_sky_pp_for_photometric_error` | none (shared used the facade import) | `photometry_shared` bodies | inject onto shared (not a patch-string follow) |
| `_coerce_bool_cell` | none | shared + `read_flux_from_csv` | inject onto shared; phase2a imports the real home |

## G-EPSF evidence (why it runs)

`_route_lc_per_frame_err` and `_get_lc` are **not** imported by
`psf_internal_lc` / `psf_photometry`. The name that is:

- `psf_internal_lc.py:472` -- `from photometry_core import ensemble_normalize`

`ensemble_normalize` moved to `photometry_lightcurve.py` and is
re-exported. Callers unchanged. R-GATE: run G-EPSF.

## Gates

Product SHA for gates: `1abba69` (extracts at `924e3d5` + source-scan retarget).

| gate | status | detail |
| --- | --- | --- |
| G1 `--fast --clean` | PASS at `1abba69` | 1638 passed, 32 skipped (1630 at E3 + E4 facade + source-scan). First attempt FAIL at `924e3d5` (source-scan tests still pointed at photometry_core). clean-tree PASS. db-quick-check WARN waived. Log: `g1.txt` |
| G2 `--full` aperture | PASS at `1abba69` | era04_aperture `d55fcc9d` n=53 / ext `cc8b532e` n=157. Pipeline 1778s. Log: `g2_full.txt` |
| G-EPSF `--full-epsf` | PASS at `1abba69` | epsf01 `c743b8ba` n=53. ePSF stage 13987s, n_stars=63, 53 PSF LCs. G3 residual BO dem=12.505 (ref 12.505), n_full=134. Aperture hashes unchanged. Pipeline 1941s. Log: `g_epsf.txt` |
| G4 live 516 | PASS before G2, after G2, after G-EPSF | csv `bfa24039778f437b...` / fits `13e77cf8a1dcb4e7...` / epsf `172f95403beae36d...`. Not written. |

G4 path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/`
(`masterstars_full_match.csv`, `MASTERSTAR.fits`, `masterstar_epsf.fits`).

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV `last_verified` stamped to `1abba69` by `--full-epsf`.

## STOPs

None. Real-module import of `photometry_provenance` / `photometry_shared`
from the filled stub *would* have been a cycle (phase2a first -> sibling
-> core -> incomplete phase2a). Late sibling imports after the 63 defs
broke it. Giants-via-footer broke the state/target vs incomplete-phase2a
cycle. Not a STOP.

## Files changed

Four module commits + source-scan retarget + this report/logs/ledger
(after remaining gates):

- `src_py/photometry_phase2a.py` (filled stub)
- `src_py/photometry_lightcurve.py` (new)
- `src_py/phase2a_state.py` (new)
- `src_py/phase2a_target.py` (new)
- `src_py/photometry_core.py` (cuts + re-exports + skip/inject)
- `src_py/photometry_shared.py` (placeholders for injected names)
- `src_py/photometry_gate_helpers.py` (lazy import of three E4 names)
- `dev/tests/test_consolidate_e4_facade.py`
- `dev/tests/test_consolidate_e2_facade.py` (C-D stay list: only `run_full`)
- `dev/tests/test_field_map_no_catalog_only.py`, `test_recur_shatext_templates.py`
- `dev/tools/iron_gates_scan.py`
- `dev/tools/docs_pdf/flow_doc_facts.py`, `build_flow_doc.py`, `docs/VYVAR_FLOW_CZ.pdf`
- `dev/results/context/session_20260902_e4/` (this REPORT + gate logs)

## Errors

G1 first attempt at `924e3d5` failed on source-scan tests that still
read `photometry_core.py`. Fixed in `1abba69`; G1/G2/G-EPSF PASS on
the retry SHA. G2 and G-EPSF passed on the first attempt at `1abba69`.
