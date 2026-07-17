# VYVAR_CODE_AUDIT.md - complete hygiene/static-analysis audit

**Status: AUDIT-ONLY (read + log; NO edits during the read).** Fixes are a separate phase: batched by
category or file, isolated, byte-identity/regression where logic could change, Milan-approves before push.
This ledger persists across sessions (source of truth for the audit, like STATE/JOURNAL).

## Method (complete + robust by construction)
"Every line / every function" is achieved by **static analysis tooling** (covers the whole tree by
construction; a human read misses unused funcs / dead params / near-dup code - tools do not), then
**targeted manual logic reads** for what tools cannot see (non-textual logic duplication, wrong parameter
*values*, silent fallbacks, test-vs-prod divergence).

Toolchain (run on repo HEAD f3b73e9): `vulture` (dead code), `ruff` (unused imports/vars/args,
redefinitions), `pylint --enable=duplicate-code` (textual duplication), custom AST/grep param-usage scan
(dead/duplicate config params), grep sweeps (broad-except, prints, hardcodes, TODO), AST function profiler
(per-function LOC/complexity/risk - Category H).

## Scope
138,553 LOC total. **Core (root *.py): 78 files, 92,297 LOC - audited first.** tests/ 13.9k, scripts/
28.6k, GAIA_DR3/ 1.2k - triaged lower (cruft there is lower-risk). Core top-10 files = ~65% of core.

## Confidence legend
[H] high (precise tool, verify only edge cases) . [C] candidate (needs dynamic-use verification:
getattr/dispatch, Streamlit callbacks, entry points, string refs, tests/scripts) . [V] verified.

---

## Category A - DEAD CODE - COMPLETED  (verified full-tree grep: core + tests + scripts + GAIA_DR3, HEAD f3b73e9)
78 vulture function/method candidates (conf>=60) classified by full-tree reference count:
- **51 DEAD** (no reference anywhere in tree) - removal-eligible after a final per-symbol eyeball.
- **13 TEST/SCRIPT-ONLY** - KEEP (test/diagnostic API; removing breaks tests/scripts).
- **14 LIVE in core** - vulture false positives; dropped from the dead list.

Distribution of the 51 dead (by file): database.py 14, pipeline.py 10, importer.py 5, photometry_core.py 2,
export_reports.py 3, param_resolver.py 2, photometry_report.py 2, time_utils.py 3, ui_variability.py 2,
others 1 each (comp_pool_rms, proc_frame_store, psf_photometry, report_methods, tess_verify, ui_calibration,
validate_lc_crossval, vyvar_blind_solver).

**51 DEAD (file:line):**
comp_pool_rms.py:54 `_norm_id_series`; database.py:1065 `update_master_source_safety`, :1104
`count_final_data_for_equipment_id`, :1113 `count_final_data_for_telescope_id`, :1464
`set_obs_draft_masterstar_path`, :1543 `qc_processing_run_exists`, :1550 `delete_qc_processing_run_by_hash`,
:2511 `get_setting_int`, :2521 `set_setting`, :3131 `get_observation_metadata`, :3174
`update_observation_import_log`, :3211 `insert_observation_files`, :3371 `fetch_draft_scanning_ids`, :3622
`finalize_draft`; export_reports.py:106 `_observer_location_configured`, :387 `_test_is_eclipsing`, :494
`_comp_quality_map_for_export`; importer.py:146 `_is_empty_or_missing`, :296 `_format_temp`, :325
`_first_fits_in_dir`, :396 `_resolve_session_lights`, :1500 `_copy_fits_folder`; param_resolver.py:479
`resolve_saturation`, :529 `resolve_exptime`; photometry_core.py:394 `_aperture_to_mask_single`, :6046
`_apply_airmass_detrend_helper`; photometry_report.py:522 `_lunar_risk_fill_color`, :1874
`_katalogy_cell_for_pdf`; pipeline.py:207 `_fits_header_positive_float`, :235 `_per_frame_noise_error_map`,
:1466 `_icrs_offset_arcmin`, :3185 `get_auto_fov`, :3821 `_try_solve_wcs_astrometry_net_or_local_cli`, :4210
`_wcs_astrometry_nearly_identical`, :10068 `_gaia_sky_match_wcs_fragment`, :10101
`_refine_masterstar_wcs_gaia_sky_match_infile`, :14196 `_has_any_usable_master_flat`, :16743
`quick_analyze_last_import`, :16824 `create_observation_from_payload`; proc_frame_store.py:283
`frame_columns`; psf_photometry.py:2192 `_psf_fit_region_mask`; report_methods.py:16
`have_psf_frame_columns`; tess_verify.py:354 `summary_text`; time_utils.py:33 `_clamp_lat`, :37 `_clamp_lon`,
:46 `_clamp_elev`; ui_calibration.py:268 `draft_runtime_status`; ui_variability.py:139
`_raw_lightcurve_from_frames`, :699 `_render_field_image_with_candidate`; validate_lc_crossval.py:102
`_row_mag`; vyvar_blind_solver.py:767 `_cluster_centroid_votes`.

Notes:
- database.py has **14 dead public methods** - confirm they are not an intended external/UI API surface
  before removal (DB methods are a common place for "called via dispatch / future use").
- `_select_comps_tiered` (photometry_core:11292) is TEST/SCRIPT-ONLY (2 script refs). Those scripts are
  archived (`scripts/archive/fixes/`) -> effectively production-dead; removal must also clean those
  archived scripts. (See Cross-link 3 in the verification section.)

**13 TEST/SCRIPT-ONLY (KEEP):** database.py update_obs_draft_center / insert_equipment / insert_telescope /
insert_location; dilution.py compute_dilution_batch; param_resolver.py resolve_pixel_um / resolve_focal_mm /
resolve_binning; photometry_core.py _get_lc_adaptive / _select_comps_tiered; psf_neighbor_sub.py
neighbor_sub_target_flux; psf_photometry.py _epsf_fwhm_native_legacy_px / _annulus_sky_per_px_custom.

## Category B - UNUSED IMPORTS / REDEFINITIONS  [H]
ruff, concrete:
- **`photometry_core.py:11715` F811 - `resolve_comp_sparse_fallback_enabled` REDEFINED** (shadows the
  import at line 34). Duplicate definition - one is dead. **Correctness-adjacent: confirm which binding
  is live and that the intended one wins.** (RESOLVED [V] benign - see verification section Cross-link 1.)
- `photometry_core.py:34` F401 `resolve_comp_sparse_fallback_enabled` imported-unused (paired w/ above).
- `photometry_core.py:11716` F401 `resolve_comp_sparse_fallback_min` imported-unused.
- `check_star_kmag.py:9` F401 `typing.Any`; `pipeline.py:35,36` F401 `_astrometry_align_mp_init/_task`.
- `night_run.py:460` F841 `missing_mag` assigned-never-used; `photometry_core.py:8098` F841 `am_piecewise`.

## Category C - DEAD / UNUSED PARAMETERS - COMPLETED (config) + listed (args)

### Config params (273 total)
- **1 fully DEAD** (no reference anywhere outside config.py): `varastro_observer_name`.
- **Candidate dead-knobs (no DIRECT production-logic read): ~8** - but MUST be confirmed per-param for
  **resolver-indirection** (a field read only via `resolve_X(cfg)` looks dead but is live). Automated
  resolver detection is UNRELIABLE here; final confirmation belongs in the per-file read.
  - Confirmed LIVE-via-resolver (NOT dead): `comp_sparse_fallback_enabled` (resolver called
    photometry_core:11728,12259 - manually verified); `comp_iterative_clip_enabled` (likely live).
  - Strongest dead candidates (no direct read, no obvious called resolver - confirm in per-file pass):
    `psf_spatial_enabled`, `psf_spatial_grid`, `psf_spatial_min_stars_per_cell` (whole PSF-spatial
    feature, only test refs - abandoned feature); `phase01_comparison_proximity_tiebreak` (the
    distance/proximity rule - already established unwired, Cross-link 2); `phase01_comparison_rms_bin_mag`;
    `comp_tier4_bprp_limit`; `comp_sparse_fallback_min` (its resolver imported but uncalled - Cross-link 1
    tail); `aperture_fwhm_factor_medium`; `masterstar_log_astroalign`.
- **131 lightly-used** (<=2 logic refs) - spot-check during the per-file read.
- **130 live** (>2 logic refs).

### Unused function arguments (47: ruff ARG001 x46 + ARG002 x1) - dead/threaded-through params
Mostly signature cruft (params declared but unused), e.g. `cfg` passed-unused (param_resolver:422,441,523,529;
photometry_core:5474; check_star_kmag:493), `window_length` x4 (tess_verify:122,212,433,485),
`vsx_local_db_path` threaded-unused (comp_selection_per_target:129,1801; photometry_core:10583,11124),
`t_bp_tgt`/`mag_t` threaded-unused in comp tier funcs (comp_selection_per_target:326,1717,1719,1794,2136).
**Correctness-adjacent cross-links (flag for the per-file read):**
- `photometry_core.py:12338` unused arg `max_psf_chi2` - consistent with the PSF-chi2 comp filter being
  effectively OFF in production (production passes `max_psf_chi2=inf` at the comp-selection call). Confirm intent.
- `pipeline.py:12307` unused `max_control_points` + `:12312` unused `max_extra_platesolve` - alignment/
  platesolve params declared but unused; connects to the dense-field control-point perf ticket (Fix C
  follow-up). (RESOLVED [V] silent-override - see verification section Cross-link 4.)

## Category D - DUPLICATE CODE  [V textual]
pylint similarities (>=8 lines, core): **41 duplicate blocks.** Hotspot files (blocks each):
comp_selection_per_target 11, night_run 10, comp_pool_rms 10, photometry_core 9, app 7, ui_variability 5.
Notable:
- **`comp_pool_rms` <-> `comp_selection_per_target`**: per-frame CSV iteration + flux->mag comp-RMS logic
  duplicated (e.g. comp_pool_rms[115:145]==comp_selection[758:788]; [211:243]==[880:911]). Candidate for a
  single canonical comp-RMS routine ("one canonical fix over per-consumer patches").
- `photometry_core[4797:4825]` <-> `ui_variability[1342:1372]`: VDI merge logic duplicated.

## Category E - BROAD-EXCEPT HYGIENE  [triage by context]
**1291 `# noqa: BLE001` (broad `except Exception`) + 110 `except: pass/continue`.** The recurring
`[TODO-BROAD-EXCEPT-HYGIENE]` anti-pattern at scale. Cannot hand-review individually. Triage workflow:
group by file + by CONTEXT - risky = those in science/fallback/protection paths that can swallow
NameError/AttributeError (these hide bugs in the safety nets); benign = UI/logging/optional-feature
guards. Fix the risky ones (narrow the except / log the swallow); leave or gradually narrow the benign.
(Cat H confirms 1355 broad/bare excepts live mostly inside the monster functions - triage them in-line
during each function's deep read.)

## Category F - DEBUG / INSTRUMENTATION CRUFT  [H]
- **171 `print(`** in production core (should be `log_event`/logging).
- **103 `_debug_`/DEBUG** references.
- **16 hardcoded catalog IDs:** 7 are the BO CVn debug funnel (`1498613634033133184`:
  comp_selection_per_target.py:34,345,591; photometry_core.py:11882,11890,12136,12193) - **all
  debug-only, zero logic effect, but cruft.** Generalize to a per-target debug toggle
  (`VYVAR_DEBUG_COMP_CIDS`, already prototyped + validated 2026-06-19) or remove. Remaining 9 IDs:
  classify (comment vs logic).

## Category G - TODO/FIXME/HACK  [inventory]
44 markers - inventory + triage (which are stale, which are real open work; cross-link to ROADMAP).

## Category H - COMPLETE FUNCTION INVENTORY + complexity risk-map  (AST profile, HEAD f3b73e9)

### Coverage (complete - "every function")
**1727 functions across 78 core files profiled.** Per-function metrics: LOC, cyclomatic-ish complexity
(branches/loops/bool-ops), broad/bare excepts inside, max nesting, magic-number density, risk score
(= 0.4*LOC + 2*cx + 5*broad + 3*depth).

### Structural headline
- **1355 broad/bare `except` inside functions** (matches the ~1291 noqa:BLE001 from Cat E; the recurring
  hygiene risk lives mostly inside the big functions below).
- **30 functions > 400 LOC; 80 > 200 LOC; 26 monsters > 500 LOC; 11 functions with >= 15 broad-excepts.**

### Monster functions (>500 LOC) - refactor + deep-read candidates
SCIENCE-CRITICAL (highest blast radius - read first):
- `pipeline.py:10180 generate_masterstar_and_catalog` - **1896 LOC, 47 broad-except** (production entry
  point; the single most dangerous function in the tree).
- `vyvar_platesolver.py:4255 solve_wcs_with_local_gaia` - 1434 LOC, 25 broad.
- `photometry_core.py:7284 _phase2a_process_one_target` - 1167 LOC, 27 broad.
- `photometry_core.py:12310 run_phase0_and_phase1` - 922 LOC, 22 broad (comp-selection orchestration).
- `pipeline.py:12302 _astrometry_align_impl_body` - 972 LOC, 27 broad (holds Cross-link 4 / Fix C cap).
- `pipeline.py:7312 detect_stars_and_match_catalog` - 989 LOC, 13 broad, 90 magic numbers.
- `pipeline.py:9010 export_per_frame_catalogs` - 898 LOC, 24 broad.
- `photometry_core.py:6339 _phase2a_prepare_shared_state` - 758 LOC, 16 broad (Fix B gate hook lives here).
- `photometry_core.py:11609 select_comparison_stars_per_target` - 699 LOC, 7 broad (comp selection).
- `astrometry_optimizer.py:266 optimize_masterstar_matches` - 899 LOC, 16 broad.
- `pipeline.py:5531 write_photometry_plan_files` - 641 LOC, 24 broad.
- `pipeline.py:6721 detect_stars_match_master_reference` - 543 LOC, 6 broad.
- `vyvar_platesolver.py:3375 _solve_wcs_validate_and_refine` - 580 LOC, 8 broad.
- `psf_photometry.py:2526 psf_photometry_stars` - 547 LOC, 12 broad; `:819 _epsf_prepare_stars` - 532 LOC.
- `night_run.py:500 run_night_pipeline` - 629 LOC, 7 broad (orchestration).

CONFIG/UI (lower science risk; structural):
- `config.py:634 __post_init__` - **1162 LOC, 376 branches** (loads+validates all 273 params; reading this
  resolves the dead-knob / resolver-indirection questions for Cat C and the 131 lightly-used params).
- UI renderers (read later, low science risk): ui_settings render_settings_dashboard (992),
  ui_aperture_photometry render_aperture_photometry (751)/_render_target_detail (607), ui_variability
  render_variability_dashboard (888), ui_quality_dashboard render_quality_dashboard (724),
  ui_masterstar_qa render_masterstar_qa (517), app.py render_live_view (692)/_render_pending_job_dispatcher
  (572)/_run_vyvar_full_pipeline (551).

### Deep manual-read priority order (plan step 3)
1. `generate_masterstar_and_catalog` (the 1896-LOC entry point; triage its 47 broad-excepts: risky=swallow
   in science/fallback path vs benign; flag silent fallbacks, dead branches, wrong param values).
2. `_phase2a_process_one_target` + `_phase2a_prepare_shared_state` (Phase-2A photometry core; Fix B hook).
3. `run_phase0_and_phase1` + `select_comparison_stars_per_target` (comp-selection orchestration; resolves
   the proximity-rule / sparse-fallback / resolver-indirection threads).
4. `_astrometry_align_impl_body` (alignment; Cross-link 4 / Fix C control-point cap in full context).
5. `solve_wcs_with_local_gaia` + `_solve_wcs_validate_and_refine` (platesolve).
6. `detect_stars_and_match_catalog` + `detect_stars_match_master_reference` + `export_per_frame_catalogs`.
7. `config.py __post_init__` (param validation; closes Cat C resolver-indirection + lightly-used checks).
8. Remaining functions by descending risk; UI renderers last.
Broad-except context triage (Cat E) is performed in-line during each function's read.

---

## High-value cross-links (correctness-adjacent - do FIRST, with care)
1. F811 `resolve_comp_sparse_fallback_enabled` redefinition (Cat B) - duplicate live/dead binding. [V] benign.
2. `phase01_comparison_proximity_tiebreak` dead param vs the intended min-distance selection rule (Cat C). [V] dead/unwired.
3. `_select_comps_tiered` dead comp-selection variant (Cat A) - divergent selection path. [V] production-dead.
4. `pipeline.py:12307 max_control_points` unused arg vs Fix C control-point cap (Cat C). [V] silent-override.

## Verification workflow for [C] candidates (before any removal)
For each candidate symbol: grep the name across the FULL tree (core + tests + scripts), check for dynamic
use (`getattr`, dispatch dicts, string refs, Streamlit `on_click`/`key=`, entry points, `__all__`),
check git history if ambiguous. Only symbols with zero real references -> mark [V] dead -> eligible for
removal in the fix phase.

## Prioritized plan
1. (DONE) Tooling + first complete category scans -> this ledger.
2. (DONE) Verify the 3 high-value cross-links + the [C] dead-function candidates (full-tree grep).
   Categories A & C inventoried + classified.
3. (DONE) Category H: complete AST function inventory + complexity risk-map -> breadth/structural audit
   phase COMPLETE (Categories A-H inventoried).
4. (NEXT) Per-file deep manual logic read in the Cat-H priority order above (catches non-textual dup, wrong
   param values, silent fallbacks - what tools miss). Resolver-indirection confirmation of dead config
   params + the database.py public-API check land here. Each monster read is a multi-pass chunked effort.
5. Broad-except context triage (Cat E) - performed in-line during each function's read.
6. Fix phase: batched by category, isolated, byte-identity/regression, Milan-approved, Cursor implements.

## Discipline
Audit-only here. No edits, no commits from the read. Every removal/fix is a separate, isolated change with
a regression check (byte-identity on a known draft where logic could be touched). Tests must stay green.

---

# Verification results - session 2026-06-19 (read-only, HEAD f3b73e9)

Full-tree grep (core + tests + scripts) + targeted logic reads. No edits. Legend: [V] verified, [C] candidate.

## High-value cross-links

### 1. F811 `resolve_comp_sparse_fallback_enabled` redefinition - [V] BENIGN (not a live/dead hazard)
- Single canonical definition: `config.py:29` (`resolve_comp_sparse_fallback_enabled`) / `config.py:38`
  (`resolve_comp_sparse_fallback_min`). There is **no second definition** - only re-imports.
- `photometry_core.py:34` imports it at module level (inside the `from config import (...)` block).
- `photometry_core.py:11715-11716` **re-imports both locally** inside `select_comparison_stars_per_target`.
  ruff flags F811 (local re-import shadows the module-level name within the function) + F401 (the
  module-level import then looks unused).
- **Both bindings resolve to the identical `config` function object** -> zero behavioral divergence; the
  "intended one wins" trivially because they are the same callable.
- Live consumers: `photometry_core.py:11728` and `:12259` (sparse-retry path, via the local import);
  `citations.py:202` (its own local import); `tests/test_comp_sparse_fallback.py`.
- `resolve_comp_sparse_fallback_min`: imported at `photometry_core.py:11716` but **never called in
  photometry_core** (genuine dead import there); used in `config.py` + tests. -> F401 correct, benign.
- **Fix-phase note:** remove ONE redundant import (drop the local re-import at 11715-11716, OR drop the
  module-level line 34). Pure hygiene; no regression risk.

### 2. `phase01_comparison_proximity_tiebreak` dead param vs min-distance rule - [V] DEAD param; rule NOT wired
- config only: `config.py:417` (default `False`), `:1666-1669` (load from dict), `:2026-2027` (serialize);
  plus the params-doc generator `scripts/_build_vyvar_params.py:212,521`.
- **Never read by any selection logic** (no ref in photometry_core / comp_selection_per_target /
  comp_pool_rms; no `getattr(cfg, "phase01_comparison_proximity_tiebreak"...)` anywhere).
- Live final-selection tiebreak is **catalog_id**, not distance: `_select_comps_by_color_then_rms`
  (`photometry_core.py:11246-11254`) ranks by `["_broeg_score"(=1/rms) desc, catalog_id asc]`. No distance
  / proximity term in the ranking. Upstream candidate ordering also sorts by `catalog_id` (mergesort).
- Corroboration that the rule was prototyped then unwired: orphaned `scripts/_validate_comp_proximity_365.py`
  ("Validate proximity tie-break on draft_000365") referencing `comparison_stars_per_target.pre_proximity.csv`
  / `photometry_summary.pre_proximity.csv` artifacts.
- **Conclusion (measured):** the min-distance/proximity selection rule is genuinely not active; the flag is
  dead. This is the likely reason the distance rule "isn't working." Fix-phase decision needed: either wire
  the rule (feature work, regression-tested) or remove the dead flag (hygiene). Correctness-adjacent.

### 3. `_select_comps_tiered` (photometry_core.py:11292) - [V] PRODUCTION-DEAD (divergent path)
- Only name-references in the tree are **archived code-gen scripts**:
  `scripts/archive/fixes/build_comp_selection_module.py:46`, `scripts/archive/fixes/gen_comp_selection.py:47`
  (symbol lists for module extraction, not calls).
- No production caller, no test, no `getattr`/string/dispatch reference. Production selection uses
  `_select_comps_by_color_then_rms` (and the `select_comparison_stars_per_target` orchestrator).
- **Conclusion:** confirmed dead divergent selection variant; eligible for fix-phase removal (isolated,
  with a byte-identity check on a known draft since it shares helpers). Remove the 2 archived-script refs too.

### 4. `pipeline.py:12307 max_control_points` unused arg - [V] SILENT-OVERRIDE (ties to Fix C control-point ticket)
- `_astrometry_align_impl_body` declares `max_control_points: int = 180` (`:12307`) and
  `max_extra_platesolve: int = 0` (`:12312`). Callers thread real values through: `app.py:502/891`,
  `night_run.py:357/900`, public wrapper `pipeline.py:13280/13425`, and many scripts pass `200`/`250`
  (e.g. `qatar8_night_run_v.py:631`, `chiandh_night_run_bvr.py:151`, `dy_peg_night_run_bvr.py:367`).
- **But `_astrometry_align_impl_body` never reads its own `max_control_points`.** The effective control-point
  count is computed independently:
  - `_align_star_cap = max(10, min(5000, alignment_max_stars))` then **`min(_align_star_cap, 200)`** (`:12347-12349`),
  - `align_cp = max(12, min(_align_star_cap, round(1.5 * n_ref_fit)))` (`:12820`), passed to the aligner
    (`:12831`) and recorded as `alignment_max_control_points_used` (`:13236`).
- **Conclusion (measured):** a caller setting `max_control_points=250` has **zero effect** at this layer -
  the cap is silently `min(alignment_max_stars, 200)` (further limited by `1.5*n_ref`). The user/script-facing
  `max_control_points` knob is dead here; the real knob is `cfg.alignment_max_stars` + the hardcoded `200`.
  This is exactly the dense-field control-point behaviour from the Fix C diagnostic (cap pinned at 200, not
  the threaded value). Note: `vyvar_alignment_frame.py:259/276` DOES consume a `max_control_points`, but it
  is fed `align_cp` from `_align_star_cap`, not the threaded arg. Fix-phase decision: wire the threaded knob
  through (and reconcile with the `200` cap / the perf ticket) or drop the dead arg. Correctness-adjacent.

## Notable Category-A dead-function candidates (spot-verified)
- `export_reports.py:387 _test_is_eclipsing` - [V] DEAD. No caller anywhere; inline self-test never invoked.
- `export_reports.py:494 _comp_quality_map_for_export` - [V] DEAD (def only; no refs).
- `pipeline.py:235 _per_frame_noise_error_map` - [V] no callers/refs in tree (incl. app.py).
- `pipeline.py:3185 get_auto_fov` - [V] no callers/refs in tree (incl. Streamlit app.py).
- `pipeline.py:3821 _try_solve_wcs_astrometry_net_or_local_cli` - [V] no callers/refs in tree (verify git
  intent before removal - possible deliberately-retained WCS fallback).
- `dilution.py:280 compute_dilution_batch` - [C] **NOT production-dead**: used by `tests/test_dilution.py`.
  Test-only API; production uses `compute_dilution_factor` (per-star). KEEP.
- `param_resolver.py`: `resolve_pixel_um`/`resolve_focal_mm`/`resolve_binning` - [C] used only by diagnostic
  script `scripts/diagnose_psf_elongation_362.py` (pipeline has its own `_resolve_focal_mm_for_plate_scale:3256`).
  Script-only - KEEP. `resolve_saturation`/`resolve_exptime` - [V] no refs beyond defs (dead candidates;
  part of a public-looking resolver API -> confirm intent before removal).

## Status of Category F item (BO CVn hardcoded-ID debug funnel)
- The `VYVAR_DEBUG_COMP_CIDS` toggle proposed in Cat F was **prototyped and validated** during the
  comp-funnel diagnostic (2026-06-19) - it cleanly generalizes the 7 hardcoded `1498613634033133184`
  debug gates and was reverted (uncommitted). Ready to lift into the fix phase (zero logic effect).

## Outstanding verification (for the per-file read, plan step 4)
- Resolver-indirection confirmation of the ~8 strongest dead-config-knob candidates (psf_spatial_*,
  phase01_comparison_rms_bin_mag, comp_tier4_bprp_limit, aperture_fwhm_factor_medium, masterstar_log_astroalign)
  - resolved by reading `config.py __post_init__` (Cat H item 7).
- database.py 14 dead public methods - confirm not an intended external/UI/dispatch API before removal.
- The 131 lightly-used (<=2 ref) config params - spot-check consumed-by-logic.
- Cat D duplicate blocks - confirmed textually by pylint [V]; canonicalization is a fix-phase decision.
- Cat E broad-except context triage - performed in-line during the Cat-H monster-function deep reads.

---

# Deep-read findings (plan step 4) - read-only, HEAD f3b73e9

Severity: INFO / LOW / MED / HIGH. Findings logged here; all fixes deferred to the approved fix phase.

## DR1 - `select_comparison_stars_per_target` (photometry_core.py:11609-12307, 699 LOC, 7 broad-except)
Read in full (signature, candidate build, metric/hard filters, RMS, sparse routing, tier assembly, return).
Comp-selection orchestrator.

- **DR1-1 [MED] within-function duplicate recursion (divergence risk).** The sparse-fallback recursion is
  invoked via TWO code paths that each re-thread the full ~35-arg call: helper `_retry_sparse_fallback`
  (def 11725; called 11879/12059/12103/12132/12191) AND an inline recursion at 12259-12301. The two
  arg-lists are hand-maintained and can drift (e.g. one passes a `_cfg_p1`-derived `cfg`, the other `cfg`).
  Consolidate to one recursion helper. (Static dup scan missed this - it is argument threading, not a clean block.)
- **DR1-2 [INFO -> doc] sparse-fallback semantics (closes the V0454 comp-path thread).** "sparse_fallback
  comp path" means: the DEFAULT pass could not reach `n_comp_min` (=3) strict comps, so the function recursed
  with `_use_iter_clip` (`skip_apriori_rms=True` at 12056 + iterative ensemble clip at 12107), RELAXING the
  `max_comp_rms` (0.05) hard cut. Exactly why V0454 (2 detectable candidates; other 12 had n_frames=0) ended
  on the sparse path with 2 kept-"suspect" comps. Behaviorally correct - add a one-line legend so
  "sparse_fallback" reads as "could not find 3 strict comps", not as an error. Reconfirms NO distance/proximity
  term in the ranking (consistent with dead `phase01_comparison_proximity_tiebreak`, Cross-link 2).
- **DR1-3 [LOW] pointless broad-except.** 12069 `except Exception: return cands` wraps only trivial variable
  aliasing (`ms_arr_x2 = ms_arr_x`) which cannot raise - dead defensive cruft; remove the try.
- **DR1-4 [LOW] silent-null broad-except - RESOLVED [V] benign + reveals a dead arg.** 12216
  `except Exception: final_lookup = None` silently nulls the lookup on any failure. **Verified downstream:**
  `_assemble_comp_selection_result_rows` (comp_selection_per_target.py:2150 arg) **ignores the passed
  `final_lookup` and rebuilds it internally** from `final_comps` (2161-2166), then uses it None-safely
  (guard `if final_lookup is not None` at 2174). So no data is dropped on a transient build error. Stronger
  finding: the caller's `final_lookup` build (12212-12217) is **entirely redundant** and `final_lookup` is a
  **dead parameter** of `_assemble_comp_selection_result_rows` (the lookup is built twice; the passed copy is
  discarded). Fix-phase: drop the caller build + the dead arg (Cat C unused-arg item). No correctness risk.
- **DR1-5 [LOW] dead/misleading signature defaults.** `max_psf_chi2: float = 3.0` (11633) implies the PSF-chi2
  comp filter is ON, but production overrides with `inf` (filter OFF - ties to unused-arg finding at 12338).
  `max_delta_bprp: float = 0.5` (11641) vs config `comp_max_delta_bprp=0.79` (caller overrides). These defaults
  are never used in production and mislead a reader; align to config or annotate.
- **DR1-6 [LOW] stale suppression.** `_ = fwhm_px` (11697) marks `fwhm_px` unused, but it IS used at 11910 and
  11973. Misleading no-op; remove.
- **DR1-7 [LOW] dual candidate sets (consistency smell).** `candidates_pre` (adaptive-mag-filtered, 11880;
  drives metric accumulation + dilution) vs `candidates` (rebuilt 12093 from `_base_mask|det_mask`; drives
  scoring/tiering). Safe in practice because `_assign_comp_tiers_to_pool` intersects with `active`
  (rms-filtered, keyed off candidates_pre); confirm no path consumes the un-intersected `candidates`.

Broad-except triage (7): 11887/11905/12146/12209 -> BO CVn debug funnel BENIGN (Cat F); 11924 -> aperture-iso
prep, ACCEPTABLE (logs warning, degrades to no-isolation); 12069 -> DR1-3 (pointless); 12216 -> DR1-4
(resolved benign). None hides a science bug.

**Net:** comp-selection logic is sound; report-visible oddities (sparse path, 2 suspect comps, no distance
rule) are fully explained and consistent with the static findings. Actions = hygiene (DR1-1 consolidation,
DR1-3/4/5/6 cruft) + one doc note (DR1-2). DR1-4 downstream check done -> benign. All deferred to fix phase.
Read-coverage: logic-bearing spans read in full; only un-quoted spans are the `_retry_sparse_fallback`
arg-list tail (11760-11820, the duplication itself) and already-audited helper delegations. Fully read.

### Next deep-read target
Recommend `run_phase0_and_phase1` (photometry_core.py:12310 - the orchestrator that CALLS this function and
holds the `max_psf_chi2=inf` override site at 12338) to close the comp-selection chain while context is
loaded, then the #1 monster `generate_masterstar_and_catalog` (pipeline.py:10180).

## DR2 - `generate_masterstar_and_catalog` (pipeline.py:10180-12076, 1896 LOC, 47 broad-except) - IN PROGRESS
The #1 monster / production entry point. **FULLY READ (PASS 1-4, 10180-12075).** PASS 1
(setup/selection/build/solve-prep): 3 LOW findings DR2-1/2/3 (DR2-1 misleading default `max_catalog_rows=12000`;
DR2-2 redundant re-floor 11197; DR2-3 silent `bin=1` fallback 10693). PASS 2/3/4 + SUMMARY below.

### PASS 2 - solve execution + WCS scale-adjust + `platesolve_only` return (10870-11101)
Hard acceptance guards are correct fail-fast (solver-not-solved raise 10878; no valid WCS raise 10889;
match_rate<0.60 raise 10902; anisotropic-after-retry raise 10967; the 10930-10970 relaxed-FOV/SIP-off retry
is a sound recovery). Findings are in the expected-scale fallback chain:

- **DR2-4 [MED likelihood / HIGH fix-priority, V] hardcoded HOME-RIG `9.77` arcsec/px universal fallback
  (pipeline.py:10994) - cross-rig robustness violation.** Verified: `_exp_scale_apx` resolves DB/bundle hint `_plate_scale_ms` -> WCS pixel
  scale (10986-10990) -> else **hardcoded `9.77`** (10993-10994); `9.77` is the ONLY such literal in
  pipeline.py and is the home rig (Carl-Zeiss + QHY294MM/IMX492) scale. On the narrow rig (~0.646-1.30"/px)
  or Brno (~0.566"/px) this is 7-17x wrong. Two verified consequences:
  1. **Shadows the config fallback (11047 dead).** Because 10994 forces `_exp_scale_apx` non-None, the
     `if _vy_plts is None: _vy_plts = _exp_scale_apx` at 11045 always fires, so the intended
     `_vy_plts = cfg.export_arcsec_per_px` at **11047 is unreachable** (confirmed by reading 11036-11047).
     The configured plate scale is never used as the fallback.
  2. **Can write a foreign plate scale to a non-home rig.** In a deep cascade (solver returns no
     `plate_scale_arcsec_px` AND `_pscale_adj` has no new/expected scale), `_vy_plts = _exp_scale_apx` = 9.77
     (11045) and is written as `VY_PLTS` into the MASTERSTAR header (11051-11060) on a non-home rig.
  This is the "home works / other rig silently wrong" failure mode the project treats as non-negotiable.
  Fix-phase: replace the `9.77` literal (10994) with `cfg.export_arcsec_per_px` (or the DB/equipment-derived
  expected scale), which also makes 11047 coherent/reachable. Cross-rig regression (home + narrow + Brno)
  required since this touches the written WCS scale. **Most science-relevant finding of the audit so far.**
- **DR2-5 [LOW] anisotropy quality gate fails OPEN on nan (10927, 10965).** If `WCS(hdr).pixel_scale_matrix`
  access raises, `scale_ratio`/`scale_ratio2` = nan; the guards (`isfinite and > _aniso_thr` at 10930; the
  post-retry raise at 10967) are then False -> anisotropy check/retry silently skipped, solve accepted. A
  quality gate failing permissive on a computation error. LOW (a WCS that passed `_has_valid_wcs` should
  compute fine). Fix-phase: log the nan case so a non-evaluable scale is visible, not treated as iso-pass.
- **DR2-6 [LOW, V] `_wcs_ok` is DEAD - the WCS-quality block is compute-and-log-only.** Verified by grep:
  `_wcs_ok` appears only at 10996/11003/11004/11021 - read solely at 11004 to gate a log line inside the
  same try; NO consumer after 11021. So `masterstar_wcs_quality` (11000) gates nothing - its only effect is
  a diagnostic log. Implication: DR2-4's contamination of this check (a false "bad quality" log on a
  9.77-vs-real mismatch) is harmless log noise; the real DR2-4 concern is the VY_PLTS write path, not this
  check. Fix-phase: either wire `_wcs_ok` to gate something (force rescale / surface a warning) or collapse
  the block to a single log; it is currently a non-functioning "quality gate".

PASS-2 broad-except triage: 10927/10965 -> DR2-5 (fail-open nan); 10991 -> DR2-4 chain (WCS-scale compute
fails -> _exp_scale_apx None -> forced 9.77); 11019/11030/11062 -> LOGGED, ACCEPTABLE; 11048 (`_vy_plts=None`)
-> defensive, guarded by `if _vy_plts is not None` at 11051, ACCEPTABLE; 11099 (-> output error key) ->
FITS already written, error surfaces, ACCEPTABLE.

**Running PASS-1+2 picture:** setup/selection/build/solve-prep (PASS 1) disciplined. Scale-adjust block
(PASS 2) is mostly sound fail-fast guards + ONE genuine cross-rig robustness bug (DR2-4) + one non-gating
"quality" block (DR2-6) + one fail-open gate (DR2-5).

### PASS 3 - catalog build + df_final + annotation + photometry plan (11101-11540)
Hard guards correct: `n_detected==0 -> raise` (11268); `platesolve_dir None -> raise` (11160); heavy calls
`detect_stars_and_match_catalog` (11205), `write_photometry_plan_files` (11509), `_annotate_masterstars_flux_zones`
(11359) are UN-wrapped -> failures propagate (GOOD).

**Flagged worry REFUTED [V]:** the 11357 `df_final` silent broad-except does NOT mask a catalog-match
failure. The match already happened (`df_out` is the matched result; `n_detected==0` raises at 11268). The
`pd.read_csv(csv_path, dtype={catalog_id:str,name:str})` at 11356 merely RE-LOADS the optimizer output from
disk (string dtypes to avoid Gaia-ID float precision loss); `except -> df_final = df_out.copy()` (11358) is a
reasonable degradation. Residual (smaller) finding -> DR2-8.

- **DR2-7 [LOW, V] dead branch via hardcoded `_skip_boj = True` (11137).** Verified: `_skip_boj` is a constant
  `True` (not a config knob), so `if _skip_boj:` always fires and the global-median background-subtraction
  else-branch (11144-11154, "BOJ o nulu") is unreachable. Fix-phase: gate the disable behind a config flag or
  remove the dead branch.
- **DR2-8 [LOW, V] silent + precision-risk `df_final` fallback (11357-11358).** Verified: the except has NO
  log, and `df_out.copy()` does not re-assert string dtype on `catalog_id`/`name` - re-introducing the exact
  Gaia-ID float-precision loss the string-dtype read at 11356 was preventing. Fix-phase: add a log and
  string-type `catalog_id`/`name` in the fallback path.
- **DR2-9 [LOW] hardcoded `time.sleep(0.5)` race band-aid (11492).** Comment: "UI may read CSV immediately
  after this returns." A fixed sleep to dodge a read-after-write race is fragile (slow disk / load). Fix-phase:
  make the UI-side CSV read robust (exists+retry) instead of pacing with a sleep.
- **DR2-10 [LOW, V] silent `_sync_comparison_stars_across_setups` failure (11517-11520, bare `pass`, no log).**
  Verified. A silent failure of the cross-filter comparison-star sync could leave B/V/R setups with
  INCONSISTENT comparison stars and no signal - relevant to multi-filter robustness. Fix-phase: log the failure.

PASS-3 broad-except triage: 11193 (cone-cache skip), 11257 (VYVAR-pairs merge skip), 11347 (catalog-ID repair
skip), 11351 (optimizer skip -> writes un-optimized df_out), 11377 (source_type annotate failed), 11477
(VY_FWHM_GAUSS fit failed, logged WITH traceback): all LOGGED, ACCEPTABLE (best-effort enhancements w/ graceful
fallback). Note 11351 writes the un-optimized catalog -> acceptable but worth a louder log (silent optimizer
skip degrades match quality). 11489 (VY_NDAO tag), 11496 (`del df_out`), 11507 (dup-artifact cleanup): BENIGN
cleanup `pass`.

**Running picture (PASS 1-3, 11101-11540 done; ~11540-12075 remains):** catalog construction is solid with
correct fail-fast on critical paths; PASS-3 excepts are best-effort enhancements (logged) or benign cleanup.
Standout remains DR2-4 (PASS 2, cross-rig 9.77). PASS 3 adds four LOW hygiene/robustness items (DR2-7..10).

### PASS 4 - MASTER_SOURCES write + stress test + return (11540-12075)
Builds a rich Gaia cross-match (MASTER_SOURCES DB table) with per-star quality flags (border/blend/sat/
variable/non-single/catalog-noise/Gaia-neighbour-veto/nonlinear-FWHM/bad-column), writes to project DB, runs
a stress test (10% relative-RMS sample -> STRESS_RMS, flag Unstable > 1.5x bin median, VSX top-3-per-bin ->
flag Variable) and a common-field bbox crop.
**Design (GOOD):** the entire MASTER_SOURCES + stress block (11546-12062) is wrapped in an outer try that
records failure to `out["master_sources_error"]` and CONTINUES. Primary deliverables (masterstar FITS,
catalog CSV, photometry plan) are produced in PASS 3 BEFORE this block, so a MASTER_SOURCES failure never
loses core output. Inner DB writes (11925/11970/12061) capture errors to `out["..._error"]` (surface to caller).

- **DR2-11 [LOW] fail-silent quality-veto cluster (no log).** Several vetos degrade silently on an internal
  exception, NO log: blend detection 11607 (`-> blended_idx=set()`), per-star FWHM 11659 (`-> all nan`,
  well-handled by 11663-11669 fallback cascade), neighbour-veto pixel-scale 11704 (`-> scale=nan ->
  veto_radius=nan -> neighbour veto skipped`), BPM JSON 11741 (`-> None -> no bad-column flag`), per-star
  neighbour veto 11851 (`pass`). Each fails toward "fewer stars excluded" (permissive) - individually
  acceptable. The gap is the SILENCE: a rig-systematic failure (e.g. WCS pixel-scale always nan on some rig)
  would silently disable a veto fleet-wide with no signal. Fix-phase: one-line log when a veto is disabled by
  an exception.
- **DR2-12 [LOW, V] silent DB-aware photometry-plan rewrite failure (11923 `pass`).** Verified: after the
  MASTER_SOURCES write, `write_photometry_plan_files` is called AGAIN with `draft_id`+`database_path`
  (11912-11921) so the plan reflects DB safe-comp exclusions; on failure it `pass`es silently (11923-11924),
  keeping the earlier plan from 11509 (written WITHOUT DB context). A silent failure means the plan may
  include comps that MASTER_SOURCES flagged unsafe, with no signal. Fix-phase: log the failure (and consider
  whether the plan should hard-depend on the DB-aware pass when a draft_id is present).

PASS-4 broad-except triage: 11607/11659/11704/11741/11851 -> DR2-11 (fail-silent vetos); 11842 (`continue`,
skip neighbour with unparseable mag) ACCEPTABLE; 11923 (`pass`) -> DR2-12; 11925/11970/12061/12063/12073 ->
capture error into `out["..._error"]` (surface to caller) ACCEPTABLE.

## DR2 SUMMARY - `generate_masterstar_and_catalog` fully read (10180-12075, 1896 LOC, 47 broad-except)
**Headline recalibration.** The "#1 most dangerous function" (flagged purely on its 47-broad-except count) is,
on a full line-by-line read, **defensively and carefully written.** The broad-excepts are overwhelmingly DB
`conn.close()` cleanup, clamped numeric fallbacks with sane defaults, logged warnings/retries, and best-effort
QA enrichment whose errors surface to the returned `out` dict. Critical paths have correct fail-fast `raise`
guards: solver-not-solved (10878), no valid WCS (10889), match_rate<60% (10902), anisotropy-after-retry
(10967), n_detected==0 (11268), platesolve_dir None (11160). Heavy science calls
(`detect_stars_and_match_catalog`, `write_photometry_plan_files`, `_annotate_masterstars_flux_zones`) are
un-wrapped and propagate.

**One genuine science bug:** DR2-4 (hardcoded home-rig `9.77` plate-scale fallback, 10994) - MED likelihood /
HIGH fix-priority; violates the cross-rig robustness rule and touches WRITTEN output (VY_PLTS) on non-home rigs.
Header corruption requires a deep fallback cascade (solver + pscale_adj both yield no scale) so likelihood is
MED, but as a silent cross-rig correctness hazard on written output it is a HIGH-priority fix. Also shadows the
`cfg.export_arcsec_per_px` fallback (11047 dead). Fix: use `cfg.export_arcsec_per_px` / DB-derived scale +
cross-rig regression (home + narrow + Brno).

**11 LOW hygiene/robustness items:** DR2-1 misleading default `max_catalog_rows=12000`; DR2-2 redundant re-floor
(11197); DR2-3 silent bin=1 (10693); DR2-5 anisotropy gate fail-open on nan; DR2-6 dead `_wcs_ok` / non-gating
WCS-quality block; DR2-7 dead BOJ branch (`_skip_boj=True`); DR2-8 silent precision-risk `df_final` fallback;
DR2-9 `time.sleep(0.5)` race band-aid; DR2-10 silent cross-filter comp-sync; DR2-11 fail-silent quality-veto
cluster; DR2-12 silent DB-aware plan rewrite.

**Real risk = SIZE / MAINTAINABILITY, not hidden bugs:** 1896 LOC, 18 returns, deep nesting (MASTER_SOURCES
block ~8 indent levels). A phased refactor (setup / select+build / solve+scale / catalog+plan /
master_sources+QA, each a named helper) would materially improve auditability but is itself high-risk and MUST
be gated by byte-identity regression on a known draft (home + one other rig). Treat the refactor as a SEPARATE,
later, carefully-gated workstream - NOT part of the hygiene fix-batch.

### Next options
1. Continue deep reads -> next monster by priority: `_phase2a_process_one_target` (photometry_core.py:7284,
   1167 LOC / 27 broad) or alignment `_astrometry_align_impl_body` (pipeline.py:12302, holds Cross-link 4 /
   Fix-C cap).
2. Pivot to the fix-batch: it now includes the first real bug (DR2-4 cross-rig 9.77) + the accumulated hygiene.

## DR3 (TARGETED) - `_astrometry_align_impl_body` (pipeline.py:12302-13273, 972 LOC, 27 broad-except, 8 nested defs)
**Scope note (honest coverage):** TARGETED read to close the two open threads that made this the recommended
next function - Cross-link 4 (dead `max_control_points`) and the Fix-C control-point cap - plus a structural
except triage. A full line-by-line pass of the per-frame alignment worker (`_flush_one_alignment` 12858, the
astroalign call path) + in-context triage of its 27 broad-excepts is DEFERRED to a focused follow-up. The
control-point/perf threads are fully resolved at source level.

- **DR3-1 [V] Cross-link 4 CONFIRMED at source - `max_control_points` is a DEAD arg.** Declared
  `max_control_points: int = 180` at 12307 (and public wrapper 13280); callers thread 200/250. **Never read
  in the body** (grep 12302-13273: only def lines + log/record sites use the *derived* values). Effective cap
  is independent: `_align_star_cap = max(10, min(5000, cfg.alignment_max_stars))` (12347) then
  **`min(_align_star_cap, 200)`** (12349, comment "use at most TOP 200 brightest stars") ->
  `ref_xy_fit = ref_xy[:_align_star_cap]` (12674) -> `align_cp = max(12, min(_align_star_cap, round(1.5*len(ref_xy_fit))))`
  (12820) -> fed to astroalign; recorded `alignment_max_control_points_used` (13236). A caller's
  `max_control_points=250` has ZERO effect; real knobs = `cfg.alignment_max_stars` + the hardcoded `200`
  (the `200` is INTENTIONAL: dense-field stability per the comment).
- **DR3-2 [V] Fix-C control-point perf lever IDENTIFIED.** On a dense field `ref_xy_fit` is capped at 200
  (12674), so `align_cp = min(200, round(1.5*200)=300) = 200` (12820); astroalign RANSAC over up to 200
  control points/frame = exactly the Fix-C pathological slowness (~654 s/frame at mcp~200; ~3-10 s at ~50).
  The perf lever is the `200` at 12349 (and/or `align_cp` at 12820). Lowering it for dense fields recovers
  perf, but `200` was chosen for stability -> needs CROSS-RIG regression (home sparse + narrow + Brno dense)
  before defaulting. Consistent with the deferred Fix-C perf ticket.

**Combined fix-phase decision (one change, both threads):** wiring `max_control_points` THROUGH to replace
the hardcoded `200` at 12349 (clamped, sane default) would simultaneously (a) make the caller-facing knob real
(closes Cross-link 4) and (b) give Fix-C dense-field perf tuning a first-class lever. Alternative: drop the
dead arg + document that `alignment_max_stars` + the `200` cap is the real knob (hygiene-only; leaves Fix-C a
separate hardcode edit). Either way correctness-adjacent -> needs the cross-rig alignment regression.

**Structural except triage (27 broad - from AST; in-context triage deferred):** matches DR2's disciplined
style - `raise` reraise (12626, good); nested-retry try (12520, 12550); log `Expr` (12578, 13095, 13134);
fallback `return` in nested helpers (12736, 12754); fallback `Assign` (12435, 12704, 12711, 12721, 12785,
12881, 13006, 13010, 13073, 13078); `pass` skip/cleanup (12390, 12440, 12523, 12553, 12652, 12691, 13021,
13042, 13188). No silent-swallow in an obvious science-math path spotted structurally, but the per-frame
alignment worker excepts (12858+) warrant an in-context pass before any are called benign - flagged for follow-up.

**Net:** both motivating threads closed - `max_control_points` is dead (DR3-1), Fix-C perf lever is the
`200`/`align_cp` chain (DR3-2). Recommend the combined fix (wire the arg to replace the 200), gated by
cross-rig alignment regression.

### Next options
1. Full line-by-line follow-up of `_astrometry_align_impl_body` (in-context triage of the 27 excepts + the
   alignment worker `_flush_one_alignment` 12858) for complete coverage.
2. Next priority monster `_phase2a_process_one_target` (photometry_core.py:7284, 1167 LOC / 27 broad) - the
   Phase-2A per-target photometry core (LC numbers + Fix B gate live here).
3. Pivot to the fix-batch - anchored by DR2-4 (cross-rig 9.77) + the Cross-link-4/Fix-C decision (DR3) +
   accumulated hygiene.

## DR4 - `_phase2a_process_one_target` (photometry_core.py:7284-8450, 1167 LOC, 27 broad-except) - PASS 1 (core math 7785-8055)
The Phase-2A per-target differential-photometry core: where each target's calibrated lightcurve + error
bars are produced. PASS 1 reads the math core: ensemble normalize -> dilution -> aperture correction ->
color term -> error model (Fix A). Cat I lens active (rig-specific value used as a universal fallback,
per Milan's binding principle).

**Core math - SOUND (verified at source, HEAD f3b73e9):**
- `ensemble_normalize` (called 7791; def 2454-2659) - flux-sum ensemble `m_ens=-2.5log10(sum 10^-0.4m)`,
  zeropoint `mag_calib = mag_inst + median_j(cat_j - inst_j)` (classic differential offset, documented
  2472-2476; NOT median(cat) which would shift by ~2.5log10(n)). Comps = good+suspect (2500-2503), ordered
  by (quality, comp_rms, id), n_comp_min always + extras under p2p threshold, capped n_comp_max. Outputs
  (mag_calib, delta_mag, ensemble_scatter) ALL sized `n_frames = len(target_mag_inst)` (2484-2487). Clean.
- Aperture correction (7878-7894) - `mag_calib_ac = mag_calib + delta_m_corr` gated on `ac_ok`; sign
  sanity log. Clean.
- Color term (7896-7948) - applied only if `apply_color_term` + group-CT `apply_gate` + an extrapolation
  guard (`_check_color_term_extrapolation` keeps target uncorrected when its BP-RP is outside the comp
  color range, 7922-7948). `c1=0` default; `mag_calib_ct` = copy when CT skipped. Matches the corrected
  CT design.
- Error model / Fix A (8037-8051) - `err = sqrt(err_photon^2 + ensemble_scatter^2)`, where
  ensemble_scatter is the per-frame SEM of the comps' zeropoint residuals (Honeycutt 1992). The inline
  comment (8038-8046) correctly documents the Fix-A rationale (replaced `std(comp inst mags)` which caused
  the ~0.58 mag / 23x inflation; dropped the double-counting `comp_rms_med/sqrt(n)` term). Math correct.

### Findings
- **DR4-1 [LOW] fragile implicit ordering coupling in the Fix-A error model (8049-8051).** `err` (from
  `target_frames = all_frames[catalog_id==target_cid]`, 8015) and `ensemble_scatter` (sized to the target
  LC) are combined POSITIONALLY, guarded only by `if _ens_sc.shape == err.shape` (8049). **Verified not an
  active bug:** both derive from the same `all_frames` filtered by the same `target_cid` in the same row
  order, so they are co-ordered today and the pairing is correct. But the coupling is implicit/unenforced,
  and on a genuine shape mismatch the ensemble-ZP term is **silently dropped** (no else / no log ->
  under-estimated err -> wrong IVW/SysRem weights + trust band). Harden: align by a key (source_file/bjd)
  rather than position, and log if the guard ever fails. (Downgraded MED->LOW after the ordering check.)
- **DR4-2 [LOW, Cat I] hardcoded `3.0` px fallback aperture (7842).** In the GS11 dilution aperture:
  `_ap_px = float(apertures_px.get(target_cid, 3.0))` then `_ap_arcsec = _ap_px * state.plate_scale_arcsec`
  (7843). The `3.0` px default is rig-dependent in angular terms (3 px = ~29" home rig vs ~1.7" Brno). Low
  risk: it is gated behind the config aperture (7838-7840, `gs11_dilution_aperture_arcsec` used when finite
  / positive), is a defensive `.get` default rarely hit, only affects the dilution correction, and IS
  scaled by the derived plate scale. Per the Cat I principle, derive the fallback from FWHM (k*FWHM) rather
  than a fixed px literal. The plate scale itself is correctly taken from `state` (derived) - Cat I clean
  there.
- **DR4-3 [LOW] CT prototype uses hardcoded `min_comp=5` / `sigma_clip_sigma=3.0` (7960-7961, 7980-7981)**
  while production color term uses `cfg.phase01_ct_min_comp` (7990). Minor inconsistency; diagnostic-only
  (the prototype runs under `_ct_prototype_enabled()` 7950, logs `gate_would_pass` via
  `_append_ct_prototype_row`, does NOT affect production photometry). Align to config or annotate.

### Broad-except triage (PASS-1 region)
- 7849 (`_cid_int=None` on ID-normalize fail), 7907 (`fv=nan` on bp_rp parse fail): defensive, benign.

### Net (PASS 1)
The differential-photometry math core is sound and the Fix-A error model is correct + well-documented. No
active bug; three LOW items (one fragility DR4-1, two minor hardcodes DR4-2/DR4-3 via the Cat I lens).
Plate scale is derived from state (Cat I clean).

### DR4 PASS 3 - tail: airmass / SG / Democratic / check-star / artifacts / summary / return (8055-8450)
**Framing correction (from PASS-1 "Remaining" note):** there is NO SysRem/IVW in this function. It is a
per-target LC producer (writes `lightcurve_<cid>.csv` + a summary row). SysRem (Tamuz 2005, operates across
light curves) is a separate downstream phase. So the "IVW consumes DR4-1 err" concern is downstream, not
here. `err` is written to the LC CSV (8192) and used for the LC PNG; no weighting math in this function.

Findings:
- **DR4-4 [LOW, V via cross-link] airmass detrend is DEAD / removed; misleading comments + dead vars/helper.**
  The "Airmass detrending" section (8056) and the SG comment "Runs after airmass detrend" (8101) advertise
  an airmass detrend that does NOT run: `am_slope` stays a NaN constant (8097, never updated), `am_piecewise`
  stays `False` (8098, F841-unused per Cat B - read only inside the dead helper), and
  `_apply_airmass_detrend_helper` (photometry_core.py:6046) is DEAD (Cat A - no caller, already listed in
  the Cat A inventory line 49). Confirmed by the summary row: `"am_detrended": bool(math.isfinite(am_slope))`
  (8419) is ALWAYS False, and the log prints `am_slope=nan` (8444). The summary honestly reports
  am_detrended=False, so this is NOT a science bug (airmass is handled by the differential comp ensemble +
  opt-in SG); it is dead + misleading. (Corroboration: `docs/VYVAR_DECISIONS.md:1012` still documents the
  airmass detrend as live, with a stale `:5732-5754` line ref - also needs a docs fix.) Fix-phase (hygiene):
  remove `am_slope`/`am_piecewise` + the dead Cat-A helper + fix the stale comments (8056, 8101) + the
  DECISIONS.md entry. Note: `airmass_arr` IS still computed + written to the LC CSV (8184) for downstream/
  external use - keep that.
- **DR4-5 [LOW, verify] LC PNG write is unwrapped while sibling artifact writes are try-wrapped.** The LC
  PNG `save_lightcurve_png` (8328, under `if _save_png:`) has NO try/except, but the cutout PNG (8342-8352)
  and the field-map PNG (8355-8366) are each wrapped (`except -> LOGGER.warning("Optional artifact write
  failed")`). So an LC PNG render failure would propagate and ABORT the target's summary bookkeeping (the
  `summary_rows.append` at 8400 + `n_lc += 1` at 8435 never run), even though the science LC CSV is already
  written at 8179. Confirm whether `save_lightcurve_png` is internally robust; if not, wrap it like its
  siblings so an artifact failure cannot lose the target's summary row.

Sound / clean in the tail:
- ALG-2 Savitzky-Golay (8100-8113) and ALG-4 Democratic Detrender (8115-8127): both config-gated opt-in
  (`savgol_detrend_enabled`, `democratic_detrend_enabled`), with cfg-driven window/polyorder. Conservative
  defaults honored.
- Check-star ensemble + sidecar (8129-8167): QA diagnostic; `_chk_n_min = max(1, min(3, len(comp_ids)))`;
  failure logged at DEBUG + skipped (ACCEPTABLE - not the science output).
- `save_lightcurve_csv` (8179-8210): the science output write; UN-wrapped (propagates on failure - correct,
  a failed LC write should surface). Writes the DR4-1 `err` (8192).
- Summary row (8400-8434): comprehensive; comp_path "sparse_fallback" detection (8394, DR1-2 thread), tier
  counts, RMS (full + ooe), dilution, CT. Honest.

Broad-except triage (PASS-3 region):
- 8166 (check-kmag sidecar, log DEBUG), 8265 (method-LC init, log WARNING), 8323 (comp_quality.json, log
  WARNING), 8351 (cutout PNG, log WARNING), 8365 (field-map PNG, log WARNING): optional artifacts, LOGGED,
  ACCEPTABLE. 8273 (catalog-only LC alias copy, `pass`): benign cleanup. 8258 (per alt-method, log WARNING),
  8294/8296 (tier parse -> 0/""): defensive, benign.

### DR4 partial summary (PASS 1 core + PASS 3 tail done; PASS 2 setup 7284-7785 remains)
The Phase-2A per-target photometry core is **SOUND**: the differential math (ensemble_normalize, AC, color
term + extrapolation gate, Fix-A error model) is correct (PASS 1), and the tail (detrend hooks, check-star,
output writes, summary) is disciplined (PASS 3). NO correctness bug found. Five LOW items total:
- DR4-1 err/ensemble_scatter positional coupling (not active; harden + log).
- DR4-2 [Cat I] `3.0` px fallback aperture (derive from FWHM).
- DR4-3 CT prototype hardcoded `min_comp=5`/`sigma=3.0` (diagnostic-only).
- DR4-4 dead airmass detrend + misleading comments + dead vars/helper (hygiene; cross-links Cat A + Cat B + DECISIONS.md).
- DR4-5 LC PNG unwrapped vs wrapped siblings (robustness consistency).
Cat I lens: the core is essentially clean (one minor `3.0` px); plate scale + apertures derive from
state/FWHM. PASS 2 (input assembly: comp LC, quality maps, aperture/AC prep, PyTICS) is the only remaining
DR4 span - lower science interest (it consumes upstream Phase-1 outputs already audited in DR1).

### Next options
1. DR4 PASS 2 (setup 7284-7785) for complete coverage of this function.
2. Next monster: `solve_wcs_with_local_gaia` (vyvar_platesolver.py:4255, 1434 LOC) - the platesolver, the
   other high-science-value + likely Cat-I/hardcode site (cone radius, mag limits, RMS thresholds).
3. Pivot to the fix-batch (now: DR2-4 implemented-pending-regression + Cross-link-4/Fix-C decision + the
   accumulated LOW hygiene incl. DR4-1..5).

## DR5 (Cat-I focus) - `detect_stars_and_match_catalog` (pipeline.py:7312-8300, 989 LOC, 13 broad-except, 3 nested defs)
DAO detection + Gaia/VSX catalog matching for one image (masterstar or per-frame). Read with the Cat-I lens
(rig-specific value used as a universal fallback) as the primary objective - this is the top hardcode-density
site in the tree (90 magic numbers per Cat H) - plus a structural except triage.

### Cat-I VERDICT - clean (no 9.77-style violation; verified at source, HEAD f3b73e9)
The 90 magic numbers are **predominantly legitimate algorithmic / astrometric constants**, NOT rig-specific
hardcodes:
- **Match-rate / iteration thresholds (algorithmic, rig-independent):** retry if match < 70% (7941),
  converged at r >= 0.95 (7952), widen step `max(*1.12, +0.45")` (7954), `med_nn > 1.15*sep` NN sanity
  (8204), tighten if `n_matched >= max(10, 0.20*n)` (7964). Fine.
- **Astrometric match tolerances are in the CORRECT unit (arcsec = sky angle, universal):** signature
  defaults `match_sep_arcsec=8.0` / `vsx=5.0` / `gaia_variable=2.0` (7320-7322), `thr_wider=min(1.5*sep, 90)`
  (7935), tighten target `_tight_sec` (7964). These are NOT rig-specific - arcsec is the rig-independent unit
  for matching detections to a sky catalog.
- **Cross-rig robustness mechanism present:** `match_sep_used = max(12.0, float(match_sep_arcsec))` (7742,
  7929) floors the match tolerance at 12" regardless of caller. This PROTECTS coarse/undersampled rigs (home
  9.77"/px: a 1-px centroid error = 9.77", so a 2-8" tolerance would fail matching; the 12" floor keeps it
  matchable). So the fixed-arcsec tolerances + the 12" floor are a deliberate, rig-robust design - Milan's
  no-hardcode principle is respected here. (Verified: no plate-scale-relative scaling of the tolerances, and
  none needed given the arcsec unit + the floor.)
- **saturate_level_fraction=0.999 (7323):** a fraction applied to the rig's saturation ADU (from DB/header)
  - derived, not a rig value. Clean.

### Findings
- **DR5-1 [LOW, Cat I] pixel-unit quality thresholds with rig-dependent sky-angle meaning.** Two thresholds
  are expressed in PIXELS, so they mean different things per rig:
  - **WCS-refine RMS reject `rms > 10` px (8032; verified `float(_rms_w) > 10.0`):** 10 px = 5.7" on Brno
    (0.566"/px) but 98" on the home rig (9.77"/px). So a coarse rig can ACCEPT a much worse astrometric
    solution than a fine rig. The fit RMS is naturally in pixels (it is the fit residual), but the ACCEPTANCE
    bar has rig-dependent astrometric meaning. Consider expressing the reject threshold in arcsec (or
    `k * plate_scale`) so WCS quality means the same across rigs. (LOW - coarse rigs are already plate-solved
    with their own validation upstream; this is the pixel-NN refine fallback path.)
  - **DAO FWHM clamp `max(1.2, min(20.0, ...))` px (verified 7485; same clamp pattern reused at the other DAO
    sizing sites):** the 1.2-px floor is a DAOStarFinder sampling requirement (sub-pixel FWHM is
    undersampled), but it matters MOST on the undersampled home rig (real FWHM ~1 px), where clamping up to
    1.2 px is a real assumption. Algorithm-natural (detection sampling limit), but worth a comment that it is
    a DAO-sampling floor, not a seeing estimate.
  These are not "a rig-correct value used universally" (the 9.77 class); they are pixel-unit detection/fit
  metrics where the per-rig behavior is worth a note. No fix is mandatory; flag for the Cat-I-principle pass.

### Structural except triage (13 broad - from AST)
7 fallback `Assign` (7402, 7407, 7421, 7426, 7470, 7507, 7595), 5 `pass` (7460, 7525, 7531, 7554, 8215),
1 `Expr` +log (8188). Pattern matches the disciplined style seen elsewhere (defensive coercion fallbacks +
cleanup skips). The 5 `pass` sites warrant an in-context check before being called benign (deferred to a
full logic read); none is in an obvious science-math computation from the structural view.

### Net
The top hardcode-density function in the tree comes back **Cat-I clean**: no 9.77-style rig hardcode; the
match tolerances are in the correct arcsec unit with a 12" floor that provides cross-rig robustness; the
only items are two pixel-unit thresholds (DR5-1) with rig-dependent behavior, both algorithm-natural. This
reinforces that DR2-4 (the 9.77 plate scale) was an ISOLATED violation, not a pervasive pattern - VYVAR's
detection/matching layer respects the derive-or-config principle. A full line-by-line logic read +
in-context triage of the 5 `pass` excepts can follow if complete coverage is wanted; the Cat-I objective
(the reason for reading this function) is met.

### Next options (post-DR5)
1. Full logic read of `detect_stars_and_match_catalog` (the 5 `pass` excepts + the 3 nested match-pass defs).
2. Last big science monster: `solve_wcs_with_local_gaia` (vyvar_platesolver.py:4255, 1434 LOC) - platesolver
   (cone radius, mag limits, RMS thresholds = next Cat-I site).
3. Pivot to the fix-batch (DR2-4 pending regression + Cross-link-4/Fix-C + accumulated LOW hygiene).

## DR6 (Cat-I focus) - `solve_wcs_with_local_gaia` (vyvar_platesolver.py:4255-5688, 1434 LOC, 25 broad-except, 0 nested defs)
The local-Gaia plate solver - the last big science monster and the most plate-scale-relevant function in the
tree. Read with the Cat-I lens + the scale-write/rescale acceptance path + a structural except triage.

### Cat-I VERDICT - CLEAN, and the POSITIVE EXEMPLAR of Milan's no-hardcode principle (verified at source, HEAD f3b73e9)
**The expected/working plate scale is fully DERIVED, with NO rig-specific hardcoded fallback (no 9.77 class).**
- `_exp_scale` provenance (4593-4639): passed arg `expected_plate_scale_arcsec_per_px` (e.g. MASTERSTAR,
  4593-4598) -> FITS header scale keyword `_scale_hdr_kw` (SECPIX/PIXSCALE/SCALE/SECPIXEL, 4618-4629) ->
  **physics: `plate_scale_arcsec_per_pixel(pixel_pitch_um, focal_length_mm)` from the header optics**
  (4630-4637) -> else **stays `None`** (and 4640 returns solved:False rather than injecting a literal). No
  literal fallback anywhere in the chain.
- Working scale path: computed from pixel pitch + focal physics, fallback to `_exp_scale` only if
  finite/positive; if both fail the solve proceeds without a fabricated scale - it does NOT inject a home-rig
  literal.
- This is **exactly the correct pattern DR2-4 (pipeline.py:10994 `9.77`) should be fixed toward.** The
  platesolver derives scale from optics/header/arg and falls back to None, never to a rig constant. DR2-4 was
  a lone deviation from the codebase's OWN established correct pattern.

**The scale-rescale acceptance path is provenance-aware and rig-safe (5350-5399) - sophisticated, correct:**
- Empirical-scale rescale (5351-5388): requires >=14 star pairs (5351); computes the empirical median plate
  scale from matched pairs (5353); **if the empirical scale differs from the header/expected scale by >10%,
  it DISTRUSTS the empirical value** (`_rel_hdr > 0.10` -> logs, sets `_emp_s=None`, 5367-5372) - an explicit
  anti-confusion guard (bad pairs once gave ~12"/px vs the real ~9.55"/px and broke the FITS WCS, per the
  inline comment 5360-5361). Rescales only within 10% + a 0.7% trigger mismatch (`trigger_relative_mismatch=0.007`, 5377).
- Expected-scale rescale (5391-5399): only if empirical did not fire, AND gated by
  `_allow_expected_cd_rescale = not (blind_solver AND _exp_scale_from_expected_arg)` (5391-5392) - it will
  NOT rescale to the expected scale when the hint is from a blind solver AND the expected scale came from a
  passed arg (anti-circular guard). The comment (5389-5390) states the rule precisely: rescale-to-expected
  helps when the expected scale comes from FITS optics, harms when it comes from DB/config under a blind hint.
- Net: the solver KNOWS the provenance of its scale (FITS optics vs DB/config vs blind) and gates the CD/PC
  rescale accordingly. This is the OPPOSITE of the DR2-4 blind-inject bug. Strongly reinforces that DR2-4 is
  isolated, not representative.

**Other Cat-I candidates - all benign (geometric / astrometric / algorithmic):**
- `0.5 * naxis1/2` (5092/5093, 5241/5242) = image center. Geometric, correct.
- `sep2d < 2.0 arcsec` (5662) = match tolerance in the correct (arcsec) unit. Fine.
- `_off_deg >= 0.05` deg (5248) = the locked cone-recenter threshold (~3'); deliberate astrometric value.
- `max(12, ransac_min_pairs)` (5465) = RANSAC min-pairs floor. Algorithmic.
- `expected_scale_rel_tol_override=1.0` (4885) = a deliberate RELAXED scale tolerance for a recovery "probe"
  path (default None at 4853/4948 = normal tolerance); a relaxation knob, not a rig value.
- `max_px=max_px_coarse * 1.35` (5062) = geometric expansion of a derived coarse pixel-match radius for a
  retry. Algorithmic.
- `_rel_hdr = abs(_emp_s/_exp_scale - 1.0)` (5364) = the 10% empirical-vs-expected comparison above.

### Findings
- **DR6-1 [LOW, Cat I] DAO FWHM pixel fallback `configured_fallback=3.0` -> `3.5` (4754-4756).** Same class
  as DR5-1: `dao_detection_fwhm_pixels(hdr0, configured_fallback=3.0)` (4754) tries the header first
  (header-derived when available), then falls back to `3.0` px (and `3.5` px if it returns None at 4756, with
  a log at 4757). A px FWHM fallback is rig-dependent in angular terms but algorithm-natural (DAO needs a
  kernel FWHM). Adjacent detection-sigma fallbacks `_sig_cfg=3.5` (4749) and `_sips_fb=2.5` (4751) are
  algorithmic detection constants. LOW - prefer deriving the kernel FWHM from data when the header lacks it;
  otherwise annotate as a DAO-sampling fallback.

### Structural except triage (25 broad - from AST)
10 fallback `Assign`, 9 `pass`, 4 `Expr` (log), 1 `return`, 1 `continue`. Matches the disciplined style of
the other monsters. The empirical-rescale `pass` (5387) is a SAFE skip (on failure it keeps the solved WCS,
the correct degradation - it does not inject a scale). The other 8 `pass` sites warrant an in-context check
before being called benign (deferred to a full logic read); none is in an obvious science-math computation
from the structural view.

### Net
The most plate-scale-relevant function in the tree comes back **Cat-I CLEAN and is the positive exemplar**:
the expected/working scale is derived (arg -> header -> pixel-pitch+focal physics -> None, never a rig
literal), and the scale-rescale path is provenance-aware with explicit anti-confusion (>10% empirical reject)
and anti-circular (no rescale-to-expected under blind+arg hint) guards. This is the correct pattern DR2-4
must be fixed toward, and it confirms DR2-4 was an ISOLATED lapse, not a codebase pattern. Only one LOW Cat-I
item (DR6-1, the DAO FWHM px fallback, same class as DR5-1). A full line-by-line logic read + in-context
triage of the 8 remaining `pass` excepts can follow for complete coverage; the Cat-I objective is met.

## Science-critical deep reads - COMPLETE (DR1-DR6)
With DR6, the high-blast-radius science functions are read: comp selection (DR1), the masterstar/catalog
entry point (DR2, full), alignment control-point/perf threads (DR3), the per-target photometry core (DR4
core+tail), detection/matching Cat-I (DR5), and the plate solver Cat-I (DR6). Consistent verdict across all
six: **the core science is sound and respects the derive-or-config principle; ONE genuine cross-rig bug
(DR2-4, the home-rig 9.77 plate-scale fallback) - found, isolated, and fixed pending regression; the real
residual risk is monster-function size/maintainability, plus ~20 LOW hygiene/robustness items.** The
remaining monsters are lower science value (orchestration `run_phase0_and_phase1`; `export_per_frame_catalogs`;
`config.py __post_init__` for the Cat-C resolver/dead-knob closure; UI renderers) - worth reading for
completeness but unlikely to change the headline.

### Recommended pivot: the fix-batch
The audit has reached the point of diminishing NEW correctness findings. Recommend pivoting to the fix-batch
(each fix isolated, byte-identity/regression where logic is touched, Milan-approves, Cursor implements):
1. **DR2-4** cross-rig 9.77 plate-scale fallback - IMPLEMENTED, pending cross-rig regression + push (the one
   real bug; fixed toward the DR6 derive-or-None pattern).
2. **Cross-link-4 / Fix-C** (DR3) - wire `max_control_points` to replace the hardcoded 200, gated by
   cross-rig alignment regression (closes the dead arg + gives the dense-field perf lever).
3. **Dead code / dead params** (Cat A 51 dead + Cat C dead knobs) - after the database.py 14-public-method
   API check + the config.py __post_init__ resolver-indirection read.
4. **BO CVn debug generalization** (`VYVAR_DEBUG_COMP_CIDS`, prototyped) + F811 import hygiene (Cat B).
5. **~20 LOW hygiene/robustness items** (DR1-1..7, DR2-1..3/5..12, DR4-1..5, DR5-1, DR6-1) - batch the
   logged-fallback / dead-branch / misleading-default / stale-comment items; the silent-veto and
   silent-plan-rewrite logs (DR2-10/11/12) are the highest-value robustness wins.
The phased refactor of the monster functions is a SEPARATE, later, byte-identity-gated workstream - NOT part
of the hygiene fix-batch.

### Next options (post-DR6)
1. Pivot to the fix-batch (recommended) - start with DR2-4 cross-rig regression, then the dead-code/hygiene
   batch.
2. Continue deep reads for completeness: `run_phase0_and_phase1` (comp-selection orchestration + the
   `max_psf_chi2=inf` override site) or `config.py __post_init__` (closes the Cat-C resolver/dead-knob threads).
3. Full logic-read follow-ups of the targeted/partial reads (DR3 alignment worker, DR4 PASS 2 setup, DR5/DR6
   remaining `pass` excepts) for byte-complete coverage.
