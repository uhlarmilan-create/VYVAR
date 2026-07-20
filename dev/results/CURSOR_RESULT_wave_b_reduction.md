CURSOR RESULT - 2026-07-18 WAVE-B-PARAM-REDUCTION

What I did
Executed the anchor-gated WAVE-B parameter reduction per the PARAM-BUDGET-AUDIT
dispositions (`dev/results/param_budget_audit.csv`) as approved by Milan. Reduced the
registered-parameter surface from 304 to 269 entries (config.json now persists 249),
one commit per step, full pytest green after each, closed by the mandatory `--full`
anchor gate vs draft_435.

## Per-step commits + diff --stat

STEP 1 - WIRE-IN calibration CCD-temp tolerance (bug fix) - `617b76f`
```
 dev/tests/test_calibration_library_match.py | 29 +++++++++++++++++++++++++++++
 src_py/app.py                               |  2 ++
 src_py/importer.py                          |  5 +++++
 src_py/night_run.py                         |  2 ++
 4 files changed, 38 insertions(+)
```
`temp_tolerance=cfg.calibration_master_ccd_temp_tolerance_c` is now passed at both
importer call sites of `find_best_calibration_library_path` (dark path). The signature
default `temp_tolerance: float = 0.5` was KEPT as a documented last-resort fallback
(callers that do not supply the arg still behave as before). New unit test
`test_dark_selection_honors_nondefault_tolerance` proves selection honors a non-default
tolerance (synthetic library rows at deltaT 0.4 / 0.6 / 5.0).
NO-OP for current runs: the key is absent from Milan's config.json, so the effective
value stays 0.5 - identical to the previously hardcoded default.

STEP 2 - DELETE-DEAD 4 keys - `08e5684`
```
 config.json                           |  4 ---
 dev/tests/test_ui_params_dashboard.py |  6 +++--
 dev/validation/params_registry.json   | 47 -----------------------------------
 docs/VYVAR_PARAMS.md                  | 16 +++++-------
 src_py/config.py                      | 24 +-----------------
 5 files changed, 11 insertions(+), 86 deletions(-)
```
Removed `aperture_fwhm_factor_medium`, `masterstar_log_astroalign`,
`phase01_comparison_proximity_tiebreak`, `phase01_comparison_rms_bin_mag` from AppConfig,
registry, generated docs, and config.json.

STEP 3 - INTERNALIZE frame dims - `03d640c`
```
 config.json                           |  2 --
 dev/tests/test_ui_params_dashboard.py |  6 +++---
 dev/tests/test_wave_b_internalized.py | 27 +++++++++++++++++++++++++++
 dev/validation/params_registry.json   |  8 ++++----
 docs/VYVAR_PARAMS.md                  | 10 +++++-----
 src_py/config.py                      | 13 +++++++++----
 6 files changed, 48 insertions(+), 18 deletions(-)
```
`frame_width_px` / `frame_height_px` remain AppConfig fields (5 real read sites keep
working, still resolve from FITS NAXIS at run time) but left the USER parameter space:
registry owner->internal, widget->hidden, tier->expert; removed from config.json
persistence (save excludes, load ignores with a one-line deprecation log). Guard test
`test_wave_b_internalized.py` asserts save output contains neither key.

STEP 4 - MERGE 14 -> 3 structured keys - `c828c9c`
```
 config.json                           |  32 ++++--
 dev/tests/test_ui_params_dashboard.py |   6 +-
 dev/tests/test_wave_b_merged_keys.py  |  81 +++++++++++++++
 dev/validation/params_registry.json   | 189 ++++------------------------------
 docs/VYVAR_PARAMS.md                  |  29 ++----
 src_py/check_star_kmag.py             |   9 +-
 src_py/comp_selection_per_target.py   |  23 +++--
 src_py/config.py                      | 168 +++++++++++++++++++++---------
 src_py/export_reports.py              |  15 ++-
 src_py/photometry_core.py             |  25 +++--
 src_py/pipeline.py                    |   8 +-
 src_py/ui_settings.py                 |  40 ++++---
 12 files changed, 332 insertions(+), 293 deletions(-)
```
- `comp_color_tiers` (list-of-dicts) replaces 8 scalar `comp_tier{1..4}_bprp_limit/_weight`.
  Accessors `comp_tier_bprp_limits()` / `comp_tier_weights()` back the read sites in
  comp_selection_per_target.py / check_star_kmag.py / photometry_core.py / export_reports.py.
- `phase01_tiers` (list) replaces 4 scalar `phase01_tier{1..4}_mag`; accessor
  `phase01_tier_mags()`.
- `aperture_snr_sizing` (mapping {small,large}) replaces `aperture_fwhm_factor_small/_large`
  (medium already deleted in STEP 2).
Backward compatibility: the config.json loader accepts the OLD scalar keys for one
transition release and maps them into the structured key with a deprecation log; save
writes ONLY the new form. Milan's config.json migrated in the same commit. Dashboard
renders the structured keys via the existing JSON/custom widget path (simpler; no new
widget). Guard tests `test_wave_b_merged_keys.py` cover defaults, accessors, save payload,
and legacy-scalar backward compatibility.

STEP 5 - DELETE-DB-DUP 9 keys + drop SETTINGS table + focal precedence - `d6c0d55`
```
 config.json                                        |  9 ---
 dev/tests/test_config_observer_location_hydrate.py | 75 ++++++++++++++--------
 dev/tests/test_master_validity_days_g6_f002.py     | 38 ++---------
 dev/tests/test_ui_params_dashboard.py              |  6 +-
 dev/tests/test_wave_b_internalized.py              | 28 ++++++++
 dev/validation/params_registry.json                |  2 +-
 docs/VYVAR_PARAMS.md                               |  6 +-
 src_py/config.py                                   | 74 ++++++---------------
 src_py/database.py                                 | 47 ++------------
 9 files changed, 117 insertions(+), 168 deletions(-)
```

STEP 6 - HARDCODE 20 solver internals - `715e754`
```
 config.json                           |  20 ---
 dev/tests/test_ui_params_dashboard.py |   5 +-
 dev/validation/params_registry.json   | 238 --------------------------------
 docs/VYVAR_PARAMS.md                  |  32 +----
 src_py/config.py                      | 252 ++++------------------------------
 src_py/pipeline.py                    |  32 +++--
 src_py/vyvar_blind_solver.py          |  47 +++----
 src_py/vyvar_platesolver.py           |  22 ++-
 8 files changed, 95 insertions(+), 553 deletions(-)
```

STEP 7 - docs & metadata sync - `df93984`
```
 docs/VYVAR_CONFIG_GUIDE_CZ.md | 100 ++++++++++++++++--------------------------
 docs/VYVAR_CONFIG_GUIDE_EN.md | 100 ++++++++++++++++--------------------------
 docs/VYVAR_DECISIONS.md       |  53 ++++++++++++++++++++++
 docs/VYVAR_JOURNAL.md         |  21 +++++++++
 docs/VYVAR_PARAMS.md          |   2 +-
 docs/VYVAR_STATE.md           |  20 ++++++---
 6 files changed, 165 insertions(+), 131 deletions(-)
```

## Final parameter count

Registry: **269 entries** (down from 304). config.json persists **249** keys.
Arithmetic: 304 ? 4 (DELETE-DEAD) ? 11 (MERGE 14->3 net) ? 20 (HARDCODE) = 269.
DB-dup (9) and internalized (2) keys deliberately KEEP registry entries (owner
db_static/fits_dynamic/internal, widget hidden) per STEP 3/5, so they still count toward
the 269 registry total but are excluded from config.json persistence. The audit's "~258"
figure was an estimate that assumed the DB-dup/internal entries would also leave the
registry; keeping them registered (so the honest full-config report can still render them)
is the reason the true landing point is 269.

## The 20 HARDCODE locations (name - old default - module:line)

vyvar_blind_solver.py:
- `blind_scale_tol_frac` = 0.10  -> `_BLIND_SCALE_TOL_FRAC` (vyvar_blind_solver.py:334)
- `blind_cluster_eps_deg` = 1.0  -> `_BLIND_CLUSTER_EPS_DEG` (vyvar_blind_solver.py:335)
- `blind_cluster_min_votes` = 4  -> `_BLIND_CLUSTER_MIN_VOTES` (vyvar_blind_solver.py:336)
- `blind_cluster_min_samples` = 3 -> `_BLIND_CLUSTER_MIN_SAMPLES` (vyvar_blind_solver.py:337)
- `blind_cluster_vote_span` = 12  -> `_BLIND_CLUSTER_VOTE_SPAN` (vyvar_blind_solver.py:338)
- `blind_cluster_coherence_cap` = 25 -> `_BLIND_CLUSTER_COHERENCE_CAP` (vyvar_blind_solver.py:339)

vyvar_platesolver.py:
- `blind_prefilter_min` = 4  -> `_BLIND_PREFILTER_MIN` (vyvar_platesolver.py:68)
- `masterstar_odds_match_floor` = 30 -> `_MASTERSTAR_ODDS_MATCH_FLOOR` (vyvar_platesolver.py:69)
- `masterstar_odds_k` = 12.0 -> `_MASTERSTAR_ODDS_K` (vyvar_platesolver.py:70)
- `masterstar_odds_min_quadrants` = 3 -> `_MASTERSTAR_ODDS_MIN_QUADRANTS` (vyvar_platesolver.py:71)
- `masterstar_false_alarm_p_max` = 1e-6 -> `_MASTERSTAR_FALSE_ALARM_P_MAX` (vyvar_platesolver.py:72)
- `masterstar_sip_force_rms_guard_ratio` = 1.15 -> `_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO` (vyvar_platesolver.py:73)

pipeline.py:
- `moffat_chi2_limit` = 50.0 -> `_MOFFAT_CHI2_LIMIT` (pipeline.py:853)
- `sky_adu_fallback` = 1581.6 -> `_SKY_ADU_FALLBACK` (pipeline.py:854)
- `masterstar_solver_use_draft_median_if_hint_sep_deg` = 1.0 -> `_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG` (pipeline.py:855)
- `masterstar_optimizer_mirror_extra_log` = True -> `_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG` (pipeline.py:856)
- `masterstar_platesolve_prewrite_rms_max_px` = 30.0 -> `_MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX` (pipeline.py:857)
- `masterstar_platesolve_prewrite_relaxed_rms_max_px` = 35.0 -> `_MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX` (pipeline.py:858)
- `masterstar_platesolve_nn_refine_max_rms_px` = None -> `_MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX` (pipeline.py:859)
- `platesolve_anisotropy_threshold` = 1.3 -> `_PLATESOLVE_ANISOTROPY_THRESHOLD` (pipeline.py:861)

(`masterstar_sip_force_rms_guard_ratio` is consumed in BOTH vyvar_platesolver.py and
pipeline.py:860; 20 unique keys total.)

## The 9 DELETE-DB-DUP keys + removal mechanism

Keys: `gain`, `read_noise`, `plate_scale_arcsec_per_px`,
`phase01_plate_scale_arcsec_per_px`, `export_arcsec_per_px`, `observer_lat`,
`observer_lon`, `observer_alt_m`, `observer_location_name`.

Mechanism: the AppConfig fields remain as run-time hydrated mirrors - the existing
hydration/resolver paths are untouched (observer_* hydrate from the DB LOCATION row
selected by `observer_location_id`; gain/read_noise/plate-scale resolve via
`param_resolver` DB/FITS-first). What changed is ONLY config.json persistence: the loader
no longer reads these keys and `to_json` no longer writes them, and they were removed from
Milan's config.json. Registry keeps the entries (owner db_static / fits_dynamic, hidden
widget) so the honest full-config report and the wave-A dashboard still render them.
Byte-identical safety: on draft_435 gain/read_noise/plate-scale already resolve DB/FITS
first (config was never consulted), and the observer coordinates in the DB LOCATION row
for id=2 match the dataclass defaults exactly - so removing the unused config.json copies
cannot change one byte. Guard test `test_db_dup_keys_absent_from_save_payload` asserts
none of the 9 ever appears in save_config_json output; `test_db_dup_keys_still_appconfig_fields`
asserts they remain AppConfig fields. Tests that previously pinned cfg values
(test_config_observer_location_hydrate.py, test_master_validity_days_g6_f002.py) were
rewritten to assert the resolver/hydration path instead.

Vestigial DB SETTINGS table: dropped. `_ensure_settings_table` / `_seed_default_settings`
constructor calls removed; `_drop_settings_table` (idempotent DROP TABLE IF EXISTS) added;
dead accessors `get_setting_int` / `set_setting` removed. Guard
`test_settings_table_dropped_config_is_authoritative` confirms masterdark/flat validity
days are config-authoritative after the drop.

## Focal-length precedence outcome

UNIFIED-BY-VERIFICATION (no code change needed). Both resolution paths already resolve
DB-optics-first with FITS-header fallback via `param_resolver.resolve_focal_mm`
(static-fact model); the audit's "two opposite-order paths" note was stale. No code
comment indicated a deliberate header-first site, so nothing was STOPped - the desired end
state was already in place and is now documented.

## sips_dao_fwhm clarification

RETRACTED. The audit's bonus `sips_dao_fwhm` claim does not correspond to any live key:
the code uses the registered `sips_dao_fwhm_px` everywhere (config.py:456/1028/2088,
pipeline.py:7352/8068/10087/11251/11892/13288, photometry_core.py:14841,
vyvar_platesolver.py:4784, night_run.py:328/882/895, app.py multiple). There is no bare
`sips_dao_fwhm` field or config key. No action beyond the retraction.

## Gate outputs

### pytest (full suite)
```
938 passed, 19 skipped, 31 warnings in 274.08s
```

### session_baseline_check.py --fast
```
git-branch                   PASS   main
git-head                     PASS   df93984
git-staged                   PASS   none
git-untracked-known          WARN   2 known untracked
git-untracked                WARN   dev/scripts/forensic_disc_ui_match2.py
git-origin-main              WARN   differs from origin/main (191be0e); consider git pull
config-paths                 PASS   all present
pytest                       PASS   938 passed, 19 skipped
ledger                       PASS   v1 14 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
OVERALL: PASS
```

### session_baseline_check.py --full (MANDATORY anchor gate vs draft_435) - PASS byte-identical
```
git-branch                   PASS   main
git-head                     PASS   df93984
git-staged                   PASS   none
git-untracked-known          WARN   2 known untracked
git-untracked                WARN   dev/scripts/forensic_disc_ui_match2.py
git-origin-main              WARN   differs from origin/main (191be0e); consider git pull
config-paths                 PASS   all present
pytest                       PASS   938 passed, 19 skipped
ledger                       PASS   v1 14 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
full-provenance              PASS   anchor git_hash=10d610c0e79d...
full-pipeline                PASS   2016s -> tmp\session_baseline\20260718T112444Z
full-science-compare         PASS   n_lc=166 failures=0
full-provenance-hash         PASS   ce37895f4882e035... (informational; git-bound, not cross-commit gate)
full-snapshot-sha-core       PASS   3d26f4692ac81fc5... n=333
full-photometry-sha-core     PASS   3d26f4692ac81fc5... n=333
full-photometry-sha-extended PASS   6420f1daa53a0d5d... n=499
full-counters-runtime        PASS   {"phase2a_empty_comp_drop": 1}
full-counters-meta           PASS   {"phase2a_empty_comp_drop": 1}
full-counters-expected       PASS   allowlisted {"phase2a_empty_comp_drop": 1} (structural empty-comp drops)
OVERALL: PASS
```
Byte-identical confirmed: core SHA `3d26f469...` n=333 and extended `6420f1da...` n=499
both match the draft_435 snapshot exactly; science-compare n_lc=166 failures=0; counters
allowlist unchanged; git_dirty_code=false (git-staged none, HEAD df93984). The 9 DB-dup
fallback removals + 20 hardcodes + merges changed zero bytes of science output, as
expected.

## Files changed (full stack)

src_py/: app.py, importer.py, night_run.py, config.py, database.py,
comp_selection_per_target.py, check_star_kmag.py, photometry_core.py, export_reports.py,
pipeline.py, ui_settings.py, vyvar_blind_solver.py, vyvar_platesolver.py.
dev/validation/params_registry.json; docs/VYVAR_PARAMS.md, VYVAR_CONFIG_GUIDE_EN.md,
VYVAR_CONFIG_GUIDE_CZ.md, VYVAR_DECISIONS.md, VYVAR_JOURNAL.md, VYVAR_STATE.md.
dev/tests/: test_calibration_library_match.py, test_wave_b_internalized.py,
test_wave_b_merged_keys.py, test_ui_params_dashboard.py,
test_config_observer_location_hydrate.py, test_master_validity_days_g6_f002.py.
config.json. Commits: 617b76f, 08e5684, 03d640c, c828c9c, d6c0d55, 715e754, df93984
(on top of audit fbac9bc).

## STEP 9 - push: GATED, awaiting Milan's explicit "push".
