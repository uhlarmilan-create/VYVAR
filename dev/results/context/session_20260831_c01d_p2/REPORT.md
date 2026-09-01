# CURSOR RESULT - CONSOLIDATE-01D phase 2 (approved config removals + dao_gaia flatten)

Date: 2026-09-01. Branch: consolidate-01. English. ASCII.
Base: origin/consolidate-01 @ 6c95ac4. Implementer: Cursor. Architect: Claude.
Milan approved (2026-08-31) every row below; nothing else from the D1 table was removed.
Live draft 516 was not written.

## What I did

P2-1: removed five LEGACY keys (one commit each), hard-wired the default path, added
KNOWN_REMOVED_KEYS, regenerated docs/VYVAR_PARAMS.md.
P2-2: removed five DEAD persist-only keys (one commit).
P2-3: dropped three alias AppConfig fields; JSON loader still maps old names.
P2-4: removed display-only phase01_use_bprp_primary; BP-RP display is unconditional.
P2-5: ROADMAP OPEN D1B-UNITS-01; CONFIG-PREREZ closed except that row.
P2-6: flattened dao_gaia CLI family onto dao_gaia_common.py + iter4; deleted stage_01/iter2/iter3.
G-CONFIG: combined old-config load test committed so G1 --clean sees it.

## Per-key commit list

HEAD after FLOW sync: 0fc53b7 (12 commits on 6c95ac4).

| commit | key / scope | superseding decision |
| --- | --- | --- |
| b3aa740 | global_comp_pool_enabled | COMP-POOL-01 (global pool always on) |
| 47a8d76 | export_err_mode | ERR-CALIB (exported err always calibrated) |
| 6ec0859 | err_background_mode | F-BINGAIN-1 (empirical always; Howell math stays as missing-sigma fallback) |
| 639b303 | masterstar_accept_mode | MASTERSTAR odds gate (odds-only) |
| a2c94a1 | psf_ac_policy | ZP-OK v2 / P4 (p4_none only; chi2_lt5_legacy deleted) |
| 470a0f1 | DEAD 5 persist-only | no consumer; key+field+registry only |
| 4479c22 | ALIAS 3 fields | loader still maps to survivors |
| e4cb6a2 | phase01_use_bprp_primary | display-only; BP-RP always |
| bbeedd8 | P2-5 ROADMAP | D1B-UNITS-01 OPEN; CONFIG-PREREZ done except that |
| 24223f9 | P2-6 dao_gaia flatten | helpers in dao_gaia_common.py; iter4 not renamed |
| a4d608c | G-CONFIG test | old config.json with every removed/aliased key |
| 0fc53b7 | FLOW doc sync | G1 fallout: drop removed keys from FLOW facts/builder/PDF |

DEAD 5: qc_fwhm_limit, qc_elong_limit, psf_spatial_grid, psf_spatial_min_stars_per_cell,
gs11_comp_suspect_dilution.

ALIAS 3 survivors: comp_sparse_fallback_enabled, blind_index_fine_path, observer_code.
JSON names remain in _LEGACY_CONFIG_KEYS (not KNOWN_REMOVED). UI writer
ui_settings.py no longer writes the alias names.

## Count reconciliation (spec 268 vs actual 267)

Spec: 279 - 5 LEGACY - 5 DEAD - 1 bprp = 268; "aliases were outside the 279".

Actual after P2-4:

```
{"db_static": 8, "config_runtime": 267, "fits_dynamic": 6, "internal": 12}
```

PARAMS.md Entries: 293 (= 8+267+6+12).

Why 267 not 268: `comp_iterative_clip_enabled` was inside the 279 (owner=config_runtime).
The other two aliases were outside: `blind_index_path` (internal) and `aavso_observer_code`
(db_static). P2-3 therefore dropped config_runtime 269 -> 268 and also db_static 9 -> 8,
internal 13 -> 12. P2-4 then dropped the bprp flag 268 -> 267.

Dashboard lock in test_ui_params_dashboard.py matches 8/267/6/12.

## Deleted-branch diffstat (production paths)

6c95ac4..HEAD on the files that held flip-only branches / flatten:

```
src_py/dao_gaia_common.py           | 393 +++++++++++++++++-
src_py/dao_gaia_stage_01.py         | 798 ------------------------------------
src_py/dao_gaia_stage_01_iter2.py   | 737 ---------------------------------
src_py/dao_gaia_stage_01_iter3.py   | 660 -----------------------------
src_py/dao_gaia_stage_01_iter4.py   |  55 ++-
src_py/dao_gaia_stage_validation.py |   2 +-
src_py/epsf_psf_merge.py            |   8 +-
src_py/photometry_core.py           | 303 +++++++-------
src_py/photometry_report.py         |   5 +-
src_py/pipeline.py                  |  21 +-
src_py/psf_internal_lc.py           |   7 +-
src_py/psf_photometry.py            |  55 +--
src_py/ui_aperture_photometry.py    |   9 +-
src_py/ui_epsf_dashboard.py         |   3 +-
src_py/ui_settings.py               |   2 -
src_py/vyvar_platesolver.py         |  34 +-
16 files changed, 615 insertions(+), 2477 deletions(-)
```

Per-key remaining path (hard-wired default):

| key | remaining behaviour | deleted branch |
| --- | --- | --- |
| global_comp_pool_enabled | always build_global_comp_pool (photometry_core.py ~16254) | False/per-target pool skip |
| export_err_mode | always calibrated export | `model` skip of ERR-CALIB |
| err_background_mode | empirical when sigma_bkg_ap present | key test `== "howell"`; Howell math stays as howell_fallback |
| masterstar_accept_mode | odds-only (vyvar_platesolver.py ~2438) | fraction accept |
| psf_ac_policy | p4_none only (epsf_psf_merge / pipeline / psf_photometry) | chi2_lt5_legacy |

Provenance strings `export_err_mode=calibrated` and `psf_ac_policy=p4_none` remain as
product stamps, not AppConfig fields.

err_background_mode: `_photometric_error_with_bkg_mode` still accepts an ignored
`err_background_mode` argument; the howell key flip is gone. Howell citation
(`citations.py` howell1989) stays. ERR_BKG_SOURCE_HOWELL_FALLBACK stays.

## P2-6 flatten inventory

Moved into src_py/dao_gaia_common.py (history notes in section comments / docstrings):

From stage_01, 2026-08:
FRAMES, SHARPNESS_OPEN, MATCH_RADIUS_PX, SkyEstimate, _wcs_from_hdr, load_frame,
_star_mask_from_gaia, estimate_sky, run_dao, g2_empty_false_accept, crop_boxes.
Sandbox paths DRAFT / PS_DIR / LIGHTS_DIR.

From iter2, 2026-08:
EDGE_MARGIN_PX = 10.0 (the masterstar_gaia_census_edge_margin_px CLI consumer;
the config key stays ACTIVE -- pipeline/accounting still read it),
OVERLAY_G_MAX, G3_GAIA_MAX, GAIA_QUERY_G, _is_edge, _local_snr, _nn_gaia_px,
g3_spurious, decompose_holes_le13.
Already in common before flatten: _is_corner, _peak_at, _saturation_limit, asinh_rgb.

From iter3, 2026-08:
SOURCE_CROWDED_MISS, _gaia_on_chip_pm.

Deleted modules: dao_gaia_stage_01.py, dao_gaia_stage_01_iter2.py, dao_gaia_stage_01_iter3.py.
iter4 not renamed. Follow-up dao_gaia_stage.py only if Milan asks.

Validation: dao_gaia_stage_validation.py imports FRAMES from dao_gaia_common;
still `import dao_gaia_stage_01_iter4 as iter4`. Production (pipeline/night_run/app)
still imports no dao_gaia_stage_01* module.

Known leftover (not fixed): dao_gaia_stage_01_iter4.py:629 unpacks
`holes, summ = decompose_holes_le13(...)` but dao_gaia_common.decompose_holes_le13
returns only a DataFrame. CLI overlay path would TypeError if that branch is run.
Pre-existing vs the iter2 return shape; this task did not invent a second return.

## Test disposition

No test imported stage_01 / iter2 / iter3. No rewrite required for flatten besides
dao_gaia_stage_validation.py import of FRAMES.

| test | disposition |
| --- | --- |
| test_dao_gaia_xfer_01.py | kept; mocks `_import_iter4`; PASS after flatten |
| test_dao_gaia_calibration.py | kept; PASS after flatten |
| test_skipproc_qc_allowlist.py per-key INFO | kept / extended for LEGACY 5 + DEAD 5 |
| test_legacy_alias_keys_map_silently | kept; aliases still map |
| test_g_config_all_removed_and_aliased_keys_load_silently | added (a4d608c); 11 removed + 3 aliased; no WARN; no fields |
| test_g7_f003_phase01_use_bprp_primary.py | rewritten as known-removed load test |
| test_g7_f003b_report_bprp_primary.py | display always BP-RP even if old JSON has flag False |
| test_g7_f003c_report_cfg_snapshot.py | snapshot probe switched off the removed key; `_use_bprp_primary` always True |
| test_ui_params_dashboard.py distribution | locked 8/267/6/12 |
| test_epsf_dashboard_pct.py | still requires substring `psf_ac_policy` in ui_epsf_dashboard.py (product stamp) |
| test_epsf_psf_merge.py | p4_none only |
| test_err_background_empirical.py | empirical always; Howell fallback still tested as data condition |
| test_masterstar_odds_acceptance.py | odds-only |

## Gates

| gate | status | detail |
| --- | --- | --- |
| G1 --fast --clean before | PASS at 6c95ac4 | 1603 passed, 32 skipped. clean-tree PASS. db-quick-check WARN waived. Log: g1_before.txt |
| G1 --fast --clean after | PASS at 0fc53b7 | 1611 passed, 32 skipped. First attempt FAIL test_flow_doc_config_facts (err_background_mode gone from config.json); FLOW sync 0fc53b7 then PASS. Log: g1_after.txt |
| G2 --full aperture | PASS | era04_aperture d55fcc9d n=53 / ext cc8b532e n=157. Pipeline 1660s. Log: g2_full.txt |
| G-EPSF --full-epsf | PASS | epsf01 c743b8ba n=53; G3 residual BO 12.505 / FW 4.629, n_full=134 both. ePSF stage 17247s (n_stars=63, 53 PSF LCs). Log: g_epsf.txt |
| G4 live 516 | PASS after G2 and after G-EPSF | csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d |
| G-CONFIG | PASS unit | test_g_config_all_removed_and_aliased_keys_load_silently (also in G1 --clean) |

G4 path: Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/
(masterstars_full_match.csv, MASTERSTAR.fits, masterstar_epsf.fits). Not written.

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV last_verified commit stamp updated to 0fc53b7 by --full (same as 01C/01D p1 ritual).

## STOPs

None that blocked the task.

Count: spec 268 vs actual 267 reconciled above (comp_iterative_clip_enabled was
config_runtime, not outside the 279).

iter4 overlay unpack vs single DataFrame return: noted, not a spec contradiction;
left as-is.

No D1 row beyond Milan's approved list was removed. D1B-UNITS-01 three px/unit pairs
were not flipped.
