CURSOR RESULT - 2026-08-07 (ZONE-SIMPLIFY, derive don't configure)

What I did
Deleted four ZONE-FIX config keys and the legacy zone path; zone boundary now derives
from dao_detection_n_equiv; saturation from frame/profile data; UI icon crash fixed;
parameter-reduction inventory for next session. Four commits pushed.

## 1. Config key count and derivations

| Stage | Registry / AppConfig fields |
|-------|----------------------------:|
| Before (HEAD `6db4b67`, incl. 4 ZONE-FIX keys) | 292 |
| After ZONE-SIMPLIFY | 288 |
| Net delta | **-4** |

Deleted keys (grep clean in `src_py/`, registry, `docs/VYVAR_PARAMS.md`; removed from
local `config.json`):

| Removed key | Replacement (rule 1 / 2) |
|-------------|--------------------------|
| `masterstar_zone_mode` | Deleted -- peak significance only |
| `masterstar_zone_sigma_linear` | `T1 = dao_detection_n_equiv` at run time |
| `masterstar_zone_sigma_noisy1` | `T2 = T1 - 1.0` (`_MASTERSTAR_ZONE_SIGMA_STEP`, rule 2) |
| `masterstar_zone_sigma_noisy2` | `T3 = T1 - 2.0` (two whole-sigma steps, rule 2) |

Task brief cited 291/287 from an earlier snapshot; at base `6db4b67` the registry held
292 entries. Net -4 zone keys is confirmed by grep.

## 2. Zone boundary: T1 = dao_detection_n_equiv (3.78 default)

Implementation: `_masterstar_zone_sigma_thresholds()` in `src_py/pipeline.py`; threaded
via `det_meta["dao_detection_n_equiv"]` from `_dao_detection_threshold_adu` and
`AppConfig.dao_detection_n_equiv` fallback at the MASTERSTAR write call site.

Sweep comparison (offline reclassify on stored CSVs; T2/T3 = T1-1, T1-2):

| Draft | T1=3.5 usable | **T1=3.78 (derived)** | T1=4.0 usable |
|-------|-------------:|----------------------:|--------------:|
| draft_452 fixture | 1831 | **1741** | 1690 |
| draft_435 | 719 | **678** | 644 |
| draft_500 | 374 | **341** | 335 |
| draft_502 | 1123 | **1083** | 1036 |

3.78 lands between the 3.5 and 4.0 rows on all four drafts as expected. No standalone
zone threshold key and no hardcoded sigma literal beyond the 1.0 whole-sigma step.
Matched/DAO_ONLY separation remains monotonic (same property as ZONE-FIX sweep).

Full distributions (with saturation fix): `dev/results/zone_simplify_measurement.json`.

## 3. Section 2b saturation measurements (draft_502)

Reference frame: `Archive/Drafts/draft_000502/non_calibrated/lights/V_60_2/TOI-1131.01.b_2025-04-22_23-05-09_V.fits`.

| Measurement | Value |
|-------------|------:|
| Frame max ADU | 98232.375 |
| Pixels at max (+/-0.5 ADU) | 1 |
| Hard truncation / pile-up | **No** |
| Empirical clip ADU | null |
| Masterstars zone=saturated (before) | 10 / 1668 |
| Targets zone_flag=saturated | 2 / 22 |

Q central/8-neighbour ratio: flagged saturated stars Q ~ 0.97-1.07; similarly-bright
unflagged Q ~ 0.98-1.23. Peaked profiles throughout -- not flat-topped.

**Fix:** `_detect_empirical_clip_level_adu` + `_resolve_peak_saturation_limit_adu` --
clip from histogram when pile-up exists; skip raw `EQUIPMENTS.SATURATE_ADU` peak test
when frame max exceeds camera ceiling or sky median indicates external pedestal.
Profile flatness path in DAO detection unchanged. No new config key.

## 4. TIC 198213332 before/after

| Field | Before | After reclassify |
|-------|--------|------------------|
| catalog_id | 1625373404725030528 | same |
| peak_max_adu | 91320.625 | unchanged in CSV |
| zone | saturated | **linear** |
| is_usable | False | **True** |
| active_targets skip | skip_photometry=True (zone_flag) | clears on re-run |

Not genuinely saturated.

## 5. P1 SHA and before/after counts

P1 headless (`dev/tools/zone_fix_p1_run_once.py`):

| Metric | Value |
|--------|-------|
| core SHA | `9b39d899be0853311d7acf0ced956f4ff9226871df23aeebb5f00c916fc7b479` |
| core n | 81 |

Unchanged from pre-ZONE-SIMPLIFY: P1 photometry uses frozen `masterstars_full_match.csv`
and does not re-run MASTERSTAR zone annotation. Science change applies on next
MASTERSTAR build.

Before/after zone + saturation (offline reclassify, T1=dao_detection_n_equiv=3.78):

| Draft | is_usable before -> after | saturated before -> after |
|-------|---------------------------|---------------------------|
| draft_452 | 1799 -> 1741 | 28 -> 31 |
| draft_435 | 1799 -> 678 | 28 -> 31 |
| draft_500 | 1713 -> 341 | 23 -> 26 |
| draft_502 | 4 -> **1083** | 10 -> **0** |

draft_502 comparison pool: global usable candidates 4 -> 1083.

## 6. Section 4 reduction inventory (read-only backlog)

Current registry: **288** keys. `scope_group` breakdown:

| scope_group | count | meaning |
|-------------|------:|---------|
| n/a | 255 | universal / not rig-scoped |
| a | 19 | site-scoped |
| b | 10 | rig-scoped (pixel keys -- D1b deletion candidates) |
| c | 3 | session-scoped |
| ? | 1 | unclassified |

**Group-(b) keys** (derive from FWHM / plate scale / arcsec, D1b territory):

| Key | Derive from |
|-----|-------------|
| `blind_verify_match_tol_px` | arcsec tolerance x plate scale (companion `blind_verify_match_tol_arcsec` exists) |
| `cog_ladder_step_px` | FWHM factor (companion `cog_ladder_step_fwhm_factor` exists) |
| `hrd_color_bg_box_px` | FWHM or arcsec box (`hrd_color_bg_box_arcsec` exists) |
| `masterstar_centre_rms_max_px` | arcsec gate (`masterstar_centre_rms_max_arcsec` exists) |
| `masterstar_sibling_rms_max_px` | arcsec gate (`masterstar_sibling_rms_max_arcsec` exists) |
| `phase01_chip_interior_margin_px` | FWHM factor (`phase01_chip_interior_margin_fwhm_factor` exists) |
| `phase01_comparison_isolation_radius_px` | FWHM factor (`phase01_comparison_isolation_radius_fwhm_factor` exists) |
| `phase01_comparison_max_dist_deg` | field radius from WCS footprint + target density |
| `qc_max_hfr` | FWHM-relative HFR limit (`resolve_hfr_limit_px`) |
| `sips_dao_fwhm_px` | measured FWHM factor (`sips_dao_fwhm_fwhm_factor` exists) |

**Derivable candidates from this task (deleted or flagged, not removed beyond zone keys):**

- Four `masterstar_zone_*` keys -- **deleted**; T1 from `dao_detection_n_equiv`.
- Saturation peak vs raw ADC -- **fixed**; derive clip from frame, profile flatness in DAO path.
- `masterstar_dao_threshold_sigma` vs `dao_detection_n_equiv`: detection threshold is
  `N_equiv * rms_conv`; `masterstar_dao_threshold_sigma` may be legacy overlap with
  `dao_detection_n_equiv` -- candidate for audit (do not delete in this task).

**Dead/redundant candidates (report only):**

- D1 already added arcsec/FWHM companion keys (`*_arcsec`, `*_fwhm_factor`) alongside
  group-(b) px keys -- px keys become redundant once companions are wired as primary.

## 7. Unchanged by design

- `ui_calibration.py:72` `:material/visibility:` -- supported material syntax.
- `app.py:2752` `page_icon="*"` -- different API.
- `saturate_limit_fraction` (0.85) -- physical margin below clip level, kept.
- `EQUIPMENTS.SATURATE_ADU` -- retained for raw-scale sanity bound only.

## 8. Tests

- `dev/tests/test_masterstar_zone_classifier.py`: 5 passed (derived thresholds, pedestal, sat skip).
- `dev/tests/test_streamlit_icon_args.py`: passed.
- `test_generated_params_md_is_fresh`: passed.
- Full suite (`-k "not test_invariants_p1"`): **1267 passed**, 20 skipped.
- Expected P1 ledger exclusions: `test_invariants_p1_golden.py` module.
- Pre-existing: `test_flow_doc_config_facts` -- local `config.json`
  `vsx_out_of_scope_types: ["ROT"]` vs `flow_doc_facts.py` expects `[]`.

No new config key. No new `WIRED_INV_IDS` entry.

## Files changed

- `src_py/pipeline.py`
- `src_py/config.py`
- `src_py/ui_calibration.py`
- `src_py/params_registry.py`
- `dev/validation/params_registry.json`
- `docs/VYVAR_PARAMS.md`
- `config.json`
- `dev/tests/test_masterstar_zone_classifier.py`
- `dev/tests/test_streamlit_icon_args.py`
- `dev/tests/test_ui_params_dashboard.py`
- `dev/tools/zone_simplify_measure.py`
- `dev/tools/zone_fix_measurement.py`
- `dev/results/zone_simplify_measurement.json`
- `dev/results/CURSOR_RESULT_zone_simplify.md`
