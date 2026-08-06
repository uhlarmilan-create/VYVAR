CURSOR RESULT - 2026-08-05 (Task D1)

What I did
Implemented inactive unit-normalisation plumbing for group (b): `src_py/unit_resolver.py`, 10 companion `None` fields in `AppConfig`, resolver wiring at science call sites, registry entries, tests, and JOURNAL plate-scale correction. No legacy defaults changed.

## 1. Premise check (plate scales verified)

| rig | arcsec/px | source |
|-----|-----------|--------|
| Carl-Zeiss 200 mm + QHY294MM (Jirny), binned | **9.77** | `docs/VYVAR_JOURNAL.md:3233-3235` (confirmed; supersedes stale 1.3 placeholder) |
| Newton 300/1200 + C3-26000 bin1 | **0.65** | `docs/VYVAR_JOURNAL.md:3362-3363` |
| Newton 300/1200 + C3-26000 bin2 | **1.30** | `docs/VYVAR_JOURNAL.md:3362-3363` |
| chi/h Persei (Dablice) | **1.302** | `docs/VYVAR_JOURNAL.md:1908` |

`phase01_comparison_isolation_radius_px=25` ? ~244 arcsec (wide) vs ~32.5 arcsec (Newton bin2) vs the arcsec twin `phase01_comparison_min_dist_arcsec=60`. Current default is wrong on both rigs, in opposite directions. D1b must choose normalised defaults; this task only adds inactive companions.

## 2. Fields, resolver, call sites

**Module:** `src_py/unit_resolver.py` - `resolve_px_from_arcsec`, `resolve_px_from_fwhm_factor`, `resolve_hfr_limit_px`, `resolve_max_dist_fallback_deg`, cfg wrappers, one-shot `[UNIT-RESOLVE]` INFO logging.

| legacy key | companion field | wired at |
|------------|-----------------|----------|
| `blind_verify_match_tol_px` | `blind_verify_match_tol_arcsec` | `vyvar_platesolver.py` (after `_known_ps` resolved) |
| `cog_ladder_step_px` | `cog_ladder_step_fwhm` | `photometry_core.py:12557` (per-frame COG, uses measured `fw`) |
| `hrd_color_bg_box_px` | `hrd_color_bg_box_arcsec` | `hrd_colorfield.py:745` (WCS plate scale from MASTERSTAR FITS) |
| `masterstar_centre_rms_max_px` | `masterstar_centre_rms_max_arcsec` | `vyvar_platesolver.py:3578` (WCS median scale) |
| `masterstar_sibling_rms_max_px` | `masterstar_sibling_rms_max_arcsec` | `vyvar_platesolver.py:5795` (`expected_plate_scale_arcsec_per_px`) |
| `phase01_chip_interior_margin_px` | `phase01_chip_interior_margin_arcsec` | `photometry_core.py:15822`, `5625` (edge filter via header WCS) |
| `phase01_comparison_isolation_radius_px` | `phase01_comparison_isolation_radius_arcsec` | `photometry_core.py:15953` |
| `phase01_comparison_max_dist_deg` | `phase01_comparison_max_dist_fov_frac` | `photometry_core.py:15912` (`resolve_max_dist_fallback_deg`) |
| `qc_max_hfr` | `qc_max_hfr_fwhm_ratio` | `pipeline.py:15194` (DAO FWHM guess from QC crop) |
| `sips_dao_fwhm_px` | `sips_dao_fwhm_fwhm_factor` | `photometry_core.py:15854` |

Registry: 10 new keys in `dev/validation/params_registry.json` (`scope=universal`, `scope_group=n/a`). Total registry keys: **287**.

## 3. Conversion table (current defaults ? target unit)

Assumed typical FWHM **4.0 px** for FWHM-multiple rows (Newton-like sampling; wide-field FWHM varies).

| key | px default | wide 9.77?/px | Newton bin1 0.65 | Newton bin2 1.30 | chi/h 1.302 | spread |
|-----|------------|---------------|------------------|------------------|-------------|--------|
| `blind_verify_match_tol_px` | 2.5 px | 24.4 arcsec | 1.63 arcsec | 3.25 arcsec | 3.26 arcsec | 15x |
| `cog_ladder_step_px` | 0.5 px | 0.125x FWHM | 0.125x FWHM | 0.125x FWHM | 0.125x FWHM | unitless |
| `hrd_color_bg_box_px` | 96 px | 938 arcsec | 62.4 arcsec | 124.8 arcsec | 125 arcsec | 15x |
| `masterstar_centre_rms_max_px` | 1.20 px | 11.7 arcsec | 0.78 arcsec | 1.56 arcsec | 1.56 arcsec | 15x |
| `masterstar_sibling_rms_max_px` | 2.0 px | 19.5 arcsec | 1.30 arcsec | 2.60 arcsec | 2.60 arcsec | 15x |
| `phase01_chip_interior_margin_px` | 50 px | 488 arcsec | 32.5 arcsec | 65.0 arcsec | 65.1 arcsec | 15x |
| `phase01_comparison_isolation_radius_px` | 25 px | 244 arcsec | 16.3 arcsec | 32.5 arcsec | 32.6 arcsec | 15x |
| `phase01_comparison_max_dist_deg` | 1.5 deg | 0.39x half-diag FOV+ | rig-dependent | rig-dependent | rig-dependent | FOV-driven |
| `qc_max_hfr` | 5.0 px HFR cap | 1.25x FWHM++ | 1.25x FWHM | 1.25x FWHM | 1.25x FWHM | unitless |
| `sips_dao_fwhm_px` | 2.5 px | 0.625x FWHM | 0.625x FWHM | 0.625x FWHM | 0.625x FWHM | unitless |

+ At 9576x6380 px and 9.77?/px, half-diagonal ? 3.8 deg; `fov_fraction=0.75` ? ~2.9 deg vs legacy fallback 1.5 deg.  
++ Using FWHM=4 px; HFR cap as ratio is the intended normalisation.

**Proposed D1b defaults (not implemented):**

| key | proposal | justification |
|-----|----------|---------------|
| `blind_verify_match_tol_arcsec` | 3.0 arcsec | ~2.3 px at Newton bin2; tolerant blind match without wide-field 244 arcsec blow-up |
| `phase01_comparison_isolation_radius_arcsec` | 60 arcsec | Match arcsec twin `min_dist_arcsec`; blending/isolation parity |
| `masterstar_centre_rms_max_arcsec` | 1.5 arcsec | ~1.2 px at Newton bin2; centre RMS gate in sky units |
| `masterstar_sibling_rms_max_arcsec` | 2.5 arcsec | Slightly looser than centre gate |
| `phase01_chip_interior_margin_arcsec` | 65 arcsec | ~50 px at chi/h scale; ROADMAP PHASE0-BORDER-MARGIN-GEOMETRY interim |
| `hrd_color_bg_box_arcsec` | 120 arcsec | ~92 px at Newton bin2; local background box ~15-20 arcsec target FWHM scale |
| `cog_ladder_step_fwhm` | 0.125 | 0.5 px / 4 px FWHM at typical sampling |
| `sips_dao_fwhm_fwhm_factor` | 0.625 | 2.5 px / 4 px FWHM initial DAO guess |
| `qc_max_hfr_fwhm_ratio` | 1.25 | 5 px / 4 px FWHM QC cap |
| `phase01_comparison_max_dist_fov_frac` | 0.39 | Recovers ~1.5 deg fallback on wide QHY FOV (half-diag ~3.8 deg) |

Tuning appears Newton/chi-h oriented for px-native defaults; wide-field leaks (documented JOURNAL :3233-3235).

## 4. `crowding_tighten_min_fwhm_px` verdict

**Legitimately px-native** - gates crowding tightening on undersampled images (pixel-domain sampling property). Reclassified to group **(a)** in `dev/tools/classify_params_scope.py` (`scope_group=a`, high confidence). **No arcsec companion added.**

## 5. P1 golden result (verbatim)

```
VYVAR_INVARIANTS_P1=1 pytest dev/tests/test_invariants_p1_golden.py::test_headless_chain_sha -q

FAILED test_headless_chain_sha
AssertionError: core SHA mismatch: aa72e97979a74d5b8297c6bc3624bee668d8bd5f28624de0a708149e286c2636 != e7976de18e4197e85e0120dcadf6bdae5ac0be73238be92f83c7cd87fa0fedee
1 failed in ~397s
```

**Bisect:** With **all** `src_py/` reverted to HEAD `7620077` (no D1 code), the same core SHA `aa72e979...` is produced. Ledger `VL-P1-GOLD` locked at commit `a9d7eb0` (2026-07-28); HEAD is `7620077` (2026-08-05 scope triage). **Ledger staleness vs current HEAD**, not a D1 regression when companions are `None`. D1 wiring re-checked: resolver paths return legacy values verbatim when companions are unset.

Other P1 tests in file: 4/5 passed on last full run (SHA test only failure).

## 6. Keys left on legacy path notes

| key | note |
|-----|------|
| `vyvar_blind_series.py` | Still scales `blind_verify_match_tol_px` into a tier-local cfg before platesolver; resolver runs at platesolver consumption. No arcsec companion on tier cfg. |
| `ui_dao_stars.py` / `ui_settings.py` | UI reads/writes legacy px keys only (companions hidden in registry). |

## Tests / suite

| part | wall time | result |
|------|-----------|--------|
| `test_unit_resolver.py` | 1.1 s | 10 passed |
| Full suite | 341.9 s | **1253 passed, 26 skipped** (baseline C'': 1243+26; +10 from new tests) |

## Files changed

- `src_py/unit_resolver.py` (new)
- `src_py/config.py`, `photometry_core.py`, `pipeline.py`, `vyvar_platesolver.py`, `hrd_colorfield.py`
- `dev/tests/test_unit_resolver.py` (new), `dev/tests/test_ui_params_dashboard.py`
- `dev/tools/classify_params_scope.py`, `dev/tools/add_d1_registry_fields.py` (new)
- `dev/validation/params_registry.json`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_PARAMS.md`
- `dev/results/CURSOR_RESULT_d1_unit_normalisation.md`
