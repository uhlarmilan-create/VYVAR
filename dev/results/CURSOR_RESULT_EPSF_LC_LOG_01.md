CURSOR RESULT - 2026-08-24T11:30:00Z (EPSF-LC-LOG-01)

What I did
Shipped an internal/diagnostic per-target PSF light-curve product and a hard
AAVSO/VarAstro submit guard (INV-PSF-SUBMIT-01). PSF LCs are additive; aperture
LCs and science exports are not rewritten. Not pushed; Milan authorizes.

Premise (0.1): compared the previous absence of truthful `lightcurve_*_psf.csv`
files (globs in photometry_core/photometry_report/export_reports were dead) with
a new writer that emits those files from post-F6 merged catalogs using the same
pinned ensemble as the aperture branch. The two products differ: aperture LCs
remain the science export; PSF LCs are relative photometry only, header-marked
NOT FOR SUBMISSION.

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G1 git pull / tip | PASS | `b1af049` SESSION-CLOSE-20260823 (already up to date) |
| G2 `--fast` OVERALL | PASS | start 1512 passed; end T4 1518 passed, 32 skipped; db-quick-check WARN via committed waiver |
| G3 anchor era | PASS | frozen `draft_000516_snapshot_era03_20260820` core `9902d918` n=121, extended `472bc9e4` n=179 |

## Implementation notes

- New writer `src_py/psf_internal_lc.py`. Epoch backbone = aperture LC
  `source_file` rows (science lights). PSF inst mag = `-2.5 log10(psf_flux)`
  only when `psf_fit_ok` and flux > 0. Ensemble ZP = `ensemble_normalize`
  flux-sum on the pinned set (INV-PIN / no new selection). Error =
  `combine_production_err_mag(psf_flux_err/psf_flux, ensemble_SEM)`.
- Failed-PSF epochs kept: `psf_fit_ok=False`, NaN PSF mag columns, aperture
  `delta_mag`/`err` still filled. `psf_ap_ratio = psf_flux / aperture flux`.
- Trigger: end of `run_epsf_psf_merge_job`; UI button on ePSF dashboard;
  headless `python src_py/psf_internal_lc.py --platesolve-dir ... --frames-root ...`.
- Part B: `export_lightcurve_reports` raises `InvariantViolation("INV-PSF-SUBMIT-01", ...)`
  for `lc_method` in {psf, adaptive}. `export_all_method_lightcurve_reports`
  iterates aperture only so internal PSF files never enter the writers.
- Phase 2A no longer writes submission-shaped `lightcurve_*_psf.csv`
  (that path would collide with the diagnostic product).

## Test evidence

| Test | Status | Detail |
|------|--------|--------|
| T1 draft 516 BO CVn | PASS | `lightcurve_1498613634033133184_psf.csv`; 134 epochs; 133 `psf_fit_ok`; 1 NaN `psf_delta_mag` row; all REQUIRED_HEADER_MARKERS present |
| T2 byte-identity | PASS | aperture LC + BO CVn AAVSO + VarAstro SHA unchanged by the writer; PSF file hash is new (positive control) |
| T3 INV-PSF-SUBMIT-01 | PASS | export_method psf and adaptive raise with `INV-PSF-SUBMIT-01` in the message; aperture still writes; batch exporter skips PSF files even if present |
| T4 `--fast` (end) | PASS | 1518 passed, 32 skipped; OVERALL PASS; db-quick-check WARN via committed waiver |

Synthetic writer test: 3 epochs, middle epoch PSF-fail preserved as NaN.

`pytest dev/tests/test_psf_internal_lc.py`: 6 passed in 7.8 s.
Registry parity: `test_registry_lists_all_wired_ids` + `test_wired_ids_have_call_sites` PASS.

All-target write on draft 516: 60/60 aperture LCs received a PSF companion
(`n_written=60`, `n_skipped=0`).

## Sample header (BO CVn)

```
# INTERNAL DIAGNOSTIC PRODUCT - NOT FOR AAVSO/VARASTRO SUBMISSION
# epsf_model_file=masterstar_epsf.fits
# epsf_model_sha256=172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20
# epsf_n_stars=67
# epsf_build_timestamp=2026-08-22T19:11:31Z
# epsf_oversampling=2
# epsf_smoothing_kernel=quadratic
# epsf_cutout_size=17
# psf_weight_mode=full_ccd
# psf_err_mode=sandwich_full_ccd
# gain_authority=g_pt=0.637067 source=g_pt
# ensemble_n_comp=4
# ensemble_pinned_ids=1497771992240531712,1499200223486564608,1497974027502858240,1497368849430107904
# ensemble_source=pinned
# git_hash=b1af0493dde53590850541376dceb233e5da0f46
# git_dirty=True
# PSF absolute scale untrusted pending EPSF-SHAPE-01;
# relative photometry only
# product=internal_psf_diagnostic
```

`n_group` is blank: F6 merge sidecars do not persist `psf_group_n` (honest empty).

## Files changed

- `src_py/psf_internal_lc.py` (new)
- `src_py/export_reports.py` (INV-PSF-SUBMIT-01 raise; aperture-only batch export)
- `src_py/epsf_psf_merge.py` (post-job writer hook)
- `src_py/photometry_core.py` (Phase 2A no longer writes PSF LC variants)
- `src_py/ui_epsf_dashboard.py` (Write internal PSF light curves button)
- `src_py/invariants_runtime.py` (`INV-PSF-SUBMIT-01` in `WIRED_INV_IDS`)
- `docs/VYVAR_INVARIANTS.md` (registry row + definition/rationale/trigger/test block)
- `dev/tests/test_psf_internal_lc.py` (new)
- Additive draft 516 products: `.../photometry/lightcurves/lightcurve_*_psf.csv` (60 files)

`src_py/psf_photometry.py` gained an optional `smoothing_kernel=` on the sandbox
build path for EPSF-SHAPE-01-M; default behavior unchanged.

Not pushed.
