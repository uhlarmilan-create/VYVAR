# VYVAR -- Production sigma_sys floor specification

Date: 2026-07-13  
Status: **IMPLEMENTED** (c4 + per-rig floor wiring; re-anchor pending Milan review)

Cross-ref: `docs/VYVAR_SIGMA_BUDGET_SPEC.md` (Broeg IVW / scintillation arc still parked).

---

## Error model (single domain)

All terms in **relative-flux** domain (LC ``err`` contract). Magnitude conversions use
``MAG_ERR_SCALE = 2.5/ln(10)`` from ``mag_constants.py`` only.

    err_total^2 = err_photon_bkg^2 + sem_ens_rel^2 + sigma_sys_rel(rig)^2

| Term | Source | Notes |
|------|--------|-------|
| ``err_photon_bkg`` | ``_photometric_error`` + empirical ``sigma_bkg_ap`` | Unchanged |
| ``sem_ens_rel`` | Honeycutt ensemble SEM with c4 correction | ``sem_mag = std(ddof=1)/c4(n)/sqrt(n)`` |
| ``sigma_sys_rel`` | Config ``sigma_sys_mag[equipment_id]`` | Converted mag -> rel once at assembly |

Implementation: ``sigma_floor_core.combine_production_err_rel``;
``photometry_core._combine_err_with_ensemble_scatter_keyed`` (after SEM join).

---

## c4 small-sample correction

Unbiased scale: ``c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)``.

Production ensemble SEM (``ensemble_normalize``) uses ``ensemble_sem_mag_from_residuals``.
Unit tests: ``tests/test_sigma_floor.py`` vs literature values to 1e-4.

At n=3..8 comps, uncorrected SEM is biased low by 3.5--11.4% in expectation.

---

## Per-rig floor fit (Part A)

Script: ``scripts/fit_sigma_floor.py``. Artifacts: ``tmp/sigma_floor/``.

**Cohorts:** constant check stars only; exclude variables, saturated (fill_p95 >= 0.85),
SS Cam sparse-path, stars with < 15 epochs.

| Rig | Draft / setup | equipment_id |
|-----|---------------|--------------|
| Wide Carl-Zeiss | draft_424 / NoFilter_60_2 | 1 |
| Newton | draft_426 g_60_4 + i_70_4 pooled | 4 |
| Newton r | pending COMP-POOL-R / SPARSE-TRUST | 4 |

**Method:** 1-D bisection on pooled reduced chi2/dof - 1 for ``sigma_sys_mag``;
bootstrap >= 500 resamples (16% CI). Anti-circularity: random half A fit / half B validate
and swap (seed 424426).

**Instability rule:** if CI spans > 2x central value, rig left un-floored (config 0.0).

**Wide consistency gate:** fitted wide-rig floor must fall in SIGMA-A3 prior band
6.5 mmag [5.5, 7.5]. Failure -> STOP report; rig un-floored.

---

## PZQ red-noise diagnostic (report-only)

Per star: binned RMS ``sigma_N`` for N in {2, 4, 8} vs white ``sigma_1/sqrt(N)``.
Fit ``sigma_N^2 = sigma_w^2/N + sigma_r^2`` (Pont, Zucker & Queloz 2006; key ``pzq2006``).

**Not wired into per-point err.** Figures: ``tmp/sigma_floor/pzq_*.png``.

Correlated noise does not average as ``1/sqrt(N)``; per-point bars use white floor only.

---

## Production wiring

| Item | Location |
|------|----------|
| c4 SEM | ``photometry_core.ensemble_normalize`` |
| Floor add | ``_combine_err_with_ensemble_scatter_keyed`` |
| Config map | ``config.sigma_sys_mag`` (equipment_id str -> mag) |
| LC column | ``sigma_sys_mag`` (constant per LC) |
| Provenance | ``pipeline_meta.sigma_floor`` |
| PDF note | ``photometry_report.py`` Methods block |

Fail-safe: unknown equipment_id -> floor 0.0, one-time INFO log.

---

## Fitted defaults (2026-07-13)

Committed in ``config.json`` (CI in fit JSON):

| equipment_id | Rig | sigma_sys_mmag | Status |
|--------------|-----|----------------|--------|
| 1 | QHY294MM wide | -- | **un-floored** (bootstrap unstable; point fit below SIGMA-A3 CI) |
| 4 | C5A-150M Newton | 18.0 [15.6, 20.2] | committed 0.018 mag |

---

## FUTURE -- SPARSE-TRUST

Howell, Warnock & Mitchell 1988 two-comp night-difference variance with photon correction
(reserved; not implemented here). See ROADMAP SPARSE-TRUST row.

---

## Citations

| Key | Reference |
|-----|-----------|
| ``merlinehowell1995`` | Merline & Howell 1995 ExA 6, 163 |
| ``pzq2006`` | Pont, Zucker & Queloz 2006 MNRAS 373, 231 |
| ``everetthowell2001`` | Everett & Howell 2001 PASP 113, 1428 |
