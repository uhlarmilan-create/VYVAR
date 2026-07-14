# VYVAR -- Production sigma_sys floor specification

Date: 2026-07-13  
Status: **ACCEPTED** (ANCHOR-CHAIN-ACCEPT 2026-07-13; exact c4 validation PASS)

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
| Newton r | draft_426 r_60_4 sparse trust (COMP-POOL-R) | 4 |

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

**Cross-checks (2026-07-14 PZQ report, tmp/pzq_sigma_r/):**
(a) Wide Carl-Zeiss median sigma_r ~ 5.5 mmag [4.7, 6.5] is consistent with the ~4.5 mmag
unexplained rig constant from SIGMA-A4 (two independent methods, same quantity).
(b) Newton g median sigma_r ~ 18.8 mmag ~ fitted production floor 18.0 mmag: the Newton
white floor absorbs predominantly **correlated** noise; per-point err bars are honest, but
binned/averaged Newton quantities remain underestimated even with the floor. PDF/report
wording must state this explicitly when quoting Newton binned scatter or PZQ on Newton rigs.

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

## Anchor chain (draft_424, wide eq1)

For acceptance the baseline chain must be explicit; comparing against a pre-bingain anchor
confounds approved err-model changes.

| Snapshot | Purpose | git_hash | core_sha (prefix) | extended_sha (prefix) |
|----------|---------|----------|-------------------|-----------------------|
| ``draft_000424_snapshot_20260708_full`` | Historical anchor (pre-bingain, pre-unit-fix) | ``750c856`` | ``92939fab`` | ``76642318`` |
| ``draft_000424_snapshot_intermediate_b5364e6_20260713`` | Acceptance baseline: bingain + unit fix + masterstar exclusion, **no c4/floor** | ``b5364e6`` | ``373e8235`` | ``0243f719`` |
| ``draft_000424_snapshot_sigma_floor_20260713`` | Accepted anchor: intermediate + **c4** (eq1 floor=0) + Newton floor (eq4) | ``8fb21b3`` | ``bf3743a1`` | ``dec5c637`` |

Exact validation (intermediate -> accepted): per-epoch c4 predictor matched all 23542 epochs
within abs diff <= 2e-6 (max 9.97e-7; median 2.93e-7). See ``tmp/anchor_chain/c2_exact_c4_validation.json``.

---

## Fitted defaults (2026-07-13)

Committed in ``config.json`` (CI in fit JSON):

| equipment_id | Rig | sigma_sys_mmag | Status |
|--------------|-----|----------------|--------|
| 1 | QHY294MM wide | -- | **un-floored** (bootstrap unstable; point fit below SIGMA-A3 CI) |
| 4 | C5A-150M Newton | 18.0 [15.6, 20.2] | committed 0.018 mag |

### Wide rig: point fit 4.8 mmag vs SIGMA-A3 band 6.5 mmag [5.5, 7.5]

These are **not contradictory**. SIGMA-A3 measured a pooled floor-like excess in the
**legacy** err model (Howell photon only; ensemble SEM added in the wrong domain). Production
now includes F-BINGAIN-1 empirical background in ``err_photon_bkg`` and c4-corrected ensemble
SEM in the correct relative-flux domain. That combination absorbs part of what SIGMA-A3
attributed to a separate white ``sigma_sys`` term. The fit script therefore finds a lower
residual floor (~4.8 mmag) with an unstable bootstrap; per the instability rule the wide rig
correctly remains **un-floored** (config 0.0). Expect wide per-point ``err`` to rise vs a
pre-bingain anchor when comparing snapshots; that shift is **not** evidence that the Newton
18 mmag floor leaked onto equipment_id=1 (see ``CURSOR_RESULT_anchor_err_verify.md``).

---

## FUTURE -- SPARSE-TRUST

**IMPLEMENTED (2026-07-14).** Howell, Warnock & Mitchell 1988 triangulation + CI trust bands;
external K sourcing (Amendment 1). Spec: ``docs/VYVAR_SPARSE_TRUST_SPEC.md``. Arc CLOSED;
see ROADMAP SPARSE-TRUST row and ``CURSOR_RESULT_arc_close.md``.

**FUTURE design note (rig-aware X2_RED):** candidate X2_RED_eff = max((0.02)^2, k * sigma_r^2)
once multi-night sigma_r on eq4 is available. Not implemented.

---

## Citations

| Key | Reference |
|-----|-----------|
| ``merlinehowell1995`` | Merline & Howell 1995 ExA 6, 163 |
| ``pzq2006`` | Pont, Zucker & Queloz 2006 MNRAS 373, 231 |
| ``everetthowell2001`` | Everett & Howell 2001 PASP 113, 1428 |
