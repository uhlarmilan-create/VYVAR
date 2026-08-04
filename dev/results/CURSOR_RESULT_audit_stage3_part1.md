CURSOR RESULT - 2026-07-30T18:00:00Z

What I did
Implemented additive LC err-budget export (no change to combined `err` value). Measured corrected P-02 (check-star reduced chi-squared) and D1-2 (linearity residual vs peak ADU) on anchor snapshot `draft_000435_snapshot_skysurface_20260716`, setup `NoFilter_60_2`.

## Output / findings

### 1.1 - LC err-budget export (prerequisite, no numeric science change)

**Implementation:** export existing terms from `_combine_err_with_ensemble_scatter_keyed` path:
- `err_photon` - photon/SNR base (relative flux)
- `err_sem_rel` - ensemble SEM term (relative flux)
- `err_sigma_sys_rel` - `sigma_sys_mag` floor (relative flux)

| Location | Change |
|----------|--------|
| `src_py/photometry_core.py` | `_err_budget_components_keyed()`; wire in target LC save path; `save_lightcurve_csv()` columns |
| `dev/tests/photometry_sha.py` | Added columns to `PHOTOMETRY_QC_COLS_LC` (excluded from science SHA) |

Combined `err` unchanged (additive columns only).

---

### 1.1 - P-02 corrected metric: check-star chi2_red

**Invalid Stage 2 metric (rejected):** `lc_rms / lc_rms_ooe` on variable targets - measures intrinsic variability, not error calibration.

**Correct metric:**
\[
\chi^2_\mathrm{red} = \frac{\sum_i ((m_i - \langle m\rangle)/\sigma_i)^2}{N-1}
\]
on **check stars** (constant comparison stars via `check_kmag_*` sidecars), with production-style \(\sigma_i\).

**Literature (R1)**

| Topic | Source | Quote / value |
|-------|--------|---------------|
| Check-star / ensemble method | Honeycutt (1992) PASP 104, 435 | Standard reference for differential ensemble photometry and check-star validation of zeropoint stability. |
| Scintillation (diagnostic only; **not** wired to production `err`, decision 3) | Osborn et al. (2015) MNRAS 452, 1707 eq. (7) | '\(\sigma_Y^2 = 10\times10^{-6} C_Y^2 D^{-4/3} t^{-1} (\cos\gamma)^{-3} \exp(-2h_\mathrm{obs}/H)\)' with '\(C_Y\) ... mean value of **1.5**' across sites. |
| Implemented scintillation | `src_py/sigma_budget.py` | `OSBORN_CY_DEFAULT = 1.5`; same equation in `scintillation_sigma()`. |

**Measurement script:** `dev/scripts/audit_stage3_part1_measure.py`  
**Data:** `tmp/audit_stage3_part1_snapshot.json`

#### Rig: `NoFilter_60_2` / QHY294PROM (BO CVn field)

| Quantity | Value |
|----------|------:|
| Check fields measured | 162 |
| **Median chi2_red** (check photon + field ensemble SEM + sys) | **40.98** |
| Median chi2_red (target `err` proxy, same `source_file`) | 0.108 |
| `scint_would_be_rel` (Osborn, C_Y=1.5, D=0.2 m, t=60 s, airmass~1.2, alt=275 m) | ~0.000826 |

**T1 Group D reconciliation (same five targets):**

| target_id | N | chi2_red **targets** (invalid baseline) | T1 prior | chi2_red **check** (same field) | chi2_red check w/ target err proxy |
|-----------|---:|----------------------------------------:|---------:|--------------------------------:|-----------------------------------:|
| 1485540612577549568 | 139 | 0.488 | 0.577 | 5.07 | 0.197 |
| 1485552329248338816 | 139 | 1.066 | 1.265 | 31.93 | 0.093 |
| 1485574899299782528 | 139 | 1.139 | 1.345 | 15.15 | 0.131 |
| 1485609538212672000 | 37 | 1.302 | 1.578 | 94.48 | 0.058 |
| 1485913828055470592 | 139 | 1.250 | 1.477 | 31.00 | 0.090 |

**Reconciliation:** T1 priors match **target** chi2_red on `mag_calib_final` (recomputed; agreement within ~0.01-0.03). They do **not** apply to check stars - targets are variable; check-star chi2 uses `kmag` sidecar series.

**Interpretation (measurement only; decision 3 unchanged):**
- With **check-specific photon err** + field ensemble SEM: median chi2_red >> 1 ? quoted uncertainties **under-estimate** check-star scatter (or `kmag` residual includes unmodeled systematics).
- With **target err** proxy: median chi2_red ~ 0.1 ? **over-estimated** vs constant check star (expected: target err includes target photon noise, often larger).
- Scintillation term (~0.08% rel) is negligible vs photon+ensemble; wiring it into production would not fix chi2 ~ 41.

**Variance fractions (median over check fields, check-err decomposition):** ensemble SEM dominates (~95% of variance); photon ~5%; `sigma_sys`=0 on this rig.

---

### 1.2 - D1-2 corrected metric: linearity residual vs peak ADU

**Invalid Stage 2 metric (rejected):** Gaia G vs peak ADU correlation (r = -0.85) - brightness-forced, not linearity.

**Correct metric:** per-frame ZP from masterstars: `zp = median(inst - cat)`; residual `= inst - cat - zp` vs `peak_max_adu`, binned to `saturate_limit_adu`.

| peak ADU bin (approx) | n | mean residual (mag) | std (mag) |
|----------------------|---:|--------------------:|----------:|
| 0 - 6553 | 342847 | +0.124 | 1.996 |
| 6553 - 13107 | 11753 | -0.172 | 0.239 |
| 13107 - 19661 | 4013 | -0.237 | 0.227 |
| ... | ... | ... | ... |
| 52428 - 58982 | 460 | -0.303 | 0.166 |
| 58982 - 65535 | 619 | -0.229 | 0.652 |

- **`saturate_limit_adu` median:** 65535
- **Trend onset (heuristic):** none flagged below saturation; low-peak bin dominated by outliers (std ~2 mag).
- **High-peak bins:** mean residual shifts from ~+0.12 mag (low peak) toward ~-0.30 mag (~400-600 mmag) - systematic, but **unfiltered flux vs Gaia G catalogue mag** confounds pure sensor linearity.

**Literature (R1)**

| Topic | Status |
|-------|--------|
| IMX294 / IMX571 manufacturer linearity spec | **UNVERIFIED** - datasheet not retrieved this session |
| Standard test method | Flat-ratio vs exposure time (classic CCD linearity test); not executed here |

**Verdict:** Residual-vs-peak plot does **not** isolate sensor linearity on this rig without band-matched catalogue magnitudes. No ADU level identified as clean sensor non-linearity onset.

---

## Errors (if any)
- D1-2 low-peak bin includes bad/unmatched sources; high-peak trend may reflect transform mismatch (CV/G) not ADU linearity.
- Check-star chi2 uses reconstructed err (check photon + target-field ensemble SEM); full 'check as target' pipeline rerun would be tighter.

## Files changed
- `src_py/photometry_core.py` - err-budget export
- `dev/tests/photometry_sha.py` - QC column allowlist
- `dev/scripts/audit_stage3_part1_measure.py` - measurements (new)
- `dev/results/CURSOR_RESULT_audit_stage3_part1.md` (this report)

## STOP GATE 1
**Part 0 + Part 1 complete. Waiting for Milan before Part 2.**
