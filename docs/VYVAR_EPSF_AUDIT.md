# VYVAR ePSF path -- read-only end-to-end audit

Audited at HEAD fe8201c, read-only, file:line. Motivated by the h & chi Per probe finding
`epsf_vs_input_fwhm_ratio` = 0.59-0.67 ("ePSF narrower than input FWHM; worth checking").

No production, config, or flag changes from this audit. PSF remains OFF in production.

---

## Path map (entry -> photometry)

| Stage | Function | Line |
|-------|----------|------|
| FWHM context (ratio denominator) | `get_epsf_fwhm_from_context` | 181 |
| | `VY_FWHM` header -> `_median_fwhm_obs_files` (53) -> 4.5 fallback | |
| | `_clamp_fwhm_px` (49) to [2, 12] | |
| Star selection | `_epsf_allowed_catalog_ids` | 84 |
| | `_epsf_positions_from_csvs` | 128 |
| | `_epsf_augment_candidates_from_detected_pool` | 596 |
| | `_epsf_prepare_stars` | 763 |
| Build | `build_epsf_model` -> `_epsf_build_imagepsf_from_stars` | 1297, 471 |
| | EPSFBuilder: osamp 2, maxiters 15, smoothing quadratic/quartic | 489-494 |
| QC in-build | `epsf_nan_fraction`, `epsf_fwhm_native_px`, ratio, `epsf_asymmetry` | 498-560 |
| Build warnings | ratio <0.5 or >2.0; asymmetry >0.1 | 1340-1346 |
| Aperture correction | `_compute_aperture_correction` (empirical flux ratio) | 378 |
| | `_compute_moffat_aperture_correction` | 412 |
| Per-star quality | `assess_psf_quality` | 2021 |
| Photometry | `psf_photometry_stars` | 2077 |

### Sky subtraction in `psf_photometry_stars` (2026-06-08 fix)

**Was:** 2-pixel cutout border median (`psf_photometry.py` ~2392) -- bright-star wings
contaminated the small fit cutout border (+7.9 ADU/px at mag 12 in V3d), producing mag-dependent
pre-AC flux bias; single AC zero-pointed bright mags leaving ~+4-5% mid-mag excess.

**Now:** Initial sky from aperture-geometry annulus on the full frame; **one-pass residual-annulus
refine** after the first PSF fit subtracts the fitted ePSF wing (and grouped neighbours) from the
annulus before the refit. Fallback: cutout border median when annulus is off-chip. Provenance:
`psf_sky_method` (`residual_annulus` | `annulus_local` | `border_fallback`). PSF-only; aperture LC
path unchanged.

**V3d re-run (seed 367):** mid-mag drift fixed by **sky-only fit weights** (Astier 2013;
Lacroix 2025; `psf_weight_mode=sky_only`). Reported errors use **sandwich variance**
(`psf_err_mode=sandwich_skyonly`): true pixel variance with sky-only weights -- required for
calibrated bright-star errors (P3 mag12 0.56 -> 1.07). Noisy post-AC: +0.8% (mag12) ->
+1.75% (mag16); drift **+0.95 pp**. V3d **PASS** all pillars mag<=17. See
`tier_v3d/v3d_sandwich_proof.md`, `v3d_weight_proof.md`.

---

## FINDING EPSF-1 -- epsf_fwhm_native half-max estimator is non-robust

**Status: CLOSED (2026-06-08)** -- replaced with azimuthally-binned radial profile in
`_epsf_fwhm_native_from_profile`; QC warning band tightened to [0.80, 1.25]; harness V3e PASS.
Report: `tests/validation/data/tier_v3e/v3e_epsf_fwhm.md`.

**Location (legacy):** `psf_photometry.py` first-pixel half-max inside `_epsf_build_imagepsf_from_stars`
(preserved as `_epsf_fwhm_native_legacy_px` for harness comparison only).

**Mechanism:**

```text
v = epsf_data.ravel(); v = v / v.max()          # normalized by single PEAK pixel
r = radius of each pixel from center, raveled
below_half = np.where(v_sorted_by_r < 0.5)[0]
epsf_fwhm_native = 2.0 * r_s[below_half[0]] / osamp   # FIRST pixel below half-max
```

Two choices bias the half-max radius **low**:

1. **No azimuthal binning** -- the first individual pixel (smallest radius) below 0.5 sets FWHM,
   not the radius where the azimuthal average crosses 0.5.
2. **Normalization by `v.max()`** -- a noisy peak makes other pixels look smaller, pulling the
   0.5 crossing inward.

=> `epsf_fwhm_native` is systematically **underestimated** => `epsf_vs_input_fwhm_ratio` biased **< 1**.

**Supporting evidence (h & chi Per probe):** 375 L (3.36 px) x 0.666 and 380 L (3.80 px) x 0.589
both yield `epsf_fwhm_native` ~ **2.238 px** -- nearly constant, independent of seeing -- consistent
with a geometry/grid-dominated rather than data-dominated estimator.

**RESOLVED (2026-06-09 decisive test, `docs/VYVAR_EPSF_FWHM_TEST.md`):**

| Cause | Holds? | Evidence |
|-------|--------|----------|
| **(3) Inflated seeing denominator** | **YES (dominant)** | OBS_FILES L median **3.84 px** vs Moffat ePSF **2.13-2.20 px** and stellar Moffat **1.92-2.10 px** on 375/380 L. Ratio<1 is benign. |
| **(1) Estimator artifact** | **Secondary** | `buggy_halfmax` pinned at **2.236 px** (= sqrt(5)) both drafts; Moffat/azimuthal ~2.09-2.29 px. |
| **(2) Narrow built ePSF** | **NO** | `fwhm_moffat` ~= `fwhm_stars` within ~10%; EPSFBuilder reproduces stellar core. |

Harness **V3e** on Tier-A synthetic: ratio=1.127 (ideal Moffat stack). Real-field ratio 0.59-0.67
is driven mainly by **denominator seeing**, not a bad ePSF model.

**Blast radius (good news):** `epsf_fwhm_native` is used ONLY for QC dict + ratio + build warning
(515, 542-544, 1342). It does **NOT** enter:

- `assess_psf_quality` (uses input `fwhm_px`, chi2, snr, pos_shift, nn_dist_fwhm)
- `_compute_aperture_correction` (empirical flux-ratio median, FWHM-independent)

=> **Diagnostic-fidelity issue**, not a photometry-correctness bug. Can mislead humans (ratio<1
looks like bad ePSF) and the <0.5 / >2.0 warning threshold (1343) is calibrated on a biased metric.

**Practical takeaway:** Do **NOT** withhold PSF on h & chi Per because of ratio<1 alone. PSF
readiness should rest on per-star quality gating (chi2/snr/shift/nn) and PSF-vs-aperture comp RMS on
real blends, not on this ratio. See `docs/VYVAR_HCHIPER_PSF_PROBE.md`.

**Fix shipped (EPSF-1, 2026-06-08):**

1. `_epsf_fwhm_native_from_profile`: bin radius (0.5 oversampled px), mean normalized flux per
   annulus, linear interpolate 0.5 crossing -> `epsf_fwhm_native = 2*r_half/osamp`.
2. QC warning at build: ratio outside **[0.80, 1.25]** (was 0.5-2.0).
3. **V3e harness** (`v3e_epsf_fwhm.py`): Moffat inject at FWHM 2.7 / 5.4 / 6.02 px; NEW ratios
   **1.038-1.049** (PASS [0.85, 1.15]); OLD ratios 1.048-1.111 (closer to 1 but still
   grid-quantized). Real h&chi Per ratio 0.59-0.67 remains a **seeing-denominator** effect, not
   flux or `assess_psf_quality`.

---

## Rest of path -- no further red flags

- EPSFBuilder params standard; single and grid builds share one code path.
- `epsf_asymmetry` (quad-fold) correct for coma; does **not** flag symmetric elongation (A3 / harness
  lesson). Smear needs ellipticity proxy (`TODO-PSF-ASYMMETRY`).
- Aperture correction empirical, FWHM-independent -- robust.
- `assess_psf_quality` multi-criteria with graceful NaN skips -- sound.
- All PSF flags remain OFF in production.

---

## Revision 2026-07-09 (HEAD f38e924) — PSF-AUDIT-FIXES

Read-only audit (Claude, 2026-07-09) found four latent PSF-arc issues; all **FIXED** with PSF
disabled in production (byte-identity vs draft_424 anchor preserved).

| ID | Status | Commit | Summary |
|----|--------|--------|---------|
| **PSF-ERR-DECOUPLED** | FIXED | `0b5eb8b` | `psf_flux_err` never reached LC err; added missing `psf_*` to `PROC_STORE_COLS`; route LC `err` from PSF sandwich when `lc_flux_method=psf`; emit `err_method` only when PSF routes exist. |
| **PSF-AC-FALLBACK** | FIXED | `0b5eb8b` | `psf_ac_applied` per-row flag; routing guards require AC applied (`rule_faint_psf`, `_get_lc_adaptive`, `_resolve_star_flux_method` via `compute_lc_flux_method`). |
| **PSF-FLAG-VESTIGIAL** | FIXED | `a3368fb` | `psf_spatial_enabled` master gate: spatial active iff `enabled AND order>0`. |
| **PSF-LEGACY-SELECTOR** | FIXED | `0b5eb8b` | Removed `_get_lc_psf_or_dao` (superseded by per-star/adaptive router). |

**Part A.1 store-projection check:** `PROC_STORE_COLS` previously had `psf_flux`, `psf_fit_ok`,
`psf_chi2`, `psf_quality`, `psf_snr` only. **Added:** `psf_flux_err`, `psf_quality_fallback`,
`psf_ac_factor`, `psf_ac_n_used`, `psf_ac_applied`. `lc_flux_method` is runtime-computed (not
proc-CSV persisted). `wcs_untrusted` remains derived from `catalog_match_mode`.

**Positive confirmations (unchanged):** fail-safe `_run_epsf` gating; quality-fallback default;
`neighbor_sub` `bright_close_regime` refuse path; grouper gate; EPSF-1 closure holds.

**Runner classification:** `psf_runner.py` = standalone dev harness, **not** production path.

**Enablement checklist (post-fix):** real-field Newton draft + ERR/AC gates **DONE** by this task;
flip `psf_photometry_enabled` only after Brno-gate field validation.

---

## Cross-references

- h & chi Per probe: `docs/VYVAR_HCHIPER_PSF_PROBE.md`
- Synthetic harness V3e: `docs/VYVAR_VALIDATION.md`, `tests/validation/recover.py`
- Audit ledger: `docs/VYVAR_AUDIT_LEDGER.md` (EPSF-1)
