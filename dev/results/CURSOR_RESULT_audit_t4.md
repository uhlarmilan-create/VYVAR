CURSOR RESULT - Science Audit Tranche 4 (2026-07-30)

Read-only: source/literature verification + M1/M2 discrepancy resolution. No code changes. No commits.

---

## -0 - Tranche-3 M3 retraction (accepted)

Tranche 3 inferred ~9? effective significance from `conv/sigma_pixel ? 0.39-0.45` using a **unit-sum Gaussian smooth**. DAOStarFinder uses Stetson (1987) FIND's **zero-sum density-enhancement kernel** with `scale_threshold=True` (photutils default), so `threshold_eff = threshold - kernel.rel_err` and nominal `N-sigma_pixel` is **exactly N? in the convolved image** for white noise.

**Local verification (photutils 3.0.0, FWHM 3.2 px, ?=46 ADU):**

| Quantity | Value |
|----------|------:|
| `kernel.rel_err` | 1.3604 |
| measured ?_conv/sigma_pixel | 1.3594 |
| significance of 3.8*sigma_pixel threshold | **3.80 ?_conv** |

**Withdraw:** S3 recommendation to switch threshold source to Background2D *because sigma_pp is unstable vs bkg2d*. M2 showed both estimators agree within ~1.2% per frame; 14-16% frame-to-frame spread is a property of resampling/dither, not estimator choice.

---

## -1 - VYVAR DAOFIND usage: correct convention

`threshold = masterstar_dao_threshold_sigma - std_dao` on median-subtracted input matches DAOFIND semantics when `std_dao` is the per-pixel background ? of the **same image being searched**. photutils implements the scaling correctly. **No defect in threshold arithmetic.**

Recalibration **2.1 ? 3.8** (AUDIT-T3 bundle) remains valid as restoring anchor-class effective threshold once P-10 removes doubled-gradient inflation of `sigma_clipped_stats`. Paper may state **"3.8? detection threshold"** literally - subject to -3 noise correlation.

---

## -2 - Real defect: correlated noise after astroalign resample

MASTERSTAR is built from `detrended_aligned` (resampled) frames. DAOFIND's white-noise identity breaks; nominal 3.8? becomes **~3.30-3.58?** effective depending on subpixel dither (Tranche 4 -3.1 simulation; literature: Fruchter & Hook 2002, Casertano et al. 2000).

**Neither sigma_pp nor Background2D fixes this** - both correctly estimate per-pixel ? of the resampled image; swapping estimators does not restore the uncorrelated-noise assumption.

**Options (literature order):**

| Option | Description |
|--------|-------------|
| **A** | Detect on pre-align preprocessed frame (no correlation) |
| **B** | `scale_threshold=False`; threshold convolved-image RMS directly |
| **C** | Monte-Carlo correlation factor per setup (WISE / drizzle standard) |
| **D** | Document nominal vs effective (6-13% optimistic), accept run-to-run drift |

Photometry path already handles correlation via Labb- empty apertures (empirical ? on resampled image). Detection should **measure noise of the quantity thresholded**, not derive from white-noise sigma_pixel.

---

## -3 - M1/M2 discrepancy **resolved** (not a measurement bug)

Tranche 4 -6 flagged: M1 post-align sigma_pp median 30.64 ADU vs M2 range 44.93-51.94 ADU for "subset" frames.

**Cause:** **Different frame sets**, not inconsistent math.

| Set | Frames |
|-----|--------|
| M1 aligned (10, evenly spaced) | 001, **017**, **034**, **050**, **067**, **083**, **100**, **116**, **133**, **150** |
| M2 top-5 | **001**, **002**, **003**, **004**, **005** |
| Overlap | **001 only** |

Post-align sigma_pp on shared frame 001: **51.94 ADU in both** (exact match).

M1 median 30.64 is dominated by frames 017-150 (sigma_pp ? 28-32 ADU). M2 frames 002-005 were **never in M1**; they sit at sigma_pp ? 45-51 ADU (less warp attenuation / different dither). The headline "align drops sigma_pp ~28%" compares **different frame indices** (early vs late night), confounded with resampling - **must not be used** until a paired before/after study on the **same** frames is run.

**Action for future measurement:** paired cal/pre/aligned sigma_pp on identical frame list; separate early vs late frame cohort if dither evolution is of interest.

---

## -4 - Citations (CITATIONS.bib)

| Reference | Status |
|-----------|--------|
| Stetson (1987) PASP 99, 191 | **Present** (`stetson1987`) |
| Fruchter & Hook (2002) PASP 114, 144 | **Present** |
| Casertano et al. (2000) AJ 120, 2747 | **Present** |
| Bertin & Arnouts (1996) | Already cited elsewhere |

No new bib entries required for Tranche 4.

---

## -5 - Where this leaves decisions

| Item | Status |
|------|--------|
| Push AUDIT-T3 bundle (P-10 + sigma_pp + 3.8) | Still strictly better than `origin/main`; not blocked |
| Anchor re-cut | Still blocked - choose A/B/C/D for detection noise on resampled frames first |
| Background2D as threshold source | **Withdrawn** as fix for correlation (Tranche 3 S3) |
| `CURSOR_RESULT_dao_sigma_stability.md` | Superseded on M3 interpretation and S3 verdict; M1/M2 headline needs errata |

---

## Files

| Path | Role |
|------|------|
| `dev/results/CURSOR_RESULT_audit_t4.md` | this report |
| `tmp/dao_sigma_stability.json` | underlying M1/M2 numbers (frame lists verified) |
