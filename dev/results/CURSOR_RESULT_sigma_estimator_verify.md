# CURSOR RESULT - SIGMA-ESTIMATOR-VERIFY (2026-07-26)

Read-only diagnostic. No code changes, no commits. Scratch: `tmp/sigma_estimator_verify.py`,
`tmp/sigma_estimator_verify.json`.

---

## What I did

Tested whether draft_450's lower DAO `bg_std` (62.0 vs 83.5 ADU) reflects genuinely lower pixel
noise, or a broken estimator. Measured gradient-immune pixel-to-pixel noise (`sigma_pp`), traced
the estimator code path, and checked MASTERSTAR stack construction.

---

## S1 - Real pixel noise (`sigma_pp`)

Method: `sigma_pp = MAD(I[i+1]-I[i]) / sqrt(2) / 0.6745` on star-masked regions (DAO mask,
40 px margin). Combined horizontal and vertical quadrature mean.

### MASTERSTAR stacks

| Draft | sigma_pp (ADU) | bg_std estimator (ADU) | bg_std / sigma_pp | DAO threshold @ 2.1 sigma (ADU) |
|-------|---------------:|-----------------------:|------------------:|--------------------------------:|
| 435 | **46.13** | **83.82** | 1.82 | 175.4 |
| 450 | **46.07** | **62.23** | 1.35 | 130.2 |

**Verdict:** `sigma_pp` is **unchanged** between drafts (46.13 vs 46.07 ADU, <0.2% delta).
`bg_std` fell **26%** (83.8 -> 62.2). **The estimator is wrong on 450; real pixel noise did not
change.**

If true noise is `sigma_pp ~ 46 ADU`, the effective detection sigmas are:

| Draft | threshold / sigma_pp | vs nominal 2.1 |
|-------|---------------------:|---------------:|
| 435 | 175.4 / 46.1 = **3.81** | 1.81x stricter than label |
| 450 | 130.2 / 46.1 = **2.83** | 1.35x stricter than label |

Using the task's reference scale (435 `bg_std` as noise proxy): `130.2 / 83.5 = 1.56 sigma_true`,
tail ratio `P(>1.56)/P(>2.10) = 3.32x` vs measured pass-1 ratio **8926/2552 = 3.50x** (~5%).

### Individual detrended frames (cross-check)

| Frame | draft | sigma_pp | bg_std | threshold @ 2.1 |
|-------|-------|----------|--------|----------------:|
| Light_001 | 435 proc | 42.56 | 102.11 | 214.0 |
| Light_001 | 450 plain | 42.30 | 72.41 | 152.1 |
| Light_050 | 435 proc | 30.60 | 69.24 | 145.0 |
| Light_050 | 450 plain | 30.47 | 48.96 | 102.8 |

Same pattern on matched frame indices: **`sigma_pp` matches; `bg_std` is lower on 450** (29-39%
lower on these examples). Frame 139 not present under 450 plain naming on disk at measurement time.

---

## S2 - Estimator code path

### Where `bg_std` comes from

**File:** `src_py/pipeline.py`
**Function:** `detect_stars_and_match_catalog` (lines 8080-8150), called for MASTERSTAR pass-1 DAO
via `generate_masterstar_and_catalog` (line 11998).

**Steps:**

1. `mean, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)` on the full float32 frame.
2. `data0 = arr - med` (global median subtract only; no sky-surface removal at detection time).
3. `_, _, std_dao = sigma_clipped_stats(data0, sigma=3.0, maxiters=3)` (or reuse step-1 stats if
   no DAO binning).
4. `threshold = masterstar_dao_threshold_sigma * std_dao` (2.1 x std_dao).

**Estimator:** `astropy.stats.sigma_clipped_stats` (iterative 3-sigma clipping, 3 maxiters) on
the **full frame** after median subtract. Not `mad_std`, not `Background2D`, not annulus-based.

### Clipping survivors (MASTERSTAR)

| Draft | clip survivors | clip total | survivor fraction |
|-------|---------------:|-----------:|------------------:|
| 435 | 2,872,275 | 2,908,554 | 0.9875 |
| 450 | 2,866,755 | 2,908,554 | 0.9856 |

Clipping is **not** dramatically more aggressive on 450 (slightly fewer survivors, which would
*raise* std, not lower it). The 26% drop in `std_dao` is **not** explained by clipping counts alone.

### Why 62.0 on 450 (evidence, not inference)

Measured facts on the **same** frames where `sigma_pp` is unchanged:

- 450 detrended inputs retain the **large-scale component** removed by order-2 sky-surface subtract
  on the 435 `proc_*` path (DAO-ONLY VERIFY: large/small variance ratio 20-60x on frame diffs).
- After global median subtract, `sigma_clipped_stats` on the full frame returns **lower** RMS on 450
  despite **equal** high-frequency noise (`sigma_pp`).

**Concrete mechanism (hypothesis grounded in measurement):** the estimator conflates large-scale
structure with noise. On 450, remaining smooth gradient redistributes pixel values after median
subtract such that the global sigma-clipped std **underestimates** the threshold-relevant noise
(`sigma_pp` unchanged). This is the **inverse** of the earlier causal claim that missing sky-surface
subtraction *lowers* bg_std because gradients add variance -- gradients add low-frequency power that
this particular estimator does not track the same way `sigma_pp` does.

**Do not carry "missing sky subtract -> lower bg_std" as proven physics.** Carry: **estimator
returns 62 vs 83 while true pixel noise is ~46 ADU on both stacks.**

---

## S3 - Stack noise character

**MASTERSTAR is not a multi-frame combination.** `build_masterstar_from_detrended`
(`pipeline.py` lines 2801-2959) copies the **single best** detrended frame (lowest `VY_FWHM`) via
`shutil.copy2`. No median stack, no sigma-clipping combine, no per-frame normalization.

| Item | draft_435 | draft_450 |
|------|-----------|-----------|
| Combination method | Best-frame copy | Best-frame copy |
| Frames in candidate pool | 139 detrended | 139 detrended |
| `VY_FWHM` on MASTERSTAR | 3.207 px | 3.521 px |
| `VY_NDAO` | 2552 | 8926 |
| Same calibrated dark/flat | yes (byte-identical masters) | yes |

Different best-frame selection (FWHM 3.21 vs 3.52) means the two MASTERSTAR arrays are not guaranteed
to be the same physical frame, but per-frame index 050 shows the **same sigma_pp / divergent bg_std**
pattern even on matched indices. The detection count ratio is not explained by a different stack
combination method -- there is none.

---

## Summary

| Question | Answer |
|----------|--------|
| Did real pixel noise fall on 450? | **No.** sigma_pp ~ 46 ADU on both stacks. |
| Is bg_std trustworthy on 450? | **No.** 62 ADU vs 83 ADU while sigma_pp unchanged. |
| Does 130 ADU threshold mean 2.1 sigma? | **No.** It is ~2.8 sigma vs sigma_pp, ~1.6 sigma vs 435's bg_std scale. |
| Primary defect | **Sigma estimator** (`sigma_clipped_stats` on median-subtracted full frame), not exposure/calibration. |

**Impact on SKY-SURFACE arc:** missing sky-surface subtract changes the **input image** to this
estimator; fixing sky-surface alone may not restore bg_std to 83 unless the estimator is also
reviewed. Treat as potentially **two defects** (see SKY-SURFACE-REGRESSION deliverable).

---

## Files changed

None (read-only).
