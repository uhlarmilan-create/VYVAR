CURSOR RESULT - 2026-08-12 (ZP-clip close-out)

What I did
Measurement + behaviour sweep on the accepted 509 zeropoint-clip cause. Read-only. No fixes. Post-processed draft 509 instrumental fluxes only for the counterfactual matrix.

## Output / findings

### Part 1 — Character of the 37 rejections

**Scattered through the night, not contiguous physical blocks.**

- Frame indices: 10, 17, 22, 24, 29–31, 38, 40, 42, 47, 50, 61, 64, 67, 77–79, 81, 84, 86, 93–94, 99, 102, 104, 108–109, 114–115, 117, 119, 129, 133, 138, 145, 148
- Strict contiguous blocks: **30** (25 singletons, 5 runs of length 2–3). Max run length = 3. Median gap between rejections = 3 frames.
- Predictors (kept vs rejected means): airmass 1.064/1.051, sky 1597/1549, FWHM 3.019/3.014, bright flux +0.3%. No physical driver.
- Bright-comp residual vs other 4: kept std 0.010, rejected std 0.007 — rejected frames are **not** when the star is noisier.

**MAD-ratio distribution for bright comp (`|z?med|/?`) across 134 frames:**

| min | p10 | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| 0.67 | 1.19 | 1.50 | 2.21 | 3.20 | 4.83 | 40.98 |

- n > 3.0: **37** (by definition the rejects)
- Among rejects: ratio min/med/max = 3.10 / 4.17 / 41.0
- Within 10% of boundary (ratio ? 3.3): **9/37 (24%)**
- Critical: MAD on reject frames is **half** of kept (median MAD 0.018 vs 0.037). Absolute |z?med| is similar (~110 vs ~99 mmag). Boundary shrinks from ~166 to ~80 mmag. **33/37** rejects have MAD < overall p25. One frame has MAD?0.002 ? ratio 41.

**Interpretation (pre-registered):** temporally scattered + denominator collapse ? clip switching on **estimator noise** with N=5 (MAD set by a thin order statistic), not a physical event on that star.

### Part 2 — Is the bright comp deviant?

Leave-one-out vs other comps (poly5 residual):

| star | G | Phase1 rms | leave-1 raw std | poly5 std | shape |
|---|---|---|---|---|---|
| bright `…531712` | 9.75 | 0.0111 | 0.0096 | 0.0092 | unimodal |
| `…564608` | 9.68 | 0.0114 | 0.0116 | 0.0113 | unimodal |
| `…698240` | 10.79 | 0.0147 | 0.0104 | 0.0100 | unimodal |
| `…2858240` | 11.17 | 0.0139 | 0.0133 | 0.0131 | unimodal |
| faint `…0107904` | 11.52 | 0.0206 | 0.0121 | 0.0118 | unimodal |

- Bright: **good comparison star**. No time structure (corr(res,airmass)?0). Clip is discarding a good star.
- Faint: mildly noisier but still unimodal ~12 mmag; not catastrophic. Dropping it every frame is a **constant** offset if membership were fixed; the damage is the **intermittent** bright drop. Per-frame membership change is the wrong tool either way — a bad star should be excluded once for the draft.

### Part 3 — Counterfactual matrix

Harness control: **D == C exactly** (max|D?C|=0). Matrix valid.

`B` = 5 comps, clip OFF, equal-weight mean ZP.  
`E` = 5 comps, clip OFF, Broeg 1/?² weights.  
`A` = 5 comps, clip ON, Broeg (production path).

| variant | comps | clip | check `…1001088` scatter | check `…4892800` scatter | target res std | shape | n |
|---|---|---|---|---|---|---|---|
| A | 5 | ON | 0.0190 | 0.0187 | 0.0201 | unimodal* | 134 |
| B | 5 | OFF | **0.0068** | **0.0085** | **0.0124** | unimodal | 134 |
| C | 3 | OFF | 0.0079 | 0.0092 | 0.0128 | unimodal | 134 |
| D | 3 | ON | 0.0079 | 0.0092 | 0.0128 | unimodal | 134 |
| E | 5 | OFF Broeg | 0.0073 | 0.0086 | 0.0125 | unimodal | 134 |

\*Harness A ZP residual is still bimodal (peaks ?0.027 / +0.011); poly5 on target mag can hide the second peak. Archived official check scatter 0.0252 / target 0.0260 bimodal — harness A tracks archived target at r=0.999 but under-reproduces absolute check scatter by ~6 mmag (tier/series path detail); **relative** matrix is unaffected.

435-quality reference: instrumental / archived check ~0.008.

**Pre-registered case: B ? C ? 435 quality (~0.008).**  
The clip is the entire cause. Removing it restores the result. Comp admission (5 vs 3) needs no change for photometry quality.  
E ? B (does not beat B); Broeg weighting alone is not required.

### Part 4 — Surviving per-frame / per-epoch rejection (behaviour sweep)

On the instrumental-flux ? exported-magnitude path:

| location | what | trigger | N / gate |
|---|---|---|---|
| **`photometry_core.py:3461–3479`** `ensemble_normalize` | drops comps from per-frame ZP | 3×MAD on (cat?inst) | **`len(z) ? 4`** — **this defect** |
| `photometry_core.py:4546–4604` `detect_outliers` | flags LC points outlier_hi/lo (does not rewrite mag; used in reporting / OOE flag filter) | 3×MAD on mag_calib | activates on ?3 finite LC points |
| `photometry_core.py:9925–9975` / `_exclude_err_scatter_unmatched_epochs` | **drops epochs** from LC export | ensemble-scatter join fail (I-04) | per-epoch, not N-comp |
| `photometry_core.py:3334–3356` | draft-level comp selection into ensemble | quality good/suspect; p2p order | n_comp_min/max — whole-draft, not per-frame |
| `photometry_core.py:3012–3080` `check_comparison_stability` | draft-level exclude/suspect | p2p > 0.10; slope gate | whole-draft membership |
| `photometry_core.py:3440–3481` Broeg weights | **down-weights** comps in ZP (no drop when clip OFF) | 1/rms² × tier | whenever rms present |

Off / diagnostic / not science mag path: Labbe empty-aperture MAD (`photometry_core.py:729`), align-residual gate default OFF (`7877+`), PSF inlier clips in `psf_photometry.py` (not used for this aperture export).

**Known recon conflicts (confirmed still present, not re-opened):**
- `detect_outliers` @ `photometry_core.py:4589–4602`
- plate-solver SIP pair clip @ `vyvar_platesolver.py:685–693` (`med + 5×1.4826×mad`)

**Only N-comp-gated per-frame rejection of the same class as this bug:** the ZP clip at 3461–3479.

### Part 5 — FWHM inconsistency (record only; finding A-1)

| quantity | 435 | 509 |
|---|---|---|
| insp FWHM (med) | 5.14 px | 5.31 px |
| VY_FWHM_GAUSS | 2.40 px | 3.30 px |
| target aperture | 2.716 px | 4.141 px |

**How computed:**
- **Inspection FWHM:** `pipeline.py:1385–1439` ? `_robust_frame_fwhm_median` (`16646+`) = median of **moment-based** FWHMs (`_moment_fwhm_elong_peak_at` `16588+`, `fwhm = 2.355 × mean(?1,?2)`) over many star-like DAO detections on each calibrated frame.
- **VY_FWHM_GAUSS:** `pipeline.py:12880–12915` ? `measure_fwhm_from_masterstar` (`photometry_core.py:469–627`) = median of **2D Gaussian fits** (`Gaussian2D`, FWHM = 2.355×stddev) on ~30 isolated stars on the **MASTERSTAR stack**.

**Definitional difference:** moment-FWHM on single frames vs fitted Gaussian FWHM on the stacked master. Both claim to be FWHM in px (same 2.355 factor). Gap is **~1.6–2.1×**, not explained by ?-vs-FWHM confusion — **genuine disagreement** between estimators (stack smoothing / fit-box / star sample / under-sampling likely). Separate task.

**Enclosed-flux fraction (2D Gaussian EE = 1?exp(?r²/(2?²))):**

| draft | ap | EE @ VY_FWHM_GAUSS | EE @ insp FWHM | r/FWHM_g | r/FWHM_insp |
|---|---|---|---|---|---|
| 435 | 2.716 | 0.972 | **0.540** | 1.13 | 0.53 |
| 509 | 4.141 | 0.987 | **0.815** | 1.25 | 0.78 |

If insp FWHM is the truth, 435’s aperture enclosed only ~54% of the PSF; 509 ~82%. Aperture followed the smaller Gaussian number in both drafts.

## Errors (if any)
None. Harness A absolute check scatter 0.019 vs archived 0.025 (relative matrix intact; D=C control pass).

## Files changed
None (read-only). Scratch: `tmp/_zp_clip_close.py`, `tmp/_zp_mad_ee.py`, `tmp/_zp_tier_check.py`, `tmp/_zp_reject_37.csv`.

---

## DECISIONS REQUIRED

**Decision 1 — Delete the per-frame ZP 3×MAD clip in `ensemble_normalize` (`photometry_core.py:3461–3479`)?**

- Evidence: matrix case **B ? C ? 435 quality**. Clip OFF with 5 comps ? check scatter 0.0068 / 0.0085; clip ON ? 0.019. Rejections are scattered estimator flips (MAD collapse), not a bad star episode. Bright TIER1 is a good comp (leave-1 std 0.009).
- Recommendation from data: **authorize deletion** (iron-rule alignment). No code in this task.

**Decision 2 — Revert `phase01_comparison_max_mag_diff` 2.0 ? 1.5?**

- Evidence: with clip OFF, **5-comp (B) equals or beats 3-comp (C)**. Admission to 2.0 is what *armed* the dormant N?4 gate, but is not an independent quality regressor once the clip is gone. Faint comp is slightly noisier (0.012) but harmless under equal/Broeg weights without rejection.
- Recommendation from data: **no quality-driven need to revert**. Optional policy choice only (pool size / purity), separate from Decision 1.

Do not couple the two: Decision 1 alone restores 509; Decision 2 is not required by the matrix.
