# MEMO: How photometry tools locate a star and decide it is saturated

**Date:** 2026-08-13
**Question:** Given a catalog star, how do established tools find its pixels on
a frame, and over which pixels do they test for saturation?
**Short answer:** They do not search for a bright pixel. The position is
determined, not discovered, and saturation is tested over the star's own pixels
at that determined position.

---

## 1. Nobody searches a box for the brightest pixel

The universal pattern is: take an initial position, then **centroid** in a small
cutout using a flux-weighted moment or a fit. The brightest pixel is not the
target of the operation.

- **photutils** `centroid_sources` takes initial guesses and makes a cutout of
  `box_size` around each - default **11 pixels** - then computes a centroid
  inside it with centre-of-mass, a 1D/2D Gaussian fit, or a quadratic fit.
- **AstroImageJ** offers two centroid methods: the Howell (2006) algorithm,
  which the AIJ paper notes gives highly repeatable results and is *not
  sensitive to the starting location*, and a centre-of-mass method preferred for
  defocused stars.
- **IRAF/DAOPHOT** `phot` centroids the aperture centre from the approximate
  position.
- Space-mission pipelines do the same: ASTERIA computes flux-weighted first
  moments; Spitzer uses an iterative first-moment centroider.

**Directly relevant:** in photutils 3.0 - the version installed in VYVAR - the
`xpeak`/`ypeak` keywords with `search_boxsize`, which search a box for the
maximum pixel value, are **deprecated and scheduled for removal**. The
documented replacement is `centroid_sources` at specific positions. The pattern
VYVAR built is the one the library is retiring.

## 2. Saturation is tested over the object's own pixels

- **SExtractor** `FLAGS` bit 4 means at least one pixel **of the object** is
  saturated. The object is its isophotal footprint from segmentation - the
  pixels that belong to that source - not a search window.
- **STDWeb** follows the same structure: objects whose footprints contain
  saturated or cosmic-ray pixels are still detected, and flagged so later stages
  can ignore them.
- **AstroImageJ** reports linearity and saturation warnings for the pixels in
  the photometric aperture, and marks the star in the interface.
- **DAOPHOT** `datamax` excludes bad pixels from the sky aperture, the
  photometry aperture and the PSF fitting aperture - always relative to the
  aperture already placed.

In every case the pixel set is defined **first**, by detection or by aperture
placement, and saturation is a property measured **over that set**. There is no
step in which a bright pixel is located and then judged to be the star.

## 3. AstroImageJ solves exactly VYVAR's failure mode - and does not centroid

The AIJ manual describes this case directly. Centroiding is enabled or disabled
per aperture, and when the first aperture is centroided, **apertures with
centroiding disabled move from image to image according to the average movement
of the centroided apertures**. The stated reason is to allow apertures to be
placed around faint stars near bright stars, which would otherwise **capture the
aperture** if centroiding were enabled.

That is the measured VYVAR defect, named and solved: comparison star
`1497974027502858240`, flux about 24000 ADU, has a neighbour about 20 px away at
37000-61000 ADU. The search took the neighbour.

AIJ's answer is not a better search or a plausibility filter. It is: **do not
try to find that star independently. Move it with the frame.**

## 4. Whole-star rejection, decided once

Where saturation removes a star, it removes it for the whole dataset. In the
L/T-transition brown dwarf study, stars that saturate or exceed the 46000 ADU
non-linearity limit are rejected outright from the reference set. Frames are
recentred by whole-pixel shifts to a common centre defined from the first dither
position, preserving flux.

This matches `INV-COMP-MEMBERSHIP`: membership decided once, never per frame.

---

## 5. What this means for VYVAR

**VYVAR invented a search problem that this field does not have.**

The position of a comparison star on a frame is not unknown. It follows from the
astrometric solution plus the frame's collective drift, both of which VYVAR
already computes. The reconcile measurement using raw WCS plus a drift measured
from the target located BO CVn on **134 of 134** frames. The master-grid centroid
lock is the same idea already implemented elsewhere in the pipeline.

Once the position is given rather than discovered:

- there is nothing to hijack, because nothing is being searched for
- the anchor test, the brightness plausibility test and the peak self-check all
  become unnecessary - they exist only to check a search that should not happen
- the peak is simply the maximum over the aperture footprint at the known
  position, which is what SExtractor, AIJ and DAOPHOT all do
- faint stars stop being harder than bright ones, because neither is being
  hunted

**Recommended structure:**

1. Position from the per-frame aligned DAO grid (master-grid lock) plus optional
   11 px centre-of-mass refinement -- see **section 6 correction** (WCS plus
   uniform drift alone is insufficient on QHY raw).
2. Frame drift measured from stars that can be centroided reliably (AstroImageJ
   rule) -- diagnostic only on VYVAR implementation.
3. Peak and saturation measured over the aperture footprint at that position.
4. Saturation decided once per star per draft.

**What this removes:** `mag_guided_centroid` on target stars, the 45x45 search
window, `PEAK_ALIGNED_MAX_DIST_PX`, `PEAK_RAW_ALIGNED_MAX_RATIO`,
`PEAK_MIN_ADU`, the ring-contrast threshold, and the `fail_frac` /
`hijack_frac` / `raw_aligned_ratio` policy that was proposed to contain their
failures. None of it has an analogue in any tool surveyed.

**What this keeps:** everything else in SAT-DIAG - the derived ceiling, the
compatibility falsification, the provenance flags, the two levels, the tier
policy, the migration.

---

## 6. Implementation correction (2026-08-13, draft 510 BO CVn)

Section 5 step 1 as originally filed ("WCS plus collective frame drift") is
**insufficient on QHY294MM raw frames**. Measured WCS positional error on raw
reaches **10-15 px** relative to aligned DAO centroids; uniform target drift
does not recover faint-comp peaks (sky-level ADU at WCS+drift seed).

**Working form in VYVAR:**

1. **Primary placement:** per-frame aligned DAO `(x, y)` on the raw pixel grid
   (same grid as `detrended_aligned` after astroalign).
2. **Refinement:** 11 px centre-of-mass centroid (`photutils` default
   `box_size`); never brightest-pixel search on comparison or target stars.
3. **Drift diagnostic:** variable-target mag-guided offset at WCS (reconcile
   model) -- reported in `sat_diag.json`, not used as the comp placement seed.

Reconcile "134/134 target placement" used target drift for diagnostics; comp
peaks at **~5800 ADU** (not **~49000** hijack) require aligned DAO + COM, not
WCS+drift alone. This correction is authoritative over section 5 step 1 wording.

---

## References

- Bradley, L. et al., photutils documentation: `centroid_sources`,
  `centroid_quadratic`, `centroid_com`; deprecation of `xpeak`/`ypeak` and
  `search_boxsize` in 3.0
- Collins, K. A., Kielkopf, J. F., Stassun, K. G., Hessman, F. V. 2017, AJ 153, 77
  (AstroImageJ), and the expanded edition arXiv:1701.04817
- Howell, S. B. 2006, Handbook of CCD Astronomy (AIJ's default centroid algorithm)
- Bertin, E., Arnouts, S. 1996, A&AS 117, 393; SExtractor Flagging documentation
- Stetson, P. B. 1987, PASP 99, 191; IRAF `daophot.datapars`
- Karpov, S. 2024, STDWeb, arXiv:2411.16470
- Vos, J. M. et al. 2019, brown dwarf variability, arXiv:1910.02638
  (whole-star rejection above the non-linearity limit)
- Knapp, M. et al. 2020, ASTERIA photometry, arXiv:2005.14155
