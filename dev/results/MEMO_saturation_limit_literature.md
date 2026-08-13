# MEMO: How saturation and linearity limits are handled - literature and tools

**Date:** 2026-08-13
**Purpose:** Decision record for the `SATURATE_ADU` question, and a workflow
review of how VYVAR applies it.
**Question:** In what units is a saturation limit stored, how is its value
determined, and what is done with a star that crosses it?

---

## 1. The unit convention is settled: image ADU, not sensor-pixel ADU

Every tool examined expresses the limit in the units of the pixel values in the
file being measured.

- **SExtractor** (Bertin & Arnouts 1996): `SATUR_LEVEL` is documented as the
  level *in ADUs* at which saturation arises, with `SATUR_KEY` naming a FITS
  header keyword (default `SATURATE`) read from the image itself. Typical
  configured values are 50000-62000 for 16-bit data.
- **DAOPHOT / IRAF** `datapars.datamax`: users should either leave it undefined
  or set it to the linearity or saturation limit of the detector; it is used to
  detect and remove bad data from the sky aperture, to flag bad data in the
  photometry aperture, and to remove bad data from the PSF fitting aperture.
- **AstroImageJ** (Collins et al. 2017): linearity and saturation warning levels
  are entered in the Aperture Photometry Settings panel, in image counts.
- **FITSH** (Pal 2009): the oversaturation mask marks pixels whose ADU value is
  above a limit defined near the maximum of the A/D conversion, or below it when
  the detector is non-linear at high signal.

No tool stores a per-sensor-pixel well depth and scales it by binning at compare
time. **The option of storing 16384 as an unbinned value and multiplying by
`XBINNING * YBINNING` is not the convention anywhere.** It is also refuted by
the measured histogram (pile-up at 65535, not at 4 x 16383 = 65532).

## 2. Two levels are standard, not one

The tools separate **linearity** from **saturation**, and they are different
numbers:

- AstroImageJ carries both as distinct settings, and the reference-star panel
  highlights a star yellow when the linearity level is reached and red when the
  saturation level is reached.
- FITSH distinguishes oversaturation from blooming, and notes the limit may sit
  *below* the A/D maximum when the response is non-linear.
- DAOPHOT's `datamax` is explicitly "the linearity **or** saturation limit".

VYVAR currently has one scalar plus a hardcoded 0.85 fraction used as a proxy
for linearity. A fixed 85% is a convention, not a measurement. Real CMOS
linearity knees are device- and mode-specific and are found by testing.

## 3. The value is measured, and the stated value is routinely wrong

This is the strongest and most consistent point in the sources.

- **PSFEx documentation** (Bertin): the `SATURATE` keyword is commonly present
  in released images but is in practice often set higher than the level at which
  the detector begins to behave markedly non-linearly. The recommended procedure
  is to inspect saturated stars visually, check whether pixel values
  systematically exceed the stated level, and if they do not, point `SATUR_KEY`
  at a non-existent keyword and force a lower value.
- **VVV survey PSF photometry** (Surot et al. 2019): the ADU linearity values in
  the instrument manual did not match what was seen in the reduced images,
  because dark correction, flat-fielding, readout mode and frame combination
  shifted the baseline counts. They derived the limit from the images and
  applied a conservative margin below it.
- **DAOPHOT practice** (Virginia observing guide): it is tempting to take the
  saturation limit to be the amplifier's limiting count, but that is not the
  actual saturation limit of the stars. Working values around 60000 are used on
  65535-count systems.
- **AAVSO** advice for CCD photometry: measure the linearity of your own camera
  with an exposure ramp against mean pixel value; do not assume the full-well
  number. The example given puts a nominally 65535-count camera's knee near
  55000.

Applied here: the QHY294 sensor specification is not the operative number. The
operative number is the level at which *these* frames, from *this* rig, in *this
readout mode and binning*, stop responding linearly.

## 4. The ceiling is the smaller of two things

Andor's technical note on saturation states the general rule: depending on the
camera configuration, saturation may occur at either the pixel well or at the
A/D converter, whichever is reached first. With a high-sensitivity setting the
limit is set by the 16-bit ADC at 65535 counts rather than by the well.

The measured draft-510 histogram - 13024 pixels at exactly 65535, one pixel at
65532, and a saturated core with all nine of its pixels above 60000 sitting
exactly at 65535 - is the signature of a container ceiling, not of four summed
14-bit wells.

## 5. Binning changes the count scale, and that is documented practice

AAVSO's guidance on binning is explicit that a count in a binned image is not
the same quantity as a count in an unbinned image, that the gain may change with
binning, and that bias and dark frames should be taken separately for each
binning configuration.

For CMOS sensors the same guidance notes that binning is performed as arithmetic
addition of the digitised counts, and that with a fixed gain the summed value
can reach the ADC ceiling while each native pixel is only partly filled.

**Consequence for VYVAR:** the operative limit is a property of
`(camera, readout mode, binning)`, not of the camera alone. A single scalar in
`EQUIPMENTS` is structurally wrong regardless of what value it holds. It will be
correct for exactly one configuration and silently wrong for every other.

## 6. Flagging versus excluding

- **AstroImageJ** flags. The reference-star panel colours a star's checkbox
  yellow at the linearity level and red at the saturation level, and the
  observer decides what to do.
- **FITSH** sets mask bits, before calibration.
- **DAOPHOT** uses `datamax` to exclude bad *pixels* from the fit, not to remove
  a star silently from a comparison ensemble.
- Survey pipelines (KiDS, KMTNet) encode saturation in bitmask planes carried
  alongside the data.

The common structure is: **the limit produces a flag; a separate, visible policy
decides what the flag does.** VYVAR currently couples them - crossing the limit
directly removes a star from the comparison pool, from PSF fitting and from the
aperture-correction reference set. With a wrong scalar that coupling silently
thinned the pool from 140 to 78 and would have dropped 2 of the 5 comparison
stars that produced the good BO CVn light curve.

## 7. A workflow issue this raises, independent of the value

The saturation test must be applied to data that still has the detector's
original ceiling.

Draft 510's `detrended_aligned` frames contain values up to about 68567 - above
the 65535 raw ceiling - because the alignment step resamples with a bicubic
kernel, which is not bounded by the range of its input samples. Any threshold
test performed on resampled frames is therefore testing a quantity that no
longer has the detector's ceiling.

Both the reference tools and the survey pipelines determine saturation on the
raw or calibrated frame and carry the result forward as a flag. FITSH is
explicit that its saturation masking is done *before* any calibration.

**This is not hypothetical for VYVAR - it is confirmed.** `peak_max_adu` in the
proc and photometry tables is taken from the **aligned float frame**
(`pipeline.py:8050+`), not from raw. Aligned frames on draft 510 reach about
69000 while the raw ceiling is 65535. Every saturation decision in the pipeline
is therefore currently made against a quantity that has passed through a bicubic
resampling and no longer carries the detector's ceiling.

Correcting the stored limit without moving the measurement stage would leave the
comparison wrong in a different way: a correct ceiling of 65535 tested against
aligned peaks that can reach 69000 would flag stars that never saturated, and
could miss stars whose warped peak undershoots.

**Recommendation:** determine the saturated/non-linear status of every star on
the raw frame, record it as a per-star per-frame flag, and let every later stage
consult that flag rather than re-thresholding resampled pixel values.

---

## 8. Recommendation for VYVAR

1. **Store two levels, in image ADU, keyed by `(equipment, readout mode,
   XBINNING, YBINNING)`** - a linearity level and a saturation level. This is
   option B4. A single scalar per camera cannot be right across binning modes.

2. **Determine both empirically per rig configuration** with an exposure ramp,
   the AAVSO procedure: mean pixel value against exposure time, and the level at
   which the response departs from the fit by more than a stated tolerance. Store
   the measurement date and the tolerance used alongside the value, so a referee
   can see how it was obtained.

3. **Until that measurement exists**, resolve the ceiling per draft from the
   frames themselves - the observed pile-up level - and apply a conservative
   margin below it, following the VVV and DAOPHOT practice. Emit a warning
   stating that the limit is derived rather than measured, so it cannot be
   mistaken for a calibrated value.

4. **Separate the flag from the action.** Crossing a level sets a flag. What the
   flag does - exclude from the comparison pool, exclude from the aperture
   correction reference set, mark the epoch in the export - is a policy that
   should be visible and individually decidable, as in AstroImageJ.

5. **Test on raw pixels, never on resampled ones**, and carry the flag forward.

6. **Gate the peak measurement itself.** Every number above depends on peaks
   being measured on raw pixels at the correct sky position. Two measurements of
   BO CVn on the same raw frames disagreed by a factor 4.8 in the median - 17492
   against 3662 - because one used fixed master-grid x,y with no WCS and was
   sampling background on most frames. A fixed-position sample on raw data is
   not a saturation test; it is a background measurement that happens to return
   a number.

   Any automated peak measurement must carry a self-check that would reject
   that failure: the located peak must be a local maximum, its ratio to the
   surrounding ring must exceed a stated threshold, and the check must be
   reported per star per frame rather than assumed. A limit derived from
   ungated peak measurements is worth nothing regardless of how the limit itself
   is stored.

## Authoritative numbers as of 2026-08-13

Superseding the earlier draft-510 figures, which came from the ungated
fixed-position measurement:

| quantity | value |
|---|---|
| BO CVn, frames >= 16384 | 94 of 134 |
| BO CVn, frames >= 13926 (85% of 16384) | 126 of 134 |
| BO CVn, frames >= 65535 | 0 of 134 |
| comparison stars surviving admission at limit 16384 | 78 of 140 |
| comparison stars surviving admission at limit 65535 | 124 of 140 |
| draft-509 BO CVn comps failing admission at 16384 | 2 of 5 |
| draft-509 BO CVn comps failing admission at 65535 | 0 of 5 |
| raw pixel ceiling, measured | 65535, 13024 pixels |
| aligned-frame maximum | approximately 69000 |

The superseded figures were: median 3662, 61 of 150 frames above 16384.

## References

- Bertin, E., Arnouts, S. 1996, A&AS 117, 393 (SExtractor); SExtractor and PSFEx documentation, astromatic.net
- Stetson, P. B. 1987, PASP 99, 191 (DAOPHOT); IRAF `daophot.datapars` documentation
- Collins, K. A., Kielkopf, J. F., Stassun, K. G., Hessman, F. V. 2017, AJ 153, 77 (AstroImageJ)
- Pal, A. 2009, PhD thesis, arXiv:0906.3486 (FITSH saturation and blooming masks)
- Surot, F. et al. 2019, A&A (VVV bulge PSF photometry), arXiv:1907.01972
- de Jong, J. T. A. et al. 2015, A&A 582, A62 (KiDS bright-star masking), arXiv:1507.00742
- Andor Technical Note FAQ063, "Understanding CCD Saturation: Pixel Well Depth vs Bit Depth"
- AAVSO CCD school material on binning and on CMOS cameras for photometry, aavso.org
