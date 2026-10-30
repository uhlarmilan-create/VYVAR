# VYVAR - a bad pixel in the data and how it was solved (BADPIX-REPAIR, 2026-10-29)

Story for the conference: found in the step-4 A/B test on BO CVn, decided with Milan, measured on his own
data, implemented and verified. Decision record: docs/VYVAR_DECISIONS.md BADPIX-REPAIR-2026-10.
Figures: dev/results/badpix_repair/ (toi3919_edge.png, badpix_methods.png, toi3919_after.png).

## 1. The finding
- Night BO CVn 2026-04-23 (draft 527, 146 frames, bin 2). The light curve of the exoplanet host TOI-3919 lost
  27 of 146 points (frames 122-150, end of the night) as `edge_fail` ("the aperture touches pixels without
  data").
- Cause: one bad pixel of the master flat. Since G01-29 a bad flat pixel is NaN (missing) in the calibrated
  frame - never divided by a broken flat value. The cubic alignment warp turns the one missing pixel into a
  4x4 block. The pipeline then dropped any star whose LARGEST grid aperture (k = 2 x FWHM, 4.5-5.4 px)
  reached such a block - although the aperture actually used was 1.7-2.0 px and the bad pixel was 4.1-5.1 px
  from the star.
- The 27 points, re-measured with the chosen aperture, sit in the scatter of the curve: residual median
  -11 mmag, robust scatter 33 mmag (normal points: 35 mmag). Figure toi3919_edge.png.
- How common: bad flat pixels per frame (IRAF ccdmask rule, ~1 false mark per flat): BO CVn 275 (0.010 %),
  Brno g 23, Brno r 90.
- Milan: "nula nula nic" - MuniWin, AIJ, SIPS simply produce the curve; only a point that really jumps
  (satellite, hot pixel, cosmic) is removed. Keep the points, note the bad pixel; do it as accurately as
  possible, mathematically and physically correct.

## 2. What other software does
| software | bad pixel inside the aperture |
|---|---|
| SExtractor (MASK_TYPE=CORRECT) | replaced by the pixel mirrored through the object centre; measurement kept; FLAGS bit 1 only above 10 % bad pixels |
| SEP | replaced by the mean of the aperture; masked annulus pixels ignored |
| IRAF | repaired before photometry (`fixpix`: linear interpolation across the narrowest span); `phot` reports bad data as error 304 |
| MuniWin / AIJ / SIPS | no masking - the curve is produced |

## 3. The measurement (injection-recovery on Milan's data)
Brno 534, 10 calibrated frames (5 g, 5 r), 400 isolated unsaturated stars, apertures k = 0.75 / 1.0 / 1.5;
a pixel inside the aperture is declared missing, repaired by each method, the aperture flux measured again.
Error in units of the star's photon noise (sigma), 7164 cases per method:

| one missing pixel, native grid | bias | robust scatter | 95 % below | 95 % below at < 0.25 FWHM |
|---|---|---|---|---|
| aperture mean (SEP) | +0.03 | 0.49 | 4.5 | 15 |
| linear interpolation (IRAF fixpix) | -0.01 | 0.17 | 1.3 | 4.5 |
| mirror pixel (SExtractor) | -0.01 | 0.22 | 1.4 | 2.8 |
| Moffat fit | +0.01 | 0.16 | 1.0 | 2.0 |
| **frame ePSF fit** | **+0.003** | **0.12** | **0.42** | **0.72** |

The same pixel after the warp (4x4 block): even the ePSF 0.81 sigma (95 % < 6), the others 1.4-5.3
(95 % < 12-52); the mirror is impossible in 18 % of cases (its mirror lies in the block).
Two conclusions (figure badpix_methods.png): repair BEFORE the alignment (one unknown pixel, not 16), and
predict the pixel from the frame's own PSF - the only method that stays inside the photon noise even when the
bad pixel hits the star's core.

## 4. The solution in VYVAR
- QC in-place (after the frame FWHM, before alignment; same path in GUI and night run):
  missing pixel = local sky (4-6 FWHM annulus median, sources masked) + the ePSF models of the stars reaching it.
- Stars: DAO detections near missing pixels; flux and position fitted on the good pixels (neighbours jointly,
  non-linear pixels masked, centre within 1 FWHM of the DAO centroid). A missing pixel at a star's peak must
  not hide the star: detection sees it as the median of its neighbours.
- ePSF: from the frame's <= 100 brightest isolated stellar detections below 0.9 x the calibrated saturation
  (calibration stamps VY_CSAT = (SAT - pedestal) / max normalised flat).
- No silent fill: without VY_CSAT, with fewer than 30 build stars, or when a star's fit fails, its pixels stay
  missing (the point is dropped as before) and the reason is logged.
- Traceability: repaired pixels listed in the VYBPFIX extension of the calibrated frame, carried through the
  alignment to badpix_repaired.json, counted per aperture; the light curve keeps the point with the note
  `bad_pixel_repaired: N px` in flag_reason.

## 5. Verification of the production code
- Brno, same 10 frames, one missing pixel in every test star at once, 3 rounds, 15 659 cases:
  bias -0.004 sigma, robust scatter 0.125 sigma, 95 % < 0.43 sigma; > 1 sigma in 0.75 %, > 3 sigma in 0.15 %
  (very bright stars, S/N 400-2000: 0.2-1.4 %, ePSF core mismatch); 14 pixels (0.09 %) left missing (failed fit).
  A repaired point is 0.8 % noisier (sqrt(1 + 0.125^2)).
- The test found two bugs before delivery: a missing peak pixel hid the star (pixel got only the sky, up to
  4 sigma error) and a bright star fitted with a faint neighbour did not converge in 100 iterations.
- BO CVn full night (A/B against the baseline d29): 275 missing pixels per frame, 87-106 filled from star
  models, the rest from the sky; 1-3 pixels left in 5 of 147 frames (logged). TOI-3919: 146/146 points,
  all 27 recovered points normal, rms 35.60 mmag (baseline 35.63). Whole night: edge_fail points 56 -> 0,
  normal points 3975 -> 4000, median lc_rms change +0.05 %, variability candidates 0, known variables 12.
  Figure toi3919_after.png.
- Brno 534 full night: g 2254 missing pixels in 98 frames (1394 filled from star models, 859 from the sky,
  1 left), r 8820 in 98 frames (5855 / 2918 / 47 left); export file names identical to the baseline, median
  lc_rms change -0.02 % (g) / -0.1 % (r).
- Cost: preprocess BO CVn 142 -> 613 s, Brno 532 -> 1047 s on the 2-worker test machine (whole run +15 % /
  +1 %), mostly the per-frame ePSF build.
