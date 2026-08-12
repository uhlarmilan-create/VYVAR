CURSOR RESULT - 2026-08-12

What I did
Fixed per-frame QA / header FWHM measurement robustness after CR cleaning was
removed. No science-pixel clipping. FWHM is now the median of moment-FWHMs over
many star-like DAO detections (bright, unsaturated, isolated, extended).

## Findings

### Frame 57/62 cause
- Draft 508 DB: frame 62 FWHM=1.45 px, frame 58=2.39 px (physically impossible).
- Header ``VY_FWHM`` on calibrated lights was ~2.0 across the night.
- Root cause: ``_qc_fwhm_elongation`` used photutils ``SourceCatalog`` segmentation
  (npixels=8) as primary. After L.A.Cosmic removal, ~83% of those "sources" were
  hot-pixel / cosmic islands with FWHM~1.6-2.5 px. Median collapsed to ~2 px.
- That poisoned (1) QA dashboard FWHM, (2) ``VY_FWHM`` used for aperture =
  factor x FWHM x 0.667, (3) masterstar pick (lowest VY_FWHM).

### Was FWHM from one detection?
- Intended path already took a median of many detections, but the *membership*
  of that sample was wrong (segmentation islands, not stars).
- New path: median over typically 50-80 star-like DAO detections per frame.

### Masterstar Frame 70 vs Frame 10
- Selection logic unchanged: copy the candidate with lowest ``VY_FWHM`` among
  ``masterstar_best_of_n``.
- Not a migration regression in the picker itself. With CR-poisoned ~2 px headers,
  ranking was noise, so 508 could prefer Frame 70 (5.13) while 435 preferred
  Frame 10 (5.08). Robust FWHM should restabilize selection on true sharpness.

## FWHM definition (new)
``_robust_frame_fwhm_median`` / ``_qc_fwhm_elongation`` / QA inspection:
1. Center-crop, background-subtract.
2. DAOStarFinder (kernel FWHM hint 4.5 px; not CR-scale).
3. Keep detections that are unsaturated, roundish (elong 0.75-1.55), isolated
   (>=10 px from brighter accepted), and extended (peak/sum <= 0.22) -- rejects
   hot pixels/CRs without sigma-clipping the FWHM list.
4. Frame FWHM = median of those moment-FWHMs (half-cutout 7 px). No sigma-clip.

Catalog matching is not available at calibrate/QC time; star-like DAO membership
is the pre-catalog equivalent of "many bright unsaturated isolated stars".

## Verification on draft 508 calibrated lights (recompute)
- n=150/150 finite
- median 5.17 px, std 0.083 (was std 0.46 with lows at 1.45)
- frame 62: 5.14 (was 1.45); frame 58: 5.37 (was 2.39); frame 7: 5.14

Fresh draft re-run still required so DB + headers + apertures pick up the new
estimator.

## Guards
- tests: ``dev/tests/test_robust_frame_fwhm.py`` PASS
- --fast: see session run
- ASCII-only; commit+push

## Still pending Milan (unchanged from clip-removal)
Borderline hard gates (max_comp_rms, p2p 0.10, slope, bpm_dark_mad_sigma,
frame_align_residual_gate) and unclipped comp_rms definition.
