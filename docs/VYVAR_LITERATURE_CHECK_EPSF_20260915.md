# VYVAR - Literature check of the EPSF-XVAL / CORE findings (2026-09-15)

Purpose: verify, claim by claim, whether the conclusions drawn from
EPSF-XVAL-A2, SHAPE-01, CORE-01..04 and VAL-01 are supported by the
published literature before D-EPSF-XVAL-DOD-03 is written. Verdicts:
SUPPORTED / CORRECTED / OPEN. Architect: Claude. Static literature
check; no new measurement.

## C1 - Aperture photometry outperforms PSF fitting for bright, isolated
## stars; PSF fitting wins for faint / crowded stars. SUPPORTED.

- Howell 1989, PASP 101, 616: optimum (small) apertures maximize S/N for
  point sources; the CCD equation must be used; growth curves are needed
  to correct small-aperture fluxes.
- Sokolovsky et al. 2017, MNRAS 464, 274 (arXiv:1609.01716): with
  SExtractor + PSFEx, "for the brightest stars in the field, the aperture
  photometry is about a factor of 2 more accurate than PSF photometry",
  attributed to PSF-variation reconstruction accuracy. Direct precedent
  for VAL-01 max r = 2.33 on the bright end.
- Nardiello et al. 2015 (ground-based) and Libralato et al. 2016, MNRAS
  456, 1137 (K2): aperture better on isolated bright stars, PSF better
  on faint stars.
- Hartman et al. 2005 (NGC 6791, arXiv:astro-ph/0504487): for the
  brightest stars aperture photometry outperforms PSF fitting; PSF
  fitting shows a constant error term attributed to model errors.
- Irwin 1997 / Naylor 1998, MNRAS 296, 339: variance-weighted PSF
  fitting is equivalent to aperture photometry for bright stars.
  Consistent with CORE-04 K3 (uniform weights move VYVAR toward PSFEx).

Consequence: the bright-end precision deficit of the PSF path is the
expected behaviour of the method class, not a VYVAR-specific defect.
The justified domain of the PSF path is faint / crowded stars; the
science-method picker fallback to aperture on psf_fit_ok=False is the
literature-consistent design.

## C2 - Undersampled PSF (FWHM ~2.4 px) produces pixel-phase errors in
## PSF fitting whenever model != truth; the ePSF cure requires dithered
## data. SUPPORTED.

- Anderson & King 2000, PASP 112, 1360: pixel-phase error traced to
  inadequate PSF modelling; the ePSF is derived by iterating between the
  PSF and star positions using dithered exposures; ~100 stars per ePSF
  recommended; 3x3 spatial grid recommended for HST chips.
- Lauer 1999, PASP 111, 1434: total flux of an undersampled PSF can
  depend on where the centre falls within a pixel (intra-pixel
  sensitivity), up to 0.03 mag (WFC F555W); not corrected by flat-field;
  correctable with a well-sampled PSF from a dithered set.
- Simulation study (arXiv:2004.06253): pixel-phase error persists for
  Gaussian-model fits until FWHM > 2.5 px for a Moffat truth; an ePSF
  built with finer sub-pixel divisions removes it.
- CORE-03 T1 surface (72 mmag ptp over one pixel of phase; ~8 mmag per
  0.1 px in the live window) is the same phenomenon.

Consequence: the VYVAR single-frame, single-phase-per-star ePSF build
(67 stars, no dither) is below the literature recipe for undersampled
data. The aligned-frame design limits the exposure (~0.1 px phase
spread) but does not remove the mechanism. Intra-pixel sensitivity of
the CMOS sensors is unmeasured (OPEN, see C7).

## C3 - The R-P2 reading "sampling alone does not fix the phase
## component" is NOT supported; CORRECTED.

- RASTI 2025 "Strategies for accurate ePSF modelling on undersampled
  images" (arXiv:2512.16764): for an approximately Gaussian synthetic
  ePSF, an oversampling factor of three yields photometric and
  astrometric pixel-phase errors of order 0.1%.
- CORE-04 osamp 3/4 rebuilds were pathological (flux-scale +713 mmag,
  ringing, non-finite). That is a builder failure (EPSF-BUILD-OSAMP-01),
  not evidence about the physics. R-P2 must be re-read in the ledger as
  "not measured (builder)"; architect error 28.

## C4 - Per-star SNR-optimal apertures without a curve-of-growth
## correction put stars on different flux scales (audit D5-1). SUPPORTED.

- Howell 1989: growth curves introduced precisely to correct fluxes
  measured with very small optimum apertures.
- Stetson 1990, PASP 102, 932 (DAOGROW): the standard growth-curve
  method; assumes linear response so all point sources share the same
  enclosed fraction at a given radius.
- Chang et al. 2015 (M37, arXiv:1503.03375): optimum aperture shrinks
  with magnitude; per-frame growth-curve correction from isolated bright
  stars is applied to all objects afterwards.
- Dolphin 2000 (HSTphot): PSF-vs-aperture systematic offsets of 0.05 to
  0.15 mag across a magnitude range are a known artefact when the PSF
  or aperture system is not tied by growth curves; Stetson 1992
  reported up to 0.25 mag over ~6 mag.

Consequence: VAL-01 criterion 2 (PSF minus per-star-aperture offset
vs G: -236 to -548 mmag) measures the aperture path's missing CoG
correction, not the PSF path's accuracy. Criterion 2 must be re-based on
a common-scale reference: Gaia-transformed catalogue magnitudes, or a
growth-curve-corrected large aperture on isolated bright stars.

## C5 - Validation methodology. SUPPORTED (DoD-02 criterion 1 is the
## literature standard; DoD-01 was not).

- All comparative studies above report rms-vs-magnitude per method on
  constant stars (Nardiello 2015; Libralato 2016; Sokolovsky 2017;
  Hartman 2005). None validates one method by the rms of its difference
  from another method against an absolute bar; two different estimators
  on the same photons do not cancel photon noise.
- Reduced chi^2 > 1 as a systematic indicator (Goessl & Riffeser 2002,
  A&A 381, 1095): chi^2 growing with brightness is a model-mismatch
  signature; consistent with F5 / R-C0 and with the picker filtering on
  psf_fit_ok.

## C6 - Aperture correction ties PSF and aperture systems. SUPPORTED.

- Stetson 1990; Dolphin 2000; Cepheid Key Project papers: the accepted
  practice is to tie PSF magnitudes to a large-aperture system through
  DAOGROW-style growth curves on bright isolated stars. VYVAR has
  `cog_aperture_correction_enabled: False` and the ePSF path uses
  `psf_ac_policy = p4_none`. Both paths are therefore un-tied; this is
  the root of the criterion-2 result.

## C7 - OPEN items the literature flags that VYVAR has not measured

- Intra-pixel sensitivity of the IMX-class CMOS sensors under a 2.4 px
  FWHM (Lauer 1999 mechanism). Measurable from dithered data.
- Brighter-fatter / charge-sharing at high signal (Guyonnet et al. 2015;
  arXiv:1407.8280 for CCD PTC nonlinearity): flux-dependent PSF shape
  biases PSF photometry of bright stars; ties to audit D1-2.

## Recommendations carried into D-EPSF-XVAL-DOD-03

1. Criterion 1 (precision): rms-ratio vs aperture on constant stars,
   read per G bin; PASS domain = the PSF path's justified domain
   (fainter stars); bright-end admission handled by psf_fit_ok + picker.
2. Criterion 2 (accuracy): linearity vs Gaia-transformed catalogue for
   BOTH paths; aperture result recorded against D5-1.
3. Phase component: two literature-backed routes - correct osamp >= 3
   build (after EPSF-BUILD-OSAMP-01) or dithered Anderson-King build
   (TODO-A). Neither is required for the aperture science product.
4. D5-1 curve-of-growth correction becomes a prerequisite for any
   common-flux-scale use of the aperture path.

## References (ADS bibcodes)

Anderson & King 2000PASP..112.1360A; Lauer 1999PASP..111.1434L;
Howell 1989PASP..101..616H; Stetson 1987PASP...99..191S; Stetson
1990PASP..102..932S; Naylor 1998MNRAS.296..339N; Dolphin
2000PASP..112.1383D; Hartman et al. 2005AJ....130.2241H; Nardiello et
al. 2015MNRAS.447.3536N; Libralato et al. 2016MNRAS.456.1137L;
Sokolovsky et al. 2017MNRAS.464..274S; Goessl & Riffeser
2002A&A...381.1095G; Chang et al. 2015AJ....150...27C; Bertin 2011
(PSFEx) 2011ASPC..442..435B. Bibcodes for the 2020 and 2025 arXiv
items to be confirmed at citation time.
