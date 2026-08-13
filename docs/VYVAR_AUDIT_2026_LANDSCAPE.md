# VYVAR audit 2026 — Wave 5: photometry landscape

**Date:** 2026-08-13  
**Purpose:** Position VYVAR against the field for methods-paper justification — not a single-tool comparison.

Survey sources: project docs, published tool manuals, ADS abstracts, and maintainer pages (2024–2026 status). **Maintained** = release/commit within ~24 months or active forum.

---

## 5.1 Tool survey summary

### Variable-star / amateur photometry

| Tool | Users / status | Aperture & background | Comp selection | Ensemble | Errors | Saturation | Time | Transforms | Exports |
|------|----------------|----------------------|----------------|----------|--------|------------|------|------------|---------|
| **AstroImageJ (AIJ)** | Exoplanet + vars; maintained | User circles; annulus modes | Manual + optional auto | User ZP | Propagation optional | Flags | JD/HJD/BJD plugins | User coeffs | AAVSO native |
| **C-Munipack / Munipack** | European amateurs; maintained | Aperture CLI/GUI | Manual | Classic diff | Basic | Header-based | HJD | Filter-dependent | Custom |
| **MaxIm DL** | Commercial; maintained | Aperture wizard | Interactive | Per-set ZP | Estimated | Full-well warn | UTC | Transform wizard | AAVSO |
| **MPO Canopus** | Asteroid/vars; maintained | Aperture | Comp lists | Ensemble | RMS | Manual | UTC | Limited | Text |
| **Tycho Tracker** | Vars; niche | Fixed/aperture | Auto suggestions | Median ZP | Simple | Clip | UTC | None | CSV |
| **AstroArt** | Commercial | PSF + aperture | Manual | ZP | Basic | Warn | UTC | Photometry module | CSV |
| **Siril** | Stacking + phot; active | Recent photometry module | Manual | Stack-based | Minimal | Clip | UTC | None | CSV |
| **PixInsight** | Advanced; active | DynamicPSF + aperture | Manual | ImageIntegration ZP | NoiseGenerator | Linear fit | UTC | Spectrophotometry tools | Custom |
| **ASTAP** | Plate solve + phot | Aperture | Manual | — | — | — | UTC | — | CSV |
| **AAVSO VPhot** | Web; AAVSO official | Web aperture | User comps | Web ensemble | Web | Flags | BJD required | Standard | AAVSO |
| **LesvePhotometry** | Windows; exoplanet | AIJ-like | Auto comp search | Ensemble | Basic | — | HJD | — | AAVSO |
| **AIP4Win** | Legacy Windows | Aperture | Manual | — | — | — | UTC | — | Text |
| **Iris** | Legacy free | Aperture | Manual | — | — | — | UTC | — | CSV |
| **Prism / FotoDif** | Commercial niche | Aperture | Manual | Diff | Basic | — | UTC | — | CSV |

### Exoplanet-specific

| Tool | Notes vs VYVAR |
|------|----------------|
| **HOPS** | TESS-centric; auto comp; no cross-binning; no local Gaia DB |
| **EXOTIC / Exoplanet Watch** | Web pipeline; WCS + aperture; less comp-tier logic |
| **AIJ transit fitting** | De facto reference for TESS amateur; manual comp control |

### Research libraries / pipelines

| Tool | Notes vs VYVAR |
|------|----------------|
| **DAOPHOT/IRAF apphot** | Gold standard PSF; no GUI workflow; VYVAR uses photutils DAO subset |
| **SExtractor / SExtractor++** | Detection + MAG_AUTO; less ensemble LC orchestration |
| **sep** | Fast Python; mesh background; VYVAR xval agrees at 7.6 mmag comp RMS |
| **photutils** | Library; VYVAR xval **3 mmag** target agreement draft_510 |
| **PSFEx** | PSF modeling for weak lensing; VYVAR ePSF is photutils EPSFBuilder |
| **PHOTOMETRYPIPELINE / AutoPhOT** | Survey-oriented; less variable-star comp QA |
| **VaST** | Vars discovery; not full cal?export |
| **ISIS/HOTPANTS** | Difference imaging; VYVAR no diff-imaging path |
| **Tractor** | Probabilistic modeling; overkill for differential vars |

### Survey / mission pipelines

| Pipeline | Relevance |
|----------|-----------|
| **LSST Science Pipelines** | Rubin ap/PSF; extreme scale; VYVAR borrows no code |
| **TESS SPOC / QLP** | Systematics removal (SysRem-like); VYVAR optional SysRem |
| **Kepler PDC** | Cotrending; analogous to ensemble CM detrend |
| **banzai / DRAGONS** | Observatory reduction; similar cal steps; no VSX workflow |
| **Gaia DPAC** | Photometric standard; VYVAR uses Gaia as astrometric/photo prior |

**Added beyond task list:** **Siril** (growing amateur phot), **LesvePhotometry**, **SExtractor++**.

---

## 5.2 VYVAR positioning

### Conventional (cite only)

- Differential ensemble photometry with weighted comps (Honeycutt 1992; Broeg 2005).
- Astrometry.net-style plate solve + SIP distortion (Calabretta & Greisen 2002).
- Howell-style noise terms (with fixes I-11, P-02).
- BJD_TDB for ephemeris work (Eastman et al. 2010).
- Gaia DR3 for astrometry and colour priors.

### Unusual (needs explicit justification)

| Choice | Field norm | VYVAR | Justification status |
|--------|------------|-------|---------------------|
| **Cross-binning cal** (bin1 masters ? bin2 science) | Rare; most tools assume match | CAL-DIAG SUM/MEAN derivation | **Justified** INV-CAL-01; VYVAR-owned verification |
| **Software resample + block SUM dark** | Many use hardware bin only | Implemented | DOCUMENTED D1-3 |
| **Comp colour tiers + weights** | AIJ/VPhot: manual | Automated tiers in config | DECISIONS; needs paper paragraph |
| **SAT-DIAG raw placed aperture** | Most use reduced frame peaks | Raw grid lock 2026-08-13 | **New**; spec in SAT_DIAG_SPEC |
| **Per-target comp lists** | Often one comp set | Phase 0+1 per variable | CQ-3 design |
| **Headless night_run orchestration** | Rare in amateur tools | Full CLI chain | Operational advantage |
| **272-parameter registry** | Most tools GUI-only | Explicit provenance | CONFIG guides |

### Missing vs field standard

| Gap | Severity | Notes |
|-----|----------|-------|
| No interactive blink / manual epoch veto UI | LOW | QC dashboard partial |
| No difference imaging | INFO | ISIS/HOTPANTS use case |
| Linearity curve per sensor (D1-2) | MED | Deferred dome-flat ramp |
| COG/aperture correction applied (A-1) | MED | Identified not wired |
| WIDE-ERR wide rig error bars | MED | Before publication claims |
| PM correction (local DR3) | LOW | DR4 wait |
| BPM sidecars not observed | LOW | Path unclear |

### Ahead (paper-worthy)

- **CAL-DIAG v2 + INV-CAL-02** zero-config calibration integrity gates (no surveyed amateur tool equivalent).
- **INV-COMP-MEMBERSHIP** explicit policy post ZP-clip postmortem.
- **SAT-DIAG placed-aperture** raw authority contract.
- **Cross-binning verification** as first-class invariant.
- **Integrated trust flag + comp QA + sparse stats** in one export row.

---

## 5.3 End-to-end comparison (BO CVn, draft_510)

**Setup:** Same 134 aligned frames, same Gaia comp ids, xval harness aperture r=5.93 px (2× FWHM_est), annulus 8.9–13.9 px.

| Stage | Agreement | Dominant difference if non-zero |
|-------|-----------|--------------------------------|
| WCS/positions | Shared WCS from VYVAR | — |
| Centroid | photutils refit in xval | Sub-pixel |
| Background | Annulus vs mesh (sep) | ~2 mmag comp RMS |
| Aperture flux | DAO vs circular | < 3 mmag target |
| Ensemble | LOO median same definition | negligible |
| Differential LC | BO CVn RMS 0.1454 vs 0.1456 | **0.02%** |

**Acceptance criterion (pre-stated):** target RMS ? < 10 mmag; comp RMS ? < 5 mmag. **PASS.**

**AstroImageJ:** Not run in this environment (Java desktop). Milan should replicate with same comp stars, r?6 px, annulus ~9–14 px on exported aligned FITS.

**Stage attribution for residuals:** comp RMS photutils lower ? **background/centroid** stage; not ensemble or time.

---

## 5.4 Synthesis for referee

VYVAR is **conventional in differential photometry math** where xval demonstrates agreement with photutils/sep at the mmag level on the anchor field. It is **unusual in calibration integrity gates and cross-binning contracts** — deliberately, because no surveyed tool validates those paths automatically. Primary **residual risks** are A-1 aperture sizing (documented), wide-rig error budget (WIDE-ERR), and unmeasured linearity knee — not disagreement with standard differential flux definitions.

---

## Wave 5 closing

**Surprised:** No surveyed amateur package combines CAL-DIAG-style derived resample convention with stage-stamped FITS products.

**Could not determine:** Live maintenance status of Iris/AIP4Win; exact AIJ default comp auto algorithms.

**Next (Wave 6 — blocked):** Disposition of unwired UI, detect_outliers API cleanup, alignment_detection_sigma wire-or-remove.
