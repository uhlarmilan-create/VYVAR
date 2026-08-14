# VYVAR audit 2026 ù Wave 7 closure

**Date:** 2026-08-14  
**Audience:** JAAVSO referee / methods reviewer  
**Authority:** Milan authorized measurement and documentation; no push.  
**Evidence:** `tmp/wave7_reexport_510_results.json`, `tmp/wave7_ee_gradient_510.json`, `tmp/xval_out_wave7/xval_results.csv`, `tmp/wave7_draft435_inventory.json`, `dev/results/CURSOR_RESULT_a1_growth_curves.md`, prior waves in `docs/VYVAR_AUDIT_2026_*.md`.

This document closes the August 2026 full-workflow audit. It states what was verified by measurement, what was not, which choices remain judgement calls, and the residual risk. The July 2026 science audit (`docs/VYVAR_AUDIT_FINAL.md`) remains authoritative for items fixed before this run; here we add the 2026 workflow audit and the A-1 re-measurement after decision (2).

---

## 1. A-1 post-change measurements (draft 510)

**Method:** Re-export per-frame proc catalogs (`export_per_frame_catalogs`, 135 sidecars, 432 s) then Phase 2A with INV-DAG-01 bypass (`run_phase2a`, 54 s). Pre-change proc CSVs backed up to `tmp/wave7_proc_bak_510/`.

### 1.1 Pre-registered predictions (re-measured)

| ID | Criterion | Before (GAUSS SNR FWHM) | After (DAO median FWHM + re-export) | Result |
|----|-----------|-------------------------|-------------------------------------|--------|
| **P1** | Target aperture radius increases; EE@production rises | `aperture_px` **4.141**; BO CVn EE **85.5%** at 4.141 px (measured gradient run) | `aperture_px` **4.261**; BO CVn EE **86.3%** at 4.261 px | **PASS** |
| **P3** | Check-star scatter ? baseline +10% (? **0.0095**) | **0.008629** | **0.008638** (+0.11%) | **PASS** |
| **P5** | Saturation admission still passes at new radius | Max comp median peak **15?856 ADU** vs admission **45?874.5 ADU** | Unchanged (same peaks) | **PASS** |

**P5 note (measurement, not assumption):** `peak_max_adu` is a 7ù7 pixel box maximum at the placed centroid on **raw** pixels (`sat_diag.apply_raw_peaks_to_proc_df`), not a peak within the photometry aperture radius. The radius change therefore does not move P5 peaks. P5 confirms no new saturation **rejections** in the comp gate; it does **not** test aperture-dependent peak growth.

**Trust:** GREEN, 5 clean comps, 134 frames (unchanged membership).

### 1.2 Cross-validation at new proc radii

**Harness:** `src_py/xval_run.py`, draft 510, 134 aligned frames, 672 Gaia sources, same LOO differential method as production.

| Metric | Before re-export | After re-export | photutils | sep |
|--------|------------------|-----------------|-----------|-----|
| BO CVn target RMS | 0.1454 | 0.1454 | 0.1456 | 0.1448 |
| BO CVn comp RMS (dao) | 0.0105 | **0.0111** | 0.0105 | 0.0100 |
| Median comp RMS (all targets) | dao **0.0102** | dao **0.0101** | **0.0078** | **0.0076** |
| Median \|target phot ? VYVAR\| | 0.0030 mag | **0.0032 mag** | ù | ù |

**Interpretation:** The VYVARùphotutils **target** agreement remains **?3 mmag** ù the strongest independent validation in this audit. The comp-RMS gap (VYVAR dao ~10 mmag vs photutils/sep ~7.6ù7.8 mmag) is **unchanged** in median; photutils/sep use a fixed **3 px** aperture in the harness, not VYVAR SNR radii, so only `comp_rms_dao` reflects the radius change (+0.9 mmag on BO CVn comps).

### 1.3 Enclosed-energy gradient at production radius

**Method:** Median curve-of-growth over 134 frames per star; production annulus sky (r_in=10, r_out=15 px); radii 0.5ù15 px step 0.25 px (`tmp/wave7_ee_gradient.py`).

| Quantity | At r = 4.141 px (old) | At r = 4.261 px (new) |
|----------|----------------------|----------------------|
| BO CVn EE | **85.5%** | **86.3%** (+0.8 pp) |
| Comp median EE | **85.8%** | **86.5%** (+0.7 pp) |
| EE gradient (BO CVn) | **6.57 pp / px** | **6.36 pp / px** |
| Comp median gradient | **6.76 pp / px** | **6.56 pp / px** |

**Consequence class:** At ~6.5 pp per pixel, the +0.12 px radius step adds ~**0.8 pp** enclosed flux ù **marginal** for differential photometry (scatter moved 0.11%), not the step change required to reach a 90% EE target (~5.0ù5.75 px on typical comps per growth-curve closure).

---

## 2. Draft 435 ù on-disk state and re-cut scope (report only; no re-cut)

Measured inventory (`tmp/wave7_draft435_inventory.json`):

| Location | Science `BO_CVn_Light_*.fits` | `proc_*.fits` | `proc_*.csv` |
|----------|--------------------------------|---------------|--------------|
| `calibrated/lights/NoFilter_60_2` | **150** | 0 | 0 |
| `processed/lights/NoFilter_60_2` | 0 | **139** | 0 |
| `detrended_aligned/lights/NoFilter_60_2` | **0** | **139** | **139** |
| `platesolve/NoFilter_60_2` | MASTERSTAR only | 0 | 0 |

**Also present:** `aperture_snr_table.json` (FWHM **2.395 px**, stack Gaussian authority), `cal_diag.json`, full photometry tree (**167** light curves), MASTERSTAR + Gaia match CSV. **Absent:** `sat_diag.json`, `per_frame_catalog_index.csv`, `pipeline_meta.json`.

**What a re-cut would require:**

1. Regenerate SNR table with DAO FWHM authority (expected FWHM ~3.3ù3.5 px from per-frame estimates vs current 2.395 px).
2. **Regenerate aligned science FITS** ù the aligned tree has proc sidecars only; science pixels live in `calibrated/` (150 frames). Re-export catalogs as on 510 requires either a full align pass from calibrated lights or a verified proc-FITS-only export path (not exercised here).
3. Re-export proc CSVs, re-run Phase 2A, refresh trust/LCs.
4. If Milan approves anchor update: `session_baseline_check.py --full` and new `VL-P1-GOLD` photometry SHA (INV-ANCHOR-00 scope: photometry-only gate).

**Anchor / INV-ANCHOR-00 impact:** `--full` copies frozen snapshot inputs and runs **photometry only**. A 435 re-cut changes LC/trust/check-scatter fingerprints and fails the photometry SHA gate until the ledger is updated; it does **not** by itself invalidate INV-CAL-01 byte-identity on calibrated FITS unless cal/preprocess/align are re-run from raw.

---

## 3. Architecture questions (measurement and literature)

### 3.1 Per-draft, per-frame, or fixed sizing?

**VYVAR choice (decision 2, implemented):** per-**draft** median of per-frame DAO moment FWHM for SNR table authority; target and comp radii still move together within an epoch (D5-1).

| Tool / pipeline | Aperture sizing | Varies with per-frame seeing? | Differential preservation |
|-----------------|-----------------|------------------------------|---------------------------|
| **DAOPHOT / IRAF PHOT** | Fixed radii in `PHOTPARS`; optional scale in FWHM units (Stetson 1987; IRAF `photpars`) | Not per star; manual re-run if seeing changes | Fixed r for all stars in a run ? standard differential practice |
| **SExtractor** | `MAG_AUTO`: Kron ellipse ~2.5ù r_Kron (Bertin & Arnouts 1996); floor `PHOT_AUTOAPERS` | **Per object** adaptive ellipse | Differential use requires same algorithm on target and comps; crowding-sensitive |
| **AstroImageJ** | Fixed radii default; optional **per-image** r = FWHM factor ù mean FWHM of apertures, or radial-profile cutoff (Collins et al. 2017, AJ 153, 77) | **Per frame** when enabled; warns against crowded fields | All apertures in an image scaled together ? preserves differential intent |
| **photutils** | User-supplied fixed `CircularAperture(r)` (docs v3.0) | Only if caller changes r each frame | Caller responsibility |
| **VaST** | Default: diameter = **6 ù median(A_IMAGE)** from preliminary SExtractor pass (Sokolovsky et al. 2018, ASPC 11, 7) | **Per image** | Same diameter applied to all sources in that image |
| **C-Munipack** | First aperture radius > FWHM from `munipack find -f` (tutorial) | Per frame via `-f` on each image | Growth-curve path available for normalization |
| **Survey pipelines (ZTF etc.)** | PSF / aperture-model photometry on stacked/calibrated products; field PSF from bright stars | PSF model updated per CCD/night | Differential not primary product |

**Conclusion:** Per-**frame** seeing adjustment is common (AIJ, VaST, SExtractor auto) but applied **uniformly to all stars in that frame**. VYVARùs per-**draft** median is more conservative (no frame-to-frame radius jitter); it matches the Honeycutt (1992) requirement that target and comp share the same aperture geometry within an epoch. No surveyed tool varies target and comp radii **independently** within a frame.

### 3.2 Is fixed enclosed-fraction (decision 4) the right end state?

**Evidence for:** Growth curves measure production EE **81ù86%** (510) and **67ù73%** (435) vs a 90% literature target; decision (2) moved 510 by only **0.8 pp** EE ù the FWHM authority fix alone does not close the EE gap. Fixed r?? from measured curves would remove the FWHM-estimator debate and set an explicit flux scale (Stetson 1990 COG; SExtractor Kron ~90% intent).

**Evidence against / cost:** Requires per-star or per-field growth-curve measurement and storage; re-export of all proc catalogs; anchor re-cut on 435 and 510; saturation/comp selection re-validation; possible crowding interaction when r grows. Implementation cost: mediumùhigh (new sizing mode + provenance + tests), comparable to a second A-1 arc.

**What decision (2) does not settle:** (a) per-magnitude SNR radii still differ across comps; (b) draft-to-draft EE inconsistency (435 underradiused); (c) absolute-magnitude / publication claims without COG; (d) comp-RMS gap vs photutils (background/centroid, not FWHM alone).

**Recommendation:** Fixed 90% EE is the **likely next architectural step** if publication claims require stated enclosed fraction or 435/510 parity; it is **not mandatory** for differential LC quality on 510 given xval and check-scatter evidence after (2).

---

## 4. Stage-by-stage closure (referee table)

### 4.1 Method, citation, verification, result

| Stage | Method (citation) | Verification performed | Result |
|-------|-------------------|------------------------|--------|
| **Import / manifest** | FITS copy + DB manifest | Wave 1 inventory trace; draft 510/435 disk audit | **PASS** contract; F-B01/F-B02 unverified on all rigs |
| **Calibration** | Dark SUM / flat MEAN resample (Howell 2006); CAL-DIAG v2 gate | INV-CAL-01: draft 435 **150/150** pixel-identical; draft 510 cal_diag PASS | **PASS** with `sigma_p=0` degeneracy edge (INV-CAL-01) |
| **Preprocess / sky** | 2D sky surface in-place on `calibrated/` | P-10 fixed; INV-CAL-02 stage stamp; no CR clean (deliberate) | **PASS**; in-place naming cost documented |
| **SAT-DIAG** | Placed-aperture raw peak; admission 70% sat (spec) | draft 510 `sat_diag.json`; comp peaks ? threshold after A-1 | **PASS** admission; linearity knee **unmeasured** (DEFAULT_FRAC) |
| **Plate solve / align** | TAN+SIP (Lang et al. 2010; Calabretta) | draft 510 WCS on 134 frames; alignment_report | **PASS** on anchor |
| **Detection / MASTERSTAR** | DAOStarFinder (Stetson 1987); N_equiv=3.78 | T4-1 fixed batch E; MASTERSTAR QA on 510 | **PASS** |
| **Aperture photometry** | Circular aperture + annulus sky (Howell 1989); SNR table | **xval:** target **3 mmag** vs photutils, 134 frames; growth curves; A-1 re-export P1/P3/P5 | **PASS** differential; EE **86%** not 90% |
| **Ensemble / ZP** | Broeg et al. 2005 weighted median; ZP MAD clip removed | check_scatter **0.00864**; ZP-clip postmortem (509 vs 435) | **PASS** on anchor |
| **Differential LC** | LOO comp median (Honeycutt 1992) | BO CVn lc_rms 0.145; xval target match | **PASS** |
| **Error budget** | Howell 1989 + scintillation (batch D) | WIDE-ERR diag on wide rig | **PASS** anchor errs; **OPEN** wide rig ~2ù underquote |
| **Time** | astropy LTT ? HJD/BJD_TDB (Eastman et al. 2010) | BJD columns in proc CSV; export gate | **IMPLEMENTED**; spot xval not re-run |
| **Trust / export** | Trust flags; AAVSO requires BJD_TDB | draft 510 GREEN; C-EXPORT-GAP: headless skips AAVSO | **PASS** trust; export path manual |

### 4.2 What was checked; what was not; judgements; residual risk

**Checked by measurement:** draft 510 full re-export + Phase 2A after DAO FWHM authority; xval photutils/sep; EE gradient; draft 435 disk inventory; CAL-DIAG byte identity (prior session); check-star scatter; saturation admission peaks.

**Not checked (and why):**

- End-to-end re-run from raw on anchor (`INV-ANCHOR-00` boundary ù by design of `--full`).
- Draft 435 A-1 re-cut (no aligned science FITS; Milan report-only).
- Dome-flat linearity curve (D1-2 ù requires observing program).
- Gaia transform coefficients hand-verification (OSC/mono not exercised).
- AAVSO/VarAstro export on headless path (C-EXPORT-GAP).
- Independent centroid/Background2D spot suite (partially embedded in xval only).

**Judgement calls (alternatives stated):**

| Choice | Alternative | Risk if wrong |
|--------|-------------|---------------|
| Per-draft DAO FWHM for SNR table | Per-frame AIJ-style; fixed r??; stack Gaussian | Residual EE bias draft-to-draft (435) |
| No CR rejection on science pixels | van Dokkum (2001) L.A.Cosmic | Occasional outlier epochs; not measured rate on 510 |
| Comparison membership once per draft | Per-frame variable comp sets | Missed intrinsic variables; mitigated by stability gates |
| Broeg weights, no ZP MAD clip | Clip at N?4 (509 regression) | Clip caused 509 failure; removed |
| In-place `calibrated/` mutation | Separate per-stage products | Provenance confusion (INV-CAL-02 mitigates) |

**Residual risk:** Wide-rig error bars (WIDE-ERR); 435 EE undersizing if A-1 not re-cut; absolute photometry / COG if referee asks for enclosed fraction; pedestal not in headers (U-PED-01).

### 4.3 Deliberate departures from field convention

1. **No cosmic-ray rejection on science pixels** ù against van Dokkum (2001); justified because resampled wide-field cores were destroyed by L.A.Cosmic (2026-08-12 removal). Science pixels are not altered; outliers handled statistically.

2. **Cross-binning calibration with derived convention** ù CAL-DIAG SUM dark resample; no surveyed tool verifies the full VYVAR path; INV-CAL-01 gate + 150/150 byte identity on anchor.

3. **Comparison-star membership decided once per draft** ù after ZP-clip postmortem (509 vs 435); Broeg ensemble kept; variables excluded by stability/RMS, not per-epoch LOO membership.

4. **Saturation measured on raw pixels at placed aperture** ù not at photometry radius; admission 70% of full well; linearity tier uses DEFAULT_FRAC unless measured.

5. **In-place mutation of `calibrated/`** with stage stamping ù no separate immutable cal product; cost: naming (`calibrated/` holds skysf-enriched product), INV-ANCHOR-00 cannot see preprocess regressions via photometry-only gate.

### 4.4 Open items after Wave 7

| ID | Status after section 1 | Notes |
|----|------------------------|-------|
| **A-1** | **Partially closed** | Decision (2) implemented + verified on **510**; 435 not re-cut; EE still <90% |
| **WIDE-ERR** | **OPEN** | Wide-rig err ~2ù underquote; fluxes OK |
| **D1-2 linearity** | **DEFERRED** | No dome-flat ramp |
| **U-PED-01 pedestal** | **OPEN** | Header OFFSET=0 vs ~24.5 ADU/bin1 in CAL-DIAG |
| **INV-CAL-01 ?_p=0** | **OPEN (edge)** | Gate PASS with zero intercept uncertainty |
| **SAT-DIAG linearity** | **OPEN** | One-sided compatibility vs measured knee |
| **INV-ANCHOR-00** | **OPEN (documented)** | `--full` photometry-only coverage gap |
| **C-EXPORT-GAP** | **OPEN** | Headless omits AAVSO/VarAstro |
| **U-P5-PRED** | **DOCUMENTED** | P5 tested peak at placed centroid (7 px box on raw), not at photometry radius; register Wave 7 |
| **U-XVAL-COMP-RMS** | **OPEN** | ~2 mmag VYVAR vs photutils comp RMS unexplained; settle with matched-radius xval |

### 4.5 Limits of this audit (five statements)

1. **Single-rig deep validation** ù Numerical closure is anchored on BO CVn / QHY294MM home rig (drafts 435/509/510). Other equipment IDs are not re-measured end-to-end in this audit.

2. **Photometry-only regression gate** ù `session_baseline_check --full` does not exercise cal, preprocess, align, or MASTERSTAR rebuild (INV-ANCHOR-00); preprocess or DAO regressions can slip until a full rebuild harness is run.

3. **Independent validation is differential-first** ù xval confirms target LCs to ~3 mmag; comp scatter offsets (~3 mmag vs photutils) and absolute EE fraction are not fully reconciled.

4. **Literature matrix is not exhaustive** ù Tool survey (ù3.1) covers major packages cited in time-domain photometry papers, not every pipeline (e.g. MaxIm, MuniWin variants, proprietary survey codes).

5. **Re-cut authorization boundary** ù Measurements on draft 510 post-change do not automatically update P1 golden SHA, draft 435 products, or publication numbers until Milan authorizes anchor re-cut and ledger update.

---

## 5. Referee judgement

**Can VYVAR differential photometry be defended in a refereed paper?**

**Yes, with stated scope**, on the evidence gathered:

- Independent photutils cross-validation on **134 frames** shows target differential RMS agreement at **?3 mmag** ù strong evidence the LOO ensemble path is implemented correctly.
- Check-star scatter **0.0086 mag** (GREEN) survived a measured +2.9% aperture increase with no saturation admissions.
- Remaining gaps are **declared**, not hidden: enclosed fraction ~**86%** not 90%; wide-rig error bars; 435 not updated; no CR cleaning; pedestal/header mismatch.

**Not yet defensible without further work:**

- **Absolute** photometry or claims requiring known enclosed fraction ? need COG or fixed r?? (decision 4) plus 435 re-cut.
- **Wide-field / wide-rig error bars** in publication ? WIDE-ERR fix and re-validation.
- **Linearity near saturation** ? D1-2 measurement, not DEFAULT_FRAC.
- **Cross-draft comparability** after A-1 ? draft 435 re-cut per ù2.

---

## Evidence index

| Artifact | Path |
|----------|------|
| Re-export + Phase 2A | `tmp/wave7_reexport_510_results.json` |
| EE gradient | `tmp/wave7_ee_gradient_510.json` |
| xval post-change | `tmp/xval_out_wave7/xval_results.csv` |
| Draft 435 inventory | `tmp/wave7_draft435_inventory.json` |
| Growth curves (pre-change) | `tmp/a1_growth_curve_results.json` |
| A-1 implementation | `dev/results/CURSOR_RESULT_a1_snr_dao_authority.md` |
| Audit map / register | `docs/VYVAR_AUDIT_2026_MAP.md`, `docs/VYVAR_AUDIT_2026_REGISTER.md` |
| Science synthesis | `docs/VYVAR_AUDIT_2026_SCIENCE.md` |

*Wave 7 closed 2026-08-14. No git push.*
