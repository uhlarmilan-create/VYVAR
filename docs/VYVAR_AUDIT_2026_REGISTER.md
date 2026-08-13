# VYVAR audit register ù 2026 full workflow audit

**Started:** 2026-08-13  
**Method:** Data-flow and contract audit (supersedes formula-only July audit).  
**Format:** `ID | wave | stage | class | severity | evidence | reference | disposition | status`

Classes: **P** physics/method wrong ù **I** implementation ? method ù **C** contract (units, naming, gate condition) ù **U** correct but unmeasured/unjustified ù **D** dead code/duplication

---

## Carry-forward from July 2026 closure (current status)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| A-1 | 4 | aperture | U/P | HIGH | Moment FWHM ~5.1ù5.3 px vs MASTERSTAR Gaussian 2.4ù3.3 px (ù1.6ù2.1); SNR table follows smaller; EE~54% if moment correct | VYVAR_AUDIT_FINAL ù5; STATE 2026-08-13 | DOCUMENTED mechanism; COG fix deferred | OPEN |
| A-9 | 4 | PSF | U | MED | PSF scale estimators disagree 2.4ù4.9 px; not blocking differential | closure register #31 | DOCUMENTED | OPEN |
| T4-1 | 4 | detection | I?FIXED | ù | N_equiv=3.78 wired E.4 | closure #10 | FIXED 2026-08-04 | CLOSED |
| P-02 | 4 | errors | I?FIXED | ù | Scintillation wired batch D | closure #25 | FIXED | CLOSED |
| I-11 | 4 | errors | I?FIXED | ù | Howell sky term batch D | closure #21 | FIXED | CLOSED |
| I-04 | 4 | ensemble | I?FIXED | ù | err_scatter unmatched epochs | closure #22 | FIXED | CLOSED |
| I-03 | 4 | errors | U | LOW | Legacy Howell terms unused on anchor | closure #23 | DOCUMENTED | OPEN |
| D10-1 | 4 | transforms | C?FIXED | ù | unfiltered?CR band | closure #18 | FIXED Milan decision | CLOSED |
| WIDE-ERR | 4 | errors | I | MED | Wide-rig quoted err ~2x underquoted; Honeycutt SEM path | wide_error_diag.md; LIMITATIONS | OPEN pre wide submission | OPEN |
| D1-2 | 4 | calibration | U | MED | Linearity curve deferred | closure #24 | DEFERRED dome-flat ramp | DEFERRED |
| D5-2 | 4 | saturation | C?FIXED | ù | admission_sat_peak_frac=0.70 | batch E | FIXED | CLOSED |
| ZP-CLIP | 3 | ensemble | C?FIXED | HIGH | `len(z)>=4` MAD clip rejected good comp | draft 509; DECISIONS ZP-CLIP-REMOVAL | REMOVED 2026-08-12 | FIXED |
| SATURATE_ADU | 3 | calibration | C | HIGH | DB value 16384 wrong units (binned?bin1) | database.py:2854; SAT-DIAG | NULLed QHY294MM; SAT-DIAG derives | FIXED |
| calibrated/ naming | 3 | preprocess | C | MED | Two-stage product in `calibrated/`; preprocess in-place | pipeline.py:18027; night_run Step 11 | INV-CAL-02 stamps stage | PARTIAL |
| INV-CAL-01 | 3 | calibration | C | MED | sigma_p=0 degenerate on some bins | cal_diag.py; inv_cal01_validate | Wired FAIL gate | OPEN (edge) |
| INV-DAG-01 | 3 | pipeline | C | MED | Re-stamp friction blocks photometry re-run | invariants_runtime.py:494; STATE | No fix | OPEN |
| F-B01/F-B02 | 3 | import | C | LOW | PASSTHROUGH records wrong CALIBRATION_MODE | calpath_audit.md | UNVERIFIED fix order | OPEN |

---

## New findings (2026-08-13 overnight audit)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| C-ALIGN-01 | 1 | config | C | MED | `alignment_detection_sigma` wired in align path | pipeline.py:14270 | FIXED 2026-08-13 | FIXED |
| C-TRUST-01 | 3 | trust | C | MED | trust uses `resolve_apply_color_term` | trust_flag_core.py:402 | FIXED 2026-08-13 | FIXED |
| C-P2P-01 | 2 | comp QA | C | MED | p2p ceiling reads `phase01_comparison_max_comp_rms` | photometry_core.py:3015 | FIXED 2026-08-13 | FIXED |
| C-MAX-RMS-01 | 2 | phase01 | C | LOW | comp_pool signature default aligned to 0.1 | comp_pool_rms.py | FIXED 2026-08-13 | FIXED |
| U-PED-01 | 3 | calibration | U | MED | Header OFFSET=0 vs ~24.5 ADU/bin1 pedestal measured CAL-DIAG | cal_diag.json draft 510 | CAL-DIAG derives P; header silent | OPEN |
| U-FWHM-XVAL | 4 | aperture | U | LOW | xval FWHM 2.96 px on aligned frame vs SNR table 2.395 px | xval_run draft 510 2026-08-13 | Third estimator; needs reconcile | OPEN |
| U-ANCHOR-GAP | 2 | QA | U | MED | INV-ANCHOR-00: `--full` never exercises cal/preprocess/align/MASTERSTAR | VYVAR_INVARIANTS.md | Documented boundary | OPEN |
| D-UI-FIN | 1 | UI | D | LOW | `ui_finalization.render_finalization()` unwired | ui_finalization.py:132 | Untested not dead | OPEN |
| I-DETECT-OUT | 2 | outliers | I | LOW | `detect_outliers` survives on variable targets; sigma clip params ignored (zero-clipping) | photometry_core.py:4529 | Partially dead API surface | OPEN |

---

## Wave 3 ù contracts (C and I)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| C-CAL-STAGE | 3 | preprocess | C | MED | `calibrated/` holds cal+skysf product; name implies "done" after step 1 | pipeline.py:18027; MAP ù2 | INV-CAL-02 mitigates | PARTIAL |
| C-OFFSET-HDR | 3 | headers | C | MED | FITS OFFSET=0 vs measured P?24.5 ADU/bin1 | cal_diag draft 510; U-PED-01 | CAL-DIAG authoritative | OPEN |
| C-GAIN-RN | 3 | equipment | C | MED | QHY294MM RN 7.6 vs 15.2 e- double-count suspected | STATE item 8 | UNVERIFIED | OPEN |
| C-SAT-UNIT | 3 | equipment | C | HIGH | SATURATE_ADU 16384 was binned in bin1 column | database.py:2854 | NULL + SAT-DIAG derive | FIXED |
| C-EXPORT-GAP | 3 | export | C | LOW | night_run omits AAVSO/VarAstro; UI/manual only | night_run.py vs export_reports.py | Document workflow | OPEN |
| I-TRUST-COLOR | 3 | trust | I | MED | bool("off") for apply_color_term | trust_flag_core.py:402 | FIXED (C-TRUST-01) | FIXED |
| I-ALIGN-SIGMA | 3 | alignment | I | MED | Settings sigma unused | C-ALIGN-01 | FIXED | FIXED |
| C-TIME-BASE | 3 | export | C | ù | Non-BJD export refused | export_reports.py | FIXED T1 | CLOSED |
| C-OSC-EXPORT | 3 | export | C | ù | oneRGGB blocked OSC-03 | invariants_runtime.py:313 | Wired FAIL | CLOSED |

### Contract promises (summary)

| Stage | Promises | Stated? |
|-------|----------|---------|
| Import | FITS copied verbatim to Raw; manifest rows | draft_manifest.json schema |
| Calibrate | Linear cal; VY_DKRSMP convention; CAL-DIAG PASS | INV-CAL-01 spec |
| Preprocess | In-place sky surface; VY_CALSTAGE stamp | INV-CAL-02 spec |
| Align | qc_metrics status=ok; WCS invertible | QC-01, INV-WCS-00 |
| Photometry | Differential mmag; Broeg weights; per-draft comp membership | DECISIONS; INV-COMP-MEMBERSHIP |
| Export | BJD_TDB for AAVSO; OSC filter codes | export_reports + OSC-03 |

---

*Append-only during audit run. Wave 6 disposition pending Milan review.*
