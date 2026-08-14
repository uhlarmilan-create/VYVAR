# VYVAR audit register - 2026 full workflow audit

**Started:** 2026-08-13  
**Method:** Data-flow and contract audit (supersedes formula-only July audit).  
**Format:** `ID | wave | stage | class | severity | evidence | reference | disposition | status`

Classes: **P** physics/method wrong - **I** implementation != method - **C** contract (units, naming, gate condition) - **U** correct but unmeasured/unjustified - **D** dead code/duplication

---

## Carry-forward from July 2026 closure (current status)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| A-1 | 4 | aperture | U/P | HIGH | CLOSED as diagnosed, not fixed. Cause: `gaussian_fwhm_px_override` is set from MASTERSTAR `VY_FWHM_GAUSS` = 3.3014 px while the night's PSF measures ~5.19 px (`VY_FWHM`). Successor: remove that override (authorized in principle; moves numbers; own measured delta). | `dev/results/CURSOR_RESULT_COG_A1_01.md` | **DIAGNOSED** | CLOSED |
| A-9 | 4 | PSF | U | MED | PSF scale estimators disagree 2.4-4.9 px; not blocking differential | closure register #31 | DOCUMENTED | OPEN |
| T4-1 | 4 | detection | I?FIXED | - | N_equiv=3.78 wired E.4 | closure #10 | FIXED 2026-08-04 | CLOSED |
| P-02 | 4 | errors | I?FIXED | - | Scintillation wired batch D | closure #25 | FIXED | CLOSED |
| I-11 | 4 | errors | I?FIXED | - | Howell sky term batch D | closure #21 | FIXED | CLOSED |
| I-04 | 4 | ensemble | I?FIXED | - | err_scatter unmatched epochs | closure #22 | FIXED | CLOSED |
| I-03 | 4 | errors | U | LOW | Legacy Howell terms unused on anchor | closure #23 | DOCUMENTED | OPEN |
| D10-1 | 4 | transforms | C?FIXED | - | unfiltered?CR band | closure #18 | FIXED Milan decision | CLOSED |
| WIDE-ERR | 4 | errors | I | MED | Diagnosed+localized (NOISE-FLOOR-01 + WIDE-ERR-LOC-01): photon/sky channel after completed Howell; full-range a,b split fails (r2~0.42, W-R1). Exported bars unchanged. | `CURSOR_RESULT_WIDE_ERR_LOC_01.md` | **DIAGNOSED+LOCALIZED**; fix = own task | OPEN |
| D1-2 | 4 | calibration | U | MED | Linearity curve deferred | closure #24 | DEFERRED dome-flat ramp | DEFERRED |
| D5-2 | 4 | saturation | C?FIXED | - | admission_sat_peak_frac=0.70 | batch E | FIXED | CLOSED |
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
| C-TRUST-01 | 3 | trust | C | MED | `bool("off")` made color-term branch always on; fixed via `resolve_apply_color_term`. **Silent on anchor:** branch fires only with TIER3+ comps and CT off; draft_510 BO CVn has five T1/T2 comps only. Real bug - would surface on fields with T3+ comps. | trust_flag_core.py:402 | FIXED 2026-08-13 | FIXED |
| C-P2P-01 | 2 | comp QA | C | MED | p2p ceiling reads `phase01_comparison_max_comp_rms` | photometry_core.py:3015 | FIXED 2026-08-13 | FIXED |
| C-MAX-RMS-01 | 2 | phase01 | C | LOW | comp_pool signature default aligned to 0.1 | comp_pool_rms.py | FIXED 2026-08-13 | FIXED |
| U-PED-01 | 3 | calibration | U | MED | Header OFFSET=0 vs ~24.5 ADU/bin1 pedestal measured CAL-DIAG | cal_diag.json draft 510 | CAL-DIAG derives P; header silent | OPEN |
| U-FWHM-XVAL | 4 | aperture | U | LOW | xval FWHM 2.96 px on aligned frame vs SNR table 2.395 px | xval_run draft 510 2026-08-13 | Third estimator; needs reconcile | OPEN |
| U-ANCHOR-GAP | 2 | QA | U | MED | INV-ANCHOR-00: `--full` never exercises cal/preprocess/align/MASTERSTAR | VYVAR_INVARIANTS.md | Documented boundary | OPEN |
| D-UI-FIN | 1 | UI | D | LOW | `ui_finalization.render_finalization()` unwired; `render_known_field_banner` active in app.py | ui_finalization.py:132 | **KEEP** (product); wire = add tab in app.py calling `render_finalization(pipeline, draft_id)` after photometry QC | OPEN |
| I-DETECT-OUT | 2 | outliers | I | LOW | `detect_outliers` survives on variable targets; sigma clip params ignored (zero-clipping) | photometry_core.py:4529 | Partially dead API surface | OPEN |
| C-SATDIAG-PROV | 3 | saturation | C | MED | `sat_diag.json` written at align start with `raw_peaks_used: false`; placed-aperture per frame sets proc meta true. Fixed: defer JSON write until after catalog pass (`commit_sat_diag_provenance`). | sat_diag.py; pipeline.py:10638,11319 | FIXED 2026-08-13 | FIXED |

---

## Wave 3 - contracts (C and I)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| C-CAL-STAGE | 3 | preprocess | C | MED | `calibrated/` holds cal+skysf product; name implies "done" after step 1 | pipeline.py:18027; MAP -2 | INV-CAL-02 mitigates | PARTIAL |
| C-OFFSET-HDR | 3 | headers | C | MED | FITS OFFSET=0 vs measured P?24.5 ADU/bin1 | cal_diag draft 510; U-PED-01 | CAL-DIAG authoritative | OPEN |
| C-GAIN-RN | 3 | equipment | C | MED | QHY294MM RN 7.6 vs 15.2 e- double-count suspected | STATE item 8 | UNVERIFIED | OPEN |
| C-SAT-UNIT | 3 | equipment | C | HIGH | SATURATE_ADU 16384 was binned in bin1 column | database.py:2854 | NULL + SAT-DIAG derive | FIXED |
| C-EXPORT-GAP | 3 | export | C | LOW | night_run omits AAVSO/VarAstro; UI/manual only | night_run.py vs export_reports.py | Document workflow | OPEN |
| I-TRUST-COLOR | 3 | trust | I | MED | bool("off") for apply_color_term | trust_flag_core.py:402 | FIXED (C-TRUST-01) | FIXED |
| I-ALIGN-SIGMA | 3 | alignment | I | MED | Settings sigma unused | C-ALIGN-01 | FIXED | FIXED |
| C-TIME-BASE | 3 | export | C | - | Non-BJD export refused | export_reports.py | FIXED T1 | CLOSED |
| C-OSC-EXPORT | 3 | export | C | - | oneRGGB blocked OSC-03 | invariants_runtime.py:313 | Wired FAIL | CLOSED |

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

## Provenance stale-write sweep (2026-08-13)

Pattern: record written by step N, invalidated by step N+k without update (INV-CAL-02 rule: writer = worker, same operation).

| Instance | Early write | Later truth | Status |
|----------|-------------|-------------|--------|
| `VY_QCBG` header | Post-cal QC sky median at cal enrich (`pipeline.py:16211`) | After in-place sky subtract pixels no longer match | OPEN - rename/stamp at preprocess or document as pre-skysf QC |
| `preprocess_calibrated_to_processed` | Name implies copy to `processed/` | In-place mutate `calibrated/lights` only | OPEN - rename alias (Wave 6 propose) |
| `sat_diag.json` `raw_peaks_used` | Align-start `run_sat_diag` | Placed-aperture per frame | **FIXED** (C-SATDIAG-PROV) |
| `resolve_obs_file_to_processed_fits` | Name says processed | Resolves calibrated path | OPEN - naming only; callers documented in MAP |
| `cal_diag.json` | Written at cal session end | Preprocess may change sky stats used downstream | LOW - CAL-DIAG scope is cal stage only; documented |

No fourth instance of the exact sat_diag flag pattern found beyond the three above.

---

## Wave 6 dispositions (2026-08-13)

Milan approved KEEP/PROPOSE as written. **W6-DEL-04 excluded** (library delete guards are unwired safeguards, not dead).

| ID | disposition | outcome | commit |
|----|-------------|---------|--------|
| W6-DEL-01 | DELETE qc hash helpers | DONE | `181811b` |
| W6-DEL-02 | DELETE legacy masterstar path setter | DONE (+ ASCII fix tracked docs in same commit) | `b82e976` |
| W6-DEL-03 | DELETE fetch_draft_scanning_ids | DONE | `1a7320e` |
| W6-DEL-04 | DELETE library delete guards | **SKIPPED** -> PROPOSE wire | - |
| W6-DEL-05 | DELETE legacy import helpers | DONE | `599e58d` |
| W6-DEL-06 | DELETE export helper stubs | DONE | `e98b354` |
| W6-DEL-07 | DELETE PDF styling helpers | DONE | `09f2b79` |
| W6-DEL-08 | DELETE ProcFrameStore.frame_columns | DONE | `ff6fba6` |
| W6-DEL-09 | DELETE shadowed param resolvers | DONE | `c6dc90d` |
| W6-KEEP-01 | KEEP ui_finalization | unchanged | - |
| W6-PROP-01..04 | PROPOSE | Milan authorized 2026-08-13; implement later | AUTHORIZED |
| W6-PROP-05 | PROPOSE wire library delete guards (ex-DEL-04) | Milan authorized; implement later | AUTHORIZED |

**Milan authorized (2026-08-13, not implemented):** W6-PROP-03 Option A (`VY_QCBG_PRE` + preprocess `VY_QCBG`); W6-PROP-01 Option A (clip constants only); W6-PROP-05 wire library delete guards; W6-PROP-02 rename shim; W6-PROP-04 reachability fix.

| A-1 | 4 | aperture | U/P | HIGH | See Wave 8: CLOSED as diagnosed, not fixed. | `CURSOR_RESULT_COG_A1_01.md` | **DIAGNOSED** | CLOSED |

All eight deletions: `--fast` PASS after each (1323 passed, 27 skipped). Draft 510 photometry re-cut 2026-08-14 (see checksum diff).

---

## Wave 7 closure (2026-08-14)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| **U-P5-PRED** | 7 | saturation | U | MED | Pre-registered P5 tested `peak_max_adu`, a 7x7 box at placed centroid on **raw** pixels - **insensitive to photometry aperture radius by construction**. P5 could not catch an unintended saturation consequence of the radius change. Defect in prediction design, not implementation. A test that **would** measure admission against the photometry aperture: **peak ADU within a circular mask of radius `aperture_r_px` at the placed centroid on raw** (or max in that annulus), compared to `admission_threshold_adu()` - not implemented. | Wave 7 S2.1; `sat_diag.py` | Record explicitly; do not treat P5 PASS as radius-verified saturation | **DOCUMENTED** |
| **U-XVAL-COMP-RMS** | 7 | aperture | U | MED | RETRACTED as a photutils/sep vs VYVAR radius mismatch claim; pointer: `dev/results/CURSOR_RESULT_U_XVAL_COMP_RMS_localization.md`. Residual gap if any is not this item. | Wave 7 S2.2; Q1-XVAL-MATCHED | **RETRACTED** | RETRACTED |

**Checksum manifests (draft 510):** `anchor_510_checksums_placed_aperture_20260813.json` (retained) -> `anchor_510_checksums_a1_dao_fwhm_20260814.json` (current). Diff: **237** files changed (`dev/validation/anchor_510_checksum_diff_20260814.json`): 135 proc CSVs, ~100 photometry outputs, `aperture_snr_table.json`, `sat_diag.json`.

---

*Append-only during audit run. Wave 7 closed 2026-08-14.*

---

## Wave 8 - iron gates + sky clip (2026-08-14)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| **ENC-STALE-01** | 8 | process | P | LOW | 7 non-ASCII docs broke `test_ascii_policy`; fixed with `ascii_migrate.py`. | `VYVAR_PROCESS.md` SHA-on-gate note | **FIXED** | CLOSED |
| **IRON-GATES-01** | 8 | invariants | P | HIGH | Five iron rules wired. INV-PIXELS-01 nanmedian fill still awaiting Milan, so this item cannot be CLOSED. | `dev/results/CURSOR_RESULT_CLOSE_IRON_GATES.md` | **WIRED** | PARTIAL |
| **SKY-CLIP-01** | 8 | photometry | P | HIGH | Unified plain annulus median; -0.058% flux median on 510 FITS recompute. `--full` anchor and P1 golden SHA invalidated by Commit B; re-cut is follow-up. | `VYVAR_DECISIONS.md` SKY-CLIP-01 | code **FIXED**; anchor **PENDING** | PARTIAL |
| **PP-KWARG-01** | 8 | preprocess | P | HIGH | `_pp_kw` passed `use_gpu_if_available` to `qc_enrich_calibrated_lights_in_place`; draft 511 TypeError. Kwarg removed; `kwarg_compat_scan.py` gate. | `dev/results/CURSOR_RESULT_PP_KWARG_01.md` | **FIXED** | CLOSED |
| **INV-CAL-02** | 8 | invariants | C | MED | Gate claimed absent stamp while `VY_CALSTAGE` present. Disk detection + fire proof. | `dev/tests/test_inv_cal_sat_gates.py` | **FIXED** | CLOSED |
| **INV-SAT-01** | 8 | invariants | C | MED | Gate claimed absent `sat_diag.json` while file present. Draft-root load + fire proof. | `dev/tests/test_inv_cal_sat_gates.py` | **FIXED** | CLOSED |
| **DRAFT-512-EXTRACT** | 8 | photometry | U | MED | Draft 512 extraction (dirty tree). Physics valid; not a reference until re-run on committed tree. | `CURSOR_RESULT_DRAFT_512_EXTRACT.md` | **DONE** | CLOSED |
| **COG-A1-01** | 8 | aperture | U | HIGH | C-R2 fired: seeing-correlated EE systematic **not established**. EE-ratio is white noise (successive-difference 1.25-1.48 vs 1.414); floor 18.21 mmag ~ series scatter 19.18 mmag; FWHM span 3.20%. Test lacked power. | `CURSOR_RESULT_COG_A1_01.md` | **NOT ESTABLISHED** | CLOSED |
| **D5-1** | 8 | aperture | U | MED | Q1: sizing FWHM and annulus geometry are per-draft frozen. Q2: per-star radius spread is `compute_snr_optimal_aperture_table`. | `CURSOR_RESULT_COG_A1_01.md` | **ANSWERED** | CLOSED |
| **Q1-XVAL-MATCHED** | 8 | photometry | U | MED | Matched-geometry xval completed. | `CURSOR_RESULT_Q1_XVAL_MATCHED.md` | **DONE** | CLOSED |
| **Decision (4)** | 8 | aperture | U | MED | Advanced: r90 measured 5.0-5.8 px, target 5.31 px, current radii enclose ~84.6%. | `CURSOR_RESULT_COG_A1_01.md` | **ADVANCED** | OPEN |
| **A-1-OVERRIDE** | 8 | aperture | P | HIGH | Remove `VY_FWHM_GAUSS` as `gaussian_fwhm_px_override`. Authorized in principle; moves numbers; own delta. | successor of A-1 | **AUTHORIZED** | OPEN |

*Wave 8 register diffs: `REGISTER_DIFF_CLOSE_IRON_GATES.md`, `REGISTER_DIFF_PP_KWARG_01.md`.*

### Architect retractions (2026-08-14)

- Auto-FWHM "arithmetic contradiction" was wrong. Limit is `median + k*MAD*1.4826` on manifest `inspection.fwhm`; prefilter runs on `VY_FWHM` -- two populations.
- Predicted 11.5 mmag seeing systematic was a Gaussian-model estimate, superseded by measured 4.36 mmag below the 18.21 mmag floor.
- `--fast` remaining green through SKY-CLIP-01 is expected: the byte-identity anchor runs only under `--full`.

### Deferred (open, no action in CLOSE-AND-PUSH)

- A-1-OVERRIDE: remove `VY_FWHM_GAUSS` override (authorized in principle; own measured delta)
- U-SKY-FALLBACK-01: whole-frame median when annulus has fewer than 5 pixels; global path `except Exception`; no counter
- INV-PIXELS-01 nanmedian fill: awaiting Milan
- `--full` anchor and P1 golden ledger re-cut after SKY-CLIP-01 commit 1
- Draft 512 re-run on a committed tree before it can be a reference
- Seeing-dependence of EE ratio: needs a night with real seeing variation and a lower-noise estimator
- U-SCATTER-DEF, WIDE-ERR, D1-2 exposure ramp, C-EXPORT-GAP, W6-PROP -- already open, unchanged

---

## Wave 9 - DAO depth (2026-08-14)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| **DAO-DEPTH-01** | 9 | detection | U | HIGH | Draft 512 ~704 proc rows vs ~1828 independent on same pixels. Localized to MASTERSTAR prematch SNR peak filter (3621->735); G12-G15 dropped. | `dev/results/CURSOR_RESULT_DAO_DEPTH_01.md` | **LOCALIZED**; successor **SNR-GATE-01** | SUPERSEDED-FIX |
| **SNR-GATE-01** | 9 | detection | I/P | HIGH | Prematch used full-frame scene std (~570 ADU) as "1.8 sigma" noise after `c9e1f8f`; F1 disqualifies it; F2 shows discarded pass-2 stars real through ~G15. Fix: sky MAD estimator + pass-2 exempt from global peak gate. Draft 512 MASTERSTAR 735 -> 3614 (Delta A); AB-A=0 on this night. Gate inert after fix (3614 in = 3614 out). | `dev/results/CURSOR_RESULT_SNR_GATE_01.md` | **FIXED** (commit SNR-GATE-01; no push) | FIXED |
| **SNR-GATE-02** | 9 | detection | C | HIGH | Surgical land: pin `_noise_floor_adu_from_image_array` so SNR aperture radii do not move; correct BO CVn baseline id; verify Part 5 zeros. | `dev/results/CURSOR_RESULT_SNR_GATE_01.md` secs 11-15 | **DONE** with SNR-GATE-01 commit | CLOSED |
| **SNR-DEPTH-01** | 9 | detection | U | MED | F2: G15 repeatability degrades (frac_median 0.507); no depth limit implemented. | SNR-GATE-01 F2 | **CLOSED** by TARGET-DEPTH-01 | CLOSED |
| **TARGET-DEPTH-01** | 9 | photometry | I/P | HIGH | Item A: QC FILE FK -> RUN_OLD healed. Item B: per-draft target depth from COMP-POOL detect_frac / NP half-SNR (T-R0); flag `below_target_depth` (not omit). 513 depth G=15.0 masks 15 targets / 4 existing LCs. | `CURSOR_RESULT_TARGET_DEPTH_01.md`; `TARGET_DEPTH_01_results.json` | **FIXED** (commit; no push) | FIXED |

### Deferred (adjacent)

- DAO-CONE-CAP-01: field_catalog_cone 100k over ~13.6 deg truncates in-footprint G15+ vs local DR3 (~567 vs ~2592 at G15)
- DAO-SNR-SIGMA-01: **CLOSED** by SNR-GATE-01 Delta A
- DAO-PASS2-vs-PREMATCH-01: **CLOSED** by SNR-GATE-01 Delta B
- FORCED_APERTURE-DEAD-01: label checked in UI/plot paths but never assigned in production writers
- SNR-TABLE-SKY-01 (new finding): SNR-table precompute still uses misnamed `med+k*plain_std` as "sky"; Phase 2A overwrites from annulus. Own delta if fixed.

---

## Wave 10 - Comparison pool (2026-08-14)

| ID | wave | stage | class | severity | evidence | reference | disposition | status |
|----|------|-------|-------|----------|----------|-----------|-------------|--------|
| **COMP-POOL-01** | 10 | ensemble | P/I | HIGH | Split pool vs assignment. Stages 1-3 local. Draft 512 pool ~187; BO loses 1/5. **Push blocked** by COMP-POOL-02 (D override wrong; chi2_red~4; draft 435 empties ensemble). | `CURSOR_RESULT_COMP_POOL_01.md`; `CURSOR_RESULT_COMP_POOL_02.md` | **HOLD** pending C2-R2 guard | OPEN |
| **COMP-POOL-02** | 10 | ensemble | P/U | HIGH | (1) D=0.2 override wrong: 200mm=focal, aperture=70mm; sys/scint~2.43 not 4.9. (2) chi2_red 4-6 SUPERSEDED by NOISE-FLOOR-01. (3) Draft 435 NP/param 1.52 / r=1.92 px empties BO pool. (4) Red stars 7-12% wider FWHM; ~30 mmag EE term; weak colour bias in admission. | `CURSOR_RESULT_COMP_POOL_02.md` | **REPORTED**; Item1 revert done; Item2 superseded; guard not implemented | OPEN |
| **COMP-POOL-SCINT** | 10 | errors | U | MED | NOISE-FLOOR-01: floor is UL ~6.8 mmag; sys/scint ~1.70+/-0.05 (UL) on 512/510 (was ~2.43 from inflated floor). Still above 1. | `CURSOR_RESULT_NOISE_FLOOR_01.md` | **UPDATED** UL ratio | OPEN |
| **NOISE-FLOOR-01** | 10 | errors | P/U | HIGH | Flatness test: no flat bright range (N-R0 UL). Completed Howell terms close ~4% of G10 variance deficit; residual ~2.3x sigma localizes WIDE-ERR to photon/sky. Diagnostic-only; admission path pinned legacy. | `CURSOR_RESULT_NOISE_FLOOR_01.md`; `NOISE_FLOOR_01_*.json` | **REPORTED** | CLOSED |
| **WIDE-ERR-LOC-01** | 10 | errors | U | HIGH | Item A: var_total expression closes (corr on extended only). Item B: a=7.45+/-0.22, b=2.20+/-0.07 but r2=0.42 (W-R1: not separable). Item C: 0.10 px SNR-radius drift = builder/product near-tie; not a Stage-2 block. | `CURSOR_RESULT_WIDE_ERR_LOC_01.md` | **REPORTED** | CLOSED |

### Named non-derived (P-R0)

- `default_lin_frac=0.85` (D1-2 linearity knee unmeasured)
- `nonparametric_min_bin_n=8` (NP curve usability only)
- Dilution percentile step p16->p10->p05 when D piles at 1.0

