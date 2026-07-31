# VYVAR -- Audit closure register (30 items)

**Date:** 2026-07-31
**Source audit:** `docs/VYVAR_AUDIT_FINAL.md`
**Status legend:** CLOSED | FIXED | MEASURED | DECISION | QUEUED | BLOCKED

Steps **1--10** below are the active closure queue (ROADMAP). Remaining items are tracked but
not in the first execution wave.

---

## Closure queue (Steps 1--10)

| Step | ID | Item | Domain | Status | Depends on |
|------|-----|------|--------|--------|------------|
| **1** | **A-1** | Implement frame selection metric `I_j = F_j^2 / (sigma_j^2 * FWHM_j^2)` for MASTERSTAR stack ranking | 7 | **QUEUED** | -- |
| **2** | **A-2** | Selection rule: N_min=10, N_max=20, quality gate I_j >= 0.5 max(I_j) | 7 | QUEUED | A-1 |
| **3** | **A-3** | Median/sigma-clip stack combination (replace single-frame copy) | 7 | QUEUED | A-2 |
| **4** | **A-4** | Mandatory stack provenance in header + `pipeline_meta.json` | 7 | QUEUED | A-3 |
| **5** | **A-5** | Recalibrate `masterstar_dao_threshold_sigma` against stack noise/PSF | 7 | QUEUED | A-3, T4-1 |
| **6** | **A-6** | Split `DAO_ONLY` health metric by magnitude vs Gaia cap (17.5) | 7 | QUEUED | A-5 |
| **7** | **C-1** | Admission gate: predicted per-epoch SNR (`g_lim_*` + Labb sigma_bkg_ap) | 7, 8 | QUEUED | -- |
| **8** | **C-2** | Flag catalogue rows CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE | 7 | QUEUED | C-1 |
| **9** | **CR-1** | Cosmic-ray rejection (L.A.Cosmic or equivalent) | 1 | QUEUED | -- |
| **10** | **T4-1** | Milan decision: detection noise on resampled frames (options A/B/C/D) | 2, 7 | **DECISION** | measurement in stage2 |

### Aperture closure Step 1 (finding A-1 -- SNR table radius)

**Status:** **A-1b MEASURED** (2026-07-31). FIX required; patch proposed, not applied.
**Report:** `dev/results/CURSOR_RESULT_closure_step1.md`
**Harness:** `dev/tools/closure_step1_aperture_fwhm_ground_truth.py`

| Finding | Verdict | Decisive measurement |
|---------|---------|----------------------|
| Focus target `aperture_r_px = 1.916 px` vs true FWHM | **A-1b** | EE at r_ap varies **8.0 pp** best-worst frame (COG); `r_ap/FWHM = 0.479` on median frame |
| SNR table clamp | binding | **2060/2649** stars on `r_min_px` (frame 063) |
| `aperture_snr_sizing` (S1) | **DEAD** on science path | defaults 0.8/2.5 FWHM used instead |
| D5-1 Q1 per-frame FWHM for apertures | **No** | draft-constant `VY_FWHM_GAUSS = 2.395` |
| Role factors on export path (S3) | label only | `aperture_comp_factor` not applied to radius |

**Proposed fix (Milan):** option (i) -- per-frame FWHM for SNR table, not MASTERSTAR Gaussian.
Unblocks aperture Steps 2-3 (placement A-2/A-3) and informs anchor re-cut after patch.

**ID note:** MASTERSTAR stack items **A-1..A-6** in this table are a separate queue; aperture
finding **A-1** in Step 1 reports refers to SNR-table radius only.

---

## Register items 11--30

| ID | Item | Domain | Status | Notes |
|----|------|--------|--------|-------|
| 11 | P-10 sky-surface sign error | 3 | **FIXED** | `pipeline.py`; tests in `test_preprocess_sky_surface.py` |
| 12 | SKYSF-DOUBLE in-place guard | 3 | **FIXED** | Read `VY_SKYSF` before re-subtract |
| 13 | I-12 PM unavailable logging | 4 | **FIXED** | WARNING when pmra/pmdec absent |
| 14 | T1 export time_base truth | 12 | **FIXED** | Refuse non-BJD_TDB AAVSO export |
| 15 | D10-2 Gaia->Johnson range guard | 10 | **FIXED** | Stage 1; 1 comp outside range on anchor |
| 16 | D5-1 aperture provenance columns | 5 | **FIXED** | Stage 1.2 proc CSV columns; **Step 1 (2026-07-31): A-1b** -- radius undersized vs COG FWHM; per-draft FWHM; clamp binding; role factors label-only on export |
| 17 | D1-3 master flat documentation | 1 | **CLOSED** | DECISIONS entry; builder gap noted |
| 18 | D10-1 unfiltered CV->CR band | 10 | **FIXED** | Milan decision; Stage 3 |
| 19 | sigma_pp drop / sigma_clipped_stats | 2 | **FIXED** | Milan decision Stage 3 |
| 20 | masterstar_dao_threshold 2.1->3.8 | 7 | **FIXED** | Bundled with P-10 |
| 21 | I-11 Howell sky on subtracted frames | 2 | **DECISION** | Options 1--3 documented; 0 prod epochs |
| 22 | I-04 ensemble scatter unmatched | 8 | **DECISION** | NaN+exclude vs inflate |
| 23 | I-03 omitted Howell terms | 2 | QUEUED | After I-11 decision |
| 24 | D1-2 linearity correction | 1 | MEASURED | No defect signature on anchor |
| 25 | P-02 scintillation in production err | 9 | **DECISION** | Do not wire without Milan |
| 26 | U-09 DATE-OBS convention per rig | 4 | MEASURED | BO CVn: shutter-open; others TBD |
| 27 | Part 0c delta pairing fix (source_file) | 7 | **QUEUED** | Harness bug; invalid tail stats |
| 28 | DAO centroid stability / aperture placement | 5, 7 | **QUEUED** | Part 0e M4; 19/156 targets > r_ap shift |
| 29 | Anchor re-cut (VL-ANCHOR-WCSINV) | all | **BLOCKED** | After T4-1 + A-5 + pairing fix |
| 30 | TODO-B proper coaddition (Zackay & Ofek) | 7 | QUEUED | After CR-1, A complete, per-frame PSF |

---

## Decision log (Milan, 2026-07-30)

| # | Decision |
|---|----------|
| 1 | Drop `sigma_pp`; revert to `sigma_clipped_stats` for DAO noise scalar |
| 2 | Unfiltered band: switch CV -> CR (Cousins R comparison mags) |
| 3 | Do NOT pick DAO threshold N from Part 2b sweep (R5) |
| 4 | GAIA-1/GAIA-2 remain deferred to DR4 |

---

## Evidence index

| Stage / part | Report |
|--------------|--------|
| Tranche 1 | `dev/results/CURSOR_RESULT_audit_t1.md` |
| Tranche 2 | `dev/results/CURSOR_RESULT_audit_t2.md` |
| Tranche 3 | `dev/results/CURSOR_RESULT_audit_t3.md` |
| Tranche 4 | `dev/results/CURSOR_RESULT_audit_t4.md` |
| Stage 0--2 | `dev/results/CURSOR_RESULT_audit_stage{0,1,2}.md` |
| Stage 3 Part 0a--0e | `dev/results/CURSOR_RESULT_audit_stage3_part*.md` |
| Closure Step 1 (aperture A-1) | `dev/results/CURSOR_RESULT_closure_step1.md` |
| MASTERSTAR spec | `docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md` |

---

*Register maintained at audit close 2026-07-31. Update item status in JOURNAL when steps complete.*
