# VYVAR -- Audit closure register (30 items)

**Date:** 2026-08-04 (audit closed)
**Source audit:** `docs/VYVAR_AUDIT_FINAL.md`
**Status legend:** CLOSED | FIXED | MEASURED | DOCUMENTED | DECISION | QUEUED | DEFERRED | OPEN

---

## Closure queue (Steps 1--10)

| Step | ID | Item | Domain | Status | Notes |
|------|-----|------|--------|--------|-------|
| **9** | **CR-1** | Cosmic-ray rejection (L.A.Cosmic or equivalent) | 1 | **FIXED** | E.3 astroscrappy; 365810 px physical re-cut |
| **10** | **T4-1** | Detection noise on resampled frames | 2, 7 | **FIXED** | E.4 N_equiv=3.78 |
| **7** | **C-1** | Admission gate + saturation gate (D5-2) | 7, 8 | **FIXED** | E.5 admission_sat_peak_frac=0.70 |
| **8** | **C-2** | CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE | 7 | **FIXED** | bundled with C-1 gate |
| **1-6** | **A-1..A-6** | MASTERSTAR stack queue | 7 | **QUEUED** | post-audit enhancement (see MASTERSTAR spec) |

A-1 aperture finding: **A-1b CONFIRMED, DOCUMENTED** (~144 mmag fixture; COG fix deferred).

---

## Register items 11--32

| ID | Item | Domain | Status | Notes |
|----|------|--------|--------|-------|
| 11 | P-10 sky-surface sign error | 3 | **FIXED** | `pipeline.py` |
| 12 | SKYSF-DOUBLE in-place guard | 3 | **FIXED** | Read `VY_SKYSF` before re-subtract |
| 13 | I-12 PM unavailable logging | 4 | **FIXED** | WARNING when pmra/pmdec absent |
| 14 | T1 export time_base truth | 12 | **FIXED** | Refuse non-BJD_TDB AAVSO export |
| 15 | D10-2 Gaia->Johnson range guard | 10 | **CLOSED** | 2026-08-21: guard verified `gaia_johnson.py:38-42`, `140-155`; exercised in D10-1 (11+3 excluded) |
| 16 | D5-1 aperture provenance columns | 5 | **DOCUMENTED** | A-1b; COG deferred |
| 16b | **D5-2** production flux vs G scaling | 5 | **FIXED** | slope -0.318 -> -0.491; gate E.5 |
| 31 | **A-9** absolute PSF scale | 5, 7 | **DOCUMENTED** | Not blocking differential |
| 17 | D1-3 master flat documentation | 1 | **CLOSED** | DECISIONS entry |
| 18 | D10-1 unfiltered CV->CR band | 10 | **CLOSED** | 2026-08-21: empirical D10-1 + D10-1b; Milan affirmed **CV**; zero code change; `CURSOR_RESULT_D10_1_CV_CR.md`, `CURSOR_RESULT_D10_1B_CV_CR.md` |
| 19 | sigma_pp drop | 2 | **FIXED** | Milan decision |
| 20 | masterstar_dao_threshold 2.1->3.8 | 7 | **FIXED** | P-10 bundle |
| 21 | I-11 Howell sky term | 2 | **FIXED** | batch D |
| 22 | I-04 ensemble scatter unmatched | 8 | **FIXED** | batch D |
| 23 | I-03 omitted Howell terms | 2 | **DOCUMENTED** | legacy unused on anchor |
| 24 | D1-2 linearity correction | 1 | **DEFERRED** | dome-flat ramp per sensor; SAT-LIMIT-01 cheap residual-vs-peak did not resolve a knee |
| 25 | P-02 scintillation in production err | 9 | **FIXED** | batch D; floor not applied |
| 26 | U-09 DATE-OBS convention per rig | 4 | **CLOSED** | home rig 515 measured (verdict a); other rigs unverified; RESULT_U09_GAIN_AUTH |
| 27 | Part 0c delta pairing fix | 7 | **FIXED** | E.1 source_file merge |
| 28 | DAO centroid stability | 5, 7 | **FIXED** | E.2 WCS guard |
| 29 | Anchor re-cut (VL-ANCHOR-WCSINV) | all | **FIXED** | physical re-cut GATE 2; SHA 5bccd85a |
| 30 | TODO-B coaddition (Zackay & Ofek) | 7 | **QUEUED** | MASTERSTAR enhancement |
| 32 | **WIDE-ERR** wide-rig err underquote ~2x | 9 | **CLOSED** | WIDE-ERR-04: physical model (g_pt + weighted SEM, s=1, sr=0); under-quote eliminated; see RESULT_04 |
| 32b | **GAIN-DOMAIN-01** 14-bit-in-16-bit vs native DB gain | 9 | **CLOSED** | Physics proven in WIDE-ERR-03; stays CLOSED |
| 32c | **SEM** weighted ensemble SEM (ratio ~0.677) | 9 | **CLOSED** | Shipped with WIDE-ERR-03/04; identity fire proofs; never alone |
| 32d | **WIDE-ERR-CROSSRIG** per-rig calib on Newton/Boyden | 9 | **OPEN** | Mechanism is per-rig; calibrate when those drafts exist |
| 32e | **CORR-ERR-01** ZP common-mode vs diagonal budget | 9 | **OPEN** | Research note (LOW): product-frame scatter below model in (11.0,11.5]; covariance-aware budget out of v1.0 |
| 32f | **SAT-LIMIT-01** catalog sat clip NaN silent-pass | 7, 8 | **CLOSED** | INV-SAT-LIMIT; 515 reclassified 24 saturated; see RESULT_SAT_LIMIT_01 |

---

## Future threads (not audit-open)

1. **WIDE-ERR-CROSSRIG** -- apply the same per-rig gain/err calibration when Newton/Boyden drafts exist.
2. **MASTERSTAR stacking** -- Steps 1-6 + TODO-B coaddition (enhancement).
3. Draft 515 LC err = physical model (WIDE-ERR-04 identity sidecar). CORR-ERR-01 research note remains OPEN (LOW).
4. SAT-LIMIT-01 <-> BIN-8-9 (bright-end LOO excess partly saturation/nonlinearity, not closed) <-> D1-2 (knee still unmeasured; dome-flat ramp).

---

## Post-audit empirical closures (2026-08-21)

| ID | Item | Status | Notes |
|----|------|--------|-------|
| **D10-1** | Unfiltered CV vs CR band letter | **CLOSED** | Ensemble + raw-counts probe on era-516 NoFilter; CV affirmed; no export code change |
| **D10-2** | Gaia->Johnson validity guard | **CLOSED** | Present, enforced, exercised in D10-1 star selection |

---

*Register closed 2026-08-04. Post-audit updates above. See `docs/VYVAR_AUDIT_CLOSURE.md` for referee deliverable.*
