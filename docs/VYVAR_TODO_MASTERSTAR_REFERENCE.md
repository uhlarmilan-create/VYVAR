# VYVAR -- MASTERSTAR reference TODO (operational index)

**Status:** NOT STARTED
**Full architecture:** `docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md`
**Audit cross-links:** Tranches 3/4, `CURSOR_RESULT_dao_sigma_stability.md`,
`CURSOR_RESULT_dao_only_verify.md`

This is the short operational checklist. Implementation detail, literature, and equations live in
the architecture doc.

---

## TODO-C -- Admission gate vs detection threshold (HIGH)

Independent of stack reference. **Suggested first** in the MASTERSTAR arc.

| Sub | Task | Done? |
|-----|------|-------|
| C-1 | Compute predicted per-epoch SNR from `g_lim_50/90` + Labbé `sigma_bkg_ap` | |
| C-2 | Threshold admission on SNR, not DAO detection in reference | |
| C-3 | Verify draft_451 Group-B spurious actives rejected | |
| C-4 | Mark deep catalogue rows CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE | |

---

## CR-REJECTION -- Cosmic rays (MED)

Standalone hygiene gap. No CR step in `src_py` today.

| Sub | Task | Done? |
|-----|------|-------|
| CR-1 | Select method (L.A.Cosmic / equivalent) | |
| CR-2 | Wire into preprocess or detrended path | |
| CR-3 | Regression guard on anchor subset | |

Prerequisite for TODO-B; improves TODO-A median stacks.

---

## TODO-A -- Stack reference from best N frames (HIGH)

Replaces `build_masterstar_from_detrended` single lowest-FWHM copy.

| Sub | Task | Done? |
|-----|------|-------|
| **A-1** | Frame metric `I_j = F_j^2 / (sigma_j^2 * FWHM_j^2)` | **<- Closure Step 1** |
| A-2 | N_min=10, N_max=20, gate I_j >= 0.5 max(I_j) | |
| A-3 | Median or sigma-clipped mean stack | |
| A-4 | Provenance: frame list, I_j values, deterministic ties | |
| A-5 | Recalibrate DAO threshold on stack (not carry 3.8) | |
| A-6 | Split DAO_ONLY metric by magnitude vs Gaia cap 17.5 | |

Do **not** rank on FWHM alone (twilight bias measured on draft_435).

---

## TODO-B -- Proper coaddition (MED, multi-session)

Zackay & Ofek (2017) ApJ 836, 188. Optimal version of A.

**Do not start before:** TODO-C (optional), CR-REJECTION, TODO-A, uncorrelated-input strategy,
per-frame PSF, PSF-based F_j.

See architecture doc SS B1--B6 for equations and prerequisites table.

---

## Detection noise on resampled frames (Tranche 4)

Stack reference (TODO-A) does **not** fix correlated noise on aligned detection frames.

| Option | Summary |
|--------|---------|
| A | Detect on pre-align preprocessed frame |
| B | `scale_threshold=False`; threshold convolved-image RMS |
| C | Monte-Carlo correlation factor per setup |
| D | Document nominal vs effective; accept drift |

**DECISION REQUIRED (Milan)** -- blocks anchor re-cut. See `CURSOR_RESULT_audit_t4.md`.

---

## Suggested order

1. TODO-C (admission gate)
2. CR-REJECTION
3. TODO-A (stack reference) -- **A-1 is closure Step 1**
4. T4-1 decision (can parallel measurement)
5. TODO-B (proper coaddition)

---

## Citations

All keys in `CITATIONS.bib` (`stetson1994`, `zackay2017detection`, `zackay2017proper`,
`fruchter2002`, `casertano2000`, etc.).
