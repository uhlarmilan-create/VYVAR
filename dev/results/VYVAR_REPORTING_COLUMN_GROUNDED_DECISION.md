# VYVAR -- Reporting column (grounded decision)

Date: 2026-06-15  
Status: **LANDED** (Workstream B, commit `2a8355b`; supersedes B1/B2)

---

## Problem

Per-target airmass detrend (former `airmass_detrend_lc` on the target's own `mag_calib`, **implementation removed 2026-06-19**) and global
MAD outlier clipping were degrading eclipse shape on V0612 (corr `mag_calib` vs `delta_mag` ~0.59).

---

## Grounded fix (three parts)

1. **Report differential + ensemble zero-point** -- colour-matched comp ensemble (Honeycutt 1992).
   Pre-detrend `mag_calib_raw` already matched `delta_mag` shape (corr ~0.998 on V0612).

2. **Drop per-target airmass detrend on the reporting path** -- redundant after colour-matched
   differential (Plavchan et al. arXiv:0704.3584; Dhillon PHY217); signal-absorbing when fitted
   to the variable target. **Landed:** helper + `airmass_detrend_lc`/`_piecewise` removed (T1-2/T1-7).

3. **Mask-first outlier guard** -- clip out-of-eclipse only for known/candidate variables (TESS
   subdwarf recipe arXiv:2402.16018; democratic detrender arXiv:2411.09753).

---

## Implementation

`apply_reporting_postprocess` in `photometry_core.py` (Workstream B).

**DoD-B (V0612):** `mag_calib` corr **0.958** vs AIJ; pre-eclipse RMS **0.011**; ingress 24/24
`normal`.

---

## Superseded

- B1/B2 "guard the airmass detrend" framing (withdrawn).
- Tier-2 comp-ensemble extinction k for wide delta-airmass -- **PARKED** (ROADMAP).
