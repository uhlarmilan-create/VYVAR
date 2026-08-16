# CURSOR RESULT - WIDE-ERR-03C

Date: 2026-08-16
Baseline tip: 2396949
Draft 515 photometry SHA: da9cce4
Push Part 0: SKIPPED (authorization blank)
Push science: NOT authorized

## Verdict

**STOP at CORR-ERR-01.** WIDE-ERR + SEM remain OPEN. GAIN-DOMAIN-01 CLOSED.
Product-frame meter implemented (C1). Escalation C2/C3 exhausted without a
passing even-half gate. No forced pass. No re-export.

## Part 0
Authorization blank -> skip. No push.

## C1 - Product-frame meter (mandatory C1d table)

Meter (XVAL-BO-01 formula, units mmag MAD*1.4826):
`mag_calib = m_inst(dao_flux) + sum_j w_j (G_j - m_j) / sum_j w_j`
with `w_j = 1/rms_j^2` after `pytics_iterative_weights`. Self-exclusion when
the evaluated comp is in the target ensemble (renormalized weights);
otherwise full ensemble. Comps only; 03B variable guard retained.
Frame: product mag_calib. Ensemble cases among 54 clean comps: all
self_excluded (each is in its primary target's ensemble). SHA da9cce4.

### C1d before any refit (existing 03B constant-sigma_r, EVEN half)

Calib used: 03B holdout odd-fit s=1.004, sigma_r0=6.466 mmag (constant).
Eval: EVEN-indexed frames, product-frame scatter / err_exported.

| bin | n | median ratio | median err [mmag] | gated |
|-----|---|-------------:|------------------:|:-----:|
| (9.0, 9.5] | 10 | 0.768 | 10.14 | Y |
| (9.5, 10.0] | 11 | 0.780 | 11.40 | Y |
| (10.0, 10.5] | 8 | 0.722 | 11.98 | Y |
| (11.0, 11.5] | 5 | 0.705 | 19.73 | Y |
| G(8,9] union | 4 | 0.736 | 9.36 | in_window=NO |

**C1d PASS? NO.** The 03B FAIL was **not** a meter artifact: mid bins moved
**farther below** 0.85 (product scatter quieter than LOO flux-sum; existing
floor overinflates). Do **not** skip to C4 alone.

Artifact: `dev/results/WIDE_ERR_03C_C1.json`

## C2 - Floor vs aperture (ran because C1d failed)

Held-out (odd fit / even score) candidates:
| form | held-out |score-1||
|------|----------|--------|
| constant | 0.0435 | **winner** |
| step r_cut=5 px | 0.0435 | tie -> constant preferred as null |
| linear a*r_ap | 0.1229 | |

Winner meta: **s=1.0 (clamped), sigma_r=0** - physical model alone.
Even-half gate: FAIL at (11.0, 11.5] n=5 ratio **0.746** (err 18.56 mmag).

BIN-8-9 physics sentence: Winner is the null constant with zero floor; no
aperture-dependent floor is selected. Does **not** support a BIN-8-9 <->
WIDE-ERR floor register pointer (bright-end residual is not absorbed as
f(r_ap) under this held-out score).

Artifact: `dev/results/WIDE_ERR_03C_C2.json`

## C3 - Per-LC floor (ran because C2 failed)

Per-curve s>=1 and floor>=0 on odd frames; gate on even.
Even-half FAIL:
| bin | n | median ratio |
|-----|---|-------------:|
| (9.5, 10.0] | 11 | 0.787 |
| (10.0, 10.5] | 8 | 0.815 |
| (11.0, 11.5] | 5 | 0.746 |

G(8,9] union in window (ratio 0.854, err 7.94 mmag) but gated mid bins fail.

**STOP implementing.** Open **CORR-ERR-01**.

Artifact: `dev/results/WIDE_ERR_03C_C3.json`

## CORR-ERR-01 evidence

Diagnosis: on the product frame, gated-bin median(scatter/err_model) is
already <0.85 with s=1 and sigma_r=0. The catalog ZP removes common-mode
that the diagonal photon+SEM+scint budget still counts. A (s, sigma_r)
calibration with s>=1 cannot fix overprediction.

Pont / check-star residual (constant-star product frame):
| target | check_kmag MAD [mmag] | target median err [mmag] | ratio |
|--------|----------------------:|-------------------------:|------:|
| BO 1498613634033133184 | 6.713 | 11.138 | 0.603 |
| FW 1497343732462852864 | 8.201 | 9.770 | 0.839 |

(Do not use target mag_calib_final MAD for BO - astrophysical variability
~193 mmag dominates.)

## C4
Gate did not pass at any level -> **no re-export**, WIDE-ERR+SEM stay OPEN,
CORR-ERR-01 OPEN. Sidecar remains WIDE-ERR-03B on disk.

## Spec defects (named)
1. C1d hoped product meter would rescue 03B mid bins; opposite occurred
   (quieter product scatter + existing floor).
2. err_model SEM path is still LOO residual-based, not product-frame SEM;
   mixes meters inside the ratio.
3. s>=1 clamp correctly forbids absorbing model overprediction; when the
   physical model alone sits below 0.85 in gated bins, the ladder must stop
   at CORR-ERR-01 (this task did).

Physics outranks: common-mode cancellation in mag_calib is expected; the
open question is how to quote err for a ZP-common-mode-removed product
(literature: Pont+2006 red-noise / sigma_r on binned residuals - evidence
above, not a silent s<1 fix).

## Files
- `dev/tools/wide_err_03c.py`
- `dev/results/WIDE_ERR_03C_C1.json`, `_C2.json`, `_C3.json`, `_summary.json`
- `docs/VYVAR_DECISIONS.md` (product-frame rule + CORR-ERR-01)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` (32e CORR-ERR-01 OPEN)

## session_baseline_check.py --fast
**OVERALL PASS** (see console; tip 2396949).
