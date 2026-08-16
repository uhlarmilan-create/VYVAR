# CURSOR RESULT - WIDE-ERR-03B

Date: 2026-08-16
Baseline tip (pre-commit): 2396949
Draft 515 photometry run SHA: da9cce4
Push: NOT authorized

## Verdict

**B2c FAIL** (held-out frame-split). WIDE-ERR + SEM stay **OPEN**.
GAIN-DOMAIN-01 stays **CLOSED**.
Smooth form + s>=1 clamp + INV-CALIB-HOLDOUT recorded in DECISIONS.
Phase 2A err re-export on draft 515 completed; mag byte-identity **PASS**.

## What I did

### B1 - Smooth clamped calibration
- Replaced per-bin (s, sigma_r) with smooth form in `src_py/err_calibration.py`:
  constant s with **s >= 1.0 clamp**, sigma_r constant or linear-in-G (non-negative),
  form chosen by held-out score.
- Held-out chose **constant_sigma_r**.
- Production sidecar refit on all frames after form lock:
  s=1.001, sigma_r0=6.893 mmag, n=54, s_clamped=false.
- Odd-fit (gate parameters): s=1.004, sigma_r0=6.466 mmag.

### B2 - Non-circular held-out gate
- Frame split: calib on odd-indexed frames (67), eval on even (67); all 54 clean comps both sides.
- Secondary star-split: calib 27 / eval 27 -> **PASS** on eval half.
- Fire proof (s=1, sigma_r=0, gain=3.17 bare native on even): **FAIL as required**.
- B2c primary (frame eval): **FAIL** -- two gated bins just below 0.85.

### B3 - Re-export
- Phase 2A re-export with g_pt authority + weighted SEM + smooth calib sidecar.
- Fixed PT aperture bug: was using `aperture_scatter_r_min_px=1.5` (biased PT, CI too wide,
  fell back to wrong DB/4=0.25). Now uses production `aperture_r_px` (~4) from pipeline_meta.
- Mag columns byte-identical vs pre-reexport backup (49/49).
- BO/FW median err inflated as expected under clamp+floor.

### B4 - Docs / register
- DECISIONS: s>=1, smooth-form rationale, **INV-CALIB-HOLDOUT**.
- Register: WIDE-ERR + SEM remain OPEN with B2c fail notes; GAIN-DOMAIN-01 CLOSED.

## Output / findings

### B1 chosen form
| field | value |
|-------|-------|
| form | constant_sigma_r |
| s (all-frame production) | 1.00112 |
| sigma_r0_mmag | 6.8926 |
| slope | 0 |
| n_stars | 54 |
| s_clamped | false |
| g_pt authority | 0.637 e-/ADU_container |

Artifacts: `Archive/.../photometry/err_calibration.json`, `gain_photon_transfer.json`

### B2 frame-split (eval = even frames; odd-fit applied)

Gated bins outside [0.85, 1.15]:
| bin | n | median ratio | shortfall |
|-----|---|--------------|-----------|
| (10.0, 10.5] | 8 | 0.840 | -0.010 |
| (11.0, 11.5] | 5 | 0.839 | -0.011 |

G(8,9] union: n=4, ratio=0.963, median err=9.36 mmag (>=2.2) -- inside window.

Star-split eval: **PASS** (no gated bins outside).

Fire proof: G(8,9] ratio=1.413 with bare gain 3.17 -- gate fails as required.

GAIN-PT-CI-01: **not opened** (held-out G>13 faint-end median ratio not <0.85 under clamp).

### B2c failure analysis (physics overranks widening)

Grid search over constant (s>=1, sigma_r>=0): **zero** parameter pairs pass all gated bins
on this LOO LC-frame meter (bright underquote vs mid overquote tension). Do **not**
widen the window. Do **not** set s<1.

**Next step:** re-measure B2c in the true mag_calib / comps-only frame (Pont/Gillon);
if still outside, residual is correlated/common-mode beyond diagonal (s, sigma_r).

### B3 before/after median err

| target | catalog_id | median err before (mmag) | after (mmag) |
|--------|------------|--------------------------|--------------|
| BO | 1498613634033133184 | 7.573 | 11.138 |
| FW | 1497343732462852864 | 6.582 | 9.770 |

Mag byte-identity: **PASS** (49/49 LCs; columns mag_inst..delta_mag).

### Spec defects (named)
1. WIDE-ERR-03 S5e was circular (fit=accept same stars) -- architect defect motivating 03B.
2. B2 acceptance meter is still LOO flux-sum LC-frame, not full pytics mag_calib.
3. Per-star primary ensemble = first target listing that uses the comp.
4. Pre-registered [0.85,1.15] cannot be met by any 2-param clamped smooth form on this meter;
   widening would be a second circularity. Physics: mid-bin shortfall ~0.01 may be meter noise
   (n=5 at 11-11.5) or real over-floor from constant sigma_r.

## Errors (if any)
- First re-export attempt: INV-DAG-01 (phase2a after postprocess) + wrong PT aperture 1.5.
  Fixed: truncate DAG stages before re-stamp; PT uses aperture_r_px~4; pass VyvarDatabase.
- B2c FAIL (expected disposition: stay OPEN).

## Files changed
- `src_py/err_calibration.py` -- smooth form, s>=1, held-out chooser
- `src_py/photometry_core.py` -- smooth sidecar apply; PT aperture from dynamic_params
- `docs/VYVAR_DECISIONS.md` -- WIDE-ERR-03B / INV-CALIB-HOLDOUT
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` -- WIDE-ERR/SEM OPEN notes
- `dev/tools/wide_err_03b_heldout.py`, `dev/tools/wide_err_03b_reexport.py`
- `dev/results/WIDE_ERR_03B_B2.json`, `WIDE_ERR_03B_B3.json`, this file
- Draft 515 on-disk: LC err columns, `err_calibration.json`, `gain_photon_transfer.json`

## session_baseline_check.py --fast
**OVERALL PASS** (1429 passed, 28 skipped) at tip 2396949.
