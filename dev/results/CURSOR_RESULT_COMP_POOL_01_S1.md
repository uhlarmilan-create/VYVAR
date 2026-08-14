CURSOR RESULT - COMP-POOL-01 Stage 1 - 2026-08-14

Register ID: COMP-POOL-01 (Stage 1 of 3)
Scope: noise model + diagnostics only. Selection unchanged.

Artifacts:
- `src_py/comp_pool_noise.py`
- `dev/tests/test_comp_pool_noise_s1.py`
- `dev/results/COMP_POOL_01_s1_summary_512.json`
- `dev/results/COMP_POOL_01_s1_stars_512.csv`
- `dev/results/COMP_POOL_01_s1_np_curve_512.csv`
- `dev/results/COMP_POOL_01_s1_curve_ratio_512.csv`

================================================================
1. Assumption
================================================================

"Bulk of field stars are non-variable; median scatter at magnitude traces noise."
(CSI 2264 / Broeg-style premise; stated, not hidden.)

================================================================
2. Noise model (draft 512)
================================================================

Howell form (RN in e-, variance in ADU^2):

  sigma_mag^2 = (2.5/ln10)^2 * (F/g + n_pix*(sky/g + (RN/g)^2)) / F^2
                + sigma_sys^2

Inputs: g=3.17 e-/ADU (DB), RN=7.6 e- (DB), sky_med=1549 ADU,
aperture area median=36.6 px^2, zp_inst=22.03 (from Gaia G vs flux).

Systematic floor fitted on the bright asymptote G in [8.0, 10.5] (n_fit=125),
where photon noise is negligible:

| quantity | value |
|----------|-------|
| sigma_sys_mag | **0.00974 +/- 0.00024** |
| scint_mag (Osborn/Young, D=0.2 m, X=1.038, 60 s, 275 m) | **0.00199** |
| ratio sys/scint | **4.90** |
| chi2_red (validation G8-13) | 4.17 |

DB TELESCOPE.DIAMETER for "Carl-Zeiss 200mm" is 70 mm; scintillation used
**D=0.2 m** override. Recorded as a data defect (diameter wrong in DB).

**P-R2 fired:** fitted floor is ~5x predicted scintillation. Do not adjust either.
Bears on open P-02 / WIDE-ERR (underquoted / extra floor on the wide rig).

================================================================
3. Non-parametric vs parametric
================================================================

Non-parametric: 0.5-mag bins; usable if n>=8. Usable from G~8.25 to ~12.75
(n peaks at 147 in G12-12.5). Below G8 and above G13 bins thin (n<8).

Curve ratio (NP / parametric), usable bins: median **1.12**, max |log10 ratio| 0.21.
Bright end (G8.25): NP 7.4 mmag vs param ~9.8 mmag (sys-dominated).
Faint usable (G12.75): NP 44 mmag vs param ~36 mmag.

**P-R1:** mild disagreement at the faint usable end (NP above param); not papered
over. Parametric remains the operative curve for sparse fields (few free params).

================================================================
4. Derived thresholds (reported only; not applied)
================================================================

| threshold | draft 512 value | rule |
|-----------|-----------------|------|
| detect_frac_min | 1.0 | p16 of detect_frac among mag_g<=14 |
| faint_limit_g | 10.75 | NP median >= 1.5 x bright asymptote (G8-10) |
| faint_limit SNR | ~83 | 1.0857/sigma at that bin (community thumb ~100) |
| bright_upturn | not visible | no rise below G9 vs mid |
| default_lin_frac | 0.85 | named default (D1-2 open; not derived) |
| stability excess | 1.92 | p84 of scatter/sigma_total (Kjeldsen~1.5; CSI~3) |
| dilution | not derived | needs dilution.py batch (Stage 2) |
| inv_eta threshold | not derived | needs separate population study |

**P-R0:** `default_lin_frac=0.85` is named as chosen (D1-2), not derived.
Dilution and inv_eta thresholds named as not yet derived.

================================================================
5. Selection unchanged (Stage 1 gate)
================================================================

| quantity | value |
|----------|-------|
| plan comparison_stars.csv | 140 |
| per-target rows | 95 |
| BO CVn trust | GREEN |
| check_scatter | 0.009300 |
| ac_scatter | 0.013283 |
| lc_rms_ooe | 0.046659 |
| comps | 5x TIER1 |

================================================================
6. Pre-registered rules
================================================================

- P-R0: fired for lin_frac (named default); dilution/inv_eta not yet derived.
- P-R1: mild NP>param at faint usable end; reported.
- P-R2: sys/scint = 4.9; reported, not adjusted.
- P-R3/P-R4: N/A (no selection change).

================================================================
7. Next stages
================================================================

Stage 2: apply derived pool admission; remove caps; dilution derivation.
Stage 3: assignment / relaxation order if needed.
Sparse-field and second-rig validation in Stage 2/3 memo.
