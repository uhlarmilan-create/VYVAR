CURSOR RESULT - 2026-08-04 (WIDE-ERR E1)

What I did
Located the ensemble-term underquote. Retracted A2b absolute gain. Traced SEM path in
photometry_core; measured clipping bias (D12-1) and in-frame comp dispersion vs quoted SEM
on 166 check-star fields. Read-only; harness dev/tools/wide_err_e1.py;
output tmp/wide_err_e1/wide_err_e1.json.

## Guard rails

| check | result |
|-------|--------|
| anchor_manifest_check.py (pre-run) | PASS (exit 0) |
| anchor_manifest_check.py (post-run) | PASS (exit 0) |

## E1.1 -- Retract the gain result

A2b M2/M3/M4 absolute-gain conclusions **retracted** in CURSOR_RESULT_wide_err_a2b.md.

| issue | detail |
|-------|--------|
| M2 | Sky PTC on **unflat-fielded** raw frames; PRNU + vignetting inflate variance, deflate g |
| M1 | 60 s dark master subtracted from 0.15 s flats (pedestal mismatch) |
| Science bound | faint5 scatter 1.119 x 201 = **225 mmag** |
| Photon floor at g=0.96 | 200*sqrt(3.17/0.96) = **364 mmag** (impossible: scatter < floor) |
| Lower bound | 200*sqrt(3.17/g) <= 225 -> **g >= 2.50 e-/ADU** |
| gain_used | **3.17 e-/ADU consistent with bound; NOT implicated** |
| Relative survives | **g_eff/g1 ~ 0.95** (~1) -> SUM binning only |

**M4 check-star correction:** ratio_orig = 0.515 is from **one** T3 bright-representative field
(target_cid **1485540612577549568**, ens 58.1 mmag, sparse 2-comp ensemble, chi2 ~ 0.66).
**NOT** the W1 median sigma_total_robust/err = **1.83** over 163 fields.

## E1.2 -- Is the SEM computed on a clipped residual set?

**Trace (read-only, file:line):**

| step | finding |
|------|---------|
| comp_resid (3430-3438) | Built from same ``comp_pairs`` as flux sum; residual = m - comp_ref_map[cid]; **NO sigma-clip** before ``ensemble_sem_mag_from_residuals`` |
| comp_ref_map (3382-3390) | Median of **all** finite frames per comp across the night; **not clipped** |
| comp_pairs vs ens_med (3392-3418) | **Identical comp set** enters flux sum and SEM each frame |
| check_comparison_stability (3007-3040) | MAD p2p filter marks comps **excluded** before ``ensemble_normalize``; excluded comps drop from **both** flux and SEM |
| ZP sigma-clip (3468-3486) | 3-sigma clip on zeropoint offsets for ``mag_calib`` only; **does not touch ensemble_scatter** |
| temporal_bin_comp_lc (2567+) | Rolling median smooth on comp LC before stability; reduces p2p, not clipping |
| ensemble_sem_mag_from_residuals (sigma_floor_core.py:37-49) | std/(c4*sqrt(n)); **no clip** |

**Set membership (5 representative fields, per frame):**

| target_cid | max |flux| - |sem|| sample n_flux / n_sem |
|------------|---------------------|-------------------------|
| 1485540612577549568 (T3) | 0 | 2 / 2 |
| 1485552329248338816 | 0 | 8 / 8 |
| 1485574899299782528 | 0 | 8 / 8 |
| 1485987254816323328 | 0 | 8 / 8 |
| 1485987254816323328 | 0 | 8 / 8 |

**Flux-set and SEM-set differ only when a comp lacks comp_ref_map** (no finite night median);
in all 5 fields tested, **diff = 0 every frame**.

## E1.3 -- Clipping bias (D12-1)

Recomputed per-frame ensemble SEM on **166 check-star fields**:
- **Production:** temporal bin + check_comparison_stability + ensemble_normalize
- **Unclipped:** all pool comps, no stability exclusion, n_comp_max uncapped

| metric | value |
|--------|-------|
| n_fields | 166 |
| SEM_unclipped / SEM_production median | **1.00** |
| IQR | 1.00 / 1.00 / 1.00 |

**Reading:** Ratio ~1.0. Stability exclusion and n_comp_max cap do **not** measurably bias
SEM on this anchor. **No sigma-clip exists on comp_resid.** D12-1 iterative clipping bias
**not applicable** on this data path (close as measured-not-present, not unmeasured).

Implied effect on sigma_total_robust/err from clipping alone: **negligible** (~0%).

## E1.4 -- Direct empirical check (RETRACTED in E2.0)

Model-free: per frame, Honeycutt SEM of comp residuals about the **flux-sum ensemble mean**
(ens_med) vs production ``ensemble_scatter`` (residuals about **night-median comp_ref_map**).

| metric | value |
|--------|-------|
| n_fields | 166 |
| median (actual / quoted) | **13.23** |
| IQR | 7.91 / 13.23 / 20.18 |

**RETRACTED (WIDE-ERR E2.0):** ``m_i - ens_med`` is **not** a measurement residual;
``ens_med = -2.5*log10(sum F_i)`` so ``m_i - ens_med ~ m_i + 2.5*log10(n)`` tracks the
ensemble **brightness spread**, exactly the quantity ``comp_ref_map`` was introduced to
remove (photometry_core.py:3423-3429, "inflated-err bug").

**Falsification:** Spearman rho(E1.4 ratio, comp mag spread max-min) = **0.714**, p = 3.6e-27,
n = 166. Spearman vs n_comp: rho = -0.056, p = 0.47 (not significant). Strong correlation
with spread **confirms artifact**; **WIDE-ERR-SEM-ARITH verdict withdrawn.**

Arithmetic check (W2): median comp spread 0.92 mag; ~8 comps -> std ~ 0.27 mag, SEM ~ 93 mmag
vs quoted ~ 9.4 mmag, ratio ~ 10 (E1.4 measured 13.2).

## Combined line (E1, superseded by E2)

~~**WIDE-ERR-SEM-ARITH**~~ **WITHDRAWN** (E2.0 falsification). Clipping (E1.3) not present
(ratio 1.0). See CURSOR_RESULT_wide_err_e2.md for comp-residual correlation analysis.

## Files created

- dev/results/CURSOR_RESULT_wide_err_e1.md (this file)
- dev/tools/wide_err_e1.py
- tmp/wide_err_e1/wide_err_e1.json

## Errors

None blocking.
