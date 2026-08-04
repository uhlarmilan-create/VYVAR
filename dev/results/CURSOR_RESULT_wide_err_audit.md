CURSOR RESULT - 2026-08-04 (WIDE-ERR AUDIT)

What I did
Read-only literature audit of ensemble photometry and error-budget code paths
(Honeycutt 1992, Broeg 2005, Stetson 1987, Everett & Howell 2001, Sokolovsky comp_qa).
Ran AUDIT 4 measurement harness on restored draft check-star fields.
Honeycutt (1992) full PDF blocked (IOP bot wall); primary Honeycutt quotes from
partial IOP fetch plus Richmond ensemble-software reproduction (Honeycutt 1992 cited).
Everett & Howell (2001) used as secondary for error-quadrature comparison where noted.

Pre/post manifest check: PASS / PASS.

## Output / findings

### AUDIT 1 -- ensemble_normalize vs Honeycutt (1992, PASP 104:435)

Literature access: Honeycutt (1992) PDF not retrievable (captcha). Secondary sources
used and named: (a) Honeycutt (1992) partial text from IOP abstract/introduction fetch;
(b) Michael Richmond, "Software for inhomogenous ensemble photometry" (explicitly
reproduces Honeycutt 1992 solvepht equations); (c) Everett & Howell (2001, PASP
113:1428) for ensemble error quadrature where Honeycutt error section unavailable.

| STEP | CODE (verbatim) | LITERATURE (verbatim or equation) | MATCHES / DIFFERS / CANNOT DETERMINE |
|------|-----------------|-----------------------------------|--------------------------------------|
| 1. Ensemble magnitude combination | `ens_flux_sum = float(np.sum(f_arr))` / `ens_med = float(-2.5 * math.log10(ens_flux_sum))` (photometry_core.py:3417-3419) | Honeycutt (1992): "calculate the difference between the instrumental magnitude of the program star and a comparison magnitude obtained from the sum of the intensities of perhaps a dozen of the brighter stars" (strict ensemble photometry) | MATCHES |
| 2. Per-comp reference | `comp_ref_map[cid] = float(np.median(_fin))` over all finite frames (photometry_core.py:3387-3390) | Honeycutt (1992): "mO is the mean instrumental magnitude of star s which would be observed in the absence of transparency variations" (LS unknown m0(s)); Richmond/Honeycutt doc: true magnitude M(i) from least-squares | DIFFERS |
| 3. Per-frame comp residual (SEM input) | `comp_resid = [(m - comp_ref_map[cid_j]) for cid_j, m in comp_pairs if cid_j in comp_ref_map ...]` (photometry_core.py:3430-3434) | Richmond/Honeycutt (1992) doc: `error(i,j) = m(i,j) - [ M(i) - e(j) ]` where e(j) is exposure zero-point offset | DIFFERS |
| 4. SEM formula | `std_ddof1 = float(math.sqrt(sum((x - sum(arr)/n)**2 for x in arr) / (n - 1)))` / `return std_ddof1 / c4 / math.sqrt(n)` (sigma_floor_core.py:48-49); c4 from Cochran/Bolch (sigma_floor_core.py:21-34) | Richmond/Honeycutt (1992) doc quality metric: `z2 = z1 / sqrt(N)` with z1 a weighted RMS of (corrected_mag - true_mag); c4 not in Honeycutt | DIFFERS (c4 added downstream, not Honeycutt; residual/weighting differ from z1/z2) |
| 5. Weighted ZP (Broeg 2005) | `rms_j = float(comp_rms_map.get(cid_j, ...))` then `weights.append((1.0 / (rms_j**2)) * tw)` (photometry_core.py:3455-3464) | Broeg, Fernandez & Neuhauser (2005, AN 326:134): weights w = 1/sigma^2; sigma from iterative pairwise CS comparison until convergence (Broeg 2005 slides: "Repeat until convergence") | DIFFERS (one pass; sigma is Phase-1 comp_rms, not Broeg-iterated per-frame sigma) |
| 6. ZP sigma-clip (Stetson 1987) | `if len(z) >= 4:` then `_keep = np.abs(z - _med) <= 3.0 * _sigma` with `_sigma = max(_mad / _MAD_CONSISTENCY, 1e-6)` once (photometry_core.py:3468-3486) | Stetson (1987, PASP 99:191; NED DAOPHOT lecture): iterative sky clip -- "recompute mean and standard deviation ... process continues until the mean and standard deviation stop changing" | DIFFERS (single-pass 3-sigma MAD, not iterative to convergence) |

### AUDIT 2 -- err budget assembly vs cited literature

| FUNCTION | DOCSTRING CLAIM | REFERENCE | MATCHES / DIFFERS / NOT CITED |
|----------|-----------------|-----------|-------------------------------|
| `_combine_err_with_ensemble_scatter_keyed` (photometry_core.py:3546-3550) | `err_total^2 = err_photon^2 + sem_rel^2 + scint_rel^2 + sigma_sys_rel^2` (relative-flux domain); SEM from Honeycutt residual std/sqrt(n) with c4 | Honeycutt (1992) -- cited for SEM origin | DIFFERS (Honeycutt/Richmond z2 uses LS-corrected residuals and input weights; not this quadrature assembly) |
| `_combine_err_with_ensemble_scatter_keyed` | same | Everett & Howell (2001) eq. (3)-(4): sigma_j,ens = [Sum_i (1/sigma_i,j^2)]^(-1/2); sigma_j^2 = sigma_j,ens^2 + sigma_j,p^2 | DIFFERS (Everett uses inverse-variance of per-star measurement errors, not empirical std of comp_ref residuals) |
| `combine_production_err_rel` (sigma_floor_core.py:64-86) | Quadrature: photon + ensemble SEM + scint + sigma_sys (relative flux) | VYVAR_SIGMA_FLOOR_SPEC.md (internal); Howell (1989) photon term upstream | DIFFERS / NOT CITED (scintillation Young/Osborn and sigma_sys floor are VYVAR additions beyond Everett eq. 4) |
| `_combine_err_with_ensemble_scatter_keyed` caller scint term | per-epoch `scintillation_mag_per_epoch` wired at photometry_core.py:9866-9886 | Young/Osborn via sigma_budget.py / Osborn et al. 2015 | NOT CITED in `_combine_err` docstring (present in code path) |
| `sigma_sys_mag` | additive in quadrature after conversion mag->rel | Everett & Howell (2001) | NOT CITED (Everett treats photometric sigma_p and ensemble sigma_ens only; no multiplicative rig floor) |
| `comp_qa_core.sokolovsky_indices` / LOO locus | `sigma_iqr`, `inv_nv`, `spike` on leave-one-out diff series (comp_qa_core.py:100-124) | Sokolovsky et al. 2017 (config.py:609 comp_qa label) | NOT CITED in err assembly (QA metadata only; production err uses Honeycutt SEM path above) |

References found in scoped functions or direct callers:
- Honeycutt (1992): photometry_core.py:3380, 3423, 3443, 3546; sigma_floor_core.py:38
- Broeg et al. (2005): photometry_core.py:3462 (ZP weights); comp selection comments
- Stetson (1987): photometry_core.py:3470 (ZP clip comment)
- Howell (1989): upstream `_photometric_error` before combine (photometry_core.py ~9881)
- Young/Osborn scintillation: sigma_floor_core.scintillation_mag_per_epoch (caller 9866-9886)
- VYVAR_SIGMA_FLOOR_SPEC: sigma_sys additive quadrature
- Sokolovsky comp_qa: comp_qa_core.py (separate from err combine)

Everett & Howell (2001) on sigma_sys: treats systematic rig floor as NOT part of their
eq. (3)-(4); VYVAR `sigma_sys_mag` is an additive magnitude floor in quadrature
(sigma_floor_core.combine_production_err_rel), not a multiplicative scaling.

Sokolovsky vs Honeycutt ensemble error: comp_qa uses LOO flux-sum ensemble diff minus
night median (`loo_diff_series`, comp_qa_core.py:76-97) and dispersion indices
(`sigma_iqr`); this is not wired into `_combine_err_with_ensemble_scatter_keyed`.

### AUDIT 3 -- check star code path

| # | CODE PATH + LINE | INTERPRETATION |
|---|------------------|----------------|
| 1 | Target err: `_combine_err_with_ensemble_scatter_keyed(...)` at photometry_core.py:9881 inside `_phase2a_process_one_target`. Check star diagnostic LC: `photometer_check_star_production_path` calls `_phase2a_process_one_target` (10886-10903) -> same combine at 9881. `compute_check_ensemble_mag_calib` (check_star_kmag.py:629-639) produces `kmag`/`ensemble_scatter` only; `save_check_kmag_sidecar` (800-860) writes no `err` column. | W1 check-star `err` (via production-path LC) uses the SAME `_combine_err_with_ensemble_scatter_keyed` path as the parent target. Kmag sidecar path does not assemble err. |
| 2 | `compute_check_ensemble_mag_calib`: `other_ids = [c for c in comp_ids if c != cid ...]` (530-531); `ensemble_normalize(comp_lc[cid], other_lc, ...)` (629-639). `photometer_check_star_production_path`: `comp_subset = parent_comps.loc[parent_comps["catalog_id"] != check_cid]` (10866); `_comp_index={..., check_cid: comp_subset}` (10877-10879); then `ensemble_normalize(target_lc, comp_lc, ...)` at 9515 with check as target. | Check-star `mag_calib`/`mag_calib_final` uses parent comparison set minus the check star (check star is target, not in comp ensemble). Docstring at 10853-10855 matches code. |
| 3 | `ensemble_sem_mag_from_residuals`: `n = len(arr)` from per-frame `comp_resid` list (sigma_floor_core.py:41-49); called from ensemble_normalize when `len(comp_resid) >= 2` (photometry_core.py:3435-3438). | sqrt(n) uses the count of comps in the reduced ensemble on that frame (parent comps minus check, after good_ids selection), not the parent-target ensemble size including check. |
| 4 | `comp_ref_map` built inside `ensemble_normalize` loop over `good_ids` (photometry_core.py:3382-3390) on each call. Check-star call passes `other_lc` only (check_star_kmag.py:625-632). | comp_ref_map is REBUILT for the check-star ensemble (excluding check star); not reused from the parent target's earlier `ensemble_normalize` call. |

Docstring mismatch note: `_combine_err_with_ensemble_scatter_keyed` docstring cites
Honeycutt SEM (3546-3547) but check-star ensemble SEM is computed on the reduced
comp set (n-1 relative to full comp list when check was listed as a comp).

### AUDIT 4 -- measurement (<comp_resid>_frame)

Harness: dev/tools/wide_err_audit4.py
Artifact: tmp/wide_err_audit/audit4_comp_resid_frame_mean.json
Check star: 1499906247391001088; draft draft_000435_snapshot_skysurface_20260716;
ensemble = parent comps minus check star (production check-star path).

Definition: per frame, `<comp_resid>_frame` = mean over comps of (m_i - comp_ref_map[cid])
with comp_ref_map = night median per comp (same as ensemble_normalize SEM inputs).

| Quantity | Value (mag) |
|----------|-------------|
| n_fields measured | 164 |
| median over fields of (median_f `<comp_resid>_frame`) | -0.000946 |
| std over fields of (median_f `<comp_resid>_frame`) | 0.001807 |
| IQR over fields of (std_f `<comp_resid>_frame`) | 0.011647 |
| median over all frames and all fields of `<comp_resid>_frame` | -0.000812 |

Per-field medians and stds: tmp/wide_err_audit/audit4_comp_resid_frame_mean.json

## Errors (if any)

None.

## Files changed

- dev/tools/wide_err_audit4.py (new harness)
- dev/results/CURSOR_RESULT_wide_err_audit.md (this report)
- tmp/wide_err_audit/audit4_comp_resid_frame_mean.json (measurement artifact)

This audit reports what the code does and does not match. It does not draw a verdict on WIDE-ERR.
