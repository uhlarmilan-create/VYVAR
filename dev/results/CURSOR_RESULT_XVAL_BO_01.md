CURSOR RESULT - 2026-08-16 XVAL-BO-01

What I did
Traced the Phase 2A check_kmag writer, dumped BO frame 001 per-star fluxes,
and reconstructed both Layer-3 flux-sum and production mag_calib series from
proc dao_flux. Run SHA da9cce4. No science-code fix. Push not authorized.

## X1 - Code path (from source)

Writer:
`_phase2a_process_one_target` -> `compute_check_ensemble_mag_calib`
(`check_star_kmag.py`) -> `ensemble_normalize` (`photometry_core.py`) ->
`save_check_kmag_sidecar` -> `lightcurves/check_kmag_<target>.csv`.

(a) Flux values consumed
- `read_flux_from_csv` takes proc CSV column `dao_flux` (not the name `flux`).
  On draft 515 frame 001, `flux == dao_flux` for every dumped star (see dump).
- Per-star aperture radius is the proc CSV `aperture_r_px` (DAO aperture that
  produced that dao_flux). On BO frame 001 these differ by star under IMPL-05
  per-mag sizing, e.g. bright comp 1500748301498613248 at 8.499 px, check at
  7.999 px, faint comps at 3.5-4.0 px. Same authority for BO and FW.

(b) Combination formula
Documented in `ensemble_normalize` docstring and DECISIONS:

1. Unweighted AIJ flux sum for the ensemble magnitude:
   `ens_med = -2.5 log10(sum_i 10^(-0.4 m_i))`
   `delta_mag = m_focus - ens_med`
2. Catalog-anchored science product:
   `mag_calib = m_focus + sum(w_j * (G_j - m_j)) / sum(w_j)`
   with `w_j = 1/rms_j^2` (Phase-1 `comp_rms`, then optionally updated by
   `pytics_iterative_weights` before the check call).
3. The sidecar column `kmag` is `mag_calib` of the check star, NOT `delta_mag`.

DECISIONS (Broeg/ZP audit, ~2026-06-15): "`delta_mag` (flux-sum) retained as
AIJ-validation / diagnostic column; reporting `mag_calib` already uses partial
Broeg (ZP offset only)." Ensemble *combine* stays unweighted flux-sum;
weights apply only to the catalog ZP offset.

(c) Per-target branch
None for BO vs FW. Same function. Differences come only from ensemble
membership, n_comp, rms (incl. pytics), and catalog G.

## X2 - Frame 001 dump

`dev/results/XVAL_BO_01_frame001_dump.json`

BO CVn / check 1497613731286514432 / `proc_BO_CVn_Light_001.csv` / SHA da9cce4.

Per-star (quantity labels in JSON): catalog_id, role, aperture_r_px, flux,
dao_flux, mag_inst_from_dao, catalog_G_mag, comp_rms_mag, zp_weight_1_over_rms2.

Epoch formulas on that frame:
- kmag_fluxsum (Layer-3): 1.0368 mag
- kmag_magcalib_phase1_rms: 8.5728 mag
- production sidecar kmag: 8.6068 mag
(Absolute mag_calib is on the catalog scale; flux-sum is differential - large
level offset is expected. Scatter of the night series is the acceptance meter.)

## X3 - Why FW matches and BO does not

Evidence (check-MAD, 1.4826*MAD*1000, n=134, check 1497613731286514432):

| Series | BO [mmag] | FW [mmag] |
|---|---:|---:|
| production sidecar | 6.7132 | 8.2010 |
| Layer-3 / delta_mag from dao_flux | 10.3294 | 8.1966 |
| mag_calib + Phase-1 rms | 8.6825 | 7.9615 |
| mag_calib + pytics rms | 6.7133 | 8.2010 |
| corr(prod, delta_mag) | 0.643 | 0.997 |
| corr(prod, mag_calib_pytics) | 1.000 | 1.000 |
| medabs(prod, mag_calib_pytics) [mmag] | 0.0002 | 0.0003 |

Conclusion:
- The architect's harness and Layer-3 formula are correct for `delta_mag`.
  BO independent 10.3294 mmag is exactly production `delta_mag` scatter.
- Production sidecar is `mag_calib` after `pytics_iterative_weights`, not
  Layer-3 `delta_mag`. Reconstructing that path recovers BO 6.7133 and FW
  8.2010 (byte-level series match to sidecar).
- FW's 0.004 mmag agreement with Layer-3 is a scatter coincidence: for the
  FW 8-comp set, `delta_mag` MAD and `mag_calib_pytics` MAD happen to agree
  (corr 0.997). For BO's 5-comp set (incl. bright G~8.0), pytics reweights
  the catalog ZP strongly (e.g. 1500748301498613248 rms 0.00806 -> 0.01234;
  quieter comps get relatively higher weight), and `mag_calib` scatter drops
  to 6.71 while unweighted flux-sum stays at 10.33.
- Not a second flux authority and not a BO-only code branch. Flux is dao_flux
  for both. Finding class: CHECK-SIDECAR-PRODUCT - sidecar `kmag` is
  `mag_calib` (catalog ZP + pytics rms), while the documented Layer-3
  AIJ cross-check product is `delta_mag`. Doc cites above; no fix in this task.

Secondary note: removing the bright comp from an unweighted sum improves
Layer-3 scatter (architect 10.33 -> 7.82) but cannot reach 6.71; that number
is only reached on the weighted ZP path after pytics.

## X4 - Blast radius

(a) Exported target magnitudes: YES affected by the same `ensemble_normalize`
    `mag_calib` path (plus later CT/AC on the target LC). Publication mags are
    the catalog-ZP product, not pure Layer-3 flux-sum.
(b) Exported MAGERR: YES - err assembly uses ensemble_scatter / photon terms
    from the same Phase 2A path (not the check sidecar).
(c) Check sidecar only: NO - the BO/FW MAD discrepancy is explained by which
    column the acceptance meter read (`kmag`=`mag_calib` vs Layer-3
    `delta_mag`), but the underlying combine is shared with target export.

Publication readiness depends on understanding (a)/(b) as the mag_calib
product; Layer-3 remains the AIJ diagnostic (`delta_mag` on the LC CSV).

## Spec defects

1. Task assumed sidecar equals Layer-3 flux-sum; production stores mag_calib.
   Corrected step: compare harness to `delta_mag` OR reconstruct mag_calib
   with pytics-updated 1/rms^2 ZP (done).
2. Column name `flux` vs code authority `dao_flux` - identical here; still
   label which column the harness used.

## Files

- dev/results/XVAL_BO_01_frame001_dump.json
- dev/results/CURSOR_RESULT_XVAL_BO_01.md
- dev/tools/xval_bo_01_dump.py

## Errors

None.
