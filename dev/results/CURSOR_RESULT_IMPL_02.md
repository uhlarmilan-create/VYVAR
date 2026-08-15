# CURSOR RESULT - IMPL-02

Date: 2026-08-15
Baseline: a27f10f (IMPL-01)
Tip: **ba3a33b**
Push: NO

## What I did

Fixed the contaminated CoG growth curve that fed the SNR aperture table, wired
seven hard gates with fire proofs, rebuilt the table, verified colour-level
sign (Part E), and established that Part D as written cannot validate the new
table via Phase 2A alone.

## Part A - Root cause (one sentence)

The CoG sky annulus (3-5xFWHM) overlapped the EE ladder (to 4.5xFWHM), so the
estimated sky included starlight and left a residual area-like term in outer EE;
separately, empty `proc_*.fits` globs left `bkg_var` unset so the SNR table used
noise-floor-as-sky with Howell reconstruction and lost magnitude dependence.

Measured numbers (draft 514, SHA a27f10f era diagnosis in
`dev/results/IMPL_02_part_a_bkg.json`):

| quantity | value |
|---|---|
| Frame star-free edge median | ~1405 ADU (not sky-subtracted) |
| CoG annulus sky (old 3-5xFWHM) | ~1404 ADU, annulus overlapped ladder |
| Saved IMPL-01 `bkg_var` | null |
| `sigma_bkg_ap` -> var/px | ~2186 ADU^2/px |
| Fixed star-free `bkg_var` | ~1873 ADU^2/px |
| Fixed sky pedestal | ~1919 ADU/px |
| Fixed r90 | **6.0 px** (Q4 band 5-6) |

## Part B - Seven invariants

Module: `src_py/snr_cog_gates.py`. Fire proofs:
`dev/tests/test_impl_02_snr_cog_gates.py` (9 tests, each gate fires on purpose).

Measured CoG path: table write refused on gate failure
(`aperture_snr_table_REJECTED.json`). Gaussian fallback: CoG gates informational
only (unit tests / intentional fallback still write).

## Part C - New table (draft 514)

Artifact: `dev/results/IMPL_02_aperture_cog.json` (+ draft
`aperture_snr_table.json`).

| Mag | r_opt [px] | EE at opt | Bound |
|---:|---:|---:|---|
| 8 | 8.499 | 0.962 | none |
| 10 | 5.499 | 0.881 | none |
| 12 | 2.999 | 0.692 | none |
| 14 | 1.999 | 0.534 | none |
| 16 | 1.699 | 0.457 | none |

FWHM=5.195, gain=3.17, bkg_var=1873, r90=6.0, flatness_outer=0.9976,
gates_ok=True, n_bound_hits=0.

**vs prediction (faint 5-6 px, EE 0.85-0.95):** bright bins land in that class;
faint bins go smaller (EE ~0.45-0.70). With this EE curve and measured background
variance, background-limited SNR optima sit interior to r90. Not automatic
wrong; reported plainly.

### AC interaction

- Production Method B `aperture_correction_enabled=True`:
  `compute_aperture_correction` (large/small DeltaM on comps). Applied as
  `mag_calib_ac = mag_calib + delta_m_corr` when ok.
- CoG AC `cog_aperture_correction_enabled=False` (default). The SNR growth curve
  does **not** feed Method B; they are separate.
- Some draft-514 summary rows show `ac_delta_m_corr ~ -0.11` mag when applied -
  Method B can absorb part of undersizing at export, so enclosed-fraction gains
  need not equal scatter gains one-for-one.

## Part D - DEFECT (corrected specification)

**Phase 2A re-export alone does not apply the new aperture table.**

Evidence (`dev/results/IMPL_02_part_d_defect.json`):

- SNR table at BO G=9.72 wants **r~6.0 px**
- `proc_*.csv` and production LC for BO still carry **aperture_r_px=4.211**
- Phase 2A `read_flux_from_csv` uses `dao_flux` and `aperture_r_px` from proc
  CSVs; it does not remeasure at SNR-table radii

**Corrected Part D:** re-run the per-frame aperture photometry stage that writes
proc CSVs (with the gated `aperture_snr_table.json`), then Phase 2A; then report
check-star / BO / FW scatter before vs after from production products.

Before (existing products, old proc apertures): BO mag_calib std ~146 mmag
(variable); FW ~14.8 mmag. check_kmag sidecars were absent on the pre-remeasure
tree.

## Part E - Colour sign

`dev/results/IMPL_02_part_e_colour.json`:

- Config k = -0.373 mag/BP-RP
- Formula: `corr = k*(target - ens) = -k*delta_colour_Q3`
- Post-correction G-controlled level: **+1.6 +/- 90.7 mmag/BP-RP**
- Consistent with zero at 2se: **True**
- Sign-inverted would be near **-746**; not observed

## `--fast`

OVERALL **PASS** (1396 passed, 27 skipped).

## Files changed

- `src_py/photometry_core.py` - CoG outer ladder, sky outside ladder, pedestal
  sky, science FITS discovery, no r_max widen, gates, FWHM raise
- `src_py/snr_cog_gates.py` - seven gates
- `src_py/pipeline.py` - science FITS discovery for precompute
- `dev/tests/test_impl_02_snr_cog_gates.py` - fire proofs
- `dev/tools/impl_02_*.py` - measure/rebuild/Part E
- `dev/results/IMPL_02_*.json` / this RESULT
