# CURSOR RESULT - IMPL-01

Date: 2026-08-15
Baseline: 2fd9071 (PRE-IMPL-01)
Tip: **f9464e5** (IMPL-01)

## What I did

Implemented measured curve-of-growth SNR aperture sizing (Item 1), Clear/unfiltered
colour **level** correction at export (Item 2), decision record for no ensemble cut
(Item 3), and OPEN register of four PRE-IMPL measurement corrections (Item 4).

## Item 1 - Aperture from measured CoG

Replaced analytic Gaussian EE in `compute_snr_optimal_aperture_table` with the draft
growth curve from existing `compute_per_frame_cog_correction` / COG-A1 machinery.

| Requirement | Implementation |
|-------------|----------------|
| Isolation | `snr_cog_isolation_fwhm=3` (no catalogue neighbour within 3 FWHM) |
| Normalisation | `cog_ref_fwhm * fwhm` (draft 514: ref_r ~15.64 px); flatness_tail_over_norm = **1.0** |
| Bounds | `bound_hit_by_mag` per bin; search extends to EE ladder end when measured |
| Fallback | `ee_path=gaussian_fallback` + `ee_fallback_reason` when n_cog < `cog_min_stars` (**8**; same as COG production gate; Q4 used 12 when available) |
| Universality | Per-draft measurement; no telephoto-tuned constants |

**Spec catch:** first CoG attempt used catalog sky on sky-subtracted frames (~0) so EE
tracked aperture area. Fixed by always estimating annulus sky on the frame used for COG.

### Draft 514 results (`IMPL_01_aperture_cog.json`)

| Mag | r_old (PRE-IMPL) | r_new | EE at new | EE old @ 2.711 | Bound |
|-----|------------------|-------|-----------|----------------|-------|
| 8 | 4.561 | 11.031 | 0.992 | 0.663 | none |
| 10 | 4.061 | 10.981 | 0.990 | 0.663 | none |
| 12 | 3.261 | 10.481 | 0.956 | 0.663 | none |
| 14 | 2.711 | 10.481 | 0.956 | 0.663 | none |
| 16 | 2.711 | 10.481 | 0.956 | 0.663 | none |

EE at production clamp 2.711 px on measured curve: **0.660** (matches Q4 0.663).
`n_bound_hits=0` on this rebuild (search window includes normalisation radius).

### Check-star scatter before/after (same raw frames)

Method: LOO median-ensemble MAD of instrumental mags at fixed r on 40 frames,
8 bright stars (G~8-12). Not a full Phase-2A re-export.

| | mmag |
|--|------|
| Before (r=2.711 px) | **29.09** |
| After (r_med=10.481 px) | **13.04** |

Scatter **decreased**. Measuring ~95% EE instead of ~66% reduced this LOO MAD by
~16 mmag on this sample.

### Q4 night EE quantity (one line)

Q4 `ee_night_variation_mad_mmag=0.008` is **frame-to-frame absolute EE of one isolated
star at fixed r** (common-mode aperture-loss amplitude, demeaned MAD), **not** the
residual after ensemble common-mode cancellation. It does not establish
aperture-loss-vs-seeing as a WIDE-ERR differential driver on that night.

## Item 2 - Colour level at export

Wired into existing colour-term path (`resolve_apply_color_term` /
`_compute_group_color_term_fit` / `apply_color_term`):

- Clear/unfiltered + finite `color_level_k_mag_per_bprp` -> mode `clear_level`
- Constant per target: `corr = k * (target_bp_rp - weighted_mean_comp_bp_rp)`
  (existing sign removes PRE-IMPL level bias `k*(ens-target)`)
- Never airmass-dependent
- `sigma_err += |delta_colour| * k_stderr` in quadrature
- LC columns: `ct_c1`, `ct_c1_stderr`, `ct_mode`, `ct_ok`, `ct_correction`
- AAVSO: `TRANS=YES` when applied; `#COLOR_LEVEL:` header lines with k +/- stderr

Config (this rig): `apply_color_term=auto`, `k=-0.373`, `stderr=0.090`, unit
`mag_per_bprp` in registry. Per-rig; not universal.

### Literature sanity

**Ship.** AAVSO CCD Photometry Guide: untransformed Clear/standard colour residuals
"as high as several tenths of a magnitude". |k|=0.373 mag/BP-RP is inside that range
for Delta(BP-RP)~1. Gaia G-V linear terms are smaller (~0.01-0.05); Clear red CMOS
to Gaia G is expected larger. Coefficient is measured per-rig, not a published
universal Clear->G transform.

## Item 3 - Decision

`docs/VYVAR_DECISIONS.md`: **no ensemble size cut in v1.0** (science flat through 90%;
cost <4%; Q1 scale absent/contaminated). Future cuts must be field-independent.

## Item 4 - OPEN register

Same file: four PRE-IMPL corrections (Q1 contaminated; Q2 one-column; Q5 wrong match;
Q4 EE quantity named). No code action.

## Acceptance

- `--fast`: re-run after test fixes (see tip commit)
- New params with units in `dev/validation/params_registry.json`
- ASCII English

## Files changed (principal)

- `src_py/photometry_core.py` - measured CoG SNR table; clear level CT; err prop
- `src_py/config.py`, `config.json` - k_level, snr_cog_isolation, apply auto
- `src_py/export_reports.py` - TRANS + COLOR_LEVEL provenance
- `src_py/band_classify.py` - legacy compare isolates clear-level k
- `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_PARAMS.md`
- `dev/validation/params_registry.json`
- `dev/tools/impl_01_measure.py`, `dev/results/IMPL_01_aperture_cog.json`
- tests

## Errors

None blocking. First CoG sky defect caught and fixed before shipping the table.
