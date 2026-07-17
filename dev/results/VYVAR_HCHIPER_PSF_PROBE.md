# h & chi Per (drafts 375 + 380) -- PSF characterization probe

Read-only diagnostic on existing solved data. PSF flags remain OFF. Production photometry
untouched. Ground truth from solved `MASTERSTAR.fits` (not raw lights with placeholder WCS).

Probe run: 2026-06-08. Source: `compute_crowding_index`, `_load_wcs_meta`,
`_epsf_prepare_stars` + `_epsf_build_imagepsf_from_stars` (QC only, no ePSF files written).
`gaia_db_max_g` = 16.0 (configured).

---

## Step 1 -- Solved metadata (per draft + setup)

| draft | setup | scale ("/px) | class | FWHM px | FWHM" | NAXIS | lights FITS |
|-------|-------|---------------|-------|---------|-------|-------|-------------|
| 375 | B_20_2 | 1.302 | coarse bin2 | 3.29 | 4.29 | 3126x2088 | 24 |
| 375 | L_20_2 | 1.302 | coarse bin2 | 3.36 | 4.37 | 3126x2088 | **30** |
| 375 | R_20_2 | 1.303 | coarse bin2 | 3.10 | 4.03 | 3126x2088 | 24 |
| 375 | V_20_2 | 1.304 | coarse bin2 | 3.14 | 4.09 | 3126x2088 | 24 |
| 380 | B_20_2 | 1.302 | coarse bin2 | 3.80 | 4.94 | 3126x2088 | 24 |
| 380 | L_20_2 | 1.303 | coarse bin2 | 3.80 | 4.95 | 3126x2088 | **30** |
| 380 | R_20_2 | 1.302 | coarse bin2 | 3.80 | 4.94 | 3126x2088 | 24 |
| 380 | V_20_2 | 1.302 | coarse bin2 | 3.80 | 4.94 | 3126x2088 | 24 |

**Scale classification:** all setups are **coarse bin2 ~1.30"/px** (Newton 2x2 binning), not
fine bin1 (~0.65"/px). Raw-light inspector read 3600"/px from placeholder `CDELT1`; solved
MASTERSTAR WCS gives the real ~1.3"/px scale.

**Richest ensemble:** **L (Luminance)** -- 30 aligned lights per draft vs 24 for B/R/V.
`per_frame_catalog_index` rows: L=15, others=12 (half the light count; paired/indexed subset).

---

## Step 2 -- Crowding index (Gaia cone + LC star blend map)

Scalars from `compute_crowding_index(..., lc_star_set=True)`:

| draft | setup | blend@1FWHM | blend@2FWHM | Gaia dens (/arcmin2) | footprint (arcmin2) | frame_lim | eff_lim | cat bottleneck |
|-------|-------|-------------|-------------|----------------------|---------------------|-----------|---------|----------------|
| 375 | B_20_2 | 0.031 | 0.071 | 0.42 | 3071 | 13.75 | 13.75 | no |
| 375 | L_20_2 | 0.037 | 0.101 | 0.89 | 3074 | 14.88 | 14.88 | no |
| 375 | R_20_2 | 0.030 | 0.073 | 0.68 | 3079 | 14.47 | 14.47 | no |
| 375 | V_20_2 | 0.032 | 0.076 | 0.66 | 3081 | 14.43 | 14.43 | no |
| 380 | B_20_2 | 0.038 | 0.085 | 0.39 | 3071 | 13.62 | 13.62 | no |
| 380 | L_20_2 | 0.044 | 0.123 | 0.83 | 3077 | 14.76 | 14.76 | no |
| 380 | R_20_2 | 0.040 | 0.103 | 0.60 | 3072 | 14.26 | 14.26 | no |
| 380 | V_20_2 | 0.039 | 0.102 | 0.59 | 3075 | 14.23 | 14.23 | no |

`catalog_limit_g` = 15.96 for all; `catalog_is_bottleneck` = false (frame SNR limit binds first).

### LC star `nn_dist_fwhm` buckets (blend worklist proxy)

| draft | setup | N LC stars | is_blended=True | <1.0 FWHM | 1.0-1.5 | 1.5-2.0 | >2.0 |
|-------|-------|------------|-----------------|-----------|---------|---------|------|
| 375 | L_20_2 | 999 | 77 | 47 | 30 | 43 | 879 |
| 380 | L_20_2 | 1014 | 87 | 50 | 37 | 70 | 857 |

(L-band has the largest LC star set; R/V on 375 have smaller photometry footprints -- 244-257
LC stars -- reflecting partial filter photometry coverage.)

**Crowding decision input:** At **~1.3"/px**, Gaia-field blend fractions are modest but **not
zero**: ~3-4% of catalog stars have a neighbour within 1 FWHM; ~10-12% within 2 FWHM on L.
Among LC stars, **77-98 are flagged `is_blended`** with **47-61 hard blends** (nn < 1.0 FWHM)
on the richest L setup. That is enough for **NEIGHBOR-SUB / deblend routing** experiments,
but the cluster is **not ultra-dense** (most stars are >2 FWHM isolated). PSF-vs-aperture at
this coarse scale would mostly test parity on isolated comps, not fine-scale blend resolution.

---

## Step 3 -- ePSF QC probe (L_20_2 only; read-only build)

| draft | n_stars | epsf_FWHM px | ePSF/input ratio | epsf_asymmetry | nan_frac |
|-------|---------|--------------|------------------|----------------|----------|
| 375 | 322 | 2.236 | **0.666** | 0.0014 | 0.0 |
| 380 | 293 | 2.236 | **0.589** | 0.0009 | 0.0 |

- **Ratio < 1:** **Resolved** (`docs/VYVAR_EPSF_FWHM_TEST.md`): dominant cause is **inflated
  seeing denominator** (OBS_FILES L ~3.84 px) vs true ePSF/stellar core ~2.0 px (Moffat fit). Secondary:
  buggy half-max pinned at 2.236 px. **Not** a narrow ePSF build. Do not withhold PSF on ratio alone.
- **Asymmetry << 0.1:** no coma/tracking-smear QC warning on real cluster comps (consistent
  with validation harness note: quad-fold metric detects **asymmetric** distortion, not
  symmetric elongation).
- **380 seeing worse** than 375 (VY_FWHM ~3.8 vs ~3.4 px on L); ePSF ratio lower (0.59).

---

## Read (one paragraph)

h & chi Per at **bin2 ~1.30"/px** is the **right field for blend/crowding/asymmetry design work**
at coarse Newton scale: L-band gives the **richest frame ensemble (30 lights)** and the highest
Gaia effective density (~0.83-0.89 stars/arcmin2 below the SNR limit), with **~4% catalog blend
fraction at 1 FWHM** and **dozens of LC targets flagged blended** (77-98 on L). It is **not** the
field for fine-scale **PSF-vs-aperture validation** (that needs ~0.65"/px bin1). ePSF QC on L
shows **clean, low-asymmetry** models (asymmetry ~0.001). Ratio 0.59-0.67 reflects **seeing
denominator inflation** (OBS ~3.8 px vs core ~2.0 px), not a bad ePSF -- proceed on per-star gating
and blend RMS.
**Prefer draft 375 L** for initial PSF sandbox (slightly better seeing, higher ePSF/input ratio);
380 is a useful second night with similar crowding numbers but broader PSF.

**Crowding recompute (2026-06-09):** with `VY_FWHM_GAUSS` (~2.73 px) instead of `VY_FWHM`
(~3.4-3.8 px), corrected is_blended = 58/53 (vs 77/87 baseline); still enough for NEIGHBOR-SUB.
See `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.

No production changes from this probe.
