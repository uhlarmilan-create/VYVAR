# h & chi Per crowding recompute with VY_FWHM_GAUSS (375 L + 380 L)

Historical diagnostic (2026-06-09). Prior crowding used `VY_FWHM` from `crowding_index._load_wcs_meta`
(DAOStarFinder search parameter, ~3.4-3.8 px). **Production fix applied (TODO-FWHM-CONSISTENCY):**
`header_core_fwhm_px` now drives `_load_wcs_meta` and `get_epsf_fwhm_from_context`; the corrected
numbers below are the **live** crowding output on 375/380 L.

Script: `tmp/hchiper_crowding_recompute.py`. JSON: `tmp/hchiper_crowding_recompute.json`.
Self-check: baseline reproduces reported probe numbers (blend_frac / 77 / 87).

---

## Step 1 -- FWHM story (native px)

| draft | VY_FWHM (DAO) | VY_FWHM_GAUSS (hdr) | ePSF Moffat | stellar Moffat | seeing L |
|-------|---------------|---------------------|-------------|----------------|----------|
| 375 L | 3.356 | **2.744** | 2.131 | 1.916 | 3.839 |
| 380 L | 3.796 | **2.730** | 2.202 | 2.104 | 3.839 |

`VY_FWHM` is the DAO detection/search scale (inflated vs core). `VY_FWHM_GAUSS` is the pipeline
2D Gaussian fit on MASTERSTAR (~2.73 px) -- narrower than `VY_FWHM`, wider than per-star Moffat
medians (~1.9-2.1 px) from the decisive ePSF test. It is the stored core proxy consistent with
the aperture path. `seeing` (OBS_FILES L) remains the outer seeing scale (~3.84 px).

---

## Step 2 -- Crowding baseline vs VY_FWHM_GAUSS corrected

### Catalog blend fractions (Gaia cone, Part B)

| draft | FWHM used | blend@1FWHM | blend@2FWHM |
|-------|-----------|-------------|-------------|
| 375 L | VY_FWHM 3.356 | 0.0373 | 0.101 |
| 375 L | VY_FWHM_GAUSS 2.744 | **0.0260** | **0.0747** |
| 380 L | VY_FWHM 3.796 | 0.0436 | 0.1227 |
| 380 L | VY_FWHM_GAUSS 2.730 | **0.0256** | **0.0731** |

Smaller FWHM -> smaller neighbour search disk -> lower catalog blend fractions (expected).

### LC star blend table (`_build_blend_targets_df`, lc_star_set=True)

| draft | FWHM | is_blended | <1.0 | 1.0-1.5 | 1.5-2.0 | >2.0 | N total |
|-------|------|------------|------|---------|---------|------|---------|
| 375 L | VY_FWHM | **77** | 47 | 30 | 43 | 879 | 999 |
| 375 L | VY_FWHM_GAUSS | **58** | 39 | 19 | 28 | 913 | 999 |
| 380 L | VY_FWHM | **87** | 50 | 37 | 70 | 857 | 1014 |
| 380 L | VY_FWHM_GAUSS | **53** | 34 | 19 | 30 | 931 | 1014 |

Corrected counts drop ~25% vs baseline but remain substantial. Hard blends (nn < 1.0 FWHM) stay
well above 10 per draft.

---

## Step 3 -- NEIGHBOR-SUB verdict

**PROCEED on h & chi Per (375 L and 380 L)** at the true core FWHM scale (`VY_FWHM_GAUSS`):

| draft | corrected is_blended | hard nn < 1.0 FWHM |
|-------|----------------------|---------------------|
| 375 L | 58 | 39 |
| 380 L | 53 | 34 |

Decision rule threshold (>= ~20-30 blended, >= ~10 hard): **met** on both drafts. The field is
not ultra-dense at core scale, but NEIGHBOR-SUB still has dozens of real targets. Fine-scale
draft 367 (0.39"/px) remains the PSF-vs-aperture validation field; h & chi Per remains valid for
coarse-scale deblend/crowding work.

---

## Production fix (DONE 2026-06-09)

**TODO-FWHM-CONSISTENCY:** shared `header_core_fwhm_px` in `masterstar_context.py`; two read sites
updated (`crowding_index._load_wcs_meta`, `psf_photometry.get_epsf_fwhm_from_context`). Aperture path
unchanged (`pipeline.py:9206`). Numeric photometry SHA `770966c3...` unchanged on draft_000366.
ePSF QC ratio on 375/380 L moves toward ~0.78-0.81 (denominator now ~2.73 px).

---

## Cross-references

- Prior probe: `docs/VYVAR_HCHIPER_PSF_PROBE.md`
- ePSF FWHM test: `docs/VYVAR_EPSF_FWHM_TEST.md`
