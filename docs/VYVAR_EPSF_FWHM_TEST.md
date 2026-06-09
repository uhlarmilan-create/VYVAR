# Decisive ePSF FWHM test -- drafts 375 L + 380 L (h & chi Per)

Read-only diagnostic. Resolves why `epsf_vs_input_fwhm_ratio` < 1 on the h & chi Per probe.
No production, config, or estimator changes.

**Method:** ePSF built in-memory via production path (`_epsf_prepare_stars` +
`_epsf_build_imagepsf_from_stars`; no persisted `masterstar_epsf.fits` on disk). Independent
FWHM measurements on the built array and on ~30 bright isolated stars in `MASTERSTAR.fits`.

**Primary FWHM method:** 2D Moffat fit (`Moffat2D` + `LevMarLSQFitter`, alpha=2.5).
**Cross-check:** 2D Gaussian fit. **Secondary:** careful azimuthal median profile (0.25 os px bins,
peak from Moffat fit, no sky re-subtraction).

Probe date: 2026-06-09. Raw JSON: `tmp/epsf_fwhm_test.json`.

---

## Results table (all FWHM in native px)

| draft | fwhm_moffat | fwhm_gauss | fwhm_azim | buggy_halfmax | fwhm_stars | seeing_L (OBS) | VY_FWHM hdr |
|-------|-------------|------------|-----------|---------------|------------|----------------|-------------|
| 375 L | **2.131** | 2.242 | 2.087 | 2.236 | **1.916** | **3.839** | 3.356 |
| 380 L | **2.202** | 2.326 | 2.287 | 2.236 | **2.104** | **3.839** | 3.796 |

Derived ratios (vs seeing_L):

| draft | buggy/seeing | moffat/seeing | moffat/stars |
|-------|--------------|---------------|--------------|
| 375 L | 0.582 | 0.555 | 1.112 |
| 380 L | 0.582 | 0.574 | 1.047 |

ePSF build used 322 / 293 stars respectively (osamp=2). Stellar Moffat: n=30 isolated fits each.

---

## Verdict per draft

| draft | Dominant explanation |
|-------|---------------------|
| 375 L | **EXPLANATION 3:** OBS_FILES L seeing (~3.84 px) is inflated vs real PSF core (~1.9-2.1 px). Ratio<1 is **benign** -- denominator problem, not a bad ePSF. |
| 380 L | **EXPLANATION 3:** same -- seeing ~3.84 px vs ePSF/stellar core ~2.1 px. |

**Secondary (both drafts): EXPLANATION 1** -- `buggy_halfmax` pinned at **2.236 px** (= sqrt(5)
native, sqrt(20) oversampled grid artifact) independent of draft and input FWHM; Moffat/azimuthal
give ~2.09-2.29 px. Estimator misreads by ~0.05-0.15 px but is **not** the dominant cause of
ratio 0.59-0.67.

**EXPLANATION 2 rejected:** `fwhm_moffat` matches `fwhm_stars` within ~10% (375: 2.13 vs 1.92;
380: 2.20 vs 2.10). EPSFBuilder is **not** producing a core systematically narrower than real
cluster stars.

**Azimuthal vs Moffat:** agree within 0.04-0.08 px when implemented carefully (peak from Moffat,
fine bins, no sky re-subtract). Defer to Moffat as primary.

---

## One-paragraph read

The h & chi Per ratio<1 mystery is **resolved**: the built ePSF has a true core FWHM of ~2.1 px
(Moffat fit), matching isolated stellar Moffat fits (~1.9-2.1 px), while the ratio denominator
(`get_epsf_fwhm_from_context` / OBS_FILES median **~3.84 px** on L) measures a **much wider**
seeing scale than the PSF core the ePSF reproduces. **Do not withhold PSF** based on ratio alone.
The legacy half-max QC (~2.236 px constant) is a **secondary diagnostic artifact** (sqrt-grid
discretization); fixing it (TODO-EPSF-1-FWHM-QC) will improve QC fidelity but will not change the
fundamental ratio gap until the **seeing denominator** is aligned with stellar/ePSF core FWHM.

---

## Cross-references

- Probe: `docs/VYVAR_HCHIPER_PSF_PROBE.md`
- EPSF-1 audit: `docs/VYVAR_EPSF_AUDIT.md` (updated with resolved cause)
- Harness V3e: synthetic Tier-A ratio=1.127 (ideal case); real-field ratio driven by denominator
