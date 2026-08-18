# VYVAR validation harness

Inject-and-recover validation under `dev/tests/validation/` verifies cited VYVAR algorithms
against **known synthetic ground truth**. This module is **not production**; it generates
its own FITS and sidecars and must not alter draft photometry byte-identity.

## Honesty caveat

Synthetic frames test what the generator injects (Moffat PSF + Poisson/read noise +
gradient): a faithful but simplified sky. The harness validates **logic and recovery**
(does comp_qa flag the bad comp? does the trust gate gate? does crowding classify blends?),
not every real systematic (scintillation, true flat residuals, detector nonlinearity).

**Necessary, not sufficient** -- real-data drafts remain the final word.

## How to run

```bash
python -m tests.validation.recover --all
```

Outputs:

- `dev/tests/validation/data/validation_report.json`
- `dev/tests/validation/data/validation_report.md`

Partial runs: `--tier-a`, `--tier-b`, `--v3`, `--a9`.

See also `dev/tests/validation/README.md` for the full matrix.

## What it proves

| Tier | Scope | Real VYVAR entry points |
|------|-------|-------------------------|
| A | Single-frame contaminations | `crowding_index._build_blend_targets_df`, `comp_qa_core.sokolovsky_indices`, `photometry_core._catalog_only_fixed_aperture_flux`, SEP cross-val |
| B | 60-frame series | LombScargle LC, Sokolovsky comp QA, `trust_flag_core.evaluate_target`, `fit_color_term_c1` |
| V3 | Targeted utilities | `time_utils` BJD, pipeline airmass, `calibration.get_processed_master`, blind WCS proxy |
| A9 | NEIGHBOR-SUB acceptance envelope | `a9_core.measure_cell` on blend grid; `plain_aperture` baseline + `neighbor_sub` scored path |
| V3d | Fine-scale PSF vs aperture vs truth | `psf_photometry_stars`, `_catalog_only_fixed_aperture_flux`, real ePSF build; accuracy/precision/P3 pillars |
| V3e | ePSF FWHM QC (EPSF-1) | `_epsf_build_imagepsf_from_stars` + `_epsf_fwhm_native_from_profile`; OLD vs NEW ratio table |

## A9 NEIGHBOR-SUB acceptance envelope (steps 1-2)

A9 defines the **plain-aperture contamination map** (the problem) and scores the joint-fit
NEIGHBOR-SUB core (`psf_neighbor_sub.neighbor_sub_target_flux`) per cell vs zone-specific PASS rules.

- Generator: `dev/tests/validation/gen_a9.py` (truth sidecar; grid sep x delta_mag)
- Measurement: VYVAR `_catalog_only_fixed_aperture_flux` with AppConfig radii (not raw photutils)
- Contexts: **coarse** (FWHM 3.2 px, 1.30"/px) and **fine** (FWHM 6.4 px, 0.65"/px)
- Reports: `dev/tests/validation/data/tier_a9/a9_envelope.json` + `.md` + plain/gain heatmap PNGs
- Scoring: `measure_cell(mode="neighbor_sub")` -- HIGH_VALUE cells must recover (>=80% contamination
  reduction); REFUSE cells must guard-refuse; CLEAN cells no-op
- PSF variants: legacy `mismatch` (stress test) and EPSF-audit `realistic` (see diagnostic below)

Run: `python -m tests.validation.run_a9` or `python -m tests.validation.recover --a9`.

Coarse neighbor_sub pass rates (2026-06-08): **ideal 85.7%**, **legacy mismatch 21.4%**.

#### PSF-mismatch diagnostic (step 2b gate)

```bash
python -m tests.validation.run_a9_mismatch_diagnostic
```

Report: `dev/tests/validation/data/tier_a9/a9_mismatch_diagnostic.md`

**Legacy `mismatch`:** fit beta=2.0 vs inject beta=2.5; neighbour inject FWHM x1.12 (model/star
**0.89** on neighbour -- inverted vs field ePSF audit). Over-aggressive; not a realistic field test.

**`realistic` (EPSF audit anchor):** model/star FWHM **1.08**, beta matched, inject ellipticity
e=0.08.

**Post step-2a guards (2026-06-08):** inclusive sep floor `<=0.8`, catalog-anchored
`neighbor_overfit` / `target_undershoot` / `subtract_harmed`. Re-run:
`python -m tests.validation.run_a9_mismatch_diagnostic`

| Metric | Pre-2a | Post-2a |
|--------|--------|---------|
| FAIL-SILENT | 14 | **0** |
| HV PASS-RECOVER | 16.7% | **17.6%** |
| REFUSE correctness | 62.5% | **100%** |
| Verdict | BLOCK_2B_GUARDS | **SAFE_LOW_YIELD** |

2b **not started** at coarse scale: fail-safe achieved but yield low at bin2.

#### Fine-scale draft 367 diagnostic (2026-06-08)

```bash
python -m tests.validation.run_a9_draft367
```

ePSF-vs-star mismatch on draft 367 Red_180_2 (0.3889 arcsec/px): ratio **0.9994** (vs h & chi Per
375 L **1.112**). A9 `draft367` variant at measured mismatch:

| Metric | draft 367 | coarse realistic |
|--------|-----------|------------------|
| HV PASS-RECOVER | **83.3%** | 17.6% |
| FAIL-SILENT | **1** | 0 |
| REFUSE correctness | **100%** | 100% |

**Pre-2b (2026-06-08):** `bright_close_regime` guard closes edge FAIL-SILENT. Re-run: FAIL-SILENT **0**,
HV **83.3%**, coarse realistic FAIL-SILENT **0**. Combined verdict: **VALIDATED_FINE_SCALE_IDLE**
(367 sparse: 9 blended, 4 hard -- defer 2b). See `docs/VYVAR_DRAFT367_CROWDING.md`.

Expected coarse shape (contamination **excess** over isolated control, %):

| sep | dM0 | dM-1 | dM-2 | dM-3 |
|-----|-----|------|------|------|
| 0.5 | ~+80 | ... | ... | ~+1300 |
| 1.0 | ~+70 | ... | ... | ~+1100 |
| 1.5 | ~+40 | ... | ... | ~+630 |
| 2.0 | ~+14 | ... | ... | ~+225 |
| 3.0 | ~0 | ~+3 | ~+9 | ~+24 |

Exact numbers differ slightly with VYVAR annulus radii (factor 1.9 x FWHM); zone structure is what matters.

### Tier B catalog source

**Fallback (b):** Gaia-structured synthetic catalog (`source_id`, ra, dec, G, bp_rp) with
stars placed via frame WCS. Documented in `series_meta.json` as partially synthetic matching.

## FAIL policy

A **FAIL is a finding**, not a bug to paper over in production. Do not change photometry,
comp_qa, trust, or solver to pass synthetic tests. FAILs in the latest report are candidates
for Milan + Claude review.

## Latest run findings (2026-06-08)

After first full harness run (`14 pass / 2 fail / 2 skip`):

| id | status | Diagnosis |
|----|--------|-----------|
| A3 | FAIL | Quad-symmetry metric on smeared Moffat cutout stays below 0.1; full ePSF ensemble path not exercised on synthetic smear |
| A7 | FAIL | photutils annulus sky vs SEP mesh background differ ~0.7% on clean stars (methodology gap, not necessarily pipeline bug) |
| A6 | SKIP | Documented gap: flat-only leaves moonlight gradient |
| V3d | PASS | Fine-scale PSF-vs-aperture-vs-truth (367-like); crossover mag ~14; see `tier_v3d/v3d_fine_scale.md` |
| V3e | PASS | NEW estimator ratios 1.038-1.049 (Moffat 2.7/5.4/6.02 px); OLD 1.048-1.111; see `tier_v3e/v3e_epsf_fwhm.md` |

Pass highlights: crowding blend thresholds (A1/A2), Sokolovsky spike on CR series (A4),
saturation peak (A5), aperture ZP after throughput calibration (A8), variability recovery (B1),
bad-comp rejection (B2), trust gating (B3), color-term sign/magnitude (B4), calibration residual (V3c).

## Production discipline

- Photometry numeric SHA on `draft_000366`: core subset (283 LC+comp_quality+comparison) must
  remain unchanged when only validation / diagnostic code is added (**`770966c3...`**). Full
  reference including comp_qa sidecars: **`edbd97e7...`** (426 files; post CQ-C fix-once locus).
- pytest `dev/tests/`: **226 passed / 6 skipped** (+ 3 slow CQ-C draft_366 tests; V3d / V3e / ...).

## V3d fine-scale PSF (2026-06-09, publication-grade)

`python -m tests.validation.run_v3d_fine_scale` -- draft-367-like optics (FWHM ~6 px, 0.39"/px).
Real functions: `psf_photometry_stars` + `_catalog_only_fixed_aperture_flux` + PSF aperture
correction from bright-star truth. Three pillars vs mag 12-17 (30 noise realizations/mag):

| pillar | result (post sky-only weights + sandwich err) |
|--------|-----------------------------------------------|
| accuracy | PSF mid-mag bias **<~2%**; drift sub-% |
| precision | PSF scatter wins from ~mag 13 |
| uncertainty | P3 ~1 (sandwich `psf_err_mode=sandwich_skyonly`) |

**Proof CLIs** (same inject-and-recover stack):

| CLI | proves |
|-----|--------|
| `run_v3d_bias_decomposition_v2` | deterministic vs noise cause (fit_shape ruled out) |
| `run_v3d_clean_sky_proof` | residual_annulus sky path |
| `run_v3d_weight_proof` | sky-only fit weights fix mid-mag bias |
| `run_v3d_sandwich_proof` | sandwich reported uncertainty (P3) |

Reports under `dev/tests/validation/data/tier_v3d/`. Production `psf_photometry_enabled` OFF.

## V3e ePSF FWHM QC (2026-06-08, EPSF-1)

`python -m tests.validation.run_v3e_epsf_fwhm` -- known Moffat FWHM inject -> build ePSF ->
measure NEW azimuthal-profile estimator. PASS: ratio in [0.85, 1.15] (NEW span 1.038-1.049;
OLD legacy estimator 1.048-1.111). Report: `tier_v3e/v3e_epsf_fwhm.md`. Diagnostic only.

## RNG seeds

| Key | Value |
|-----|-------|
| gen_frame | 42 |
| gen_series | 43 |
| gen_a9 | 44 |
| v3d_fine | 367 |
| v3e_epsf | 370 |

Run V3e: `python -m tests.validation.run_v3e_epsf_fwhm`.

Deterministic regeneration: `python -m tests.validation.recover --all`.

## Anchor fingerprint (VL-ANCHOR-WCSINV, batch D)

**Status (2026-08-04):** Batch E **GATE 2 authorized**; anchor fingerprints **pushed** (physical re-cut).

| tier | SHA (draft_516 frozen snapshot, active anchor) | n |
|------|-----------------------------------------------|---|
| core | `477dc8cfc292ed63910ecca6ea1dacfda279fee2850422229739a5cf7db90956` | 97 |
| extended | `f71e07226893a6b07e24999927bad0da8c16e6407656fc97ee02e0d57494be5d` | 145 |

Superseded 435 sky-surface (retired by design): core `5bccd85a...` n=497 / extended `7fdcdca4...` n=744.

Superseded batch D (GATE 1): core `b9c9489a...` / extended `65bc826c...` (n=325/487).
Superseded pre-batch-D: `b7f980c0...` / `2c43bbbf...`.
Ledger: `dev/validation/VYVAR_VALIDATION_LEDGER.json` (VL-ANCHOR-WCSINV).

