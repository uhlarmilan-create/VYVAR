# VYVAR validation harness

Inject-and-recover validation under `tests/validation/` verifies cited VYVAR algorithms
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

- `tests/validation/data/validation_report.json`
- `tests/validation/data/validation_report.md`

Partial runs: `--tier-a`, `--tier-b`, `--v3`, `--a9`.

See also `tests/validation/README.md` for the full matrix.

## What it proves

| Tier | Scope | Real VYVAR entry points |
|------|-------|-------------------------|
| A | Single-frame contaminations | `crowding_index._build_blend_targets_df`, `comp_qa_core.sokolovsky_indices`, `photometry_core._catalog_only_fixed_aperture_flux`, SEP cross-val |
| B | 60-frame series | LombScargle LC, Sokolovsky comp QA, `trust_flag_core.evaluate_target`, `fit_color_term_c1` |
| V3 | Targeted | `time_utils` BJD, pipeline airmass, `calibration.get_processed_master`, blind WCS proxy |
| A9 | NEIGHBOR-SUB envelope | `a9_core.measure_cell` on blend grid; `plain_aperture` baseline + `neighbor_sub` scored path |

## A9 NEIGHBOR-SUB acceptance envelope (steps 1-2)

A9 defines the **plain-aperture contamination map** (the problem) and scores the joint-fit
NEIGHBOR-SUB core (`psf_neighbor_sub.neighbor_sub_target_flux`) per cell vs zone-specific PASS rules.

- Generator: `tests/validation/gen_a9.py` (truth sidecar; grid sep x delta_mag)
- Measurement: VYVAR `_catalog_only_fixed_aperture_flux` with AppConfig radii (not raw photutils)
- Contexts: **coarse** (FWHM 3.2 px, 1.30"/px) and **fine** (FWHM 6.4 px, 0.65"/px)
- Reports: `tests/validation/data/tier_a9/a9_envelope.json` + `.md` + plain/gain heatmap PNGs
- Scoring: `measure_cell(mode="neighbor_sub")` -- HIGH_VALUE cells must recover (>=80% contamination
  reduction); REFUSE cells must guard-refuse; CLEAN cells no-op
- PSF variants: legacy `mismatch` (stress test) and EPSF-audit `realistic` (see diagnostic below)

Run: `python -m tests.validation.run_a9` or `python -m tests.validation.recover --a9`.

Coarse neighbor_sub pass rates (2026-06-08): **ideal 85.7%**, **legacy mismatch 21.4%**.

#### PSF-mismatch diagnostic (step 2b gate)

```bash
python -m tests.validation.run_a9_mismatch_diagnostic
```

Report: `tests/validation/data/tier_a9/a9_mismatch_diagnostic.md`

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
| V3e | PASS | Synthetic Tier-A ratio=1.127; real h&chi Per still 0.59-0.67 (field-specific QC) |

Pass highlights: crowding blend thresholds (A1/A2), Sokolovsky spike on CR series (A4),
saturation peak (A5), aperture ZP after throughput calibration (A8), variability recovery (B1),
bad-comp rejection (B2), trust gating (B3), color-term sign/magnitude (B4), calibration residual (V3c).

## Production discipline

- Photometry numeric SHA on `draft_000366` (283 LC+comp files) must remain unchanged when
  only validation code is added.
- Existing pytest: green (+ V3d / neighbor_sub / A9 / draft367 tests).

## V3d fine-scale PSF (2026-06-08)

`python -m tests.validation.run_v3d_fine_scale` -- draft-367-like optics (FWHM ~6 px, 0.39"/px).
Real functions: `psf_photometry_stars` + `_catalog_only_fixed_aperture_flux` + PSF aperture
correction from bright-star truth. Three pillars vs mag 12-18 (30 noise realizations/mag):

| pillar | result |
|--------|--------|
| accuracy | PSF bias <5% mag12-17; aperture faint-end bias +19% at mag18 |
| precision | PSF scatter wins from mag ~14 |
| uncertainty | PSF reported err / actual scatter ~0.8-1.1 (mag<=17) |

Report: `tests/validation/data/tier_v3d/v3d_fine_scale.md`. Production `psf_photometry_enabled` OFF.

## RNG seeds

| Key | Value |
|-----|-------|
| gen_frame | 42 |
| gen_series | 43 |
| gen_a9 | 44 |
| v3d_fine | 367 |

Deterministic regeneration: `python -m tests.validation.recover --all`.
