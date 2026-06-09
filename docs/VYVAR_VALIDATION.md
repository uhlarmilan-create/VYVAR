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

Partial runs: `--tier-a`, `--tier-b`, `--v3`.

See also `tests/validation/README.md` for the full matrix.

## What it proves

| Tier | Scope | Real VYVAR entry points |
|------|-------|-------------------------|
| A | Single-frame contaminations | `crowding_index._build_blend_targets_df`, `comp_qa_core.sokolovsky_indices`, `photometry_core._catalog_only_fixed_aperture_flux`, SEP cross-val |
| B | 60-frame series | LombScargle LC, Sokolovsky comp QA, `trust_flag_core.evaluate_target`, `fit_color_term_c1` |
| V3 | Targeted | `time_utils` BJD, pipeline airmass, `calibration.get_processed_master`, blind WCS proxy |

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
| V3d | SKIP | Fine-scale PSF-vs-aperture requires ~0.65"/px config |
| V3e | PASS | Synthetic Tier-A ratio=1.127; real h&chi Per still 0.59-0.67 (field-specific QC) |

Pass highlights: crowding blend thresholds (A1/A2), Sokolovsky spike on CR series (A4),
saturation peak (A5), aperture ZP after throughput calibration (A8), variability recovery (B1),
bad-comp rejection (B2), trust gating (B3), color-term sign/magnitude (B4), calibration residual (V3c).

## Production discipline

- Photometry numeric SHA on `draft_000366` (283 LC+comp files) must remain unchanged when
  only validation code is added.
- Existing pytest: **183 passed, 6 skipped** (unchanged).

## RNG seeds

| Key | Value |
|-----|-------|
| gen_frame | 42 |
| gen_series | 43 |

Deterministic regeneration: `python -m tests.validation.recover --all`.
