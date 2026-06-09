# VYVAR inject-and-recover validation harness

Synthetic ground-truth frames test whether cited VYVAR algorithms recover what was
injected. This is **validation only** under `tests/validation/` -- it does not modify
production photometry or draft artifacts.

## Honesty caveat

Synthetic frames test what the generator injects (Moffat PSF + Poisson/read noise +
gradient): a faithful but simplified sky. The harness validates **logic and recovery**
(does comp_qa flag the bad comp? does the solver recover WCS? does the trust gate gate?),
not every real systematic (scintillation, true flat residuals, detector nonlinearity).
**Necessary, not sufficient** -- real-data drafts remain the final word.

## Quick start

From the repo root:

```bash
python -m tests.validation.recover --all
```

Reports land in `tests/validation/data/`:

- `validation_report.json`
- `validation_report.md`

Regenerate synthetic data only:

```bash
python -m tests.validation.gen_frame
python -m tests.validation.gen_series
```

## RNG seeds (deterministic)

| Module | Seed | Purpose |
|--------|------|---------|
| `gen_frame` | 42 | Tier-A single frame star positions + noise |
| `gen_series` | 43 | Tier-B 60-frame series |
| `gen_a9` | 44 | A9 blend-grid envelope |
| CR pixels (Tier A) | 42 | Cosmic-ray pixel locations |

## Matrix

### Tier A -- single-frame contamination (`gen_frame.py`)

| id | Injected truth | VYVAR function | PASS criterion |
|----|----------------|----------------|----------------|
| A1 | Unresolved blend ~1.0 FWHM | `crowding_index._build_blend_targets_df` | `is_blended=True`; nn in [0.8,1.3] |
| A2 | Resolvable pair ~2.5 FWHM | crowding threshold 1.5 FWHM | `is_blended=False`; nn in [2.2,2.9] |
| A3 | Tracking smear ellip=0.45 | epsf_asymmetry QC | asymmetry > 0.1 |
| A4 | CR spikes in flux series | `comp_qa_core.sokolovsky_indices` | spike index elevated |
| A5 | Saturated star | peak vs `SATURATE` | peak >= 0.85 * SATURATE |
| A6 | Moonlight gradient | (GAP) | SKIP -- document tilt |
| A7 | Clean comps | photutils vs SEP | agree ~0.2%/frame |
| A8 | Known ZP | aperture photometry | bias < 50 mmag |
| A9 | Blend grid sep x delta_mag | `a9_core.measure_cell` plain + `neighbor_sub` | zone structure PASS; scores joint-fit core (ideal + PSF-mismatch) |

### A9 -- NEIGHBOR-SUB acceptance envelope (`a9_core.py`, `gen_a9.py`)

Grid: separations 0.5-3.0 FWHM x neighbour delta_mag 0,-1,-2,-3. Measures target flux through
**VYVAR's** `photometry_core._catalog_only_fixed_aperture_flux` (AppConfig annulus radii).
Contamination excess = blend bias minus isolated-control bias. Three zones:

- **REFUSE** (sep <= 0.8): future NEIGHBOR-SUB guard must refuse
- **HIGH_VALUE** (sep ~0.8-1.5, blended): recover + >=80% contamination reduction
- **CLEAN** (wide sep / faint neighbour): NEIGHBOR-SUB no-op

```bash
python -m tests.validation.run_a9
python -m tests.validation.recover --a9
```

Outputs: `data/tier_a9/a9_envelope.json`, `a9_envelope.md`, `a9_envelope_coarse.png`.
`measure_cell(mode="neighbor_sub")` calls `psf_neighbor_sub.neighbor_sub_target_flux` (gated ON in A9
only). Production measurement sites remain unwired until step 2b.

**Mismatch diagnostic (step 2b gate):** `python -m tests.validation.run_a9_mismatch_diagnostic`
writes `a9_mismatch_diagnostic.md` comparing legacy `mismatch` vs EPSF-audit `realistic` variant.

**Fine-scale draft 367 diagnostic:** `python -m tests.validation.run_a9_draft367` runs ePSF-vs-star
mismatch audit (Red_180_2) + A9 `neighbor_sub` at measured mismatch -> `a9_draft367_diagnostic.md`.

**Draft 367 crowding:** `python -m tests.validation.crowding_audit_367` -> `tmp/crowding_audit_367.json`.

### Tier B -- series (`gen_series.py`)

Catalog source: **fallback (b)** Gaia-structured synthetic catalog (`source_id`, ra, dec,
G, bp_rp). Positions are injected, not a live Gaia DR3 cone.

| id | Injection | Function | PASS |
|----|-----------|----------|------|
| B1 | Target sine A=0.15 mag | LombScargle on proc LC | A and P within tolerance |
| B2 | Variable comp + CR spikes | Sokolovsky indices | flagged; not in comp pool |
| B3 | Weak vs strong trust cases | `trust_flag_core.evaluate_target` | weak YELLOW/RED; strong GREEN |
| B4 | Color slope in mag vs bp_rp | `fit_color_term_c1` | c1 ~ injected slope |
| B5 | Constant comps | RMS on proc flux | no spurious trend |
| B6 | Seeing jitter + CRs | frame metadata | FWHM 2.8-3.6 px |

### Tier V3 -- targeted checks

| id | Check | Notes |
|----|-------|-------|
| V3a | Blind WCS recovery | SKIP if no `GAIA_DR3/*.pkl`; else geometry proxy |
| V3b | BJD + airmass | `time_utils` vs pipeline AltAz |
| V3c | Flat/dark calibration | `calibration.get_processed_master` |
| V3d | PSF vs aperture blends | SKIP unless fine-scale ~0.65"/px |
| V3e | ePSF FWHM native vs injected | ratio in [0.85,1.15]; FAIL documents EPSF-1 until fix |

## FAIL policy

A **FAIL is a finding**, not a bug to paper over. Do not change production photometry,
comp_qa, trust, or solver to pass synthetic tests. Record FAILs in the report for Milan
+ Claude review.

## Production discipline

This harness must **not** alter the photometry byte-identity baseline (numeric SHA on
`draft_000366` LC + comp artifacts). Existing pytest suite must stay green.
