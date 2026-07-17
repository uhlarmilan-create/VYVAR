# Draft 367 crowding characterization (fine-scale NEIGHBOR-SUB use-case gate)

Read-only diagnostic (2026-06-08). Uses production `compute_crowding_index` with
`header_core_fwhm_px` / `VY_FWHM_GAUSS` (post FWHM-CONSISTENCY). Harness:
`tests/validation/crowding_audit_367.py`. JSON: `tmp/crowding_audit_367.json`.

---

## Per-filter summary (richest setup per filter)

| filter | setup | FWHM (px) | gaia_density | blend@1 | blend@2 | is_blended | hard <1.0 |
|--------|-------|-----------|--------------|---------|---------|------------|-----------|
| Blue | Blue_180_2 | 7.064 | 0.40 | 0.000 | 0.008 | 10 | 4 |
| Green | Green_60_2 | 5.074 | 1.09 | 0.000 | 0.009 | 4 | 3 |
| Red | **Red_180_2** | **6.020** | **1.11** | **0.000** | **0.022** | **9** | **4** |

Richest SNR filter: **Red** (16 frames, 180 s). Plate scale ~0.389 arcsec/px.

LC star blend buckets (Red_180_2, n=158): hard nn<1.0 **4**, sep 1.0-1.5 **5**, sep 1.5-2.0 **4**.

---

## Verdict vs h & chi Per

| field | is_blended | hard <1.0 FWHM | NEIGHBOR-SUB use case |
|-------|------------|----------------|------------------------|
| h & chi Per 375 L | 58 | 39 | dozens of real targets |
| draft 367 Red_180_2 | **9** | **4** | **sparse** |

Decision threshold (>= ~20-30 blended, >= ~10 hard): **not met** on 367.

**VALIDATED_FINE_SCALE_IDLE:** A9 proves NEIGHBOR-SUB works at fine scale (mismatch ~1.0, HV ~83%,
FAIL-SILENT 0); draft 367 has too few real blends to justify immediate 2b wiring. Keep
`psf_neighbor_sub_enabled` OFF until a blended fine-scale field appears (e.g. Brno data, once
characterized per `VYVAR_DECISIONS.md`).

---

## Cross-references

- ePSF mismatch: `docs/VYVAR_EPSF_FWHM_TEST.md` (367 section)
- A9 yield: `tests/validation/data/tier_a9/a9_draft367_diagnostic.md`
- h & chi Per recompute: `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`
