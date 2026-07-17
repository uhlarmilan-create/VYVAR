# VYVAR -- Pipeline rework draft (Claude, 2026-06-25)

**Status:** draft / not yet integrated. This file does **not** supersede `VYVAR_PIPELINE_CZ.md`
(the Czech article manual). It captures planned refresh items from the 2026-06-25 architecture
review; full rewrite pending Milan review.

## Purpose

Separate working notes for a future `VYVAR_PIPELINE_CZ.md` revision aligned with current production
(`d0a9a0a` tree): trust gate (2-axis production reality), offline-only SEP/xval, calibration
radiometry gaps, and audit-closed items (F-HOWELL-3, F-BJD-1, GAIA-ID guard).

## Planned integration topics

1. **Trust gate** -- production = comp_qa + check-star scatter + `lc_quality_flag`; SEP/xval offline only.
2. **Calibration radiometry** -- see CAL-DIAG workstream (`VYVAR_ROADMAP.md`, `VYVAR_DECISIONS.md`):
   camera-agnostic post-dark sky sanity + dark-resample SUM/MEAN convention cross-check.
3. **Binning / gain** -- F-BINGAIN-1 latent on wide rig (header_index_mapped 3.17; see AUDIT_LEDGER).
4. **Single-night product scope** -- canonical publishable unit (DECISIONS 2026-06-25).

## Related docs

- Live manual: `VYVAR_PIPELINE_CZ.md`
- State: `VYVAR_STATE.md`
- Process: `VYVAR_PROCESS.md`
