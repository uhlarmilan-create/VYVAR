CURSOR RESULT — 2026-06-08

What I did
Pre-2b publication-grade gate: added conservative `bright_close_regime` guard, re-ran A9 diagnostics,
characterized draft 367 real crowding, documented Brno standing rule. No 2b wiring; gated OFF.

## Output / findings

### Part A — edge FAIL-SILENT closed

Regime guard: refuse when `delta_mag_nn <= -2.5` AND `nn_dist_fwhm <= 1.1` (`bright_close_regime`).
Closes sep1.0/dM-3 without affecting clean recoveries (dM-2 at sep1.0, all sep>=1.3).

| diagnostic | FAIL-SILENT | HV PASS-RECOVER | REFUSE correctness |
|------------|-------------|-----------------|-------------------|
| draft 367 | **0** | **83.3%** | 100% |
| coarse realistic | **0** | 17.6% | 100% |

Provenance: `NeighborSubResult` carries `neighbor_subtracted`, `refused`, `refuse_reason` on every
path; design doc updated for step-2b column plan.

### Part B — 367 real crowding (Red_180_2, VY_FWHM_GAUSS)

| metric | value |
|--------|-------|
| gaia_density | 1.11 / arcmin^2 |
| blend_frac @ 2 FWHM | 0.022 |
| is_blended (LC) | **9** |
| hard (nn < 1.0 FWHM) | **4** |

vs h & chi Per 375 L: 58 blended / 39 hard. **SPARSE** — no immediate use case on 367.

### Combined decision

**VALIDATED_FINE_SCALE_IDLE** — NEIGHBOR-SUB validated at fine scale; defer 2b until blended
fine-scale field (e.g. Brno, after characterization gate).

### Next PSF milestone (flag only)

**TODO-PSF-V3d-FINE-SCALE**: inject-and-recover PSF-vs-aperture-vs-truth at 367 (mismatch ~0).

## Errors (if any)

None.

## Files changed

- `psf_neighbor_sub.py`, `config.py` — bright_close_regime guard
- `tests/validation/a9_core.py` — scoring for regime refuse; combined decision
- `tests/validation/crowding_audit_367.py` — crowding harness
- `tests/test_psf_neighbor_sub.py`, `tests/test_a9_*.py`
- `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`, `VYVAR_DRAFT367_CROWDING.md`, `VYVAR_DECISIONS.md`,
  `VYVAR_PROCESS.md`, `VYVAR_JOURNAL.md`, `VYVAR_VALIDATION.md`, `VYVAR_EPSF_FWHM_TEST.md`,
  `VYVAR_ROADMAP.md`, `VYVAR_AUDIT_LEDGER.md`
- `CURSOR_RESULT.md`

pytest: **209 passed, 6 skipped**. Numeric SHA `770966c3` unchanged (read-only + gated OFF).
