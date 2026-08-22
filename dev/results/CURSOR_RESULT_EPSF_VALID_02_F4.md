CURSOR RESULT - 2026-08-22 (EPSF-VALID-02 F4 + Part C)

What I did
Added InstrumentedEPSFBuilder (per-iteration status 1/2/3 curve), Part C build-star gates
(zone linear, clean source_state, science scope, edge-safe cutout, existing quality + isolation),
interim top-N=200 cap (INTERIM in meta). Ran gated builds on 517 (production path) and 516 (sandbox).

## Output / findings

### Mechanism
- `InstrumentedEPSFBuilder._process_iteration(stars, epsf, iter_num)` hooks photutils 3.x API
- Curve persisted in `masterstar_epsf_meta.json` -> `iteration_failure_curve` (always, success or fail)

### Funnel counts (516, gated build)
| Gate | n |
|------|---|
| n_csv_input | 3610 |
| n_after_zone_linear | 2264 |
| n_after_clean_source_state | 2200 |
| n_after_science_scope | 68 |
| n_after_edge_safe_cutout | 68 |
| n_after_interim_top_n | 68 |
| n_after_isolation | 67 |
| n_stars_used (build) | 67 |

Production model on 516 (unchanged): **1475** stars.

### Funnel counts (517, gated build)
| Gate | n |
|------|---|
| n_after_science_scope | 67 |
| n_stars_used | 66 |
| iter 6 n_fail | **0** (curve flat; P1-B iter-6 cascade eliminated) |

### 517 test case
**BUILD SUCCEEDS** with gated selection. Artifact:
`dev/results/context/session_20260822_epsf_valid_02_f4/build_517_gated/masterstar_epsf.fits`

### 516 sandbox rebuild
Written to `dev/results/context/session_20260822_epsf_valid_02_f4/sandbox_516_gated_build/`
(not production path). n_stars_used=67 vs production 1475; iteration curve all-zero fail counts.

### Interim N cite
DAOPHOT/allstar practice: tens to low hundreds of PSF stars per field (Stetson 1987 PASP;
Harris et al. DAOPHOT II documentation). **N=200 INTERIM** pending Part D empirical tuning.

## Docs impact

- `docs/VYVAR_INVARIANTS.md` unchanged in F4
- Meta JSON self-documents gates under `build_selection`

## Gate status

F4 is build-path only; no anchor re-cut. `--fast` status same as F2/F3.

## Errors

Initial `InstrumentedEPSFBuilder` signature mismatch (fixed: 3-arg photutils hook).

## Files changed

- `src_py/psf_photometry.py`
- `dev/tests/test_epsf_csv_prepare.py` (fixture: zone/source_state + photometry dir)
- `dev/sandbox/epsf_valid_02_f4_build.py` (sandbox harness)

**STOP** -- architect review before model swap or Part B/D.
