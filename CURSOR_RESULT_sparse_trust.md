CURSOR RESULT — 2026-07-14

What I did
Implemented SPARSE-TRUST + PZQ-SIGMA-R per spec: PZQ report (Part 1), sparse_trust_core + wiring (Part 2), tests and validation harness (Part 3). Push from prior ANCHOR-CHAIN task was already completed (origin/main at 7ed7459).

## Output / findings

### Push (prior task — confirmed)
- `git push origin main` succeeded: `b5364e6..7ed7459`

### Part 1 — PZQ sigma_r (report-only)
- Script: `scripts/pzq_sigma_r_report.py`
- Output: `tmp/pzq_sigma_r/pzq_sigma_r_summary.json` (36 stars, 3 rig histograms)
- Interpretation: median sigma_r ~5 mmag on wide cohort confirms red-noise not captured by white SEM; supports T_green=1.5 default

### Part 2 — SPARSE-TRUST implementation
- Spec: `docs/VYVAR_SPARSE_TRUST_SPEC.md` (commit `4cc600b`)
- Core: `sparse_trust_core.py` — triangulation, photon correction, sigma_ZP sparse, chi2 CI, stability test, trust band
- Wiring: `check_star_kmag.py` (`CheckEnsembleResult`, n>=2, sidecar columns), `photometry_core.py`, `trust_flag_core.py` (sparse CI path), `config.py` (T_green/T_red/X2_RED)
- Citation: `howellwarnockmitchell1988` in `CITATIONS.bib`

### Part 3 — Validation
- S1: `tests/test_sparse_trust_core.py` — 7 fast tests PASS; 3 slow synthetic coverage tests (N=15/25/139) marked `@pytest.mark.slow`
- S2-S4: `scripts/sparse_trust_validate.py` (requires regen sidecars on draft_424/426 for full run)
- Full fast pytest: **803 passed**, 11 skipped
- ruff: clean on touched files

## Errors (if any)
None.

## Files changed
- `docs/VYVAR_SPARSE_TRUST_SPEC.md` (new)
- `sparse_trust_core.py` (new)
- `tests/test_sparse_trust_core.py` (new)
- `check_star_kmag.py`, `photometry_core.py`, `trust_flag_core.py`, `config.py`
- `scripts/pzq_sigma_r_report.py`, `scripts/sparse_trust_validate.py`, `scripts/backfill_check_kmag_sidecars.py`, `scripts/select_constant_calibrators.py`
- `CITATIONS.bib`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_ROADMAP.md`
- **Not pushed** — awaiting Milan review per task gate
