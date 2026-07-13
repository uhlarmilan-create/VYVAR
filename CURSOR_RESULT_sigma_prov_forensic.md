CURSOR RESULT -- 2026-07-13T12:00:00Z

What I did
SIGMA-PROV-FORENSIC: Part A provenance sweep on archive drafts; Part B fresh
run_full_photometry_pipeline for draft_426 i_70_4 (output preserved under
tmp/sigma_prov_forensic/fresh_i_70_4/); Part C unit-consistency fix at
_combine_err_with_ensemble_scatter_keyed (mag SEM -> rel flux before quadrature)
with updated tests. Stale draft_426 archive tree left read-only.

## Predictions P1-P3 (as written pre-run)

P1: draft_426 LC provenance predates 2026-06-18 (005716d), OR provenance absent
with LC-assembly mtimes < 2026-06-18.

P2: Fresh i_70_4 LC err total ~0.009-0.010 mag on V0611-cohort stars
(photon ~0.005 + SEM ~0.0067, x1.0857 unit tolerance if pre-fix).

P3: Check-star chi2 on FRESH i_70_4 LC ~6-8 (OVERdispersed) via production_lc_err.

## Part A -- provenance table + verdict

| Setup   | provenance | LC mtime (UTC)    | LC err mag (V0611) | normalize SEM | LC/normalize |
|---------|------------|-------------------|--------------------|---------------|--------------|
| g_60_4  | ABSENT     | 2026-06-26        | 0.0174             | 0.0059        | 2.73x        |
| i_70_4  | ABSENT     | 2026-06-26        | 0.0554             | 0.0067        | 7.46x        |
| r_60_4  | ABSENT     | 2026-06-26        | 0.0235             | 0.0054        | 4.19x        |
| z_90_4  | ABSENT     | 2026-06-26        | (no V0611 LC)      | --            | --           |

Controls:
- draft_424 NoFilter_60_2: provenance PRESENT (git 13eecd62..., stamped 2026-07-08);
  includes Fix A + PROV-FIX -- healthy wide-rig legitimate.
- draft_425 B_20_2: provenance PRESENT (git 7b2d2859..., post-Fix-A).

**P1 strict verdict: FAIL.** All draft_426 setups lack provenance; LC mtimes are
2026-06-26 (AFTER Fix-A date 2026-06-18, BEFORE PROV-FIX 2026-07-08). No git_hash
to map pre-005716d.

**P1 semantic verdict: PASS (stale pre-Fix-A err fingerprint).** i_70_4
LC-implied ensemble median 0.0482 mag matches pre-Fix-A brightness-spread
0.1-0.3/sqrt(7) range; normalize SEM 0.0067 mag (current code, C0-verified).
Ratio 7.46x cannot arise from mag/flux join (cap 1.0857x). Files cannot have
been written by current single combine path.

## Part B -- fresh i_70_4 numbers + verdicts

Fresh run: tmp/sigma_prov_forensic/fresh_i_70_4/photometry (git 20d7e46 dirty,
provenance stamped 2026-07-13). Pre-Part-C combine at run time.

| Star (check)        | stale err mag | fresh err mag | stale chi2 | fresh chi2 |
|---------------------|---------------|---------------|------------|------------|
| V0611               | 0.0554        | 0.0175        | 0.24       | 2.13       |
| SS Cam              | (stale high)  | 0.0034        | 122        | 24.9       |
| 1112130898824233216 | --            | 0.0063        | --         | 10.1       |
| 1111749368289526912 | --            | 0.0074        | --         | 5.6        |

V0611 fresh decomposition: photon median 0.0169 mag, normalize SEM 0.0067 mag,
LC-implied ensemble 0.0043 mag (unit-mix artifact in quadrature).

**P2 verdict: FAIL.** V0611 err median 0.0175 mag, not 0.009-0.010. Photon+bkg
(~0.017 mag, F-BINGAIN empirical) dominates; architect assumed photon ~0.005.

**P3 verdict: FAIL (V0611).** chi2/dof = 2.13, not 6-8. Sign flip confirmed
(stale 0.24 -> fresh 2.13, under- to mild OVER-dispersion). Other pooled check
stars show stronger overdispersion (5.6-24.9); Newton problem character changes
cohort-wide on fresh LC.

Science compare (fresh vs stale): row-count mismatch (26 vs 25 frames); err
column differs by construction (Fix A). Non-err science columns not re-compared
on shared epochs this session (frame alignment needed).

## Part C -- unit fix + test

Fix: sem_rel = sem_mag / _PSF_ERR_MAG_SCALE before sqrt(err^2 + sem_rel^2) at
_combine_err_with_ensemble_scatter_keyed. Docstring documents single-domain contract.

Effect: max +8.6% on ensemble-dominated err (correctness fix, NOT the 7x anomaly).
Re-anchor required for LC err column (bundle with PROD-SIGMA-FLOOR).

Tests: tests/test_g2_f004_err_scatter_keyed.py updated (6 tests, domain hand-check).

## Blast radius

Strict sweep (mtime < 2026-06-18 or provenance < Fix A): **empty** (no drafts).

Semantic stale-err (absent provenance + LC/normalize > 2.5x):
- draft_000426 g_60_4 (2.73x)
- draft_000426 i_70_4 (7.46x) **SEVERE**
- draft_000426 r_60_4 (4.19x) **SEVERE**

AAVSO exports from draft_426 g/i/r/z carried inflated err bars. draft_424/425
controls are post-Fix-A with provenance.

## Retractions (per discipline)

1. **SIGMA-NEWTON N1 baseline on draft_426 archive LC is INVALIDATED** -- stale
   pre-Fix-A err, not a live-code sigma bug.
2. **SIGMA-SEM-CAUSE "mag/flux join as dominant cause" is SUPERSEDED** -- join
   is a real bug (Part C fixed) but capped at 1.0857x; 7x came from stale LC.

## Production recommendation (Milan decision)

Fresh Newton baseline: V0611 i_70_4 chi2=2.13 (mild overdispersion); pooled check
stars 2-25. PROD-SIGMA-FLOOR still motivated. One re-anchor after unit fix + floor.
Do NOT apply ~0.5x ensemble scale. SS Cam trust band tied to fresh chi2=24.9.

## Errors

- Part B first post-process failed (proc_dir None for tmp path); fixed via archive
  proc_dir. Science compare blocked by frame-count mismatch.

## Files changed

- scripts/sigma_prov_forensic.py (new)
- photometry_core.py (Part C unit fix)
- tests/test_g2_f004_err_scatter_keyed.py
- CURSOR_RESULT_sigma_prov_forensic.md
- docs/VYVAR_ROADMAP.md, docs/VYVAR_STATE.md, docs/VYVAR_JOURNAL.md
- tmp/sigma_prov_forensic/ (gitignored artifacts)

## pytest

770 passed, 15 skipped (1 test added: domain-consistent quadrature hand-check).
