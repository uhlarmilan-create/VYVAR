CURSOR RESULT -- 2026-07-13T12:30:00Z

What I did
Part 0: pushed local chain 5e6f91b..c8d6e80 to origin/main; session_baseline_check --fast PASS.
Part A: preserved stale draft_426 at Archive/evidence/draft_000426_stale_20260626; regenerated
g/i/r/z via run_full_photometry_pipeline on HEAD (c8d6e80 + unit fix). Part B: fresh Newton
baseline (tmp/sigma_newton_fresh/). Part C: provenance_guard + harness wiring + VL-PROVENANCE.

## Part 0 -- push confirmation

- Pushed: 5e6f91b..c8d6e80 -> origin/main (Milan-authorized 2026-07-13).
- session_baseline_check --fast: PASS (770 pytest, ledger v1 10 items -> 11 after this task).

## Part A -- regen + frame-count answer

Evidence path: Archive/evidence/draft_000426_stale_20260626 (full draft tree moved, copy restored).

| Setup  | provenance | carrier OK | LCs | check_kmag |
|--------|------------|------------|-----|------------|
| g_60_4 | YES        | YES        | 6   | 6          |
| i_70_4 | YES        | YES        | 6   | 6          |
| r_60_4 | YES        | YES        | 6   | **0**      |
| z_90_4 | YES        | n/a (no V0611 LC) | 3 | **0** |

**Frame-count (i_70_4, 26 vs 25):** extra epoch is source_file ``proc_MASTERSTAR.csv``.
Stale LC excluded it; current HEAD includes it because photometry_core loads all
``proc_*.csv`` under detrended_aligned/lights (glob ~line 933). Canonical behavior;
do not suppress.

**Anomaly (reported, not fixed):** r_60_4 regenerated with **zero** check_kmag sidecars;
stale evidence had 6. Chi2 on r pooled stars is unreliable without sidecars. Milan
review before trusting r Newton numbers.

z_90_4: only 3 measurable targets (no V0611 LC); same sparse z coverage as before.

## Part B -- fresh SIGMA-NEWTON baseline (production_lc_err)

Artifacts: tmp/sigma_newton_fresh/sigma_newton_fresh_summary.json

| Setup  | Star (role) | err med (mag) | ens share | chi2 [CI] | n |
|--------|-------------|---------------|-----------|-----------|---|
| g_60_4 | V0611       | 0.0111        | 0.50      | 3.20 [2.17, 3.90] | 24 |
| i_70_4 | V0611       | 0.0174        | 0.06      | 2.17 [1.50, 2.53] | 26 |
| i_70_4 | SS Cam      | 0.0032        | 0.89      | 27.6 [18.7, 32.6] | 26 |
| r_60_4 | V0611       | 0.0124        | 0.17      | 131.6* [86, 159] | 26 |

*V0611 r chi2 uses LC mag fallback (no check_kmag sidecar) -- treat as suspect.

Forensic delta (i V0611): chi2 2.131 -> 2.173 (+0.042); err stable at ~0.0174 mag.

Science compare (shared epochs, excl. proc_MASTERSTAR.csv): non-err columns show
milli-mag level drift vs stale (Fix A err path, aperture_correction / mag_calib_ac,
empirical bkg F-BINGAIN). Expected post-June; not byte-identical to stale.

## Part C -- PROVENANCE-GUARD

- scripts/provenance_guard.py: refuses unstamped pipeline_meta; --allow-unstamped override.
- Wired: chi2_sigma_gate.py (main), bingain_fix_validate.py.
- session_baseline_check.py --full: full-provenance step on anchor snapshot.
- VL-PROVENANCE added to validation/VYVAR_VALIDATION_LEDGER.json.
- tests/test_provenance_guard.py (5 tests).

## AAVSO flag

draft_426 g/i/r/z exports from the **stale** tree (Archive/evidence/...) carried
**inflated** err bars (conservative direction). Whether any submission needs
resubmission is **Milan's decision** -- no export action taken.

## Errors

- r_60_4 check_kmag sidecars not regenerated (0 vs stale 6) -- production path issue
  flagged for Milan; not auto-fixed per task constraint.
- z_90_4 sparse coverage unchanged (3 LCs).

## Files changed

- scripts/draft_426_regen.py, scripts/sigma_newton_fresh_run.py (new)
- scripts/provenance_guard.py (new)
- scripts/chi2_sigma_gate.py, scripts/bingain_fix_validate.py, scripts/session_baseline_check.py
- tests/test_provenance_guard.py, tests/test_validation_ledger.py
- validation/VYVAR_VALIDATION_LEDGER.json
- CURSOR_RESULT_426_regen.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_STATE.md, docs/VYVAR_JOURNAL.md
- Archive/evidence/draft_000426_stale_20260626 (on-disk, not git)
- Archive/Drafts/draft_000426 photometry regenerated (on-disk)
- tmp/draft_426_regen/, tmp/sigma_newton_fresh/ (gitignored)

## pytest

775 passed, 15 skipped (+5 provenance guard tests).
