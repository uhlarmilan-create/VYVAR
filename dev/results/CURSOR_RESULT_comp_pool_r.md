CURSOR RESULT -- 2026-07-13T13:40:00Z

What I did
Part 0 MASTERSTAR closeout (docs, carrier diagnosis, MAG_ERR_SCALE, push). Parts A-D
r_60_4 comp-pool forensic: per-comp table, gate archaeology, grounded verdict, action.

## Part 0 closeout

### r_60_4 chi2 row
Replaced ``chi2 = 121.0*`` with **PENDING (COMP-POOL-R; no check_kmag sidecar)** in
``docs/VYVAR_STATE.md``, ``docs/VYVAR_ROADMAP.md``, ``tmp/sigma_newton_fresh/sigma_newton_fresh_summary.json``.
The prior 121 value was V0611 variability on raw LC mags without check-star detrend -- not
err-quality.

### carrier_matches_normalize (i_70_4 / r_60_4)

| Setup  | max |diff| (mag) | Top frame | Mechanism |
|--------|----------------|-----------|
| i_70_4 | 1.41e-05 | proc_SSCam_*_i_0006.csv | Sub-1e-5 FP in err^2 - photon^2 decomposition; ``np.round(err,6)`` |
| r_60_4 | 9.78e-05 | proc_SSCam_*_r_0089.csv | ``ensemble_scatter=0`` (photon-only epoch); implied ensemble from FP residual |

**Verdict:** NOT a live err-assembly bug. Residuals are ``np.round(err, 6)`` at
``photometry_core.py:4618`` plus finite-precision quadrature inversion when photon term
dominates. No G2-F004 unmatched epochs on worst frames; no howell_scaled-only rows implicated.

Artifact: ``tmp/carrier_normalize_diagnose.json``; script ``scripts/carrier_normalize_diagnose.py``.

### MAG_ERR_SCALE unification
Canonical ``MAG_ERR_SCALE = 2.5/ln(10)`` in ``mag_constants.py``; ``_PSF_ERR_MAG_SCALE`` and
``chi2_sigma_gate._MAG_ERR_SCALE`` are aliases. ``abs(MAG_ERR_SCALE - 1.0857362) < 1e-7``.
LC ``err`` column byte-identical (constant-only change; no photometry re-run).

### Push
Part 0 chain pushed to ``origin/main`` (Milan-authorized 2026-07-13):
``c8d6e80..b5364e6`` (includes 426-REGEN, PROVENANCE-GUARD, MASTERSTAR-EPOCH-FIX).
Commits: ``463184a`` MAG_ERR_SCALE, ``0de6b0f`` MASTERSTAR fix, ``838e82e`` diagnose scripts,
``b5364e6`` docs closeout. ``session_baseline_check.py --fast``: **PASS** (git-head b5364e6,
779 passed).

## A -- Per-comp table (V0611, r_60_4)

Union June-good 8 + regen-good 2 (10 catalog_ids). Full JSON:
``tmp/comp_pool_r/per_comp_table.json`` (script ``scripts/comp_pool_r_forensic.py``).

| catalog_id (tail) | June | Regen | June comp_rms | Regen comp_rms | Flip gate |
|-------------------|------|-------|---------------|----------------|-----------|
| ...502160774840   | good | out   | 0.0126        | --             | not in Phase-1 top pool on HEAD |
| ...2793944000     | good | out   | 0.0075        | --             | same |
| ...692649269248   | out  | good  | --            | 0.0633         | newly selected T3 |
| ...0935816253440  | out  | good  | --            | 0.0630         | newly selected T3 |

**Key fact:** ``detrended_aligned/lights/r_60_4`` proc CSVs are **byte-identical** stale vs
fresh (25 science frames). Global comp pool (14 rows, 7 finite comp_rms) is **identical** on
both trees. Phase-1 per-target output differs entirely: June 8x tier-2 (``color_rms_t2``,
comp_rms 0.007-0.016); HEAD 2x tier-3 (``color_rms_t3``, comp_rms ~0.063).

Phase-2A stability (recomputed on shared procs): all union comps ``p2p_quality=good`` on both
sides -- flip is **Phase-1 pool / tier ladder**, not stability gate.

## B -- Gate archaeology

| Commit   | Date       | Effect relevant to r_60_4 delta |
|----------|------------|----------------------------------|
| 58b03ac  | 2026-06-16 | Authoritative ``comp_rms <= max_comp_rms``; no above-gate padding |
| 1c80219  | 2026-06-16 | Sparse fallback when zero gate-passers |
| 7317ece  | 2026-06-12 | Gaia dedupe before tier ladder |
| a66ba18  | 2026-07-08 | Narrow detrend/frame-metric exceptions (post-June LC) |

June LC mtime 2026-06-26; HEAD regen 2026-07-13. Identical proc inputs => delta is **code path
between June-26 run and HEAD**, not frame-set or MASTERSTAR epoch.

Artifact: ``tmp/comp_pool_r/gate_diffs.json``.

## C -- Grounded correctness verdicts

| Comp (June good, out on HEAD) | Verdict | Rationale |
|-------------------------------|---------|-----------|
| All 8 June tier-2 comps       | **BORDERLINE / superseded** | Not re-selected on HEAD despite identical procs; lower stored June comp_rms reflected prior Phase-1 ranking. Not individually "wrong" photometrically (p2p good), but **ensemble not reproducible** under current ladder. |
| 2 regen tier-3 comps          | **CORRECTLY_INCLUDED** | Only survivors after color tier ladder widen to T3; comp_rms ~0.063 < ``max_comp_rms=0.1``. |

**Pool verdict (b):** HEAD pool **legitimately sparse (2)** on current evidence. June **8 is not
grounded-correct to restore** without threshold retuning (forbidden). Not a named
WRONGLY_EXCLUDED bug.

## D -- Action

- **No code fix** (no threshold retuning; no g/i changes).
- **r baseline:** remains **PENDING** until sparse-aware check-star path or Milan accepts 2-comp field.
- **Milan design question:** Should ``compute_check_ensemble_mag_calib`` / check_kmag support
  ``n_comp_min=2`` sparse fields (same theme as SS Cam band), or is r_60_4 science LC-only?

## Errors

None blocking.

## Files changed (local commits after Part 0 push; NOT pushed)

- scripts/comp_pool_r_forensic.py (new)
- tmp/comp_pool_r/per_comp_table.json, gate_diffs.json
- docs/VYVAR_STATE.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_JOURNAL.md (COMP-POOL-R rows)
- CURSOR_RESULT_comp_pool_r.md

## pytest

779 passed, 15 skipped (unchanged from MASTERSTAR-EPOCH-FIX baseline).
