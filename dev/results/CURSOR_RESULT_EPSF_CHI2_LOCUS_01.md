CURSOR RESULT - 2026-09-29 EPSF-CHI2-LOCUS-01 (Phase 1)

Date: 2026-09-29. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 5a5b07a (SAT-CHI2-01).
Class: IMPLEMENTATION Phase 1. Phase 2 LOCK deferred (await Milan "LOCK").

## What I did

Replaced fixed ``psf_chi2_threshold`` SET with shared chi2(flux) locus
criterion (EPSF-CHI2-LOCUS-01). Wired iterative and grouped PSF paths
through ``apply_chi2_locus_to_rows``; night finalize before
INV-PSF-FRAME-01 in merge/export. Registered ``psf_chi2_locus_nsigma``
(default 5.0); marked ``psf_chi2_threshold`` LEGACY in VYVAR_PARAMS.md.
Added T1-T6 recurrence tests. Verification on work-copy
``tmp/session_baseline/20260929T153639Z`` (516 fresh run; first
``--full-epsf`` hit INV ordering bug, fixed, then night finalize + LC
write on that copy).

## Code cites

| item | location |
|---|---|
| Locus fit (MAD-clipped OLS, 3 rounds) | ``src_py/psf_chi2_locus.py:78-138`` |
| MIN_N_FRAME_LOCUS (=3343, SAT-CHI2 k_err x3) | ``src_py/psf_chi2_locus.py:29-41`` |
| Shared SET | ``apply_chi2_locus_to_rows`` ``psf_chi2_locus.py:210-308`` |
| Iterative path defer | ``psf_photometry.py`` -> ``apply_chi2_locus_to_rows`` |
| Grouped path defer | same helper (nonfinite chi2 fails both) |
| Night finalize + n_ok refresh | ``finalize_night_locus_for_inv_psf_frame_01`` ``psf_chi2_locus.py:442-475`` |
| Before INV-PSF-FRAME-01 | ``epsf_psf_merge.py`` (post-merge), ``frame_export.py`` |
| Also after stage merge | ``epsf_stage.py:203-217``, ``astrometry_align.py:1078-1091`` |
| Persisted columns | ``proc_frame_store.py``, ``pipeline_catalog.py`` |

## Tests (T1-T6)

``dev/tests/test_epsf_chi2_locus_01.py``: T1 bright on-locus vs fixed 50;
T2 outliers; T3 nonfinite chi2; T4 night fallback; T6 invariant n_ok
refresh. Recurrence: ``--fast`` includes suite.

``--fast --clean`` OVERALL PASS (log:
``dev/results/context/session_20260929_epsf_chi2_locus_01/fast_clean.log``).

## Verification V1-V6

Work-copy: ``tmp/session_baseline/20260929T153639Z``.
Artifacts: ``dev/results/context/session_20260929_epsf_chi2_locus_01/``.

| gate | result |
|---|---|
| V1 aperture bytes | PASS core ``87197716`` n=53; ext ``dd92e99d`` n=157 (era05-identical) |
| V2 PSF LC census | before era05: ge0.9=1, (0,0.5)=15, eq0=37; after: ge0.9=49, eq0=4 |
| V3 pin drops | 536 epochs; 0 on-locus violations; all ``nonfinite_chi2`` (true outliers) |
| V4 G3 BO | n_full=134, cov=1.0, dem RMS=15.372 mmag |
| V4 G3 FW | n_full=134, cov=1.0, dem RMS=5.360 mmag |
| V4 refs | era04 12.505 / 4.629; SAT-CHI2 M4-B 12.785 / 7.757 (pin-filter sandbox) |
| V5 BO | ratio dem RMS / median pred err = 1.116 |
| V5 FW | ratio = 0.624 |
| V6 locus night | k=1.052 (min=med=max), scatter=0.286, n=28958, 134/134 frames source=night |

Remaining cov=0 targets (4): pinned comps
``1500467303261764096`` / ``1500579870061241088`` (FW-like killers):
nonfinite chi2 every epoch -> V3 outlier, not locus.

First ``--full-epsf`` run FAIL: INV-PSF-FRAME-01 before night finalize
(all frames n_ok=0). Fixed by ``finalize_night_locus_for_inv_psf_frame_01``
before ``finalize_epsf_frame_job``. End-to-end ``--full-epsf`` OVERALL
not re-stamped in this session (runtime); product numbers above from
fixed code path on the same work-copy.

## Acceptance A1-A4

| id | verdict |
|---|---|
| A1 V1 byte-identical aperture | PASS |
| A2 no pin-drop on-locus; remaining drops named | PASS (0 on-locus; 536 nonfinite_chi2) |
| A3 BO/FW n_full=134 | PASS |
| A4 T1-T4 + ``--fast --clean`` | PASS |

Phase 1: commit + push consolidate-01. STOP for Milan LOCK (no era06 lock).

## Phase 2 (LOCK)

Not executed. Await Milan "LOCK" for era06 G3 refs, epsf01 re-cut, ledger.

## Files changed (Phase 1)

- ``src_py/psf_chi2_locus.py`` (new)
- ``src_py/psf_photometry.py``, ``epsf_psf_merge.py``, ``frame_export.py``,
  ``epsf_stage.py``, ``astrometry_align.py``, ``proc_frame_store.py``,
  ``pipeline_catalog.py``, ``config.py``
- ``config.json``, ``dev/validation/params_registry.json``, params docs/guides
- ``dev/tests/test_epsf_chi2_locus_01.py`` + recurrence fixes
- ``dev/results/CURSOR_RESULT_EPSF_CHI2_LOCUS_01.md`` + session artifacts
