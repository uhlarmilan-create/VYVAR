CURSOR RESULT -- 2026-07-13T14:30:00Z

What I did
Implemented PROD-SIGMA-FLOOR: c4-corrected ensemble SEM, per-rig sigma_sys production
wiring, fit script + PZQ diagnostics, spec/docs, tests. Started draft_424 re-anchor
(two fresh runs).

## Fit results per rig (Part A)

| Rig | equipment_id | sigma_sys_mmag | CI mmag | Split-half validate | Status |
|-----|--------------|----------------|---------|---------------------|--------|
| Wide Carl-Zeiss (424) | 1 | ~4.8 (uncommitted) | [0, 6.3] | A->B 1.03, B->A 0.97 | **un-floored** (bootstrap unstable) |
| Newton g+i pooled (426) | 4 | **18.0** | [15.6, 20.2] | A->B 1.02, B->A 0.98 | **committed** |
| Newton r (426) | 4 | -- | -- | -- | **pending** COMP-POOL-R |

**Wide SIGMA-A3 gate:** point fit ~4.8 mmag is **below** prior band [5.5, 7.5]; bootstrap
CI spans zero. Per task rules: wide left un-floored (config default 0.0); Newton wiring
proceeds.

**g_60_4 epoch audit:** anchor LC 24 epochs, 25 proc CSVs; **0 NaN delta_mag drops**.
One science frame not in anchor LC join (not a nondetection gate on listed frames).

Artifacts: ``tmp/sigma_floor/sigma_floor_fit.json``, ``pzq_*.png``; script
``scripts/fit_sigma_floor.py``.

## PZQ sigma_r (median per rig)

| Rig | median sigma_r (mag) | Note |
|-----|------------------------|------|
| wide_Carl-Zeiss | ~0.005 | red component report-only |
| Newton_g | ~0.01-0.04 per star | figures ``tmp/sigma_floor/pzq_Newton_g.png`` |
| Newton_i | similar | ``pzq_Newton_i.png`` |

Correlated noise **not** added to per-point LC err (PZQ 2006 limitation statement in spec).

## Wiring (Part C)

- ``sigma_floor_core.py``: c4, quadrature combine, resolve_sigma_sys_mag, PZQ fit
- ``ensemble_normalize``: c4 SEM via ``ensemble_sem_mag_from_residuals``
- ``_combine_err_with_ensemble_scatter_keyed``: + ``sigma_sys_mag`` quadrature
- ``config.sigma_sys_mag``: ``{"4": 0.018}`` (equipment 1 omitted = 0.0 fail-safe)
- LC column ``sigma_sys_mag``; ``pipeline_meta.sigma_floor`` block
- ``photometry_report.py`` Methods note on floor vs red noise

## Re-anchor (Part E)

**PASS (2026-07-13):** two fresh runs byte-identical.

| | SHA (prefix) | n |
|--|--------------|---|
| core | ``bf3743a1...`` | 357 |
| extended | ``dec5c637...`` | 535 |

Science compare vs old anchor (``draft_000424_snapshot_20260708_full``): **0 science failures**
(non-err columns identical; err diverges by design from c4 + wide un-floored / c4-only on 424).

**Err median by mag tertile (old vs new anchor, 178 LCs pooled):**

| Tertile | mag range | old err median | new err median | ratio |
|---------|-----------|----------------|----------------|-------|
| faint | 8.16 - 12.69 | 0.0150 | 0.0226 | 1.51 |
| mid | 12.69 - 13.72 | 0.0376 | 0.0635 | 1.69 |
| bright | 13.72 - 21.37 | 0.0858 | 0.1439 | 1.68 |

Wide rig (eq 1): c4-only inflation (no sigma_sys committed). Artifact: ``tmp/reanchor_424/err_tertiles.json``.

Snapshot locked: ``Archive/Drafts/draft_000424_snapshot_sigma_floor_20260713``.
``session_baseline_check.py`` SHA constants updated.
``session_baseline_check.py --full``: **OVERALL PASS** (2026-07-13).

## Errors

- First re-anchor shell failed (log path missing); restarted with ``tmp/reanchor_424`` created.
- Wide rig: no STOP halt (instability + SIGMA-A3 mismatch -> un-floored, documented).

## Files changed

- sigma_floor_core.py (new)
- photometry_core.py (c4, floor, LC column, pipeline_meta)
- config.py, config.json
- scripts/fit_sigma_floor.py, scripts/reanchor_424_sigma_floor.py (new)
- tests/test_sigma_floor.py (new)
- docs/VYVAR_SIGMA_FLOOR_SPEC.md (new)
- docs/VYVAR_SIGMA_BUDGET_SPEC.md, VYVAR_STATE.md, VYVAR_ROADMAP.md, VYVAR_JOURNAL.md
- CITATIONS.bib (merlinehowell1995, pzq2006, everetthowell2001)
- photometry_report.py

## pytest

794 passed, 15 skipped (+15 new in ``tests/test_sigma_floor.py``).

## Commits

Logical units prepared; **NOT pushed** (await Milan review per task + COMP-POOL-R ``8fb21b3``).
