CURSOR RESULT -- 2026-07-14 CAL-DIAG-IMPL

What I did
Part 0 baseline verification. Confirmed CAL-DIAG v1.1 implementation already on main
(2026-07-07 commits); re-validated on HEAD 13341b3. Flipped spec to APPROVED (Milan 2026-07-14).
Updated DECISIONS/ROADMAP/STATE/JOURNAL. NOT PUSHED -- Milan review first.

## Part 0 -- Baseline verification

| Check | Result |
|-------|--------|
| origin/main HEAD | **13341b3** (docs finalize K2-STATS-FIX on top of c8aed4b) |
| session_baseline_check --fast | **PASS** (830 passed, 15 skipped) |

Why 13341b3 vs c8aed4b: K2-STATS-FIX added one docs-finalize commit (13341b3) after c8aed4b;
CAL-DIAG code unchanged since 2026-07-07 implementation chain.

## Anchor-drift notes (spec line refs vs HEAD 13341b3)

Re-verified; semantics unchanged, line numbers shifted:

| Spec anchor | Spec (2026-07-07) | HEAD 13341b3 | Notes |
|-------------|-------------------|--------------|-------|
| `_dark_np_for_calibration_path` cache | pipeline.py:671-690 | pipeline.py:697+ | Same role; CAL-DIAG uses `dark_np_for_cal_diag` in cal_diag.py |
| Sequential calibrate gate | pipeline.py:14903-14942 | pipeline.py:15382-15478 | Pre-gate + abort skip wired |
| RAM-QC gate | pipeline.py:1925-1978 | pipeline.py:2023-2141 | Same pattern |
| MP batch gate | pipeline.py:17009+ | pipeline.py:17546+ / worker 14986+ | Variant (a) below |
| `resample_master_to_light_binning` | calibration.py:199 | calibration.py:244+ | `dark_resample_mode` present |
| Phase 2A cal_diag merge | photometry_core.py:5318 | photometry_core.py:9418+ | Additive `cal_diag` block |
| `_match_and_crop_pair` | pipeline.py:14307 | pipeline.py:14588 area | Geometry rule intact |
| D3 comment param_resolver | :155 | :158-161 | Present |
| D3 docstring database | :2928 | :2921-2924 | Present |

No semantic STOP required; no spec formula changes.

## MP variant choice

**Variant (a) -- parent pre-dispatch.** `run_cal_diag_pregate` runs in parent before MP workers;
session exported via `_cal_diag_export_for_workers` / `_cal_diag_session_from_export`.
Workers consult cached gate results; no per-frame re-decision. Keeps MP speed (nw>1 when enabled).

## Synthetic test matrix (tests/test_cal_diag_gate.py)

| Test | Result |
|------|--------|
| resample dark mean vs sum bin2 | PASS |
| matched SUM PASS | PASS |
| averaged-driver AUTO-CORRECT + WARN | PASS |
| garbage dark FAIL-CLOSED | PASS |
| bf=1 pairing fail (wording) | PASS |
| gate off/on byte-identical arrays | PASS |
| pregate session export roundtrip | PASS |
| dark cache same convention | PASS |
| write cal_diag.json | PASS |
| near-zero sky WARN not fail | PASS |
| PASSTHROUGH headers | PASS |
| path-coverage pregate same key | PASS |
| fail-closed sibling groups | PASS |

**14/14 passed** (2026-07-14).

## draft_424 regression (gate ON)

Harness: tmp/caldiag_d424_regression/ (read-only archive inputs).

Calibrate (gate ON, library dark+flat):
- n=150 calibrated frames
- **150/150 VY_DKRSMP=SUM**, 0 WARN, 0 FAIL
- cal_diag_aborted_groups=0
- Sample array compare vs archive calibrated: 5/5 byte-identical

Photometry (session_baseline_check --full work dir tmp/session_baseline/20260714T144230Z):
- **core SHA bf3743a1... MATCH** (n=357 vs snapshot)
- science compare: **0 failures**, n_lc=178 (benign=True)
- extended SHA differs from snapshot dec5c637... (expected): pipeline_meta additive
  `cal_diag` block (merged from archive cal_diag.json) + provenance git_hash drift +
  except_fix_summary asymmetry; science outputs unchanged
- pipeline_meta diff keys vs snapshot: `cal_diag` (work only), `provenance`, `except_fix_summary`

Gate OFF: 0 VY_DKRSMP/VY_CDSKY/VY_CDSTAT on calibrated outputs (tmp/caldiag_d424_regression/cal_off).

Note: archive draft_000424/calibrated/lights predate header wiring (no VY_* on disk); fresh
calibrate to work dir proves header provenance.

## Errors (if any)

None blocking.

## Files changed (this closeout)

docs/VYVAR_CAL_DIAG_SPEC.md (APPROVED status)
docs/VYVAR_DECISIONS.md (CAL-DIAG status update)
docs/VYVAR_ROADMAP.md (pending review wording)
docs/VYVAR_STATE.md
docs/VYVAR_JOURNAL.md
CURSOR_RESULT_cal_diag_impl.md

Implementation code unchanged (already on main since 2026-07-07).

## pytest count

830 passed, 15 skipped (--fast on 13341b3 before closeout commits).

NOT PUSHED -- Milan review first.
