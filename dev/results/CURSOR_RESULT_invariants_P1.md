CURSOR RESULT -- 2026-07-19 -- INVARIANTS P1

What I did
Built golden mini-dataset `draft_000435_p1mini` (16-frame even DATE-OBS stride
from draft_000435), locked VL-P1-GOLD after two identical headless runs, added
opt-in E2E suite (headless SHA, UI-order identity, census bands, physics),
updated ROADMAP/STATE/DECISIONS per DOCS-SYNC.

## Scope note

In-draft Raw/darks and Raw/flats have **no local masters** (0 darks, 0 flats;
CalibrationLibrary supplies them). Raw lights exist (150). Mini starts at
**photometry-ready** stage: calibrated + detrended_aligned proc products for 16
stride frames (aligned-only pool = 139 QC survivors) + parent platesolve
MASTERSTAR/catalogs. Chain coverage matches `session_baseline_check --full`
(`run_full_photometry_pipeline`). Calibrate/QC/align not re-run on the mini.

## Frame list (16)

| # | DATE-OBS | file |
|---|----------|------|
| 1 | 2026-04-23T19:35:20.355 | BO_CVn_Light_001.fits |
| 2 | 2026-04-23T19:55:30.615 | BO_CVn_Light_011.fits |
| 3 | 2026-04-23T20:13:39.634 | BO_CVn_Light_020.fits |
| 4 | 2026-04-23T20:33:49.611 | BO_CVn_Light_030.fits |
| 5 | 2026-04-23T20:51:58.634 | BO_CVn_Light_039.fits |
| 6 | 2026-04-23T21:10:07.634 | BO_CVn_Light_048.fits |
| 7 | 2026-04-23T21:32:18.634 | BO_CVn_Light_059.fits |
| 8 | 2026-04-23T21:50:27.635 | BO_CVn_Light_068.fits |
| 9 | 2026-04-23T22:12:38.634 | BO_CVn_Light_079.fits |
| 10 | 2026-04-23T22:30:47.635 | BO_CVn_Light_088.fits |
| 11 | 2026-04-23T22:48:56.644 | BO_CVn_Light_097.fits |
| 12 | 2026-04-23T23:09:06.634 | BO_CVn_Light_107.fits |
| 13 | 2026-04-23T23:27:15.634 | BO_CVn_Light_116.fits |
| 14 | 2026-04-23T23:51:27.634 | BO_CVn_Light_128.fits |
| 15 | 2026-04-24T00:11:37.639 | BO_CVn_Light_138.fits |
| 16 | 2026-04-24T00:35:49.612 | BO_CVn_Light_150.fits |

inputs_manifest_sha256:
`86ab0d9ea6e41264323badb291dfba77756192370cd3486db0f5eb049088bb91`

## Lock runs (reproducibility before ledger)

| run | elapsed | core SHA | core n | extended SHA | ext n |
|-----|---------|----------|--------|--------------|-------|
| 1 | 611.6 s | 074ae881...adfeec | 333 | 66285d3f...dd03ba | 497 |
| 2 | 574.7 s | 074ae881...adfeec | 333 | 66285d3f...dd03ba | 497 |

Science compare run1 vs run2: benign=True, n_lc=166, science_failures=0.
**Byte-identical -- VL-P1-GOLD locked.**

## Census lock values

- dao_pass1_vy_ndao: 2552
- n_detected_mean: 2777.5625
- n_matched_mean: 2638.5625
- identity_n_parent: 2842
- identity_p95_parent: 1.536099044764055
- n_summary_targets: 169

## Chain-identity outcome

**IDENTICAL.** UI-order (`_find_phase2a_paths` + `run_full_photometry_pipeline`)
vs headless direct paths: science comparator benign (166 LCs, 0 science/time
failures); core SHA match `074ae881...`. No F-431-class divergence found.
P1 suite under `VYVAR_INVARIANTS_P1=1`: 5 passed in ~9.5 min (headless skipped
when mini already at locked SHA; UI chain ~8 min).

## Docs impact

- docs/VYVAR_ROADMAP.md -- QUEUED P1 row -> DONE (2026-07-19) + result pointer
- docs/VYVAR_STATE.md -- INVARIANTS P1 section -> completed, VL-P1-GOLD active
- docs/VYVAR_DECISIONS.md -- new INVARIANTS-P1-GOLDEN-MINI entry (design + scope)
- FLOW doc / flow_doc_facts.py: **none** (dev infrastructure only; no pipeline
  flow / parameter semantics change)

## Files

- dev/tools/build_p1_golden_mini.py
- dev/tests/test_invariants_p1_golden.py
- dev/validation/VYVAR_VALIDATION_LEDGER.json (VL-P1-GOLD)
- dev/tests/test_validation_ledger.py (REQUIRED_IDS)
- docs/VYVAR_ROADMAP.md, VYVAR_STATE.md, VYVAR_DECISIONS.md
- dev/results/CURSOR_RESULT_invariants_p1.md (this file)

## Errors (if any)
None at lock time.
