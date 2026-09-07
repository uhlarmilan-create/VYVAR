CURSOR RESULT - 2026-09-07 LEDGER-SYNC-20260907

What I did
Recorded D-MP-CALIB-OFF-01 and D-FWHM-AUTH-01-CLOSE (Milan 2026-09-07)
at the top of docs/VYVAR_DECISIONS.md. Synced ROADMAP SHA prose to
30c37eb. Prepended a CONSOLIDATE-01 close block on VYVAR_STATE.md and
updated Last updated to 2026-09-07. Docs-only; no science or code change.

--fast --clean: OVERALL PASS (1621 passed, 32 skipped; clean-tree PASS).
HEAD at gate start: 30c37eb. Branch: consolidate-01.

## Refutations

None. Spec claims held. Line-number notes (not contradictions):

- VYVAR_CALIBRATE_MP gate is src_py/pipeline_calibrate.py:71-74
  (task said :72-73).
- Worker init 3-param: pipeline_calibrate.py:2022-2026.
- Pool initargs 3-tuples: pipeline.py:614 and pipeline_calibrate.py:2615.
- Spawn guard test_calibrate_batch_mp_spawn_passthrough_roundtrip stays
  in dev/tests/test_calibrate_mp_spawn.py.
- FWHM file chain: aperture_policy.resolve_frame_fwhm_px :191-206;
  qc_metrics.csv at pipeline_catalog.py:580.

## 2.5e-4 measurement artifact

Located (not handoff-only). CONSOLIDATE-01B A2 MEASURE FIRST
dev/results/context/session_20260831_c01b/REPORT.md:24
"max abs delta FWHM (px) | 2.511e-4 (Light_028)"; cause FITS card
rounding. Table a2_r_out_table.csv / a2_summary.json beside that
report. Decision text keeps the specified 2.5e-4 rounding.

## ROADMAP

- Prose SHA line now: origin/main == origin/consolidate-01 == 30c37eb
  (CONSOLIDATE-01 fast-forward, Milan PUSH_AUTH 2026-09-07).
- MP-CALIB-PARITY-01: no ROADMAP row existed; none added; closed only
  in D-MP-CALIB-OFF-01.
- A-1-OVERRIDE: not touched (still OPEN, measured-delta).

## STATE

Leads with LEDGER-SYNC 2026-09-07 CONSOLIDATE-01 close paragraph plus
one line each for D-MP-CALIB-OFF-01 and D-FWHM-AUTH-01-CLOSE.
Last updated: 2026-09-07.

## Gates

```
pytest                       PASS   1621 passed, 32 skipped
clean-tree                   PASS   worktree=b1b_clean_46b46888; pytest 32 passed; ruff PASS; pyflakes PASS
OVERALL: PASS
```

No --full, no --full-epsf.

## Files changed

- docs/VYVAR_DECISIONS.md
- docs/VYVAR_ROADMAP.md
- docs/VYVAR_STATE.md
- dev/results/CURSOR_RESULT_LEDGER_SYNC_20260907.md
