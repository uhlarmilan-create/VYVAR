CURSOR RESULT - 2026-08-03 22:15 UTC+2

**Status: BATCH E NOT STARTED -- blocked on wide-rig sigma_sys floor anomaly (GATE 1 task rule).**

What I did
Batch D GATE 1 closed (fingerprints pushed). Wide-rig `sigma_sys_mag` floor fit for
equipment_id 1 landed at **~15 mmag** on Part 1c check stars (outside 2-5 mmag sanity).
Per R8 and task rule: floor **not applied**; batch E **not started**.

## Blocker

| check | result |
|-------|--------|
| Floor fit on check stars (n=162) | **~15 mmag** (needs ~15 mmag to reach chi2_red ~1.0) |
| Everett & Howell sanity (2-5 mmag) | **FAIL** |
| Task rule (stop if sanity fails) | **STOP** |

See `dev/results/CURSOR_RESULT_batch_D.md` GATE 1 append for the three chi2_red values
(before / scint only / scint + simulated floor).

## Planned batch E scope (when unblocked)

Per `CURSOR_TASK_run_D_and_E.md` Stage 2 and `CURSOR_TASK_batch_E_recut.md`:

| item | change |
|------|--------|
| E.1 | Part 0c pairing on `source_file` |
| E.2 | DAO centroid guard with WCS fallback |
| E.3 | CR-1 cosmic-ray rejection (L.A.Cosmic / astroscrappy) |
| E.4 | T4-1 Option B: single measured `N_equiv` (confirm 3.78 vs 4.71 from Part 2b) |
| E.5 | D5-2 saturation admission gate at 70% of `saturate_limit_adu_85pct` |

Re-cut #2 (`--fast` PASS, `--full` once), per-change separable delta, stop at **GATE 2**
(Milan authorizes final fingerprints).

## Milan decision needed

1. **Investigate** photon/ensemble mis-scaling before batch E (recommended given 15 mmag floor).
2. **Override** sanity band and apply fitted floor anyway, then proceed to batch E.
3. **Proceed batch E** without floor (chi2_red remains ~3.55 on wide rig).

## Files changed

None (batch E not started).

## Errors (if any)

None. Intentional stop per task rule.
