# CURSOR RESULT - PFS-SEMANTICS-01

Date: 2026-08-17
Compared with: da9cce4 accepted 49-LC product (METER-DRAFT-DEP-01) vs third
515 rebuild after teaching PFS to key on recorded skip_reason.
Draft 515 photometry SHA (core): **36a53b0**
(`36a53b0cacd58e9fdb922726023e806dd6fbf5a42fb2caeb70f37187a822799d`, n=97)
Full SHA (with comp_qa): `609c6d19a4b1fd1551a3bbac45c8d23a98ee2dd9147fd03c4ef973128035ff38` (n=193)
Quarantined SHA (do not cite): 8f107cf (VL-PFS-8F107CF).
Git rebuild stamp: `1a516c75082d147054f9e43d09013cce5d7dd891` plus uncommitted
PFS-SEMANTICS-01 working tree.
Push: NOT authorized (Milan go required).

Premise: 8f107cf emitted 96 LCs because PFS treated any skip_photometry as
saturation-legacy. This task teaches PFS (does not disable it), rebuilds 515,
and reports gating vs the da9cce4 49-set. Meters and export live in the
companion RESULT files.

## Part 1 - quarantine 8f107cf

Ledger id `VL-PFS-8F107CF` in `dev/validation/VYVAR_VALIDATION_LEDGER.json`:
passes=false, status=quarantined_defect. Reason = PFS rescue of
non-saturation skips. Pointer = `CURSOR_RESULT_SAT_RERANK_01.md`.
No export and no acceptance number may cite 8f107cf. On-disk product was
overwritten by Part 3; the ledger row is the durable record.

## Part 2 - the fix

### 2.1 Rescue scope

`pfs_rescue_eligible` / `decide_target_saturation_policy` /
`apply_per_frame_saturation_to_active_targets`
(`src_py/photometry_core.py:7170-7504`) key on recorded skip_reason, not
bare skip_photometry.

Rescue-eligible: zone_flag==saturated (or likely_saturated) or an explicit
saturation skip_reason (`zone_flag`, `saturovany ciel`,
`per_frame_saturation`, `likely_saturated`, `saturated`).

NEVER rescue `zone_noise` or `below_target_depth` (TARGET-DEPTH-02 outranks
PFS).

Phase 2A re-force (`photometry_core.py:10411-10418`): noise always
re-forced; saturated re-forced only when PFS is OFF. Not the whole
{saturated, noise} set.

### 2.2 Threshold unification

One peak-test authority: `pipeline.inv_sat_limit_peak_test_adu()` =
0.80 x 65535 = **52428 ADU**. Source string:
`INV-SAT-LIMIT peak-test 0.80x container_clip_65535`.
Container clip is a separate named field (`SAT_LIMIT_CONTAINER_CLIP_ADU` =
65535 ADU). Per-frame flags prefer `peak_max_adu > peak_test` over stale
`is_saturated`. Provenance on this run:

| field | value | unit | domain |
|-------|------:|------|--------|
| per_frame_sat_peak_test_adu | 52428 | ADU | pipeline_meta PFS block, SHA 36a53b0 |
| per_frame_sat_peak_test_source | INV-SAT-LIMIT peak-test 0.80x container_clip_65535 | -- | same |
| per_frame_sat_container_clip_adu | 65535 | ADU | named separately, not used as the clean test |

### 2.3 Guard tests (order a, b, c)

`dev/tests/test_pfs_semantics_01.py`

(a) Pre-fix inlined policy: PFS ON + zone_noise + all-clean frames ->
    skip_photometry cleared (the hole). PASS.

(b) Post-fix apply_per_frame_saturation: PFS ON does not clear zone_noise
    or below_target_depth; saturation-zone target with clean frames IS
    rescued (n_rescued=1). PASS.

(c) Per-frame test value == catalog saturate_limit_adu_85pct == 52428 ADU.
    CV-CVn-like fixture peak=55000 ADU (between 52428 and 65535) is NOT
    clean; skip_reason=per_frame_saturation, clean_frac=0. PASS.

### 2.4 Config hygiene

`config.json` `per_frame_saturation_enabled` = false (AppConfig default).
Registry n = 291 (no new key). Rebuild harness
`dev/tools/draft_515_headless_phase012a.py` loads AppConfig then sets
`cfg.per_frame_saturation_enabled = True` (existing per-run instance
override). Provenance snapshot records the override.
`test_flow_doc_config_facts` must pass without editing flow_doc_facts
(DOC_CONFIG_FACTS still False).

UTF-8 file log: `tmp/draft_515_pfs_semantics_01.log` (not PowerShell `*>`).

## Part 3 - third 515 rebuild

Harness: `dev/tools/draft_515_headless_phase012a.py`
START_UTC 2026-08-17T11:15:10Z. PFS_AFTER_LOAD False, PFS_RUN_OVERRIDE True.
PREFLIGHT n_is_saturated=24. LC_TRIM of leftover lightcurve_*.csv before run.

### 3.1 Per-phase runtime (wall clock from harness progress)

| phase | t_start s | t_end s | duration s | domain |
|-------|----------:|--------:|-----------:|--------|
| Phase 0 | 0.1 | 0.4 | 0.3 | 218 actives |
| Phase 1 | 6.4 | 4063.5 | 4057.1 | 97 photometry-set targets, 134 frames |
| Phase 2A to hotovo | 4063.5 | 4594.1 | 530.6 | includes Comp QA 4335.8-4593.8 = 258.0 s |
| Process wall to crash | -- | -- | 5014.0 | INV-CFG-01 after 48 LCs written |

48 LCs were on disk at Faza 2A hotovo. Invariants then crashed (named
defect below). After INV-CFG-01 fix + leftover-LC cleanup,
`run_end_of_run_invariants(..., stamp_postprocess=True)` OK.

### 3.2 Gating (photometry set 97)

Phase 1 photometry set = 218 actives minus 121 vsx_type_out_of_scope = 97.
Compared to da9cce4: 97 -> 49 (45 zone_noise + 3 below_target_depth).

This SHA: **48 LCs**. Legitimate 49 -> 48.

| skip_reason (97-set) | n | vs da9cce4 |
|----------------------|--:|------------|
| zone_noise | 45 | same set |
| below_target_depth | 3 | same set |
| per_frame_saturation | 1 | NEW: CV CVn |
| LC emitted | 48 | 49 - 1 |

CV CVn `1497007144465726080`: zone=saturated, G=5.943, sat_clean_frac=
0.4477611940298507 (< 0.5), skip_reason=per_frame_saturation, no LC.
Peak-test 52428 ADU (was 65535 on 8f107cf, which reported clean_frac=1.0).
Physics outranks the spec: do not force 49.

Unexpected measured IDs: none. Unexpected skip IDs other than CV CVn: none.
stop_and_report: false.

PFS meta: rescued=0, skipped=1, fallback=0, snapshot PFS true.

JSON: `dev/results/PFS_SEMANTICS_01_gating.json`.

### 3.3 B2

0 of 24 SAT-LIMIT saturated IDs in any ensemble (b2_n=0).

### 3.4 SHA stamp

Core 36a53b0 (n=97). Full 609c6d19 (n=193). 8f107cf is not referenced
except as the quarantined state.

## Named defects / physics vs spec

1. 8f107cf quarantined (PFS rescued noise/depth). Ledger VL-PFS-8F107CF.
2. INV-CFG-01 read non-existent meta["config"] instead of
   provenance.config_snapshot (which has vsx_out_of_scope_types=["ROT"]).
   Exposed because PFS no longer blanks skip_reason. Fixed in
   `src_py/invariants_runtime.py` validate_config_behavior.
3. Leftover 8f107cf LC files on disk after 48-LC run. Cleaned; harness
   LC_TRIM added.
4. Rebuild crashed after 48 LCs + Comp QA + Trust (INV-CFG-01). Science
   LCs already written. Invariants stamped after the fix.
5. C5 0.0001 mmag formula identity vs 6-decimal LC storage / 3-decimal
   AAVSO MAG -- see EXPORT-HDR-01 RESULT.
6. BIN-8-9 LOO is on proc CSVs, not rebuilt frames -- see SAT-RERANK-01B.
7. P1 A/B pair (task 6.3) not obtained: INV-CAL-01 cal_diag missing on
   draft_000435_p1mini after the headless chain (443.4 s). 1 passed
   (mini present), 4 ERROR at fixture setup. P1-RECUT remains OPEN.
   Physics: mini is photometry-ready without cal_diag.json; FAIL-close is
   correct for a dark-applied stamp. Do not fake cal_diag to green the pair.
8. First `--fast` FAIL was BLE001 on the harness tee (`except Exception`
   without noqa). Marked. Second FAIL was leftover VYVAR_INVARIANTS_P1
   making P1 golden ERROR inside `--fast` (0 skipped). Unset; OVERALL PASS.

## Part 6 close (this machine)

`session_baseline_check.py --fast` OVERALL PASS. pytest 1442 passed, 28
skipped. flow_doc_config_facts green via config.json default false (2.4),
not via edited flow_doc_facts. Git head at the check was 1a516c7 plus
this working tree; content commits follow. Push NOT authorized.

P1 A/B: see named defect 7. P1-RECUT remains OPEN.

## Docs impact

- docs/VYVAR_DECISIONS.md -- PFS-SEMANTICS-01 (replaces blocked SAT-RERANK note)
- docs/VYVAR_ROADMAP.md -- SAT-RERANK-01 DONE; BIN-8-9 OPEN; SAT-LIMIT CLOSED; D1-2 OPEN
- docs/VYVAR_STATE.md -- 36a53b0 product; 8f107cf quarantine
- docs/VYVAR_JOURNAL.md -- this close
- FLOW / flow_doc_facts.py: none (default remains false; 2.4)

## Recurrence

Recurrence: new test test_pfs_semantics_01 (a/b/c fire proofs).

## Files

- src_py/photometry_core.py
- src_py/pipeline.py
- src_py/invariants_runtime.py
- dev/tests/test_pfs_semantics_01.py
- dev/tools/draft_515_headless_phase012a.py
- dev/tools/pfs_semantics_01_gating.py
- dev/results/PFS_SEMANTICS_01_gating.json
- this file
