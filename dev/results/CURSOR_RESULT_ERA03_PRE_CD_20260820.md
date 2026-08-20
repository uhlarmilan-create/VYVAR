CURSOR RESULT - 2026-08-20T13:15:00Z

What I did
Completed ERA-03 pre-C/D P1-P3 and Part C freeze/P1 recut setup. Fixed
--fast blockers (INV-PIN registry, P1-seed SHA, params registry, Part A
report schema). Applied check-pin distance fallback when pinned check is
not in comp pool (BO KNAME continuity). `--full` gate running on new
snapshot; `--fast` re-run in progress.

## Output / findings

### P1 -- Check-star pinning (PASS)
- `pinned_ensembles.csv`: check_catalog_id + check_kname per target (48 targets; 2 unique check stars)
- FW `check_kmag`: anchor check `1497613731286514432`; MAD **8.201 mmag** (anchor match)
- Bug fix: compute check-target separation from MASTERSTAR when check not in comp_pt (was NaN -> distance_violation -> dynamic fallback)

### P2 -- ASCII + --fast blockers
- INV-PIN-01..04 added to `WIRED_INV_IDS` + `VYVAR_INVARIANTS.md`
- `test_masterstar_gaia_01`: accept E1_tightened schema
- `params_registry.json`: +14 MASTERSTAR fields; `VYVAR_PARAMS.md` regenerated
- `--fast`: prior FAIL on P1-seed (SHA 477dc8cf -> 9902d918); **re-run in progress**

### P3 -- SHA scope 97 -> 121
- Table: `dev/results/context/session_20260819_era03/sha_scope_97_to_121.json`
- Core **9902d918** n=121; extended **472bc9e4** n=179
- Ledger VL-ANCHOR-WCSINV updated (supersedes 477dc8cf / cleanrebuild snapshot)
- Phase0 funnel: 265 active; 60 photometry LCs; CV CVn `per_frame_saturation`
- `session_baseline_check.py`: funnel fingerprints + snapshot name updated

### Part C -- Freeze + P1 recut
- Snapshot: `Archive/Drafts/draft_000516_snapshot_era03_20260820` (SHA match live 516)
- Evidence: `dev/results/context/session_20260819_era03/part_c_freeze.json`
- P1 mini rebuilt; VL-P1-GOLD: core **6af4539c** n=115, manifest **7c3796db**
- P1 golden pytest: PASS (after ledger lock update)
- **`--full`**: photometry sub-gates PASS (9902d918 n=121; science 60/0; funnel 265); first run OVERALL FAIL only on stale pytest (pre UI-params fix); **re-run in progress**

### Part D -- Partial / pending
- Exports exist on live 516 (BO/FW AAVSO + VarAstro)
- **BO KNAME**: still `1497442379271632384` on disk (pre distance-fix product); needs photometry re-run + export refresh for SUBMIT-01 KNAME check vs anchor `1497613731286514432`
- FW KNAME: correct (`1497613731286514432`)
- Docs: STATE + JOURNAL updated; ROADMAP/DECISIONS/PARAMS/CHANGELOG commit series pending
- **PUSH-STAMP-01**: not requested (await Milan)

## Errors (if any)
- None blocking pre-C/D code path. Live BO export KNAME stale until photometry re-run.

## Files changed
- `src_py/invariants_runtime.py`, `docs/VYVAR_INVARIANTS.md`
- `src_py/photometry_core.py` (check-pin distance fallback)
- `dev/scripts/session_baseline_check.py`
- `dev/validation/VYVAR_VALIDATION_LEDGER.json`
- `dev/validation/params_registry.json`, `docs/VYVAR_PARAMS.md`
- `dev/tests/test_invariants_p1_seed.py`, `dev/tests/test_masterstar_gaia_01.py`
- `dev/tools/build_p1_golden_mini.py`, `dev/tests/test_invariants_p1_seed.py`
- `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`
- Snapshot on disk: `draft_000516_snapshot_era03_20260820`, rebuilt `draft_000516_p1mini`
