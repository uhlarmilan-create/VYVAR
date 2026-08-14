CURSOR RESULT - PP-KWARG-01 (2026-08-14)

Register ID: **PP-KWARG-01**
Base: `origin/main @ 4a3e855` (implementation uncommitted on working tree)

---

## What I did

1. Re-verified F1-F3 on `4a3e855`.
2. Removed dead `use_gpu_if_available` from `_pp_kw` in `app.py` and `night_run.py`.
3. Added static kwarg signature scanner (`dev/tools/kwarg_compat_scan.py`) + wired tests.
4. Switched two preprocess tests from deprecated alias to production entry point.
5. Audited `src_py` call sites; documented unexercised pipeline paths (3.3).

---

## Section 0 -- Re-verification

| Finding | Reproduced? | Evidence |
|---------|-------------|----------|
| **F1** kwarg mismatch on production path | YES | `qc_enrich_calibrated_lights_in_place` has no `use_gpu_if_available`; `_pp_kw` in `app.py:794-801`, `night_run.py:233-240`; splat at `app.py:838,865`, `night_run.py:278,290` |
| **F2** kwarg inert on deprecated alias | YES | `preprocess_calibrated_to_processed` accepts and discards it (`pipeline.py:18124-18139`) |
| **F3** no gate caught it | YES | `--full` is photometry-only (INV-ANCHOR-00); tests called deprecated alias |

Not caused by uncommitted CLOSE-IRON-GATES work (confirmed on base tree).

---

## Part 1 -- Fix

### Change

Removed one line from each `_pp_kw` dict:

```python
use_gpu_if_available=False,  # removed
```

Removed unused `preprocess_calibrated_to_processed` imports from `app.py` and `night_run.py`.

**Did not** add the parameter to `qc_enrich_calibrated_lights_in_place` (inert legacy API).

### Call-site audit (`src_py`)

Static scan (`kwarg_compat_scan.py`) over all `src_py/**/*.py`:

| Before fix | After fix |
|------------|-----------|
| 4 mismatches (app.py x2, night_run.py x2) | 0 mismatches |

Manual check: `_pp_kw` is only consumed by the four `qc_enrich_calibrated_lights_in_place(**_pp_kw)` splats. No other keys removed.

Other `use_gpu_if_available` sites (valid):

| Location | Target | Status |
|----------|--------|--------|
| `pipeline.py:19212-19240` | `quick_preprocess_last_import` -> deprecated alias | Valid (alias accepts kwarg) |

### `preprocess_calibrated_to_processed` callers

| Caller | Role |
|--------|------|
| `pipeline.AstroPipeline.quick_preprocess_last_import(run=True)` | **Production** (legacy quick-preprocess API) |
| `dev/scripts/chiandh_inject_platesolve_phot.py` | Dev script |
| `dev/scripts/palomar7_continue367_bgr.py` | Dev script |
| `dev/scripts/pilot_palomar7_continue364.py` | Dev script |

**Proposal (do not implement here):** keep deprecated alias until `quick_preprocess_last_import` is migrated to call `qc_enrich_calibrated_lights_in_place` directly; emit `DeprecationWarning` from alias; remove alias after one release when quick-preprocess path is updated. Tests no longer depend on the alias for preprocess behaviour.

---

## Part 2 -- Close the gap

### 3.1 Signature-compatibility check

**Module:** `dev/tools/kwarg_compat_scan.py`
**Test:** `dev/tests/test_kwarg_compat.py` (wired via `--fast` pytest)

**Method:** AST walk over `src_py`:
- Collects function/method signatures (params + `**kwargs` presence).
- Resolves imports to qualified names.
- Within each function body, tracks `dict(...)` / `{...}` assignments.
- At `Call` nodes, checks literal kwargs + `**tracked_var` keys against resolved callee.

**Can see:**
- Literal keyword args at call sites.
- `**var` splats where `var = dict(key=...)` or `var = {key: ...}` in the same function.
- Module-level and class-method callees when import/name resolves.

**Cannot see (stated limits):**
- Dynamically built dicts (`dict.update`, comprehension, function return).
- Cross-function kwarg dict passing (built in caller A, splatted in nested B without same-scope assign).
- `getattr(fn, name)(**kw)` or computed callee names.
- Third-party / C-extension signatures not in `src_py`.
- Inline `{**a, **b}` merge at splat site.
- Kwargs routed through `*args, **kwargs` forwarding without local dict literal.

### Fire proof @ 4a3e855 (before Part 1 fix)

Command: `python dev/tools/kwarg_compat_scan.py`

```
app.py:838: qc_enrich_calibrated_lights_in_place() use_gpu_if_available -- unexpected keyword 'use_gpu_if_available' for qc_enrich_calibrated_lights_in_place
app.py:865: qc_enrich_calibrated_lights_in_place() use_gpu_if_available -- unexpected keyword 'use_gpu_if_available' for qc_enrich_calibrated_lights_in_place
night_run.py:271: qc_enrich_calibrated_lights_in_place() use_gpu_if_available -- unexpected keyword 'use_gpu_if_available' for qc_enrich_calibrated_lights_in_place
night_run.py:285: qc_enrich_calibrated_lights_in_place() use_gpu_if_available -- unexpected keyword 'use_gpu_if_available' for qc_enrich_calibrated_lights_in_place
FAIL: 4 kwarg mismatch(es)
```

After fix: `OK: no kwarg mismatches`

Unit fire proof: `test_pp_kwarg01_fire_proof_detects_splat_mismatch` (fixture with `_pp_kw` splat).

### 3.2 Test changes

| Test file | Before | After | Why |
|-----------|--------|-------|-----|
| `test_skysf_double_guard.py::test_t6_*` | `preprocess_calibrated_to_processed(...)` | `qc_enrich_calibrated_lights_in_place(...)` | Production path |
| `test_skipproc_qc_allowlist.py::test_preprocess_creates_no_processed_directory` | deprecated alias | `qc_enrich_calibrated_lights_in_place(...)` | Production path |

Other tests in those files already call `_qc_enrich_calibrated_in_place` (internal worker); left unchanged.

### 3.3 Unexercised production paths (report only)

Stages per INV-DAG-01. "Unexercised" = no automated test or `--fast` gate runs the **production orchestration entry** end-to-end.

| Stage | Production entry | Test / gate coverage | Gap |
|-------|------------------|----------------------|-----|
| Import / scan | `_run_vyvar_full_pipeline` scan + import steps (`app.py`, `night_run.py`) | Partial manifest/import unit tests | Full RUN VYVAR import chain not gated |
| Calibration | `run_draft_ram_calibration_qc_to_obs_files`, importer stack combine | `test_cal_*`, INV-CAL gates | No E2E calibrate-from-raw in `--fast` |
| **Preprocess** | `_vyvar_execute_preprocess_pending` / `_night_run_preprocess` with `_pp_kw` splat | `_qc_enrich_*` unit tests; **now kwarg gate** | Orchestration wrapper was untested until PP-KWARG-01 gate |
| Align | `_vyvar_execute_platesolve_pending`, `_night_run_platesolve` | `test_astrometry_fault_isolation` hits `astrometry_align_and_build_masterstar` directly | UI/night_run wrapper kwargs/orchestration not scanned for all stages |
| MASTERSTAR + DAO detect | Inside `astrometry_align_and_build_masterstar` | Partial (`test_masterstar_*`, fault isolation) | DAO detection threshold path not in anchor |
| Per-frame photometry | `run_full_photometry_pipeline` per-frame leg | `--full` anchor (frozen inputs, photometry only) | INV-ANCHOR-00: cal/preprocess/align/detect invisible |
| Phase 0+1 | comp selection, aperture BPM | Unit tests on components | Full phase01 orchestration from RUN VYVAR not gated |
| Phase 2A | `run_phase2a` / ensemble | `--full`, golden tests on outputs | Entry kwargs from UI path not fully scanned |
| Postprocess / export | `export_reports`, AAVSO writers | OSC-03 gate, sparse export tests | Full export orchestration from completed draft |

**Root cause class:** production migrated from deprecated alias to new function; tests and gates stayed on the old symbol. PP-KWARG-01 gate catches signature drift on `src_py` static calls including splats.

---

## Part 4 -- Draft 511

**Local workspace:** `Archive/Drafts/draft_000511` **not present** in this repo checkout; state inferred from architect report + `infolog_20260814_085528.txt`.

**Reported state at failure (06:58:04):**
- Draft 511 created.
- Calibration + QC completed for **all 150** frames.
- Auto FWHM limit **5.362 px**; **134/150** frames selected for continuation.
- Failed at **MAKE MASTERSTAR** with PP-KWARG-01 TypeError during preprocess step inside RUN VYVAR.

**Before clean restart (Milan authorization):**
- Remove or archive entire `draft_000511` tree (partial calibrated/QC artifacts must not be resumed).
- Clear any DB rows / draft registry entries for draft 511 if created.
- Remove partial `platesolve/` / `detrended/` / `masterstar` outputs if any were written after the failure point (architect: failed before MASTERSTAR completed -- likely no platesolve tree, but verify infolog + disk).
- Do **not** resume mid-pipeline; restart RUN VYVAR from scan/import.

**Not started in this task** (requires Milan + CLOSE-IRON-GATES settlement).

---

## `--fast` @ 4a3e855 (uncommitted tree)

```
pytest                       PASS   1333 passed, 27 skipped
OVERALL: PASS
```

---

## Files changed

| Path | Change |
|------|--------|
| `src_py/app.py` | Remove dead kwarg + unused import |
| `src_py/night_run.py` | Same |
| `dev/tools/kwarg_compat_scan.py` | NEW scanner |
| `dev/tests/test_kwarg_compat.py` | NEW gate + fire proof |
| `dev/tests/test_skysf_double_guard.py` | Production entry in test_t6 |
| `dev/tests/test_skipproc_qc_allowlist.py` | Production entry in preprocess test |
| `dev/results/REGISTER_DIFF_PP_KWARG_01.md` | Authorization diff |

**Not committed. Not pushed.**

---

## Could not do

| Item | Why |
|------|-----|
| Inspect draft 511 artifacts locally | Tree absent from workspace |
| Start draft 511 re-run | Task forbids; needs Milan authorization |
