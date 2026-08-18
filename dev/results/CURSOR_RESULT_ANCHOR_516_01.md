# CURSOR RESULT - ANCHOR-516-01

Date: 2026-08-18
Task: verify draft_000516 vs draft_000515 identity; re-cut anchors onto 516.
Verdict: **STOP at Part B** (product SHA mismatch). Part C not executed.
Push: NOT authorized.

Premise (0.1): draft_000516 (2026-08-17 evening) claimed identical to
draft_000515 product SHA **de6f7c8** (48 LCs, GAIN-PT stack). Compared:
516 vs 515 export photometry on the same BO CVn night, not da9cce4/8f107cf
eras. Outcome: **not byte-identical**; architect review required before
any anchor re-cut.

## Part A - Provenance of 516 (read)

| field | draft_000516 | draft_000515 (reference) |
|-------|--------------|--------------------------|
| draft_manifest updated_utc | 2026-08-17T22:04:59+00:00 | 2026-08-17T15:14:42+00:00 (pipeline stamp) |
| infolog run window (UTC) | 20:21:19 - 22:04:58 | (prior session) |
| pipeline_meta git_hash | **d5ef039** | **6b23633** |
| git_dirty / dirty_code | True / False | True / True |
| entry_point | run_phase2a | run_phase2a |
| calibration_mode | vyvar_calibrated | vyvar_calibrated |
| cal_diag.json | present (CAL-DIAG-v2, INV-CAL-01 ok) | present |
| per_frame_saturation_enabled | **False** (config default) | **True** (harness override) |
| PFS at Phase 2A | OFF | ON |

516 ran on **d5ef039** (XVAL-AIJ-02 stamp tip). Requirement met. It did
**not** re-run on the same code+config state as 515 (515 stamped 6b23633
with PFS ON).

Run summary (infolog): Phase 2A finished 21:58:03; 48 LCs from 134 frames;
HRD + PDF 22:04:58; pipeline OK.

## Part B - Identity measurement 516 vs 515

### Product SHA

| draft | core SHA | core n | extended n | n LC |
|-------|----------|-------:|-----------:|-----:|
| 515 | **de6f7c8** | 97 | 193 | 48 |
| 516 | **6dc6ef2e** | 97 | 145 | 48 |

Expected de6f7c8 if truly identical: **FAIL**.

### Per-target science (48 shared LCs)

- **MAG:** all science mag columns (`mag_calib_final`, `delta_mag`, etc.)
  max |delta| = 0.0 on all 48 LCs. Science-meaningful compare: **benign**.
- **ERR:** differs on **46/48** LCs. Max |delta err| = **6.59 mmag**
  (target `1498321301379345408`). BO CVn median err 8.945 mmag (515) vs
  8.532 mmag (516). Byte-identical ERR export claim: **FAIL**.
- **Byte files:** all 48 `lightcurve_*.csv` differ at byte level (QC/meta
  columns: `err`, `err_photon`, `ct_n_comp`, etc.). 5 `comp_quality_*.json`
  differ. `comparison_stars_per_target.csv` differs (1374 shared-row cell
  diffs). `comp_qa_*.json`: 515 n=96, 516 n=48.

### Gating (active_targets.csv, n=218)

| skip_reason | 515 | 516 |
|-------------|----:|----:|
| zone_noise | 45 | 45 |
| below_target_depth | 3 | 3 |
| per_frame_saturation | 1 | 0 |
| zone_flag | 0 | 1 |
| accepted (LC written) | 48 | 48 |

One target differs: `1497007144465726080` (CV CVn) -- 515
`per_frame_saturation` vs 516 `zone_flag`. Same 48-LC product count; reason
label changed with PFS OFF on 516.

### ERR_MODEL (AAVSO BO CVn export)

Both drafts:

`#ERR_MODEL=mode=calibrated; gain=g_pt=0.6371 e-/ADU_container; calib=none`

Line matches. ERR numeric column still drifts between runs.

### STOP rationale

Task rule: if SHA differs, per-file diff and **STOP for architect review**.
516 is **not** a byte replay of 515 de6f7c8. Likely drivers: (1) 515 ran
at 6b23633 with PFS ON; 516 at d5ef039 with PFS OFF; (2) non-deterministic
or config-sensitive err assembly / comp sidecars; (3) comp_qa emission count.

**Part C not executed** (P1 golden re-cut, --full anchor, ledger update).

## Part D - Retirement inventory (read-only; re-cut blocked)

Live references that **must be repointed to 516** before Milan deletes
older drafts (grep of tests/tools/ledger/docs; JOURNAL/DECISIONS prose
only = historical, OK to delete with drafts):

### Gates / tests (MUST repoint)

| file | reference |
|------|-----------|
| `dev/scripts/session_baseline_check.py` | `DRAFT_ID=435`, `SNAPSHOT_NAME=draft_000435_snapshot_skysurface_20260716`, SHA constants `5bccd85a`/`7fdcdca4`, fingerprints for draft 435 |
| `dev/tests/test_invariants_p1_golden.py` | `MINI_NAME=draft_000435_p1mini`, `DRAFT_ID=435` |
| `dev/tests/test_invariants_p1_seed.py` | `SNAPSHOT=draft_000435_snapshot_skysurface_20260716` |
| `dev/tools/build_p1_golden_mini.py` | `SOURCE_DRAFT=draft_000435`, `MINI_NAME=draft_000435_p1mini` |
| `dev/validation/VYVAR_VALIDATION_LEDGER.json` | `VL-ANCHOR-WCSINV` (435 snapshot SHAs), `VL-P1-GOLD` (435_p1mini SHAs) |
| `dev/scripts/session_baseline_check.py --fast` | manifest-db-parity hint `draft_id=435` |

### Tools still pointing at 435 snapshot or 515 (repoint or archive)

`dev/tools/wide_err_*.py`, `dev/tools/xval_bo_01_dump.py`, `dev/tools/sat_limit_01_*.py`,
`dev/tools/d515_accept_01*.py`, `dev/tools/batch_e_physical_recut.py`,
`dev/tools/anchor_manifest_check.py`, `dev/scripts/audit_stage3_*.py`,
`dev/scripts/anchor_recut_sigma_proof.py`, closure_step1*.py (435 snapshot),
and others under `dev/tools/` / `dev/scripts/` (full grep in repo).

### DELETE-OK list for Milan (after architect accepts 516 anchor)

**Do not delete until Part C passes on confirmed-identical 516.**

Candidate retirement (Archive/Drafts/):

- `draft_000435` (live trim candidate; keep until 516 snapshot cut)
- `draft_000435_snapshot_skysurface_20260716` (superseded by 516 snapshot)
- `draft_000435_p1mini` (superseded by 516_p1mini)
- `draft_000436`, `draft_000437` (historical anchor attempts)
- `draft_000509`, `draft_000513`, `draft_000514` (superseded experiment drafts)
- `draft_000515` (superseded if 516 confirmed as product carry)

**Keep:** `draft_000516` (+ future `draft_000516_snapshot_*`, `draft_000516_p1mini`).

Cursor does not delete draft directories.

## Spec defects

1. Premise "516 identical to 515 de6f7c8" is false on measured SHA and err.
2. 516 was run with **PFS OFF**; 515 product de6f7c8 used **PFS ON**
   (per-run override). Not comparable as a replay without matching PFS.
3. 515 pipeline stamp is **6b23633**, not d5ef039/a0d326c content-only replay.

## Architect options (not executed)

A. Re-run 516 Phase 2A at d5ef039 with **PFS ON** (same override as 515)
   and re-measure SHA vs de6f7c8.
B. Accept 516 as new canonical with SHA **6dc6ef2e** (mag-identical, err drift
   documented) and proceed Part C with new golden SHAs.
C. Treat 516 as failed identity check; keep 515 as product reference.

## Docs impact

Minimal STOP note only (this RESULT). STATE/ROADMAP/DECISIONS/ledger
unchanged pending architect decision. No code change.

Recurrence: n/a (measurement gate / not a bug-class)

## Files

- `dev/results/CURSOR_RESULT_ANCHOR_516_01.md` (this file)
- `dev/results/ANCHOR_516_01_identity.json`

## Errors

None blocking measurement. Part C/D re-cut **intentionally skipped**.
