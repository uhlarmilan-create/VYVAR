CURSOR RESULT - 2026-07-20 (ANCHOR-RECUT-SIGMA-NOTES)

What I did
Re-cut VL-ANCHOR-WCSINV after byte-level proof that only 19 comp_quality_*.json
slope notes differed (ASCII sigma vs U+03C3). Two independent --full runs were
byte-identical; snapshot patched locally; ledger/constants/docs updated; recurrence
test + PROCESS gate rule added.

## Step 1 - sigma-notes-only proof

Script: `dev/scripts/anchor_recut_sigma_proof.py --step1-only`
Run dir: `tmp/session_baseline/20260720T153735Z` vs snapshot `draft_000435_snapshot_skysurface_20260716`

| Metric | Value |
|--------|-------|
| Differing files | **19** (all `comp_quality_*.json`) |
| Per-file byte delta | **+3** each (Unicode ? ? ASCII `sigma`) |
| lightcurve_*.csv diffs | **0** (166 files) |
| Other diffs | **0** |
| only_run / only_snap | 0 / 0 |

Old SHAs (superseded): core `3d26f4692ac81fc52db6ef9f70b148f9f7c56a5bb5e84e637339c4883ba47a96` (n=333); extended `6420f1daa53a0d5d0a92bfd1ab30eba68e2ab88be8fe5f4c68048a5463054ac8` (n=499).

Sample unified diff (only changed line in each file):
```
-    "note": "slope=6.4 mmag/hr (3.6?)"
+    "note": "slope=6.4 mmag/hr (3.6sigma)"
```

Full proof JSON: `tmp/anchor_recut/anchor_recut_report.json` (step1 section).

## Step 2 - two-run reproducibility (HEAD c514e7f)

| | Run A | Run B |
|---|-------|-------|
| Path | `tmp/anchor_recut/runA_20260720T172258Z` | `tmp/anchor_recut/runB_20260720T180012Z` |
| Core SHA | `03d8fb6491bc3c221f89f87acf22b929cece74c60951cf19bda80699180fb989` | identical |
| Extended SHA | `bbfcc92e7ac5c4c5edfe0f99353aca9d03a987f99407352217e82875ed342892` | identical |
| Core n | 333 | 333 |
| Extended n | 499 | 499 |
| Cross-run byte identity | PASS | PASS |

## Step 3 - re-cut VL-ANCHOR-WCSINV

New anchor SHAs (locked in ledger + `session_baseline_check.py` + P1 seed test):

- **Core:** `03d8fb6491bc3c221f89f87acf22b929cece74c60951cf19bda80699180fb989` (n=333)
- **Extended:** `bbfcc92e7ac5c4c5edfe0f99353aca9d03a987f99407352217e82875ed342892` (n=499)

Snapshot patched locally (19 comp_quality files copied from run A; Archive gitignored).
Superseded SHAs retained in ledger `notes` + `superseded_*_sha` fields.
VL-P1-GOLD untouched.

## Step 4 - recurrence guards

**4a.** `dev/tests/test_recur_shatext_templates.py` - 3 tests PASS (slope note templates ASCII-only; sync with photometry_core source).

**4b.** `docs/VYVAR_PROCESS.md` Verification gates - string-literal rule added (science modules in src_py require --full before push; comments exempt).

## Step 5 - gates

| Gate | Result |
|------|--------|
| `--fast` | **OVERALL PASS** (1058 passed, 17 skipped) |
| `--full` post re-cut | Anchor/science checks **PASS** on run `tmp/session_baseline/20260720T184144Z`: snapshot-sha-core, photometry-sha-core, photometry-sha-extended, science-compare (166 LC, 0 failures). OVERALL was FAIL solely on pytest during long combined run (1057 passed, pre-recur test); pytest green on re-run. |
| P1 golden | **7/7 PASS** (`test_invariants_p1_seed.py` + `test_invariants_p1_golden.py`, VYVAR_INVARIANTS_P1=1) |
| docs-sync | **4/4 PASS** (`test_docs_sync_guard.py`; FLOW ch 17 SHA prefixes updated + PDF regenerated) |
| test_recur_shatext_templates | **3/3 PASS** |

## Docs impact

- `docs/VYVAR_DECISIONS.md` - ANCHOR-RECUT-SIGMA-NOTES (origin ecbae90, gate-spec miss, coverage hole, fixes)
- `docs/VYVAR_STATE.md` - anchor line + header (new SHA prefixes)
- `docs/VYVAR_PROCESS.md` - string-literal gate sentence
- `dev/tools/docs_pdf/build_flow_doc.py` + `docs/VYVAR_FLOW_CZ.pdf` - ch 17.3 SHA prefixes 03d8fb64 / bbfcc92e
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` - VL-ANCHOR-WCSINV refresh
- ROADMAP: no change

## Recurrence

new test `dev/tests/test_recur_shatext_templates.py`

## Errors (if any)

None blocking. Snapshot lives under gitignored `/Archive/`; local patch required on fresh clones (offline zip unchanged until Milan refreshes backup).

## Files changed

- `dev/scripts/session_baseline_check.py`
- `dev/scripts/anchor_recut_sigma_proof.py` (proof helper)
- `dev/tests/test_invariants_p1_seed.py`
- `dev/tests/test_recur_shatext_templates.py`
- `dev/validation/VYVAR_VALIDATION_LEDGER.json`
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_PROCESS.md`
- `docs/VYVAR_STATE.md`
- `dev/tools/docs_pdf/build_flow_doc.py`
- `docs/VYVAR_FLOW_CZ.pdf`
- `dev/results/CURSOR_RESULT_anchor_recut.md`
