CURSOR RESULT - EPSF-VALID-02 S1-S4 (STOP-A)

Date: 2026-08-22. HEAD: `2ba3d58`. Parent: EPSF-VALID-02 FINAL task.
**STOP-A COMPLETE** ù S1ùS4 all pass; architect review before S5/S6.

---

## S1 ù Restore draft 516 aperture columns (EXECUTED)

**Action:** Restored all 134 `proc_BO_CVn_Light_*.csv` from `proc_backup_pre_accept/`; re-merged `psf_*` columns from accept-rerun live files by `catalog_id` (map, preserving backup row order). Deleted `proc_MASTERSTAR.csv`.

| Check | Result |
|-------|--------|
| Files restored | **134** |
| proc_MASTERSTAR deleted | **Yes** |
| Non-PSF numeric verify vs backup | **0 file mismatches** (proper `read_vyvar_csv` + `isclose` atol=1e-9) |

Artifacts:
- `dev/results/context/session_20260822_epsf_valid_02_s1s4/s1_restore_summary.json`
- `dev/results/context/session_20260822_epsf_valid_02_s1s4/s1_post_restore_hash_table.csv`
- `dev/sandbox/epsf_valid_02_s1_restore.py`, `epsf_valid_02_s1_verify.py`

**References to proc_MASTERSTAR:** excluded from epoch collection (`proc_frame_store.is_masterstar_proc_name`); no production path requires the aligned-lights sidecar. Safe to delete.

---

## S2 ù F5 enumerator fix

**Commit:** `57046dd` ù `EPSF-VALID-02 F5: science-light frame enumerator for ePSF accounting.`

- `list_epsf_science_light_fits()` / `is_non_science_aligned_fits()` in `src_py/epsf_frame_accounting.py`
- Excludes `MASTERSTAR.fits` and `proc_*` cal frames
- Test: `dev/tests/test_epsf_science_light_fits.py` ù draft 516 asserts **134** science lights

---

## S3 ù F6 PSF-only merge

**Commits:** `c218921` (merge path), `2ba3d58` (registry wire)

- `src_py/epsf_psf_merge.py` ù `run_epsf_psf_merge_job`, `merge_psf_into_sidecar`, **INV-PSF-ADDITIVE-01**
- UI RUN ePSF (`app.py`) calls merge job only ù not full `export_per_frame_catalogs`
- Full export gated behind `full_catalog_export=True` (pipeline internal only)
- Moffat pass skipped when `_psf_merge_only=True`
- **INV-PSF-ADDITIVE-01** documented in `docs/VYVAR_INVARIANTS.md`; wired in `WIRED_INV_IDS`
- **EXPORT-PARITY-01** recorded HIGH in `docs/VYVAR_ROADMAP.md` (R5 evidence; not investigated here)
- Tests: `dev/tests/test_epsf_psf_merge.py`

---

## S4 ù Acceptance re-run v2 (COMPLETE)

**Script:** `dev/sandbox/epsf_valid_02_s4_accept_v2.py`  
**Model:** frozen production `masterstar_epsf.fits` on draft 516  
**Path:** F5+F6 PSF merge (science-light frames only)  
**Runtime:** ~159 min (exit 0)

| Metric | Result |
|--------|--------|
| frames_total | **134** |
| frames written (PSF merge ok) | **134** |
| Aperture hash vs S1 baseline | **0 mismatches** |
| CSS target psf_ok | **134 / 134** |
| INV-PSF-FRAME-01 | **ok** (0/134 zero-ok) |
| INV-PSF-ADDITIVE-01 | **silent** (0 aperture mismatches) |

Artifacts: `dev/results/context/session_20260822_epsf_valid_02_s1s4/s4_accept_summary.json`, `s4_epsf_job_summary.json`, `s4_aperture_hash_table.csv`, `s4_css_target_coverage.csv`

---

## Gate status

| Gate | HEAD | Result |
|------|------|--------|
| `--fast` | `2ba3d58` | **OVERALL PASS** ù 1504 passed, 32 skipped |
| `--full` recut | ù | **Not run** ù S1 restore touches Archive draft 516 only (outside snapshot); F6 code change does not require recut for STOP-A |

Evidence: `dev/results/context/session_20260822_epsf_valid_02_s1s4/fast_baseline_stdout_v2.txt`

Prior `--full` SHAs (unchanged by this work): core `9902d918ù` n=121, extended `472bc9e4ù` n=179.

---

## Commits (this session)

| Hash | Description |
|------|-------------|
| `57046dd` | F5 science-light enumerator |
| `c218921` | F6 PSF-only merge + INV-PSF-ADDITIVE-01 |
| `2ba3d58` | F6 WIRED_INV_IDS registration |

---

## Files changed (production)

| File | Change |
|------|--------|
| `src_py/epsf_frame_accounting.py` | science-light FITS list |
| `src_py/epsf_psf_merge.py` | new PSF merge job |
| `src_py/pipeline.py` | merge-only guard, full_catalog_export flag |
| `src_py/app.py` | RUN ePSF ? merge job |
| `src_py/invariants_runtime.py` | INV-PSF-ADDITIVE-01 wired |
| `docs/VYVAR_INVARIANTS.md` | invariant doc |
| `docs/VYVAR_ROADMAP.md` | EXPORT-PARITY-01 |
| `dev/tests/test_epsf_science_light_fits.py` | F5 tests |
| `dev/tests/test_epsf_psf_merge.py` | F6 tests |

Live draft 516: 134 proc CSVs restored; `proc_MASTERSTAR.csv` removed.

**STOP-A COMPLETE** ó ready for architect review before S5/S6.
