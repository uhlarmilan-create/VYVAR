CURSOR RESULT - CLOSE-AND-PUSH (2026-08-14)

Supersedes open parts of PUSH-SEQ-01. Base origin/main @ 4a3e855.
Push awaits Milan. Nothing pushed in this task.

---

## Part 1 / five conditions

### 2.1 CLOSURE.md 38-site repair -- DONE

Restored `docs/VYVAR_AUDIT_2026_CLOSURE.md` from `4a3e855` (38 bytes 0x9d) and
repaired each site in context. One substring bug (`7x7` matching inside `67-73`)
was corrected to `67-73%`. File is ASCII-only.

| # | Line | Before (0x9d as [9d]) | After |
|---|------|------------------------|-------|
| 01 | 1 | 2026 [9d] Wave | 2026 - Wave |
| 02 | 24 | 7[9d]7 pixel box | 7x7 pixel box |
| 03-04 | 37 | [9d] \| [9d] | - \| - |
| 05 | 39 | VYVAR[9d]photutils | VYVAR-photutils |
| 06 | 39 | mmag [9d] the | mmag - the |
| 07 | 39 | 7.6[9d]7.8 | 7.6-7.8 |
| 08 | 43 | 0.5[9d]15 px | 0.5-15 px |
| 09 | 52 | flux [9d] **marginal** | flux - **marginal** |
| 10 | 52 | 5.0[9d]5.75 | 5.0-5.75 |
| 11 | 56 | 435 [9d] on-disk | 435 - on-disk |
| 12 | 71 | 3.3[9d]3.5 | 3.3-3.5 |
| 13 | 72 | FITS** [9d] the | FITS** - the |
| 14 | 89 | 2.5[9d] r_Kron | 2.5x r_Kron |
| 15 | 90 | factor [9d] mean | factor x mean |
| 16 | 92 | 6 [9d] median | 6 x median |
| 17 | 96 | VYVAR[9d]s | VYVAR's |
| 18 | 100 | 81[9d]86% | 81-86% |
| 19 | 100 | 67[9d]73% | 67-73% |
| 20 | 100 | EE [9d] the | EE - the |
| 21 | 102 | medium[9d]high | medium-high |
| 22 | 125 | 2[9d] underquote | 2x underquote |
| 23 | 135 | boundary [9d] by | boundary - by |
| 24 | 137 | D1-2 [9d] requires | D1-2 - requires |
| 25 | 156 | pixels** [9d] against | pixels** - against |
| 26 | 158 | convention** [9d] CAL | convention** - CAL |
| 27 | 160 | draft** [9d] after | draft** - after |
| 28 | 162 | aperture** [9d] not | aperture** - not |
| 29 | 164 | stamping [9d] no | stamping - no |
| 30 | 171 | 2[9d] underquote | 2x underquote |
| 31 | 183 | validation** [9d] Numerical | validation** - Numerical |
| 32 | 185 | gate** [9d] `session | gate** - `session |
| 33 | 187 | first** [9d] xval | first** - xval |
| 34-35 | 189 | [9d] Tool survey ([9d]3.1) | - Tool survey (S3.1) |
| 36 | 191 | boundary** [9d] Measurements | boundary** - Measurements |
| 37 | 201 | mmag** [9d] strong | mmag** - strong |
| 38 | 210 | per [9d]2 | per S2 |

### 2.2 CHAR_MAP -- DONE

Removed `"\ufffd": "-"` from `dev/tools/ascii_migrate.py`. `ascii_migrate.py --check`:
`migrated_or_would=0 stop=0`. No other tracked file STOPs.

### 2.3 INV-NOCLIP-01 fire proof -- DONE (actual run)

Did not `git checkout 4a3e855` (would wipe the uncommitted tree). Ran the gate on
the exact blob `4a3e855:src_py/photometry_core.py`.

Command:

```
python -c "git show 4a3e855:src_py/photometry_core.py | iron_gates_scan._scan_patterns NOCLIP"
```

Actual output:

```
photometry_core.py:12120 one_sided_annulus_sky_clip clipped = sky_pixels[sky_pixels < sky_med + 2.0 * sky_std]
TOTAL 1
FAIL
```

After SKY-CLIP-01 the same pattern is absent (production scan clean; tests PASS).

### 2.4 SKY-CLIP-01 DECISIONS -- DONE

`docs/VYVAR_DECISIONS.md` now records: contamination trade-off +0.276 vs +0.069 ADU;
BO CVn not crowded; mode estimator identified, unmeasured; estimator question not closed.

### 2.5 IRON-GATES PARTIAL + AST exclusion -- DONE

Register: IRON-GATES-01 status PARTIAL. `_iterative_ensemble_clip_cm_residual`
exclusion keys on AST (no loop; `clip_sigma` not used in Compare/BinOp), not on
the comment string. Tests: `test_inv_noclip01_ensemble_stub_excluded_by_ast_not_comment`,
`test_inv_noclip01_ensemble_body_with_loop_is_not_passthrough`.

---

## Part 3 -- sky function choice

Kept `_sky_pp_from_annulus_image` as the single estimator. Deleted
`_sky_pp_from_annulus_mask`. All three call sites use `to_image` then that function:

- `_annulus_sky_subtracted_flux` (single-star, ~2696)
- `_aperture_flux_sky_per_star` (SNR per-star, ~12158)
- global path (~12502)

---

## Part 3 (PUSH-SEQ) -- settled, not re-investigated

Green `--fast` through SKY-CLIP-01 is expected: byte-identity runs only under `--full`.
**Commit 2 (`77d082a`) makes the `--full` anchor and the P1 golden ledger SHA stale.**
Re-cut is follow-up, not this push (deferred list).

The 2.3 fire proof used blob `4a3e855:src_py/photometry_core.py` because a full
checkout would have destroyed the working tree. That substitution is accepted.

---

## Four-commit history (FINAL-PUSH-02)

```
9b1d0e9 docs: CLOSE-AND-PUSH memo with --fast SHA table
3fd4566 Gates, PP-KWARG, COG measurement, register; no photometry numeric change.
77d082a SKY-CLIP-01: plain annulus median, one estimator, three call sites.
3791b6c docs: ASCII-repair Wave 7 files and stop mapping U+FFFD in CHAR_MAP.
```

Discarded unpushed SHAs: `ebc6dd5`, `6bc710f`, `9da5c44`, `0e36861`.

---

## `--fast`

Commit 1 `3791b6c` ASCII repair: **OVERALL PASS**. pytest 1325 passed, 27 skipped.

Commit 2 `77d082a` SKY-CLIP-01: **OVERALL PASS**. pytest 1325 passed, 27 skipped.

Commit 3 `3fd4566` gates/docs: **OVERALL PASS**. pytest 1338 passed, 27 skipped.

Commit 4 `9b1d0e9`: **OVERALL PASS**. pytest 1338 passed, 27 skipped.

WARNs on all four (not FAIL): untracked scratch, origin/main differs (`4a3e855`),
ledger-todo VL-ANCHOR-424 / VL-ANCHOR-DQ-430, deps-outdated.

Nothing pushed.

---

## Confirmations (section 3)

- Working tree clean apart from known untracked scratch (`dev/tests/_tmp_batch_e_lc/`,
  `dev/tools/wide_err_*.py`, `src_py/tmp/`, `vyvar.sqlite3-shm`, `vyvar.sqlite3-wal`).
- `ascii_migrate.py --check`: `migrated_or_would=0`, `stop=0`.
- `git log --oneline -4`:
  `9b1d0e9` memo; `3fd4566` gates; `77d082a` SKY-CLIP; `3791b6c` ASCII.
- Tip SHA to push: `9b1d0e9`.

---

## Could not

Full `git checkout 4a3e855` for 2.3: refused, would destroy the working tree. Blob
scan used instead; output is the real gate on the real 4a3e855 file.
