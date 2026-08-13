CURSOR RESULT - 2026-08-13

What I did
Part A: fixed `robust_frame_fwhm` tests (commit `d758c83`), `--fast` OVERALL PASS.
Part B: resolved undecided restore items, wrote pre-restore checksum, final plan ready.
Part C: not executed (awaiting Milan approval for restore + push).

---

## Part A -- robust_frame_fwhm tests

### A.1 Confirmation

| item | value |
|------|-------|
| Assertion | `raise FileNotFoundError(f"frame {n}")` |
| Path | `Archive/Drafts/draft_000508/calibrated/lights` |
| File:line | `dev/tests/test_robust_frame_fwhm.py:19` (commit `80912e0`) |

Both tests failed since introduction; never passed on machines without draft_000508.

### A.2 Fix chosen

**Committed npz fixture** at `dev/tests/fixtures/robust_fwhm_frame62.npz`.

Why not skip: skip hides regression; the point is to exercise FWHM logic.
Why not live draft path: draft_000508 absent here; draft_000435 has the same
BO_CVn_Light_062 frame. Why npz not fits: `*.fits` is gitignored; 1000x1000
center crop compresses to ~3.2 MB vs 11 MB full frame.

Test loads fixture with `use_center_crop=False` (crop pre-applied).

### A.3 FWHM verification on fixture source

Draft 435 `BO_CVn_Light_062.fits` (same field/frame as draft 509 reference):

- `_robust_frame_fwhm_median`: **5.30 px**, n_fwhm_sample=80
- `_qc_fwhm_elongation`: **5.31 px**

Within test bounds (4.5-6.0 and >4.0). Both tests PASS.

### A.4 Commit

**`d758c83`** `test(qc): commit frame-62 FWHM fixture; drop draft_508 path dependency`
(separate from `5dd2a4d` / `7eb125a`; not amended)

### A.5 `--fast` raw output

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   d758c83
git-staged                   PASS   none
git-untracked-known          WARN   23 known untracked
git-untracked                WARN   CURSOR_TASK.md; dev/results/MEMO_ensemble_zp_clip_literature.md; dev/results/MEM
git-origin-main              WARN   differs from origin/main (682f40c); consider git pull
config-paths                 PASS   all present
pytest                       PASS   1294 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+95 other) - gated upgrade, see docs/DEPS_POLICY.md
------------------------------------------------------------------------
OVERALL: PASS
```

Note: first `--fast` after fixture commit failed on `test_no_spec_files_in_docs` because
`docs/VYVAR_SAT_DIAG_SPEC.md` (prior session, untracked) violated docs guard. Moved to
`dev/results/specs/VYVAR_SAT_DIAG_SPEC.md` (correct location); not committed.

---

## Part B -- Draft 435 restore

### B.1 Undecided items resolved

See `dev/results/CURSOR_RESULT_draft435_restore_plan.md` (updated):

- **`draft_manifest.json`**: KEEP CURRENT (v3 backfill Aug-11, manifest migration; not Aug-12 re-run)
- **`_hrd_cache/summary.json`**: KEEP CURRENT (Jul-17 regen; enrich_attempts match zip)

### B.2 Pre-restore checksum (written)

`dev/validation/anchor_435_checksums_pre_restore_20260813.json` -- 2778 files, sha256.

### B.3 Final plan -- **awaiting Milan approval**

Restore ~549 photometry output files from July zip; keep manifest, HRD cache, BO CVn LC,
calibrated/raw/FITS, report caches. Full plan in restore doc above.

### B.4 Not executed

Waiting for approval in this session.

---

## Part C -- Push (pending)

### C.1 Commit stack to push (after B complete)

| commit | summary |
|--------|---------|
| `5dd2a4d` | fix(photometry): remove per-frame MAD clip from ensemble zeropoint |
| `7eb125a` | fix(encoding): restore ASCII-only policy on ZP-clip commit artifacts |
| `d758c83` | test(qc): commit frame-62 FWHM fixture; drop draft_508 path dependency |

### C.2 / C.3

Not pushed. Awaiting explicit go-ahead after B.4 completes.

---

Tree is **not clean** (modified docs, many untracked results). **Not pushed.**
Next item after push: **SAT-DIAG** (architect recommendations awaiting Milan authorization).
