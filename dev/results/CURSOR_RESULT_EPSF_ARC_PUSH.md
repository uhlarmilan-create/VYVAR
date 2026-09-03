CURSOR RESULT - 2026-08-23 19:40 UTC+2

What I did
Completed EPSF-ARC CLOSE: fixed ASCII policy blockers in arc result docs (amended SESSION-CLOSE), ran `--fast` gate at tip, pushed 15 commits to `origin/main`, verified hash parity, wrote this report (not committed per task).

## Output / findings

### Pre-push ASCII fix (amended `9459af2` -> `b1af049`)
- Migrated smart punctuation in 5 `CURSOR_RESULT_EPSF_*.md` files via `ascii_migrate.py`
- Replaced U+FFFD in `CURSOR_RESULT_EPSF_VALID_02_S1S4.md`
- Removed UTF-16 stdout captures from commit: `session_20260822_epsf_valid_02_r1r4/fast_baseline_stdout.txt`, `r4_stdout.txt`
- `pytest dev/tests/test_ascii_policy.py::test_tracked_text_files_are_ascii` -> PASS

### Pre-push gate (`session_baseline_check.py --fast` @ `b1af049`)
```
OVERALL: PASS
pytest                       PASS   1512 passed, 32 skipped
manifest-db-parity           PASS   draft_id=516
db-quick-check               WARN   WAIVED (expected)
git-head                     PASS   b1af049
git-staged                   PASS   none
```
Runtime ~11 min. DB malformed warnings during MASTER_SOURCES retirement (waived path).

### Push
```
git push origin main
   ea4e593..b1af049  main -> main
```

### Hash verification
```
git rev-parse HEAD
b1af0493dde53590850541376dceb233e5da0f46

git rev-parse origin/main
b1af0493dde53590850541376dceb233e5da0f46
```
MATCH.

### Commits pushed (`0e1a484` .. `b1af049`, 15 total)

| Hash | Subject |
|------|---------|
| `0e1a484` | EPSF-VALID-02 G0: db-quick-check waiver + DB-RETIRE-01 decision docs |
| `35096e0` | EPSF-VALID-02 F1: science-set dashboard scope and PSF pct display fix |
| `748e4bf` | EPSF-VALID-02 F2: fail-loud per-frame PSF accounting (INV-PSF-FRAME-01) |
| `8202c7c` | EPSF-VALID-02 F3: PSF measurement set uses science set (333 on 516) |
| `93b3194` | EPSF-VALID-02 F4: gated build-star selection and iteration failure curve |
| `57046dd` | EPSF-VALID-02 F5: science-light frame enumerator for ePSF accounting |
| `c218921` | EPSF-VALID-02 F6: PSF-only sidecar merge and INV-PSF-ADDITIVE-01 |
| `2ba3d58` | EPSF-VALID-02 F6: register INV-PSF-ADDITIVE-01 in WIRED_INV_IDS |
| `f97615a` | Add production ePSF edge-star build guard (EPSF-VALID-02 S6) |
| `8b98156` | EPSF-VALID-02 S6: gated ePSF swap close, docs, and merge harness |
| `777f10e` | Fix ASCII encoding in EPSF-VALID-02 S5/S6 result docs |
| `8f41031` | Update EPSF-VALID-02 S6 result with gate evidence |
| `086fb44` | fix(epsf-ui): show full science table and report gated epoch drops |
| `a0319fe` | feat(psf): FD-A full CCD variance model for PSF fit weights |
| `ad19e14` | docs(epsf): FD-A acceptance results and EPSF-SHAPE-01 roadmap |
| `b1af049` | SESSION-CLOSE-20260823: ePSF arc close (VALID-02 + BRIGHT-01) |

Base before push: `ea4e593` (origin/main).

### Arc close state (SESSION-CLOSE)
- EPSF-VALID-02: CLOSED
- EPSF-BRIGHT-01: CLOSED (Phase 1 UI + Phase 3 FD-A)
- EPSF-SHAPE-01 + EXPORT-PARITY-01: OPEN HIGH (documented in STATE/JOURNAL)

## Errors (if any)
None blocking push. DB malformed warnings during gate (waived db-quick-check).

## Files changed
- Amended `b1af049` (ASCII fixes in 5 result MDs; removed 2 UTF-16 stdout files from tree)
- This report: `dev/results/CURSOR_RESULT_EPSF_ARC_PUSH.md` (untracked; lands in next push/session)
