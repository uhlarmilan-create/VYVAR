# CURSOR RESULT - MERGE-MAIN-01

Date: 2026-09-01. Branch: consolidate-01. English. ASCII.
Base: origin/consolidate-01 @ 620d2bf. Implementer: Cursor. Architect: Claude.
No ePSF gate (nothing on the ePSF graph changed). Live draft 516 was not written.
main is not pushed by Cursor (PUSH_AUTH protocol, section 4).

## What I did

1. Fixed the P2-6 leftover unpack in dao_gaia_stage_01_iter4.py overlay loop.
2. Smoke test for that overlay branch. G1 --clean then showed a gitignored
   tmp/dao_gaia_stage_01_iter4.py shadowing src_py; pinned the import.
3. ROADMAP: EXPORT-PARITY-01 moved to CLOSED this arc; FRAME-QC-PARITY remaining
   scope annotated (INV-FRAME-QC-01 already landed). No other ROADMAP rows.
4. G1 --fast --clean, G2 --full aperture, G4 live 516. G-EPSF skipped.
5. Pushed consolidate-01. MERGE_SHA is the tip after this report commit
   (printed in the Cursor hand-off; this file does not self-hash).

## Fix diff

`decompose_holes_le13` (dao_gaia_common.py:333) returns one DataFrame
(true_miss G<=13 rows only). Overlay PNGs (`render_overlay_final`) never used
`summ`. The only consumer was `holes_le13_decompose_final.csv` on MASTERSTAR,
built from an internal `summary` count dict that is not in the returned
DataFrame (EDGE/BLENDED/SATURATED would be missing if derived from holes).
Dropped the unpack and that CSV write. Kept `holes_le13_final.csv`.

```
-        holes, summ = decompose_holes_le13(census, res.gaia_le16, res.data0, fwhm)
+        holes = decompose_holes_le13(census, res.gaia_le16, res.data0, fwhm)
         if frame_label == "MASTERSTAR":
             holes.to_csv(ctx / "holes_le13_final.csv", index=False)
-            summ.to_csv(ctx / "holes_le13_decompose_final.csv", index=False)
```

Commit: 8667149.

## Smoke test

`dev/tests/test_dao_gaia_iter4_overlay.py` loads `src_py/dao_gaia_stage_01_iter4.py`
by file path (`importlib.util.spec_from_file_location`), asserts `__file__` is
the src_py path, mocks `load_frame` / `run_frame_i6_i7` / `render_overlay_final`
/ `FRAMES` / `EMPTY_SKY_CSV`, calls `main()`, asserts `holes_le13_final.csv`
exists and `holes_le13_decompose_final.csv` does not. That is the TypeError
class of `holes, summ = <DataFrame>`.

Follow-up (commit 58dcccf): G1 pytest on the working tree first failed because
gitignored `tmp/dao_gaia_stage_01_iter4.py` (stale sandbox, still unpacks two
values) sat on `sys.path` after `test_pinned_ensembles.py`. Isolation of the
overlay test passed; the combined suite did not.

- `dao_gaia_stage_validation._import_iter4` now strips `tmp` from `sys.path`,
  inserts `src_py` first, pops a cached `dao_gaia_stage_01_iter4` module, then
  imports. Does not load gitignored tmp copies.
- Overlay test loads src_py by path so tmp cannot shadow it.

tmp/ was not deleted (gitignored disposable scratch). Combined rerun
`test_dao_gaia_calibration.py` + overlay + `test_dao_gaia_xfer_01.py`:
13 passed.

## ROADMAP diff

Commit: 1f31843. No other ROADMAP rows changed.

- OPEN **EXPORT-PARITY-01** removed (was "standing two-path defect").
- CLOSED this arc: **EXPORT-PARITY-01** -- CLOSED this arc (v2, d6c84e0:
  one production entry, G7 --parity permanent).
- OPEN **FRAME-QC-PARITY** annotated: remaining Layer A log honesty + n_stars
  outlier gate (frame 29, 263 vs ~100). Landed INV-FRAME-QC-01: `_dqf` None
  raises; provenance stamp (`src_py/night_run.py` ~638).

SEL-GHOST-01 stays OPEN until Milan pushes main and
`git ls-remote origin main` equals MERGE_SHA. Standing line still:
`origin/main` stays `7c086e8` until Milan writes PUSH_AUTH.

## Gates

| gate | status | detail |
| --- | --- | --- |
| G1 --fast --clean | PASS at 58dcccf | 1612 passed, 32 skipped. clean-tree PASS. db-quick-check WARN waived. Log: g1.txt |
| G2 --full aperture | PASS at 58dcccf | era04_aperture d55fcc9d n=53 / ext cc8b532e n=157. Pipeline 1363s. Log: g2_full.txt |
| G4 live 516 | PASS after G2 | csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d |
| G-EPSF | SKIP | nothing on the ePSF graph changed |

G4 path: Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/
(masterstars_full_match.csv, MASTERSTAR.fits, masterstar_epsf.fits). Not written.

Full G4 SHAs (unchanged vs P2):

- csv `bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a`
- fits `13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345`
- epsf `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20`

Ledger VL-COUNTERS-ZERO / VL-ANCHOR-WCSINV last_verified commit stamp updated
to 58dcccf by --full (same ritual as 01C/01D).

Pytest 1612 = P2 1611 + overlay smoke test.

## MERGE_SHA / ancestry / Milan commands

MERGE_SHA is `git rev-parse HEAD` of consolidate-01 after this report commit
and the Cursor `git push origin consolidate-01:consolidate-01`. The Cursor
hand-off prints the full 40-char SHA.

Ancestry (must succeed; fast-forward is valid; consolidate-01 contains the
sel-ghost-01 history):

```
git merge-base --is-ancestor 7c086e8 MERGE_SHA
```

### Verbatim for Milan (two steps)

1. create `dev/PUSH_AUTH_main_20260901.txt` containing MERGE_SHA
   (full 40-char SHA, one line)
2. `git push origin MERGE_SHA:main`

Banned forever: `git push origin HEAD`, bare `git push`.
The pre-push guard refuses main without the PUSH_AUTH file.

Cursor does not create PUSH_AUTH and does not push main.
After Milan pushes: verify `git ls-remote origin main` == MERGE_SHA,
then one follow-up commit on consolidate-01 (then identical to main):
update the ROADMAP standing line and close SEL-GHOST-01 as MERGED.

## Files changed (this arc, 620d2bf..)

- 8667149 MERGE-MAIN-01: fix iter4 overlay unpack of decompose_holes_le13.
- 1f31843 MERGE-MAIN-01: close EXPORT-PARITY-01; annotate FRAME-QC-PARITY remaining scope.
- 58dcccf MERGE-MAIN-01: pin iter4 import to src_py so gitignored tmp/ cannot shadow production.
- (this commit) session REPORT, G1/G2 logs, ledger last_verified at 58dcccf.

## Errors

None that blocked the task. First G1 at 1f31843 FAIL (tmp shadow) is the
58dcccf follow-up, not a remaining defect.
