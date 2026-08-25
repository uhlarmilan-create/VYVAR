CURSOR RESULT - 2026-08-25 (SEL-GHOST-01 B-STOP-1b)

What I did
Measure-only STOP before B3-B5: clean-tree `--fast`, 516 catalog_id
control at `c592ecf` vs B1 sandbox, INV-SOURCE-STATE-01 label fix,
520 pre-optimizer WCS inventory. No push, no anchor re-cut, no B3-B5.
D1-D3 still outstanding.

HEAD `6dad937` on `main` (ahead of `origin/main` `b1f5b8c`). Live 520/516
SHAs unchanged. Session: `dev/results/session_20260825_sel_ghost_01_b1b/`
plus Rule 0.2 copy `dev/results/context/session_20260825_sel_ghost_01_b1b/`.

## Premise (Rule 0.1)

**What is compared:** (M1) `--fast` on a worktree with no `tmp/` and no
untracked STAGE-01 inputs, vs the dirty-tree PASS at `01f6f77`. (M2)
skip-solve 516 MASTERSTAR catalog_id sets: live production vs control
sandbox at `c592ecf` (pre-B1, same harness) vs B1 sandbox at `01f6f77`.
The 15-row "live minus B1 sandbox" mix is not the B1 effect. (M3)
`source_state` on the same skip-solve 520/516 sandboxes before vs after
INV-SOURCE-STATE-01. (M4) on-disk 520 `g_60_4` WCS artifacts vs the
Part A 1.44 px solve; this is an inventory, not a re-solve.

**How they differ:** B-STOP-1 `--fast` PASS used gitignored `tmp/`
4-tuple unpacks. The 516 delta 3571 vs 3584 mixed skip-solve jitter
with B1. DETECTED_P1 on injects was a labelling defect. The 520
"healthy WCS" question is whether a pre-optimizer header still exists.

## M1 -- `--fast` on a CLEAN tree (blocking; first)

Clean worktree: `.worktrees/b1b_clean`, no `tmp/`, Archive/GAIA
junctions only.

| Tree | HEAD | pytest | OVERALL |
|------|------|--------|---------|
| dirty main | `01f6f77` | 1536 passed, 32 skipped | PASS (B-STOP-1; depended on gitignored `tmp/`) |
| clean | `01f6f77` | 1523 passed, 37 skipped | **FAIL** `test_validation_gate_516_runs_and_reports` (`ModuleNotFoundError: dao_gaia_stage_01_iter4`) and `test_certificate_file_gate` (missing `final_scores.csv`) |
| clean | `58a2187` | 1528 passed, 37 skipped | **FAIL** same validation test: `KeyError: 'frame'` because `empty_positions_main.csv` was local-only (120515 bytes, never tracked) |
| clean | `b39982c` | 1531 passed, 35 skipped | **FAIL** `test_pp_kwarg01_production_scope_clean`: `dao_gaia_stage_01_iter2.py:462 numpy.load() allow_pickle` (STAGE-01 now in `src_py`, so PP-KWARG-01 sees it) |
| clean | `6dad937` | 1532 passed, 35 skipped, 1 failed | **FAIL only** `test_generated_params_md_is_fresh`: `project_root` default is the folder basename (`b1b_clean` vs committed `VYVAR`). Not a missing-file hole. The three former holes pass in this worktree (61.8 s). |
| committed tree named `VYVAR` | `6dad937` | 1537 passed, 32 skipped | **OVERALL PASS** |

Commits that make the tree self-hosting:

| Hash | What |
|------|------|
| `0684ba9` | B1 follow-up: lock 4-tuple callers (`src_py/dao_gaia_stage_01*.py` + tracked `final_scores.csv`) |
| `b39982c` | B1 follow-up: track `empty_positions_main.csv`; `g2_empty_false_accept` returns NaN if the `frame` column is absent |
| `6dad937` | B1 follow-up: drop `np.load(..., allow_pickle=False)` so PP-KWARG-01 stays clean |

This is the 2026-08-24 ASCII-gate lesson again: a dirty tree looking
green. `tmp/` 4-tuple unpacks were the first hole; the empty-sky CSV
was the second; `np.load(..., allow_pickle=False)` in newly tracked
STAGE-01 was the third. A worktree whose folder is not named `VYVAR`
fails `test_generated_params_md_is_fresh` even when the tree is
complete -- do not treat that as a missing-file FAIL.

## Gates

| Gate | Result | Evidence |
|------|--------|----------|
| G1 live 520 CSV SHA | PASS | `5ce9b07fe0490103b2e16f6fbe3b18ffc7cd987fbee8a334722cc2fd46c6a683` |
| G2 live 516 CSV SHA | PASS | `bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a` |
| G3 live 516 ePSF SHA | PASS | `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` |
| G4 `--fast` at `6dad937` | **PASS** | 1537 passed, 32 skipped, OVERALL PASS (folder `VYVAR`). Worktree named `b1b_clean` fails only `test_generated_params_md_is_fresh` (`project_root` = basename). |
| G5 no live draft writes | PASS | `draft_id=None`; sandboxes under session dir |

## M2 -- 516 catalog_id delta

### M2a two diffs

Control: skip-solve 516 MASTERSTAR at detached `c592ecf` (3-tuple lock
API; worktree tmp STAGE-01 unpacks adjusted; empty-sky dir junctioned
so the certificate gate could run). Live SHA guard held.

| Set | n catalog_id | vs |
|-----|--------------|----|
| live 516 | 3584 | production (full solve, pre-B1) |
| `c592ecf` skip-solve sandbox | 3581 | same harness as B1 sandbox, no B1 |
| `01f6f77` skip-solve sandbox | 3571 | B1 on that harness |

**Harness (not B1):** live vs `c592ecf` sandbox.

- only in live: `1496997386298488832`, `1497063283984301696`,
  `1498903316693061248`, `1500387696044768384` (4)
- only in `c592ecf`: `1504304603139151872` (1)

**B1 effect:** `01f6f77` sandbox vs `c592ecf` sandbox.

- only in `c592ecf` (removed by B1): 10 IDs, listed in M2b
- only in B1 sandbox: none

The B-STOP-1 "3571 vs 3584; +1 / -14" is harness 5 plus B1 10.

### M2b B1-effect table (10 rows)

Ensemble files read (not assumed): there is **no** `pinned_ensembles.csv`
on draft 516. Aperture LCs are `photometry/lightcurves/lightcurve_*.csv`
excluding `*_psf.csv`: **60** targets (task said 48; measured 60).
Ensemble members: `photometry/comparison_stars_per_target.csv`
(`catalog_id` where `target_catalog_id` is one of those 60). 67 unique
comp IDs. None of the 10 B1-effect IDs is a target or a comp.

x,y,source_state,vy_match_mode,vy_dao_pass from the `c592ecf` sandbox
(pre-clear). d_px = hypot(det - Gaia xy) on **live 516 WCS** from
`field_catalog_cone.csv`. `match_sep_arcsec` on those rows is
d_px times plate scale 9.774 "/px (~20-26"), not a second geometry.

| catalog_id | G | x | y | d_px | source_state | vy_match_mode | vy_dao_pass | branch | LC/ens |
|------------|---|-----|-----|------|--------------|---------------|-------------|--------|--------|
| 1485338783474459648 | 13.45 | 8.04 | 1370.92 | 2.293 | EDGE | locked | 1 | B1e pass1 lock | no |
| 1485338886553675008 | 14.15 | 8.02 | 1361.83 | 2.630 | DETECTED_P1 | locked | 1 | B1e pass1 lock | no |
| 1485339809969865472 | 11.84 | 36.83 | 1387.63 | 2.234 | DETECTED_P1 | locked | 1 | B1e pass1 lock | no |
| 1485906509431611264 | 14.05 | 29.46 | 1301.79 | 2.092 | DETECTED_P1 | locked | 1 | B1e pass1 lock | no |
| 1496736389727532416 | 13.34 | 2048.34 | 1390.27 | 2.046 | EDGE | locked | 1 | B1e pass1 lock | no |
| 1504287766867392896 | 13.95 | 4.56 | 131.36 | 2.170 | EDGE | locked | 1 | B1e pass1 lock | no |
| 1504303366188592384 | 13.58 | 21.93 | 48.60 | 2.275 | DETECTED_P1 | locked | 1 | B1e pass1 lock | no |
| 1504326559012001408 | 13.64 | 5.53 | 41.99 | 2.525 | EDGE | locked | 1 | B1e pass1 lock | no |
| 1504327005688591104 | 14.35 | 14.74 | 23.17 | 2.634 | DETECTED_P1 | locked | 1 | B1e pass1 lock | no |
| 1504327177487283456 | 14.21 | 0.45 | 16.93 | 2.648 | EDGE | locked | 1 | B1e pass1 lock | no |

All ten are **pass1**, chip-edge (x or y within ~40 px). None is pass2
born-owned. `gaia_dao_resid_px` is not on this CSV; sky sep is
`match_sep_arcsec`.

### M2c governing tolerance (file:line)

**Governing lock radius on this 516 skip-solve is 2.5 px, not config 3.0.**

- `config.py:774` `masterstar_dao_pass2_center_tol_px = 2.0` (default).
- `config.py:786` `masterstar_lock_pair_tol_px = 3.0` (default).
- Production lock uses `tolerance_overrides=det_meta["dao_gaia_derived_tol"]`
  (`pipeline.py:13653`). 516 certificate
  `dao_gaia_calibration.json` derived: `lock_pair_tol_px=2.5` and
  `pass2_center_tol_px=2.5` (`residual_p95_px=1.364`, plate scale
  9.774 "/px). `pipeline_meta` still echoes the config default 3.0;
  that stamp is not what ran.

**What B1e compares:** `lock_existing_and_leftover_assign`
(`masterstar_gaia_accounting.py:570-633`) distance is
`hypot(det_x - gx_g, det_y - gy_g)` where `gx_g, gy_g` are Gaia table
`x_gaia`/`y_gaia` (`:594-619`). `locked_pairs` xy is stored from the
detection (`:1080`) and unpacked as `_lx, _ly` (`:613`) **and never
used**. So B1e is "det vs Gaia pixel", not "det vs locked-pair xy".
For these rows `d_px_vs_locked_pair=0` (the pair xy *is* the det), so
this is not a two-geometry defect. The rows really sit 2.05-2.65 px
from Gaia.

Pass2 acceptance (`dao_pass2_try_at_position:159-167`) is vs the Gaia
**seed** at cutout time, `centroid_px <= pass2_center_tol` (derived
2.5 on this 516). These 10 are `vy_dao_pass=1`, so the "pass2 within
2.0 cannot fail a 3.0 lock" puzzle does not apply: they are not pass2,
and the lock that ran is 2.5 not 3.0.

Replay of current 4-tuple lock at 2.5 px on the `c592ecf` table vs live
WCS Gaia xy: 8 geometry-rejects. 4 of the 10 B1-effect IDs have d_px
> 2.5 on that projection; 6 sit 2.05-2.29. Production
`n_lock_geometry_reject=10` matches the 10-cid set drop, so production
`gaia_on_chip` xy (pipeline `world_to_pixel_values` on the same cone,
`:13620-13628`) puts the remaining six over 2.5. B1e is right: the
rows are off Gaia by more than the **derived** lock radius. B1e is not
the defect.

Extra replay rejects at d=3.55-5.27 px (bright G~9, pass1) keep their
catalog_id via a surviving membership/other row; they are not in the
10-cid B1-effect set.

### M2d verdict line for Milan

B1 changes 516 by **10** catalog_ids, **0** of them in LC ensembles;
`--full` expected byte-identical: **YES**.

`--full` copies frozen `masterstars_full_match.csv` and does not rerun
DAO/lock (INV-ANCHOR-00). A full-chain 516 rebuild would drop those
10 IDs from the MASTERSTAR table; aperture LC SHA is still expected
unchanged (K=0). No anchor re-cut in this task.

## M3 -- H-LABEL

**H-LABEL TRUE.** Pre-fix (B1 commit `d8c18a7` / B-STOP-1 sandbox):
`_has_det = _pd > 0.0 or "vy_dao_pass" in out.columns` labelled any
cid-bearing row DETECTED once the pass2 column existed. B-STOP-1 520
leftover ghost `1111922300852743808` and 10 G<12 injects: `peak_dao`
NaN, `vy_dao_pass` NaN, `d_px=0`, `source_state=DETECTED_P1`,
`vy_match_mode=locked`.

Overwrite path: `expand_detection_to_catalog_membership` sets
`vy_match_mode="catalog_membership"` (`masterstar_gaia_accounting.py:969`).
`enrich_masterstar_gaia_complete` runs after expand (`pipeline.py:13641`)
and the lock pass wrote `vy_modes[i]="locked"` onto every det within
lock_tol of Gaia xy, including injects at d=0.

Fix commit **`58a2187` INV-SOURCE-STATE-01: detected means detected**:

- `_row_is_dao_detected` (`:540-558`): DETECTED_Pn only if this row's
  `vy_dao_pass` in {1,2} AND `peak_dao` > 0.
- else census / `catalog_membership` / `CATALOG_ONLY` (`:1288-1298`).
- lock does not overwrite existing `vy_match_mode=="catalog_membership"`
  (`:1129-1131`).
- test `dev/tests/test_inv_source_state_01.py`; invariant in
  `docs/VYVAR_INVARIANTS.md`.

Post-fix skip-solve (HEAD `58a2187`, session `sandbox_520` / `sandbox_516`):

| | 520 `source_state` | 516 `source_state` |
|--|-------------------|-------------------|
| DETECTED_P1 | 7 | 2210 |
| DETECTED_P2 | 0 | 398 |
| catalog_membership | 106 | 963 |
| DAO_ONLY | 678 | 35 |

Ghost `1111922300852743808`: now `catalog_membership` /
`vy_match_mode=catalog_membership` / peak NaN / pass NaN. 10/11 G<12
injects the same; `1111749157833870208` remains `DETECTED_P1` /
`locked` / `peak_dao=1869` / `vy_dao_pass=1` (the one real gated
match). 516 catalog_id set **did not move**: 3571 vs B-STOP-1 sandbox
3571, symmetric difference empty. Labels are not identity.

## M4 -- 520 healthy WCS inventory (no seed run)

**M4a.** No pre-optimizer WCS is stored. The live `g_60_4/MASTERSTAR.fits`
header still *mentions* the 1.44 px solve in HISTORY, then Grip overwrote
the WCS cards.

| path | exists | sha256 |
|------|--------|--------|
| `Archive/Drafts/draft_000520/platesolve/g_60_4/MASTERSTAR.fits` | yes | `2862552040ec30f98481521a512497f7e86d2a0a0203185884a2d3f1ef07c997` |
| `.../g_60_4/masterstars_full_match.csv` | yes | `5ce9b07fe0490103b2e16f6fbe3b18ffc7cd987fbee8a334722cc2fd46c6a683` |
| `.../g_60_4/dao_gaia_calibration.json` | yes | `1ce66e277a0ef288adbbf22f9b6a8f47f14bec0bf95aa539532e197d10f5ba91` |
| `.../g_60_4/MASTERSTAR.fits.bak` | **no** | |
| `.../g_60_4/*.wcs` | **no** | |
| backup WCS cards (ORIG/BAK/OLD) | **no** | HISTORY only |
| sibling `i_70_4/MASTERSTAR.fits` | yes | `9e25117badf78bc1ae6a970ee63fc3295fcd56f810d8892f9e704724967f4850` |
| sibling `r_60_4/MASTERSTAR.fits` | yes | `01178cac2a0ea0648d881e070f0f755968c9522c380d0a32e363aaacac62cce6` |
| sibling `z_90_4/MASTERSTAR.fits` | yes | `da5c86a93e2cb764d5ec0bd1c5e53ffacdd635a3e23130d0231596661d4b1d92` |
| sibling `i_70_4/dao_gaia_calibration.json` | yes | `9d048b4cb444f6b1d3be45e9fe7648625ebaaf56a6cde46954f5ee1689bcd62f` |
| sibling `r_60_4/dao_gaia_calibration.json` | yes | `0d35cb765bc630eec957e3bdd086167e421ab2e3c5a947da46bbcad2df4aca4a` |
| sibling `z_90_4/dao_gaia_calibration.json` | **no** | |

HISTORY (g header): `SIP rejected by RMS guard (lin=1.442 sip=2.144)`
then `Grip1 SIP3 rms_px lin=80.997 -> sip=84.665` (and later 81.267 /
85.123). Current PC matrix is the destroyed WCS. `VY_WCSRT=True` is
not a backup.

Certificate `built_utc=2026-08-24T18:45:27`, status **PASS**,
`residual_p95_px=2.095`. That is a DAO-Gaia diagnostic on the then-
healthy match, **not** a stored WCS. Validation `completeness_pct`
on the 516-hand MASTERSTAR row in that file is 95.78%; Light_148
`g1_eye_seed_le145=0.9735` is the ~97% STAGE-01 figure (516 sandbox
hand scores copied onto the 520 cert), not a 520 pixel WCS.

**M4b.** Not run. Nothing to seed from. Sibling FITS exist (M4a) and
are MULTIFILTER-WCS-01, not this task.

**M4c.** `retry_520_sandbox.py` failure:
`VYVAR solver: nenasiel som zhodny trojuholnik`. Infolog
`session_20260825_sel_ghost_01_b/infolog_20260825_144309.txt`:
caller FOV 0.6 deg vs optics diag 19.13 deg (F=200 mm); triangle
filter uses 15.511 "/px; solve starts at Scale=0.566 "/px (F=5480 mm);
Gaia 535 stars, triangle mag_cap 13.5 -> **n_cat_tri=113**; DAO
fallback sigma 4.50 -> 1.00; adaptive detection cap **33**
(`k=0.08`, bounds [250,800] then capped by n_cat_tri). STOP here.

## Commits this task (not pushed)

| Hash | What |
|------|------|
| `0684ba9` | B1 follow-up: lock 4-tuple callers |
| `58a2187` | INV-SOURCE-STATE-01: detected means detected |
| `b39982c` | B1 follow-up: track empty-sky CSV for --fast |
| `6dad937` | B1 follow-up: STAGE-01 np.load kwargs for PP-KWARG-01 |

## Errors

Clean `--fast` at `01f6f77`, `58a2187`, `b39982c` FAIL (M1 holes).
Worktree `b1b_clean` at `6dad937` fails only PARAMS.md basename.
c592ecf control needed 3-tuple STAGE-01 copies plus the empty-sky CSV.
No live-draft writes.

## Files changed

`src_py/masterstar_gaia_accounting.py`, `src_py/dao_gaia_stage_01.py`,
`src_py/dao_gaia_stage_01_iter2.py`, `src_py/dao_gaia_stage_01_iter3.py`,
`src_py/dao_gaia_stage_01_iter4.py`, `dev/tests/test_inv_source_state_01.py`,
`docs/VYVAR_INVARIANTS.md`,
`dev/results/context/session_20260819_daostage01_iter4/final_scores.csv`,
`dev/results/context/session_20260819_msgaia01/empty_positions_main.csv`,
this result, STATE/JOURNAL, session JSON under
`dev/results/session_20260825_sel_ghost_01_b1b/` and context copy.

## Recurrence

Dirty-tree `--fast` PASS twice in two days (`tmp/` 4-tuple, then
untracked empty-sky CSV). Same shape as the ASCII-gate. Clean-tree
`--fast` is the gate, not the working copy.

## Docs impact

STATE/JOURNAL one-liners. INV-SOURCE-STATE-01 already in INVARIANTS
(`58a2187`). No DECISIONS (no D1-D3). No ROADMAP re-cut. No push.
